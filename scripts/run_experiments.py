#!/usr/bin/env python3
"""Cross-setting reliability and fairness audit for BRSET -> mBRSET.

The script writes aggregate tables and figures only. It never writes joined
image-level rows, identifiers, predictions, fitted models, or embeddings.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


BACKBONES: Mapping[str, Mapping[str, str]] = {
    "vits16": {
        "brset": "Embeddings_brset_dinov3_vits16.csv",
        "mbrset": "Embeddings_mbrset_dinov3_vits16.csv",
    },
    "convnext_tiny": {
        "brset": "Embeddings_brset_dinov3_convnext_tiny.csv",
        "mbrset": "Embeddings_mbrset_dinov3_convnext_tiny.csv",
    },
}

SOURCE_C_VALUES = (0.0001, 0.001, 0.01, 0.1, 1.0, 10.0)
METRIC_COLUMNS = (
    "auroc",
    "auprc",
    "brier",
    "ece",
    "calibration_intercept",
    "calibration_slope",
    "sensitivity",
    "specificity",
    "ppv",
    "fpr",
    "fnr",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=20260902)
    parser.add_argument("--bootstrap-replicates", type=int, default=500)
    parser.add_argument("--target-calibration-fraction", type=float, default=0.10)
    parser.add_argument("--minimum-group-size", type=int, default=50)
    parser.add_argument("--minimum-group-events", type=int, default=10)
    return parser.parse_args()


def normalize_identifier(values: pd.Series) -> pd.Series:
    return (
        values.astype("string")
        .str.strip()
        .str.replace(r"^.*[/\\]", "", regex=True)
        .str.replace(r"\.[^.]+$", "", regex=True)
        .str.casefold()
    )


def binary_from_mixed(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    text = values.astype("string").str.strip().str.casefold()
    mapped = text.map(
        {
            "yes": 1.0,
            "y": 1.0,
            "true": 1.0,
            "present": 1.0,
            "referable": 1.0,
            "no": 0.0,
            "n": 0.0,
            "false": 0.0,
            "absent": 0.0,
            "non-referable": 0.0,
        }
    )
    return numeric.fillna(mapped)


def load_metadata(raw_dir: Path, dataset: str) -> pd.DataFrame:
    if dataset == "brset":
        path = raw_dir / "labels_brset.csv"
        metadata = pd.read_csv(path, low_memory=False)
        required = {"image_id", "patient_id", "patient_age", "patient_sex", "DR_ICDR"}
        missing = required - set(metadata.columns)
        if missing:
            raise ValueError(f"BRSET metadata lacks columns: {sorted(missing)}")
        frame = pd.DataFrame(
            {
                "image_key": normalize_identifier(metadata["image_id"]),
                "patient_key": metadata["patient_id"].astype("string"),
                "age": pd.to_numeric(metadata["patient_age"], errors="coerce"),
                "sex_raw": pd.to_numeric(metadata["patient_sex"], errors="coerce"),
                "icdr": pd.to_numeric(metadata["DR_ICDR"], errors="coerce"),
            }
        )
        frame["sex"] = frame["sex_raw"].map({1.0: "Male", 2.0: "Female"})
    elif dataset == "mbrset":
        path = raw_dir / "labels_mbrset.csv"
        metadata = pd.read_csv(path, low_memory=False)
        required = {"file", "patient", "age", "sex", "final_icdr"}
        missing = required - set(metadata.columns)
        if missing:
            raise ValueError(f"mBRSET metadata lacks columns: {sorted(missing)}")
        frame = pd.DataFrame(
            {
                "image_key": normalize_identifier(metadata["file"]),
                "patient_key": metadata["patient"].astype("string"),
                "age": pd.to_numeric(metadata["age"], errors="coerce"),
                "sex_raw": pd.to_numeric(metadata["sex"], errors="coerce"),
                "icdr": pd.to_numeric(metadata["final_icdr"], errors="coerce"),
            }
        )
        frame["sex"] = frame["sex_raw"].map({0.0: "Female", 1.0: "Male"})
        if "insurance" in metadata.columns:
            insurance = binary_from_mixed(metadata["insurance"])
            frame["insurance"] = insurance.map({0.0: "No", 1.0: "Yes"})
        if "educational_level" in metadata.columns:
            education = pd.to_numeric(metadata["educational_level"], errors="coerce")
            frame["education"] = pd.cut(
                education,
                bins=[-np.inf, 2, 5, np.inf],
                labels=["Primary incomplete or less", "Primary to secondary", "Tertiary"],
            ).astype("string")
    else:
        raise ValueError(f"unknown dataset: {dataset}")

    frame["target"] = np.where(frame["icdr"].notna(), (frame["icdr"] >= 2).astype(float), np.nan)
    frame["age_group"] = pd.cut(
        frame["age"],
        bins=[-np.inf, 49, 64, np.inf],
        labels=["<50", "50-64", "65+"],
    ).astype("string")
    frame["sex_age"] = frame["sex"].astype("string") + " | " + frame["age_group"]

    if frame["image_key"].duplicated().any():
        raise ValueError(f"{dataset} metadata contains duplicate image identifiers")
    return frame.drop(columns=["sex_raw"])


def load_dataset(raw_dir: Path, dataset: str, embedding_file: str) -> Tuple[pd.DataFrame, np.ndarray]:
    metadata = load_metadata(raw_dir, dataset)
    embeddings = pd.read_csv(raw_dir / embedding_file)
    identifier = "image_id" if dataset == "brset" else "file"
    if identifier not in embeddings.columns:
        raise ValueError(f"{embedding_file} lacks {identifier!r}")
    keys = normalize_identifier(embeddings[identifier])
    if keys.duplicated().any():
        raise ValueError(f"{embedding_file} contains duplicate image identifiers")

    feature_frame = embeddings.drop(columns=[identifier]).apply(pd.to_numeric, errors="coerce")
    feature_frame.columns = [f"f_{index}" for index in range(feature_frame.shape[1])]
    feature_frame.insert(0, "image_key", keys)
    merged = metadata.merge(feature_frame, on="image_key", how="inner", validate="one_to_one")
    coverage = len(merged) / max(1, len(feature_frame))
    if coverage < 0.98:
        raise ValueError(
            f"only {coverage:.1%} of {dataset} embeddings matched metadata; check dataset versions"
        )

    merged = merged.loc[merged["target"].notna() & merged["patient_key"].notna()].copy()
    feature_columns = [column for column in merged.columns if column.startswith("f_")]
    x = merged[feature_columns].to_numpy(dtype=np.float32, copy=True)
    finite_rows = np.isfinite(x).all(axis=1)
    if not finite_rows.all():
        merged = merged.loc[finite_rows].copy()
        x = x[finite_rows]
    merged.reset_index(drop=True, inplace=True)
    return merged, x


def split_source_patients(frame: pd.DataFrame, seed: int) -> Dict[str, np.ndarray]:
    patient_table = frame.groupby("patient_key", sort=False)["target"].max().reset_index()
    train_patients, holdout_patients = train_test_split(
        patient_table["patient_key"],
        test_size=0.40,
        random_state=seed,
        stratify=patient_table["target"],
    )
    holdout_table = patient_table[patient_table["patient_key"].isin(holdout_patients)]
    validation_patients, test_patients = train_test_split(
        holdout_table["patient_key"],
        test_size=0.50,
        random_state=seed + 1,
        stratify=holdout_table["target"],
    )
    return {
        "train": frame["patient_key"].isin(train_patients).to_numpy(),
        "validation": frame["patient_key"].isin(validation_patients).to_numpy(),
        "internal_test": frame["patient_key"].isin(test_patients).to_numpy(),
    }


def fit_linear_probe(x: np.ndarray, y: np.ndarray, c_value: float) -> Pipeline:
    model = Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    C=c_value,
                    penalty="l2",
                    solver="lbfgs",
                    max_iter=3000,
                    random_state=0,
                ),
            ),
        ]
    )
    model.fit(x, y)
    return model


def choose_c(
    x: np.ndarray, y: np.ndarray, train_mask: np.ndarray, validation_mask: np.ndarray
) -> Tuple[float, pd.DataFrame]:
    rows: List[Dict[str, float]] = []
    for c_value in SOURCE_C_VALUES:
        model = fit_linear_probe(x[train_mask], y[train_mask], c_value)
        probability = model.predict_proba(x[validation_mask])[:, 1]
        rows.append({"C": c_value, "validation_auroc": roc_auc_score(y[validation_mask], probability)})
    scores = pd.DataFrame(rows)
    best = scores.sort_values(["validation_auroc", "C"], ascending=[False, True]).iloc[0]
    return float(best["C"]), scores


def clipped_logit(probability: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(probability, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def fit_platt(probability: np.ndarray, y: np.ndarray) -> LogisticRegression:
    calibrator = LogisticRegression(penalty=None, solver="lbfgs", max_iter=2000)
    calibrator.fit(clipped_logit(probability).reshape(-1, 1), y)
    return calibrator


def apply_platt(calibrator: LogisticRegression, probability: np.ndarray) -> np.ndarray:
    return calibrator.predict_proba(clipped_logit(probability).reshape(-1, 1))[:, 1]


def threshold_at_sensitivity(y: np.ndarray, probability: np.ndarray, sensitivity: float = 0.90) -> float:
    positive_scores = np.asarray(probability)[np.asarray(y) == 1]
    if len(positive_scores) == 0:
        raise ValueError("threshold cohort contains no positive cases")
    try:
        return float(np.quantile(positive_scores, 1 - sensitivity, method="lower"))
    except TypeError:  # NumPy < 1.22
        return float(np.quantile(positive_scores, 1 - sensitivity, interpolation="lower"))


def adaptive_ece(y: np.ndarray, probability: np.ndarray, bins: int = 10) -> float:
    y = np.asarray(y, dtype=float)
    probability = np.asarray(probability, dtype=float)
    if len(y) == 0:
        return math.nan
    order = np.argsort(probability)
    chunks = np.array_split(order, min(bins, len(order)))
    total = float(len(y))
    return float(
        sum(
            len(indices) / total
            * abs(float(np.mean(probability[indices])) - float(np.mean(y[indices])))
            for indices in chunks
            if len(indices)
        )
    )


def calibration_parameters(y: np.ndarray, probability: np.ndarray) -> Tuple[float, float]:
    y = np.asarray(y, dtype=int)
    if len(np.unique(y)) < 2:
        return math.nan, math.nan
    model = LogisticRegression(penalty=None, solver="lbfgs", max_iter=2000)
    model.fit(clipped_logit(probability).reshape(-1, 1), y)
    return float(model.intercept_[0]), float(model.coef_[0, 0])


def safe_metric(function, y: np.ndarray, p: np.ndarray) -> float:
    try:
        return float(function(y, p))
    except ValueError:
        return math.nan


def metrics(y: np.ndarray, probability: np.ndarray, threshold: float) -> Dict[str, float]:
    y = np.asarray(y, dtype=int)
    probability = np.asarray(probability, dtype=float)
    predicted = (probability >= threshold).astype(int)
    intercept, slope = calibration_parameters(y, probability)
    negative = y == 0
    positive = y == 1
    specificity = float(np.mean(predicted[negative] == 0)) if negative.any() else math.nan
    sensitivity = float(np.mean(predicted[positive] == 1)) if positive.any() else math.nan
    return {
        "n_images": float(len(y)),
        "n_events": float(y.sum()),
        "prevalence": float(y.mean()) if len(y) else math.nan,
        "auroc": safe_metric(roc_auc_score, y, probability),
        "auprc": safe_metric(average_precision_score, y, probability),
        "brier": safe_metric(brier_score_loss, y, probability),
        "ece": adaptive_ece(y, probability),
        "calibration_intercept": intercept,
        "calibration_slope": slope,
        "threshold": float(threshold),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "ppv": safe_metric(lambda a, b: precision_score(a, b, zero_division=0), y, predicted),
        "fpr": 1 - specificity if np.isfinite(specificity) else math.nan,
        "fnr": 1 - sensitivity if np.isfinite(sensitivity) else math.nan,
    }


def choose_target_calibration_patients(
    frame: pd.DataFrame, fraction: float, seed: int
) -> Tuple[np.ndarray, np.ndarray]:
    patient_table = frame.groupby("patient_key", sort=False)["target"].max().reset_index()
    calibration_patients, evaluation_patients = train_test_split(
        patient_table["patient_key"],
        train_size=fraction,
        random_state=seed,
        stratify=patient_table["target"],
    )
    return (
        frame["patient_key"].isin(calibration_patients).to_numpy(),
        frame["patient_key"].isin(evaluation_patients).to_numpy(),
    )


def aggregate_group_metrics(
    frame: pd.DataFrame,
    probability: np.ndarray,
    threshold: float,
    backbone: str,
    method: str,
    setting: str,
    minimum_size: int,
    minimum_events: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    group_rows: List[Dict[str, object]] = []
    gap_rows: List[Dict[str, object]] = []
    groupings = ["sex", "age_group", "sex_age"]
    if setting.startswith("external"):
        groupings.extend([column for column in ("insurance", "education") if column in frame.columns])

    for attribute in groupings:
        valid_rows_for_gap: List[Dict[str, object]] = []
        for label, indices in frame.groupby(attribute, dropna=True).indices.items():
            index_array = np.asarray(indices, dtype=int)
            y_group = frame.iloc[index_array]["target"].to_numpy(dtype=int)
            n_events = int(y_group.sum())
            n_nonevents = int(len(y_group) - n_events)
            eligible = (
                len(y_group) >= minimum_size
                and n_events >= minimum_events
                and n_nonevents >= minimum_events
            )
            row: Dict[str, object] = {
                "backbone": backbone,
                "method": method,
                "setting": setting,
                "attribute": attribute,
                "group": str(label),
                "n_patients": int(frame.iloc[index_array]["patient_key"].nunique()),
                "n_events": n_events,
                "n_nonevents": n_nonevents,
                "threshold": float(threshold),
                "eligible_for_gap": bool(eligible),
            }
            row.update(metrics(y_group, probability[index_array], threshold))
            # ``metrics`` exposes generic floating-point count fields for the
            # image-level evaluation tables.  Restore the exact subgroup
            # support fields here so this aggregate table has an integer
            # count schema and the declared threshold cannot be overwritten.
            row["n_events"] = n_events
            row["n_nonevents"] = n_nonevents
            row["threshold"] = float(threshold)
            if n_events:
                true_positives = int(
                    np.sum(np.asarray(probability[index_array])[y_group == 1] >= threshold)
                )
                estimate = true_positives / n_events
                z = 1.959963984540054
                denominator = 1.0 + z**2 / n_events
                centre = (estimate + z**2 / (2.0 * n_events)) / denominator
                half_width = (
                    z
                    * math.sqrt(
                        estimate * (1.0 - estimate) / n_events
                        + z**2 / (4.0 * n_events**2)
                    )
                    / denominator
                )
                row["sensitivity_ci_low"] = max(0.0, centre - half_width)
                row["sensitivity_ci_high"] = min(1.0, centre + half_width)
            else:
                row["sensitivity_ci_low"] = math.nan
                row["sensitivity_ci_high"] = math.nan
            group_rows.append(row)
            if eligible:
                valid_rows_for_gap.append(row)

        for metric_name in ("auroc", "auprc", "brier", "ece", "sensitivity", "fpr", "fnr", "ppv"):
            values = np.asarray([row[metric_name] for row in valid_rows_for_gap], dtype=float)
            finite = values[np.isfinite(values)]
            if len(finite) >= 2:
                gap_rows.append(
                    {
                        "backbone": backbone,
                        "method": method,
                        "setting": setting,
                        "attribute": attribute,
                        "metric": metric_name,
                        "gap_max_minus_min": float(finite.max() - finite.min()),
                        "eligible_groups": int(len(finite)),
                    }
                )
    return pd.DataFrame(group_rows), pd.DataFrame(gap_rows)


def cluster_bootstrap_ci(
    frame: pd.DataFrame,
    probability: np.ndarray,
    threshold: float,
    replicates: int,
    seed: int,
) -> pd.DataFrame:
    patient_to_indices = {
        patient: np.asarray(indices, dtype=int)
        for patient, indices in frame.groupby("patient_key", sort=False).indices.items()
    }
    patients = np.asarray(list(patient_to_indices), dtype=object)
    rng = np.random.default_rng(seed)
    estimates: Dict[str, List[float]] = {name: [] for name in METRIC_COLUMNS}
    for _ in range(replicates):
        sampled = rng.choice(patients, size=len(patients), replace=True)
        indices = np.concatenate([patient_to_indices[patient] for patient in sampled])
        result = metrics(
            frame.iloc[indices]["target"].to_numpy(dtype=int), probability[indices], threshold
        )
        for name in METRIC_COLUMNS:
            estimates[name].append(float(result[name]))

    rows: List[Dict[str, float]] = []
    for name, values in estimates.items():
        finite = np.asarray(values, dtype=float)
        finite = finite[np.isfinite(finite)]
        rows.append(
            {
                "metric": name,
                "lower_95": float(np.quantile(finite, 0.025)) if len(finite) else math.nan,
                "upper_95": float(np.quantile(finite, 0.975)) if len(finite) else math.nan,
                "bootstrap_valid": int(len(finite)),
            }
        )
    return pd.DataFrame(rows)


def group_metric_value(y: np.ndarray, probability: np.ndarray, threshold: float, name: str) -> float:
    y = np.asarray(y, dtype=int)
    predicted = (np.asarray(probability) >= threshold).astype(int)
    if name == "auroc":
        return safe_metric(roc_auc_score, y, probability)
    if name == "sensitivity":
        return float(np.mean(predicted[y == 1] == 1)) if np.any(y == 1) else math.nan
    if name == "fpr":
        return float(np.mean(predicted[y == 0] == 1)) if np.any(y == 0) else math.nan
    raise ValueError(f"unsupported group metric: {name}")


def cluster_bootstrap_gap_ci(
    frame: pd.DataFrame,
    probability: np.ndarray,
    threshold: float,
    replicates: int,
    seed: int,
    minimum_size: int,
    minimum_events: int,
) -> pd.DataFrame:
    patient_to_indices = {
        patient: np.asarray(indices, dtype=int)
        for patient, indices in frame.groupby("patient_key", sort=False).indices.items()
    }
    patients = np.asarray(list(patient_to_indices), dtype=object)
    attributes = ["sex", "age_group", "sex_age"]
    attributes.extend([column for column in ("insurance", "education") if column in frame.columns])
    metric_names = ("auroc", "sensitivity", "fpr")
    samples: Dict[Tuple[str, str], List[float]] = {
        (attribute, metric_name): []
        for attribute in attributes
        for metric_name in metric_names
    }
    rng = np.random.default_rng(seed)
    for _ in range(replicates):
        sampled = rng.choice(patients, size=len(patients), replace=True)
        indices = np.concatenate([patient_to_indices[patient] for patient in sampled])
        sampled_frame = frame.iloc[indices].reset_index(drop=True)
        sampled_probability = probability[indices]
        for attribute in attributes:
            eligible_groups: List[Tuple[np.ndarray, np.ndarray]] = []
            for _, group_indices in sampled_frame.groupby(attribute, dropna=True).indices.items():
                group_indices = np.asarray(group_indices, dtype=int)
                y_group = sampled_frame.iloc[group_indices]["target"].to_numpy(dtype=int)
                events = int(y_group.sum())
                if (
                    len(group_indices) >= minimum_size
                    and events >= minimum_events
                    and len(group_indices) - events >= minimum_events
                ):
                    eligible_groups.append((group_indices, y_group))
            for metric_name in metric_names:
                values = [
                    group_metric_value(
                        y_group,
                        sampled_probability[group_indices],
                        threshold,
                        metric_name,
                    )
                    for group_indices, y_group in eligible_groups
                ]
                finite = np.asarray(values, dtype=float)
                finite = finite[np.isfinite(finite)]
                samples[(attribute, metric_name)].append(
                    float(finite.max() - finite.min()) if len(finite) >= 2 else math.nan
                )

    rows: List[Dict[str, object]] = []
    for (attribute, metric_name), values in samples.items():
        finite = np.asarray(values, dtype=float)
        finite = finite[np.isfinite(finite)]
        rows.append(
            {
                "attribute": attribute,
                "metric": metric_name,
                "lower_95": float(np.quantile(finite, 0.025)) if len(finite) else math.nan,
                "upper_95": float(np.quantile(finite, 0.975)) if len(finite) else math.nan,
                "bootstrap_valid": int(len(finite)),
            }
        )
    return pd.DataFrame(rows)


def cohort_summary(frame: pd.DataFrame, dataset: str) -> Dict[str, object]:
    return {
        "dataset": dataset,
        "n_images_analyzed": int(len(frame)),
        "n_patients": int(frame["patient_key"].nunique()),
        "referable_dr_images": int(frame["target"].sum()),
        "prevalence": float(frame["target"].mean()),
        "age_mean": float(frame["age"].mean()),
        "age_sd": float(frame["age"].std()),
        "female_images_percent": float((frame["sex"] == "Female").mean() * 100),
        "male_images_percent": float((frame["sex"] == "Male").mean() * 100),
        "missing_age_images": int(frame["age"].isna().sum()),
        "missing_sex_images": int(frame["sex"].isna().sum()),
    }


def calibration_curve_points(y: np.ndarray, probability: np.ndarray, bins: int = 10) -> pd.DataFrame:
    order = np.argsort(probability)
    rows = []
    for index, indices in enumerate(np.array_split(order, min(bins, len(order)))):
        if len(indices):
            rows.append(
                {
                    "bin": index + 1,
                    "mean_predicted": float(np.mean(probability[indices])),
                    "observed": float(np.mean(y[indices])),
                    "n": int(len(indices)),
                }
            )
    return pd.DataFrame(rows)


def save_figures(summary: pd.DataFrame, gaps: pd.DataFrame, curves: pd.DataFrame, output: Path) -> None:
    sns.set_theme(style="whitegrid", context="paper")
    external = summary[summary["setting"].str.startswith("external")].copy()
    if not external.empty:
        figure, axes = plt.subplots(1, 3, figsize=(11, 3.5))
        for axis, metric_name, title in zip(
            axes,
            ("auroc", "brier", "sensitivity"),
            ("AUROC", "Brier score", "Sensitivity at source threshold"),
        ):
            sns.barplot(data=external, x="backbone", y=metric_name, hue="method", ax=axis)
            axis.set_title(title)
            axis.set_xlabel("")
        figure.tight_layout()
        figure.savefig(output / "performance_overview.png", dpi=300, bbox_inches="tight")
        plt.close(figure)

    plotted_gaps = gaps[
        gaps["setting"].str.startswith("external")
        & gaps["metric"].isin(["sensitivity", "fpr", "auroc"])
        & gaps["attribute"].isin(["sex", "age_group", "sex_age"])
    ].copy()
    if not plotted_gaps.empty:
        grid = sns.catplot(
            data=plotted_gaps,
            x="attribute",
            y="gap_max_minus_min",
            hue="method",
            row="metric",
            col="backbone",
            kind="bar",
            errorbar=None,
            height=2.5,
            aspect=1.25,
            sharey=False,
        )
        grid.set_axis_labels("Sensitive attribute", "Maximum minus minimum")
        grid.savefig(output / "fairness_gaps.png", dpi=300, bbox_inches="tight")
        plt.close(grid.figure)

    if not curves.empty:
        plot_data = curves[curves["setting"].str.startswith("external")]
        grid = sns.relplot(
            data=plot_data,
            x="mean_predicted",
            y="observed",
            hue="method",
            col="backbone",
            kind="line",
            marker="o",
            height=3.5,
            aspect=1,
        )
        for axis in grid.axes.flat:
            axis.plot([0, 1], [0, 1], linestyle="--", color="black", linewidth=1)
            axis.set_xlim(0, 1)
            axis.set_ylim(0, 1)
        grid.savefig(output / "external_calibration.png", dpi=300, bbox_inches="tight")
        plt.close(grid.figure)


def compare_fairness_transfer(gaps: pd.DataFrame) -> pd.DataFrame:
    keys = ["backbone", "attribute", "metric"]
    internal = gaps[(gaps["method"] == "source_platt") & (gaps["setting"] == "internal_test")]
    external = gaps[(gaps["method"] == "source_platt") & (gaps["setting"] == "external_full")]
    comparison = internal[keys + ["gap_max_minus_min"]].merge(
        external[keys + ["gap_max_minus_min"]],
        on=keys,
        suffixes=("_internal", "_external"),
        validate="one_to_one",
    )
    comparison["gap_amplification_external_minus_internal"] = (
        comparison["gap_max_minus_min_external"] - comparison["gap_max_minus_min_internal"]
    )
    return comparison


def compare_target_recalibration(gaps: pd.DataFrame) -> pd.DataFrame:
    keys = ["backbone", "attribute", "metric"]
    before = gaps[(gaps["method"] == "source_platt") & (gaps["setting"] == "external_heldout")]
    after = gaps[
        (gaps["method"] == "target_10pct_platt_local_threshold")
        & (gaps["setting"] == "external_heldout")
    ]
    comparison = before[keys + ["gap_max_minus_min"]].merge(
        after[keys + ["gap_max_minus_min"]],
        on=keys,
        suffixes=("_before", "_after"),
        validate="one_to_one",
    )
    comparison["gap_change_after_minus_before"] = (
        comparison["gap_max_minus_min_after"] - comparison["gap_max_minus_min_before"]
    )
    return comparison


def compare_setting_transport(summary: pd.DataFrame) -> pd.DataFrame:
    keys = ["backbone", "method"]
    internal = summary[
        (summary["method"] == "source_platt") & (summary["setting"] == "internal_test")
    ]
    external = summary[
        (summary["method"] == "source_platt") & (summary["setting"] == "external_full")
    ]
    metric_names = ["auroc", "auprc", "brier", "ece", "sensitivity", "specificity", "ppv"]
    comparison = internal[keys + metric_names].merge(
        external[keys + metric_names],
        on=keys,
        suffixes=("_internal", "_external"),
        validate="one_to_one",
    )
    for name in metric_names:
        comparison[f"{name}_external_minus_internal"] = (
            comparison[f"{name}_external"] - comparison[f"{name}_internal"]
        )
    return comparison


def main() -> int:
    args = parse_args()
    if not 0 < args.target_calibration_fraction < 0.5:
        raise ValueError("target calibration fraction must be between 0 and 0.5")
    raw_dir = args.data_root.expanduser().resolve() / "raw"
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    summary_rows: List[Dict[str, object]] = []
    subgroup_frames: List[pd.DataFrame] = []
    gap_frames: List[pd.DataFrame] = []
    ci_frames: List[pd.DataFrame] = []
    gap_ci_frames: List[pd.DataFrame] = []
    tuning_frames: List[pd.DataFrame] = []
    curve_frames: List[pd.DataFrame] = []
    cohort_rows: List[Dict[str, object]] = []
    adaptation_rows: List[Dict[str, object]] = []

    for backbone_index, (backbone, files) in enumerate(BACKBONES.items()):
        source, x_source = load_dataset(raw_dir, "brset", files["brset"])
        target, x_target = load_dataset(raw_dir, "mbrset", files["mbrset"])
        y_source = source["target"].to_numpy(dtype=int)
        y_target = target["target"].to_numpy(dtype=int)
        split = split_source_patients(source, args.seed)
        calibration_mask, external_evaluation_mask = choose_target_calibration_patients(
            target, args.target_calibration_fraction, args.seed + 100
        )

        if backbone_index == 0:
            cohort_rows.extend([cohort_summary(source, "BRSET"), cohort_summary(target, "mBRSET")])

        best_c, tuning = choose_c(x_source, y_source, split["train"], split["validation"])
        tuning.insert(0, "backbone", backbone)
        tuning_frames.append(tuning)
        model = fit_linear_probe(x_source[split["train"]], y_source[split["train"]], best_c)
        source_raw = model.predict_proba(x_source)[:, 1]
        target_raw = model.predict_proba(x_target)[:, 1]

        source_calibrator = fit_platt(source_raw[split["validation"]], y_source[split["validation"]])
        source_calibrated = apply_platt(source_calibrator, source_raw)
        target_source_calibrated = apply_platt(source_calibrator, target_raw)
        deployment_threshold = threshold_at_sensitivity(
            y_source[split["validation"]], source_calibrated[split["validation"]], sensitivity=0.90
        )

        target_calibrator = fit_platt(
            target_source_calibrated[calibration_mask], y_target[calibration_mask]
        )
        target_recalibrated = apply_platt(target_calibrator, target_source_calibrated)
        target_local_threshold = threshold_at_sensitivity(
            y_target[calibration_mask], target_recalibrated[calibration_mask], sensitivity=0.90
        )
        adaptation_rows.append(
            {
                "backbone": backbone,
                "calibration_fraction": args.target_calibration_fraction,
                "calibration_images": int(calibration_mask.sum()),
                "calibration_patients": int(target.loc[calibration_mask, "patient_key"].nunique()),
                "calibration_events": int(y_target[calibration_mask].sum()),
                "source_locked_threshold": deployment_threshold,
                "target_local_threshold": target_local_threshold,
            }
        )

        evaluations = [
            (
                "raw",
                "internal_test",
                source.loc[split["internal_test"]].reset_index(drop=True),
                source_raw[split["internal_test"]],
                threshold_at_sensitivity(
                    y_source[split["validation"]], source_raw[split["validation"]], sensitivity=0.90
                ),
            ),
            (
                "source_platt",
                "internal_test",
                source.loc[split["internal_test"]].reset_index(drop=True),
                source_calibrated[split["internal_test"]],
                deployment_threshold,
            ),
            (
                "raw",
                "external_full",
                target.reset_index(drop=True),
                target_raw,
                threshold_at_sensitivity(
                    y_source[split["validation"]], source_raw[split["validation"]], sensitivity=0.90
                ),
            ),
            (
                "source_platt",
                "external_full",
                target.reset_index(drop=True),
                target_source_calibrated,
                deployment_threshold,
            ),
            (
                "source_platt",
                "external_heldout",
                target.loc[external_evaluation_mask].reset_index(drop=True),
                target_source_calibrated[external_evaluation_mask],
                deployment_threshold,
            ),
            (
                "target_10pct_platt_source_threshold",
                "external_heldout",
                target.loc[external_evaluation_mask].reset_index(drop=True),
                target_recalibrated[external_evaluation_mask],
                deployment_threshold,
            ),
            (
                "target_10pct_platt_local_threshold",
                "external_heldout",
                target.loc[external_evaluation_mask].reset_index(drop=True),
                target_recalibrated[external_evaluation_mask],
                target_local_threshold,
            ),
        ]

        for evaluation_index, (method, setting, frame, probability, threshold) in enumerate(evaluations):
            row: Dict[str, object] = {
                "backbone": backbone,
                "method": method,
                "setting": setting,
                "best_C": best_c,
                "n_patients": int(frame["patient_key"].nunique()),
            }
            row.update(metrics(frame["target"].to_numpy(dtype=int), probability, threshold))
            summary_rows.append(row)

            subgroup, gaps = aggregate_group_metrics(
                frame,
                probability,
                threshold,
                backbone,
                method,
                setting,
                args.minimum_group_size,
                args.minimum_group_events,
            )
            subgroup_frames.append(subgroup)
            gap_frames.append(gaps)

            ci = cluster_bootstrap_ci(
                frame,
                probability,
                threshold,
                args.bootstrap_replicates,
                args.seed + 1000 * backbone_index + 100 * evaluation_index,
            )
            ci.insert(0, "setting", setting)
            ci.insert(0, "method", method)
            ci.insert(0, "backbone", backbone)
            ci_frames.append(ci)

            gap_ci = cluster_bootstrap_gap_ci(
                frame,
                probability,
                threshold,
                args.bootstrap_replicates,
                args.seed + 50000 + 1000 * backbone_index + 100 * evaluation_index,
                args.minimum_group_size,
                args.minimum_group_events,
            )
            gap_ci.insert(0, "setting", setting)
            gap_ci.insert(0, "method", method)
            gap_ci.insert(0, "backbone", backbone)
            gap_ci_frames.append(gap_ci)

            curve = calibration_curve_points(frame["target"].to_numpy(dtype=int), probability)
            curve.insert(0, "setting", setting)
            curve.insert(0, "method", method)
            curve.insert(0, "backbone", backbone)
            curve_frames.append(curve)

    summary = pd.DataFrame(summary_rows)
    subgroups = pd.concat(subgroup_frames, ignore_index=True)
    gaps = pd.concat(gap_frames, ignore_index=True)
    confidence_intervals = pd.concat(ci_frames, ignore_index=True)
    gap_confidence_intervals = pd.concat(gap_ci_frames, ignore_index=True)
    tuning = pd.concat(tuning_frames, ignore_index=True)
    curves = pd.concat(curve_frames, ignore_index=True)

    summary.to_csv(output / "summary_metrics.csv", index=False)
    subgroups.to_csv(output / "subgroup_metrics.csv", index=False)
    gaps.to_csv(output / "fairness_gaps.csv", index=False)
    confidence_intervals.to_csv(output / "bootstrap_confidence_intervals.csv", index=False)
    gap_confidence_intervals.to_csv(output / "fairness_gap_confidence_intervals.csv", index=False)
    tuning.to_csv(output / "hyperparameter_selection.csv", index=False)
    curves.to_csv(output / "calibration_bins.csv", index=False)
    pd.DataFrame(cohort_rows).to_csv(output / "cohort_summary.csv", index=False)
    pd.DataFrame(adaptation_rows).to_csv(output / "target_adaptation_summary.csv", index=False)
    compare_fairness_transfer(gaps).to_csv(output / "fairness_transfer.csv", index=False)
    compare_target_recalibration(gaps).to_csv(
        output / "fairness_recalibration_change.csv", index=False
    )
    compare_setting_transport(summary).to_csv(output / "setting_transport_deltas.csv", index=False)

    manifest = {
        "seed": args.seed,
        "bootstrap_replicates": args.bootstrap_replicates,
        "target_calibration_fraction": args.target_calibration_fraction,
        "outcome": "referable diabetic retinopathy, ICDR >= 2",
        "source_split": "60% train, 20% validation, 20% internal test at patient level",
        "deployment_threshold": "chosen on source validation for at least 90% sensitivity",
        "target_adaptation": (
            "Platt scaling on 10% target patients; evaluated both with the source-locked "
            "threshold and a target-local threshold chosen for 90% sensitivity"
        ),
        "privacy": "aggregate outputs only; no identifiers, embeddings, or row-level predictions",
    }
    (output / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    save_figures(summary, gaps, curves, output)
    print(f"Completed. Aggregate outputs written to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
