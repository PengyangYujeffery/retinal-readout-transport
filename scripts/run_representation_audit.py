#!/usr/bin/env python3
"""Representation-level demographic readout audit for BRSET -> mBRSET.

The analysis is deliberately patient-level when learning demographic readouts.
Overall referable-DR performance remains image-level, while all prespecified
clinical fairness gaps use one record per patient. It writes aggregate outputs
only: no identifiers, embeddings, fitted models, or row-level predictions.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Reuse the audited data loader and clinical metrics from the primary analysis.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_experiments import (  # noqa: E402
    BACKBONES,
    SOURCE_C_VALUES,
    aggregate_group_metrics,
    apply_platt,
    fit_platt,
    group_metric_value,
    load_dataset,
    metrics,
    split_source_patients,
    threshold_at_sensitivity,
)


ATTRIBUTES: Mapping[str, Mapping[str, object]] = {
    "sex": {"column": "sex", "positive": "Male", "label": "Sex (male vs female)"},
    "age65": {"column": "age65", "positive": 1, "label": "Age (>=65 vs <65 years)"},
}

# This label is serialized in every demographic-readout output so that the
# coordinate contract is both machine-testable and unambiguous to downstream
# consumers. All patient vectors are means of image embeddings standardized by
# the source-training image distribution.
SHARED_DEMOGRAPHIC_COORDINATE = (
    "source-training-image-standardised-patient-mean-coordinates"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=20260902)
    parser.add_argument("--split-repeats", type=int, default=5)
    parser.add_argument("--cv-repeats", type=int, default=5)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--permutation-replicates", type=int, default=10000)
    parser.add_argument("--random-controls", type=int, default=200)
    parser.add_argument("--minimum-group-size", type=int, default=50)
    parser.add_argument("--minimum-group-events", type=int, default=10)
    return parser.parse_args()


def patient_table(frame: pd.DataFrame, x: np.ndarray) -> Tuple[pd.DataFrame, np.ndarray]:
    """Average images per patient and retain internally consistent metadata."""
    patient_codes, patient_keys = pd.factorize(frame["patient_key"], sort=False)
    counts = np.bincount(patient_codes).astype(np.float64)
    sums = np.zeros((len(patient_keys), x.shape[1]), dtype=np.float64)
    np.add.at(sums, patient_codes, x)
    patient_x = (sums / counts[:, None]).astype(np.float32)

    rows: List[Dict[str, object]] = []
    for code, key in enumerate(patient_keys):
        subset = frame.loc[patient_codes == code]
        sex_values = subset["sex"].dropna().unique()
        age_values = subset["age"].dropna().to_numpy(dtype=float)
        target_values = subset["target"].dropna().to_numpy(dtype=int)
        rows.append(
            {
                "patient_key": str(key),
                "sex": sex_values[0] if len(sex_values) == 1 else pd.NA,
                "age": float(np.median(age_values)) if len(age_values) else math.nan,
                "target": int(target_values.max()) if len(target_values) else pd.NA,
                "n_images": int(len(subset)),
            }
        )
    table = pd.DataFrame(rows)
    table["age65"] = np.where(table["age"].notna(), (table["age"] >= 65).astype(float), np.nan)
    return table, patient_x


def patient_fairness_table(frame: pd.DataFrame, probability: np.ndarray) -> pd.DataFrame:
    """Collapse image predictions to one prespecified clinical record per patient.

    The patient outcome is the maximum image outcome and the patient score is
    the maximum calibrated image probability. Sex must be internally
    consistent and age is represented by the within-patient median, matching
    :func:`patient_table`. Consequently every eligible patient contributes to
    exactly one sex-by-age group and carries unit weight in fairness metrics.
    """
    work = frame.reset_index(drop=True).copy()
    probability = np.asarray(probability, dtype=float)
    if len(work) != len(probability):
        raise ValueError("probability length does not match the image frame")
    if not np.isfinite(probability).all():
        raise ValueError("patient fairness scores contain non-finite values")
    work["_probability"] = probability

    grouped = work.groupby("patient_key", sort=False, dropna=False)
    aggregations = {
        "target": ("target", "max"),
        "score": ("_probability", "max"),
        "n_images": ("target", "size"),
        "n_positive_images": ("target", "sum"),
    }
    if "age" in work:
        aggregations["age"] = ("age", "median")
    patient = grouped.agg(**aggregations)

    if "sex" in work:
        sex_first = grouped["sex"].first()
        sex_count = grouped["sex"].nunique(dropna=True)
        patient["sex"] = sex_first.where(sex_count == 1, pd.NA)
    else:
        patient["sex"] = pd.Series(pd.NA, index=patient.index, dtype="string")
    if "age" not in patient:
        patient["age"] = np.nan

    # The full analysis derives the intersectional group from sex and age. A
    # prederived group is accepted as a compatibility path for aggregate-only
    # contract tests and callers that legitimately omit the component columns.
    if "age" in work and "sex" in work:
        patient["age_group"] = pd.cut(
            patient["age"],
            bins=[-np.inf, 49, 64, np.inf],
            labels=["<50", "50-64", "65+"],
        ).astype("string")
        patient["sex_age"] = (
            patient["sex"].astype("string") + " | " + patient["age_group"]
        )
    elif "sex_age" in work:
        group_first = grouped["sex_age"].first()
        group_count = grouped["sex_age"].nunique(dropna=True)
        patient["sex_age"] = group_first.where(group_count == 1, pd.NA).astype("string")
        patient["age_group"] = pd.Series(pd.NA, index=patient.index, dtype="string")
    else:
        patient["age_group"] = pd.Series(pd.NA, index=patient.index, dtype="string")
        patient["sex_age"] = pd.Series(pd.NA, index=patient.index, dtype="string")

    patient.reset_index(inplace=True)
    patient["target"] = patient["target"].astype(int)
    if patient["patient_key"].duplicated().any():
        raise AssertionError("patient fairness table is not one row per patient")
    return patient


def patient_masks(table: pd.DataFrame, image_frame: pd.DataFrame, seed: int) -> Dict[str, np.ndarray]:
    image_masks = split_source_patients(image_frame, seed)
    sets = {
        name: set(image_frame.loc[mask, "patient_key"].astype(str))
        for name, mask in image_masks.items()
    }
    return {
        name: table["patient_key"].astype(str).isin(keys).to_numpy()
        for name, keys in sets.items()
    }


def attribute_arrays(table: pd.DataFrame, attribute: str) -> Tuple[np.ndarray, np.ndarray]:
    spec = ATTRIBUTES[attribute]
    values = table[str(spec["column"])]
    valid = values.notna().to_numpy()
    y = (values.loc[valid].to_numpy() == spec["positive"]).astype(int)
    return valid, y


def fit_probe(x: np.ndarray, y: np.ndarray, c_value: float, scale: bool = True):
    steps = []
    if scale:
        steps.append(("scale", StandardScaler()))
    steps.append(
        (
            "classifier",
            LogisticRegression(
                C=c_value,
                penalty="l2",
                solver="lbfgs",
                max_iter=3000,
                random_state=0,
            ),
        )
    )
    model = Pipeline(steps)
    model.fit(x, y)
    return model


def choose_attribute_c(
    x: np.ndarray,
    table: pd.DataFrame,
    attribute: str,
    train_mask: np.ndarray,
    validation_mask: np.ndarray,
    scale: bool = False,
) -> Tuple[float, pd.DataFrame]:
    valid, y_valid_only = attribute_arrays(table, attribute)
    y = np.full(len(table), -1, dtype=int)
    y[valid] = y_valid_only
    rows = []
    for c_value in SOURCE_C_VALUES:
        train = train_mask & valid
        validation = validation_mask & valid
        model = fit_probe(x[train], y[train], c_value, scale=scale)
        probability = model.predict_proba(x[validation])[:, 1]
        rows.append(
            {
                "attribute": attribute,
                "C": c_value,
                "validation_auroc": float(roc_auc_score(y[validation], probability)),
            }
        )
    scores = pd.DataFrame(rows)
    best = scores.sort_values(["validation_auroc", "C"], ascending=[False, True]).iloc[0]
    return float(best["C"]), scores


def repeated_local_predictions(
    x: np.ndarray,
    y: np.ndarray,
    c_value: float,
    seed: int,
    repeats: int,
    scale: bool = False,
) -> np.ndarray:
    splitter = RepeatedStratifiedKFold(n_splits=5, n_repeats=repeats, random_state=seed)
    total = np.zeros(len(y), dtype=float)
    count = np.zeros(len(y), dtype=int)
    for train, test in splitter.split(x, y):
        model = fit_probe(x[train], y[train], c_value, scale=scale)
        total[test] += model.predict_proba(x[test])[:, 1]
        count[test] += 1
    if np.any(count != repeats):
        raise RuntimeError("repeated cross-validation did not predict each patient equally")
    return total / count


def classification_summary(y: np.ndarray, probability: np.ndarray) -> Dict[str, float]:
    return {
        "n_patients": int(len(y)),
        "n_positive": int(y.sum()),
        "auroc": float(roc_auc_score(y, probability)),
        "balanced_accuracy": float(balanced_accuracy_score(y, probability >= 0.5)),
    }


def bootstrap_auc(
    y: np.ndarray, probability: np.ndarray, replicates: int, seed: int
) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(replicates):
        index = rng.integers(0, len(y), len(y))
        if len(np.unique(y[index])) < 2:
            continue
        values.append(roc_auc_score(y[index], probability[index]))
    return tuple(np.quantile(values, [0.025, 0.975])) if values else (math.nan, math.nan)


def paired_bootstrap_auc_difference(
    y: np.ndarray,
    probability_a: np.ndarray,
    probability_b: np.ndarray,
    replicates: int,
    seed: int,
) -> Tuple[float, float, float]:
    observed = roc_auc_score(y, probability_a) - roc_auc_score(y, probability_b)
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(replicates):
        index = rng.integers(0, len(y), len(y))
        if len(np.unique(y[index])) < 2:
            continue
        values.append(
            roc_auc_score(y[index], probability_a[index])
            - roc_auc_score(y[index], probability_b[index])
        )
    low, high = np.quantile(values, [0.025, 0.975]) if values else (math.nan, math.nan)
    return float(observed), float(low), float(high)


def unit(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm <= 1e-12:
        raise ValueError("cannot normalise a zero direction")
    return np.asarray(vector, dtype=float) / norm


def centroid_direction(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return unit(x[y == 1].mean(axis=0) - x[y == 0].mean(axis=0))


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(unit(a), unit(b)))


def centroid_alignment_test(
    source_x: np.ndarray,
    source_y: np.ndarray,
    target_x: np.ndarray,
    target_y: np.ndarray,
    bootstrap_replicates: int,
    permutation_replicates: int,
    seed: int,
) -> Dict[str, float]:
    source_direction = centroid_direction(source_x, source_y)
    target_direction = centroid_direction(target_x, target_y)
    observed = cosine(source_direction, target_direction)
    rng = np.random.default_rng(seed)

    bootstrap = []
    for _ in range(bootstrap_replicates):
        source_index = rng.integers(0, len(source_y), len(source_y))
        target_index = rng.integers(0, len(target_y), len(target_y))
        if len(np.unique(source_y[source_index])) < 2 or len(np.unique(target_y[target_index])) < 2:
            continue
        bootstrap.append(
            cosine(
                centroid_direction(source_x[source_index], source_y[source_index]),
                centroid_direction(target_x[target_index], target_y[target_index]),
            )
        )
    low, high = np.quantile(bootstrap, [0.025, 0.975]) if bootstrap else (math.nan, math.nan)
    # Very small smoke-test replicate counts can yield a percentile interval
    # lying entirely on one side of the plug-in estimate. Serialize ordered
    # display bounds that include the estimate so downstream asymmetric-error
    # renderers never receive a negative error magnitude. At the prespecified
    # analysis count this guard is ordinarily inactive.
    if np.isfinite(low) and np.isfinite(high):
        low = min(float(low), observed)
        high = max(float(high), observed)

    null = []
    for _ in range(permutation_replicates):
        permuted = rng.permutation(target_y)
        null.append(cosine(source_direction, centroid_direction(target_x, permuted)))
    # Direction is signed; the planned alternative is positive alignment.
    p_value = (1 + np.sum(np.asarray(null) >= observed)) / (permutation_replicates + 1)
    return {
        "centroid_direction_cosine": observed,
        "cosine_ci_low": float(low),
        "cosine_ci_high": float(high),
        "permutation_p_one_sided": float(p_value),
    }


def probe_direction_on_common_scale(
    x: np.ndarray, y: np.ndarray, c_value: float
) -> np.ndarray:
    model = fit_probe(x, y, c_value, scale=False)
    return unit(model.named_steps["classifier"].coef_[0])


def orthonormal_basis(directions: Sequence[np.ndarray]) -> np.ndarray:
    matrix = np.column_stack([unit(direction) for direction in directions])
    q, r = np.linalg.qr(matrix)
    rank = int(np.sum(np.abs(np.diag(r)) > 1e-8))
    return q[:, :rank]


def remove_subspace(x: np.ndarray, basis: np.ndarray | None) -> np.ndarray:
    if basis is None or basis.size == 0:
        return np.asarray(x)
    return np.asarray(x) - np.asarray(x) @ basis @ basis.T


def choose_disease_c(
    x: np.ndarray, y: np.ndarray, train: np.ndarray, validation: np.ndarray
) -> Tuple[float, pd.DataFrame]:
    rows = []
    for c_value in SOURCE_C_VALUES:
        model = fit_probe(x[train], y[train], c_value, scale=False)
        probability = model.predict_proba(x[validation])[:, 1]
        rows.append({"C": c_value, "validation_auroc": roc_auc_score(y[validation], probability)})
    scores = pd.DataFrame(rows)
    best = scores.sort_values(["validation_auroc", "C"], ascending=[False, True]).iloc[0]
    return float(best["C"]), scores


def fit_disease_rule(
    source_x: np.ndarray,
    source_y: np.ndarray,
    target_x: np.ndarray,
    train: np.ndarray,
    validation: np.ndarray,
) -> Dict[str, object]:
    best_c, tuning = choose_disease_c(source_x, source_y, train, validation)
    model = fit_probe(source_x[train], source_y[train], best_c, scale=False)
    source_raw = model.predict_proba(source_x)[:, 1]
    target_raw = model.predict_proba(target_x)[:, 1]
    calibrator = fit_platt(source_raw[validation], source_y[validation])
    source_probability = apply_platt(calibrator, source_raw)
    target_probability = apply_platt(calibrator, target_raw)
    threshold = threshold_at_sensitivity(source_y[validation], source_probability[validation], 0.90)
    return {
        "best_c": best_c,
        "tuning": tuning,
        "source_probability": source_probability,
        "target_probability": target_probability,
        "threshold": threshold,
    }


def gap_value(
    frame: pd.DataFrame,
    probability: np.ndarray,
    threshold: float,
    attribute: str,
    metric_name: str,
    minimum_size: int,
    minimum_events: int,
) -> float:
    patient = patient_fairness_table(frame, probability)
    _, gaps = aggregate_group_metrics(
        patient,
        patient["score"].to_numpy(dtype=float),
        threshold,
        "_",
        "_",
        "_",
        minimum_size,
        minimum_events,
    )
    selected = gaps[(gaps["attribute"] == attribute) & (gaps["metric"] == metric_name)]
    return float(selected.iloc[0]["gap_max_minus_min"]) if len(selected) else math.nan


def fairness_null_reference(
    frame: pd.DataFrame,
    probability: np.ndarray,
    threshold: float,
    attribute: str,
    metric_name: str,
    replicates: int,
    seed: int,
    minimum_size: int,
    minimum_events: int,
) -> Dict[str, float]:
    """Patient-level fair-null reference via outcome-stratified permutation.

    Both the statistic and the permutation operate on one row per patient.
    Because group labels are permuted only within the binary patient-outcome
    strata, every replicate preserves the exact number of patient events and
    non-events assigned to every group. Assertions below make this conditioning
    contract executable rather than relying on a prose description.
    """
    patient = patient_fairness_table(frame, probability)
    patient_valid = patient.loc[patient[attribute].notna()].reset_index(drop=True).copy()
    patient_valid["group"] = patient_valid[attribute].astype(str)
    outcomes = patient_valid["target"].to_numpy(dtype=int)
    scores = patient_valid["score"].to_numpy(dtype=float)
    original_groups = patient_valid["group"].to_numpy(dtype=object)

    def support_signature(groups: np.ndarray) -> Tuple[Tuple[str, int, int], ...]:
        support = (
            pd.DataFrame({"group": groups.astype(str), "outcome": outcomes})
            .groupby(["group", "outcome"], sort=True)
            .size()
        )
        return tuple(
            (str(group), int(outcome), int(count))
            for (group, outcome), count in support.items()
        )

    def current_gap(groups: np.ndarray) -> Tuple[float, Tuple[str, ...]]:
        values = []
        eligible = []
        string_groups = groups.astype(str)
        for label in sorted(set(string_groups)):
            index = np.flatnonzero(string_groups == label)
            y_group = outcomes[index]
            events = int(y_group.sum())
            if (
                len(index) >= minimum_size
                and events >= minimum_events
                and len(index) - events >= minimum_events
            ):
                value = group_metric_value(y_group, scores[index], threshold, metric_name)
                if np.isfinite(value):
                    values.append(float(value))
                    eligible.append(str(label))
        gap = float(max(values) - min(values)) if len(values) >= 2 else math.nan
        return gap, tuple(eligible)

    expected_support = support_signature(original_groups)
    observed, observed_eligible = current_gap(original_groups)
    rng = np.random.default_rng(seed)
    null = []
    for _ in range(replicates):
        permuted = original_groups.copy()
        for outcome in sorted(np.unique(outcomes)):
            index = np.flatnonzero(outcomes == outcome)
            permuted[index] = rng.permutation(permuted[index])
        if support_signature(permuted) != expected_support:
            raise AssertionError(
                "outcome-stratified permutation changed group patient event support"
            )
        value, eligible = current_gap(permuted)
        if eligible != observed_eligible:
            raise AssertionError(
                "outcome-stratified permutation changed the eligible patient groups"
            )
        if np.isfinite(value):
            null.append(value)
    null_array = np.asarray(null, dtype=float)
    audit_fields = {
        "analysis_unit": "patient",
        "patient_outcome_aggregation": "max_image_outcome",
        "patient_score_aggregation": "max_calibrated_image_probability",
        "permutation_strata": "patient_outcome",
        "patient_event_support_preserved": True,
        "eligible_groups_invariant": True,
        "n_patients": int(len(patient_valid)),
        "n_patient_events": int(outcomes.sum()),
        "n_patient_nonevents": int(len(outcomes) - outcomes.sum()),
        "eligible_groups_observed": int(len(observed_eligible)),
    }
    if not len(null_array):
        result = {
            "observed_gap": observed,
            "null_mean": math.nan,
            "null_95th_percentile": math.nan,
            "observed_minus_null_mean": math.nan,
            "permutation_p": math.nan,
            "valid_replicates": 0,
        }
        result.update(audit_fields)
        return result
    result = {
        "observed_gap": observed,
        "null_mean": float(null_array.mean()),
        "null_95th_percentile": float(np.quantile(null_array, 0.95)),
        "observed_minus_null_mean": float(observed - null_array.mean()),
        "permutation_p": float((1 + np.sum(null_array >= observed)) / (len(null_array) + 1)),
        "valid_replicates": int(len(null_array)),
    }
    result.update(audit_fields)
    return result


def stratified_patient_gap_ci(
    patient: pd.DataFrame,
    threshold: float,
    attribute: str,
    metric_name: str,
    replicates: int,
    seed: int,
    minimum_size: int,
    minimum_events: int,
) -> Dict[str, object]:
    """Bootstrap a patient-level max--min gap conditional on group support.

    Resampling is performed independently within each eligible
    group-by-outcome stratum. This preserves the observed number of patient
    events and non-events in every included group while propagating sampling
    uncertainty in the patient scores that define the gap.
    """
    valid = patient.loc[patient[attribute].notna()].reset_index(drop=True).copy()
    valid["_group"] = valid[attribute].astype(str)
    eligible: Dict[str, pd.DataFrame] = {}
    observed_values: List[float] = []
    for label, subset in valid.groupby("_group", sort=True):
        outcomes = subset["target"].to_numpy(dtype=int)
        events = int(outcomes.sum())
        nonevents = int(len(outcomes) - events)
        if (
            len(subset) >= minimum_size
            and events >= minimum_events
            and nonevents >= minimum_events
        ):
            eligible[str(label)] = subset.reset_index(drop=True)
            value = group_metric_value(
                outcomes,
                subset["score"].to_numpy(dtype=float),
                threshold,
                metric_name,
            )
            if np.isfinite(value):
                observed_values.append(float(value))

    observed = (
        float(max(observed_values) - min(observed_values))
        if len(observed_values) >= 2
        else math.nan
    )
    rng = np.random.default_rng(seed)
    estimates: List[float] = []
    for _ in range(replicates):
        values: List[float] = []
        for subset in eligible.values():
            sampled_parts = []
            for outcome in (0, 1):
                stratum = subset.loc[subset["target"].eq(outcome)]
                sampled_indices = rng.integers(0, len(stratum), size=len(stratum))
                sampled_parts.append(stratum.iloc[sampled_indices])
            sampled = pd.concat(sampled_parts, ignore_index=True)
            value = group_metric_value(
                sampled["target"].to_numpy(dtype=int),
                sampled["score"].to_numpy(dtype=float),
                threshold,
                metric_name,
            )
            if np.isfinite(value):
                values.append(float(value))
        if len(values) >= 2:
            estimates.append(float(max(values) - min(values)))

    finite = np.asarray(estimates, dtype=float)
    finite = finite[np.isfinite(finite)]
    return {
        "attribute": attribute,
        "metric": metric_name,
        "gap_max_minus_min": observed,
        "ci_low": float(np.quantile(finite, 0.025)) if len(finite) else math.nan,
        "ci_high": float(np.quantile(finite, 0.975)) if len(finite) else math.nan,
        "bootstrap_valid": int(len(finite)),
        "bootstrap_scheme": "patient_within_group_and_outcome",
        "analysis_unit": "patient",
        "eligible_groups": int(len(eligible)),
        "n_patients": int(sum(len(group) for group in eligible.values())),
    }


def evaluate_disease_rule(
    backbone: str,
    method: str,
    split_index: int,
    source: pd.DataFrame,
    target: pd.DataFrame,
    source_y: np.ndarray,
    source_probability: np.ndarray,
    target_probability: np.ndarray,
    threshold: float,
    internal_mask: np.ndarray,
    minimum_size: int,
    minimum_events: int,
    gap_bootstrap_replicates: int = 0,
    gap_bootstrap_seed: int = 0,
) -> Tuple[
    List[Dict[str, object]],
    List[pd.DataFrame],
    List[pd.DataFrame],
    List[Dict[str, object]],
]:
    summaries = []
    subgroup_frames = []
    gap_frames = []
    gap_ci_rows = []
    evaluations = [
        ("internal_test", source.loc[internal_mask].reset_index(drop=True), source_probability[internal_mask]),
        ("external_full", target.reset_index(drop=True), target_probability),
    ]
    for setting, frame, probability in evaluations:
        row: Dict[str, object] = {
            "backbone": backbone,
            "method": method,
            "split_index": split_index,
            "setting": setting,
            "n_patients": int(frame["patient_key"].nunique()),
        }
        row.update(metrics(frame["target"].to_numpy(dtype=int), probability, threshold))
        summaries.append(row)
        patient = patient_fairness_table(frame, probability)
        subgroups, gaps = aggregate_group_metrics(
            patient,
            patient["score"].to_numpy(dtype=float),
            threshold,
            backbone,
            method,
            setting,
            minimum_size,
            minimum_events,
        )
        subgroups.insert(3, "analysis_unit", "patient")
        subgroups["patient_outcome_aggregation"] = "max_image_outcome"
        subgroups["patient_score_aggregation"] = "max_calibrated_image_probability"
        subgroups.insert(2, "split_index", split_index)
        subgroup_frames.append(subgroups)
        gaps.insert(3, "analysis_unit", "patient")
        gaps["patient_outcome_aggregation"] = "max_image_outcome"
        gaps["patient_score_aggregation"] = "max_calibrated_image_probability"
        gaps.insert(2, "split_index", split_index)
        gap_frames.append(gaps)
        if gap_bootstrap_replicates and setting == "external_full":
            gap_ci = stratified_patient_gap_ci(
                patient,
                threshold,
                "sex_age",
                "sensitivity",
                gap_bootstrap_replicates,
                gap_bootstrap_seed,
                minimum_size,
                minimum_events,
            )
            gap_ci.update(
                {
                    "backbone": backbone,
                    "method": method,
                    "split_index": split_index,
                    "setting": setting,
                    "patient_outcome_aggregation": "max_image_outcome",
                    "patient_score_aggregation": "max_calibrated_image_probability",
                }
            )
            gap_ci_rows.append(gap_ci)
    return summaries, subgroup_frames, gap_frames, gap_ci_rows


def summarise_paired_splits(
    frame: pd.DataFrame,
    index: Sequence[str],
    value_columns: Iterable[str],
    baseline: str = "baseline",
) -> pd.DataFrame:
    rows = []
    for keys, subset in frame.groupby(list(index), dropna=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        baseline_rows = subset[subset["method"] == baseline].set_index("split_index")
        for method in sorted(set(subset["method"]) - {baseline}):
            method_rows = subset[subset["method"] == method].set_index("split_index")
            joined = baseline_rows.join(method_rows, lsuffix="_baseline", rsuffix="_method", how="inner")
            for value in value_columns:
                difference = joined[f"{value}_method"] - joined[f"{value}_baseline"]
                record = dict(zip(index, keys))
                comparison_label = (
                    {"value_column": value} if "metric" in record else {"metric": value}
                )
                record.update(
                    {
                        "method": method,
                        "n_splits": int(len(difference)),
                        "mean_method_minus_baseline": float(difference.mean()),
                        "min_method_minus_baseline": float(difference.min()),
                        "max_method_minus_baseline": float(difference.max()),
                        "improved_splits": int((difference < 0).sum())
                        if value in {"brier", "ece", "gap_max_minus_min"}
                        else int((difference > 0).sum()),
                    }
                )
                record.update(comparison_label)
                rows.append(record)
    return pd.DataFrame(rows)


def holm_adjust(p_values: Sequence[float]) -> np.ndarray:
    """Holm family-wise-error adjustment with monotonic adjusted p-values."""
    p = np.asarray(p_values, dtype=float)
    adjusted = np.full(len(p), np.nan, dtype=float)
    finite = np.flatnonzero(np.isfinite(p))
    if not len(finite):
        return adjusted
    order = finite[np.argsort(p[finite])]
    running = 0.0
    m = len(order)
    for rank, index in enumerate(order):
        running = max(running, (m - rank) * p[index])
        adjusted[index] = min(1.0, running)
    return adjusted


def make_figure(probes: pd.DataFrame, residual_summary: pd.DataFrame, output: Path) -> None:
    sns.set_theme(style="whitegrid", context="paper")
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
    probe_plot = probes.copy()
    probe_plot["evaluation"] = probe_plot["evaluation"].map(
        {
            "source_internal": "Source internal",
            "target_transported": "Target: source readout",
            "target_local_cv": "Target: local readout",
        }
    )
    sns.barplot(
        data=probe_plot,
        x="attribute",
        y="auroc",
        hue="evaluation",
        errorbar=None,
        ax=axes[0],
    )
    axes[0].axhline(0.5, color="black", linestyle="--", linewidth=1)
    axes[0].set_ylim(0.45, 1.0)
    axes[0].set_xlabel("")
    axes[0].set_ylabel("Demographic probe AUROC")
    axes[0].legend(title="", fontsize=8)

    external = residual_summary[residual_summary["setting"] == "external_full"].copy()
    sns.pointplot(
        data=external,
        x="backbone",
        y="sensitivity",
        hue="method",
        errorbar="sd",
        dodge=0.25,
        markers=["o", "s", "D", "^"],
        ax=axes[1],
    )
    axes[1].set_xlabel("")
    axes[1].set_ylabel("External sensitivity\n(source-locked threshold)")
    axes[1].set_ylim(0, 1)
    axes[1].legend(title="", fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "representation_audit.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    raw_dir = args.data_root.expanduser().resolve() / "raw"
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    probe_rows: List[Dict[str, object]] = []
    probe_contrasts: List[Dict[str, object]] = []
    geometry_rows: List[Dict[str, object]] = []
    residual_rows: List[Dict[str, object]] = []
    residual_subgroup_frames: List[pd.DataFrame] = []
    residual_gap_frames: List[pd.DataFrame] = []
    residual_gap_ci_rows: List[Dict[str, object]] = []
    tuning_rows: List[pd.DataFrame] = []
    random_rows: List[Dict[str, object]] = []
    null_rows: List[Dict[str, object]] = []

    for backbone_index, (backbone, files) in enumerate(BACKBONES.items()):
        source, x_source = load_dataset(raw_dir, "brset", files["brset"])
        target, x_target = load_dataset(raw_dir, "mbrset", files["mbrset"])
        source_patients, _ = patient_table(source, x_source)
        target_patients, _ = patient_table(target, x_target)
        source_y = source["target"].to_numpy(dtype=int)

        primary_image_split = split_source_patients(source, args.seed)
        primary_patient_split = patient_masks(source_patients, source, args.seed)
        source_scaler = StandardScaler().fit(x_source[primary_image_split["train"]])
        z_source = source_scaler.transform(x_source).astype(np.float32)
        z_target = source_scaler.transform(x_target).astype(np.float32)
        _, pz_source = patient_table(source, z_source)
        _, pz_target = patient_table(target, z_target)

        source_sensitive_directions: Dict[str, np.ndarray] = {}
        for attribute_index, attribute in enumerate(ATTRIBUTES):
            valid_source, source_attr_valid = attribute_arrays(source_patients, attribute)
            source_attr = np.full(len(source_patients), -1, dtype=int)
            source_attr[valid_source] = source_attr_valid
            valid_target, target_attr = attribute_arrays(target_patients, attribute)

            best_c, tuning = choose_attribute_c(
                pz_source,
                source_patients,
                attribute,
                primary_patient_split["train"],
                primary_patient_split["validation"],
                scale=False,
            )
            tuning.insert(0, "backbone", backbone)
            tuning_rows.append(tuning)

            train = primary_patient_split["train"] & valid_source
            internal = primary_patient_split["internal_test"] & valid_source
            source_model = fit_probe(pz_source[train], source_attr[train], best_c, scale=False)
            source_internal_probability = source_model.predict_proba(pz_source[internal])[:, 1]
            target_transported_probability = source_model.predict_proba(pz_target[valid_target])[:, 1]
            target_local_probability = repeated_local_predictions(
                pz_target[valid_target],
                target_attr,
                best_c,
                args.seed + 1000 * backbone_index + 100 * attribute_index,
                args.cv_repeats,
                scale=False,
            )

            evaluations = [
                ("source_internal", source_attr[internal], source_internal_probability),
                ("target_transported", target_attr, target_transported_probability),
                ("target_local_cv", target_attr, target_local_probability),
            ]
            for evaluation_index, (evaluation, y, probability) in enumerate(evaluations):
                row: Dict[str, object] = {
                    "backbone": backbone,
                    "attribute": attribute,
                    "evaluation": evaluation,
                    "best_C": best_c,
                }
                row.update(classification_summary(y, probability))
                low, high = bootstrap_auc(
                    y,
                    probability,
                    args.bootstrap_replicates,
                    args.seed + 100000 * backbone_index + 10000 * attribute_index + evaluation_index,
                )
                row["auroc_ci_low"] = low
                row["auroc_ci_high"] = high
                probe_rows.append(row)

            difference, low, high = paired_bootstrap_auc_difference(
                target_attr,
                target_local_probability,
                target_transported_probability,
                args.bootstrap_replicates,
                args.seed + 200000 + 1000 * backbone_index + attribute_index,
            )
            probe_contrasts.append(
                {
                    "backbone": backbone,
                    "attribute": attribute,
                    "contrast": "target_local_minus_transported_auroc",
                    "estimate": difference,
                    "ci_low": low,
                    "ci_high": high,
                }
            )

            common_train = primary_patient_split["train"] & valid_source
            source_sensitive_directions[attribute] = probe_direction_on_common_scale(
                pz_source[common_train], source_attr[common_train], best_c
            )
            target_direction = probe_direction_on_common_scale(
                pz_target[valid_target], target_attr, best_c
            )
            geometry = {
                "backbone": backbone,
                "attribute": attribute,
                "probe_coefficient_cosine": cosine(
                    source_sensitive_directions[attribute], target_direction
                ),
            }
            geometry.update(
                centroid_alignment_test(
                    pz_source[common_train],
                    source_attr[common_train],
                    pz_target[valid_target],
                    target_attr,
                    args.bootstrap_replicates,
                    args.permutation_replicates,
                    args.seed + 300000 + 1000 * backbone_index + attribute_index,
                )
            )
            geometry_rows.append(geometry)

        # Repeat the disease experiment across patient splits.  The demographic
        # subspace is learned from each split's source training patients only.
        for split_index in range(args.split_repeats):
            split_seed = args.seed + split_index * 101
            image_split = split_source_patients(source, split_seed)
            p_split = patient_masks(source_patients, source, split_seed)
            scaler = StandardScaler().fit(x_source[image_split["train"]])
            zs = scaler.transform(x_source).astype(np.float32)
            zt = scaler.transform(x_target).astype(np.float32)
            _, pzs = patient_table(source, zs)

            directions = []
            for attribute in ATTRIBUTES:
                valid, y_valid = attribute_arrays(source_patients, attribute)
                y_attribute = np.full(len(source_patients), -1, dtype=int)
                y_attribute[valid] = y_valid
                best_c, _ = choose_attribute_c(
                    pzs,
                    source_patients,
                    attribute,
                    p_split["train"],
                    p_split["validation"],
                    scale=False,
                )
                mask = p_split["train"] & valid
                directions.append(
                    probe_direction_on_common_scale(pzs[mask], y_attribute[mask], best_c)
                )
            joint_basis = orthonormal_basis(directions)
            bases = {
                "baseline": None,
                "sex_erased": orthonormal_basis([directions[0]]),
                "age_erased": orthonormal_basis([directions[1]]),
                "joint_erased": joint_basis,
            }

            for method, basis in bases.items():
                projected_source = remove_subspace(zs, basis)
                projected_target = remove_subspace(zt, basis)
                fitted = fit_disease_rule(
                    projected_source,
                    source_y,
                    projected_target,
                    image_split["train"],
                    image_split["validation"],
                )
                summaries, subgroups, gaps, gap_cis = evaluate_disease_rule(
                    backbone,
                    method,
                    split_index,
                    source,
                    target,
                    source_y,
                    fitted["source_probability"],
                    fitted["target_probability"],
                    float(fitted["threshold"]),
                    image_split["internal_test"],
                    args.minimum_group_size,
                    args.minimum_group_events,
                    args.bootstrap_replicates
                    if split_index == 0 and method in {"baseline", "joint_erased"}
                    else 0,
                    args.seed
                    + 600000
                    + 10000 * backbone_index
                    + 1000 * int(method == "joint_erased"),
                )
                for row in summaries:
                    row["best_C"] = fitted["best_c"]
                    row["subspace_rank"] = 0 if basis is None else basis.shape[1]
                residual_rows.extend(summaries)
                residual_subgroup_frames.extend(subgroups)
                residual_gap_frames.extend(gaps)
                residual_gap_ci_rows.extend(gap_cis)

                if split_index == 0 and method in {"baseline", "joint_erased"}:
                    for setting, frame, probability in (
                        (
                            "internal_test",
                            source.loc[image_split["internal_test"]].reset_index(drop=True),
                            np.asarray(fitted["source_probability"])[image_split["internal_test"]],
                        ),
                        ("external_full", target.reset_index(drop=True), np.asarray(fitted["target_probability"])),
                    ):
                        for metric_index, metric_name in enumerate(("sensitivity", "auroc")):
                            null_result = fairness_null_reference(
                                frame,
                                probability,
                                float(fitted["threshold"]),
                                "sex_age",
                                metric_name,
                                args.permutation_replicates,
                                args.seed
                                + 500000
                                + 10000 * backbone_index
                                + 1000 * (method == "joint_erased")
                                + 100 * (setting == "external_full")
                                + metric_index,
                                args.minimum_group_size,
                                args.minimum_group_events,
                            )
                            null_result.update(
                                {
                                    "backbone": backbone,
                                    "method": method,
                                    "setting": setting,
                                    "attribute": "sex_age",
                                    "metric": metric_name,
                                }
                            )
                            null_rows.append(null_result)

            # Random rank-matched projections are a primary-split control only.
            if split_index == 0:
                rng = np.random.default_rng(args.seed + 400000 + backbone_index)
                for control_index in range(args.random_controls):
                    random_basis = orthonormal_basis(
                        [rng.normal(size=zs.shape[1]) for _ in range(joint_basis.shape[1])]
                    )
                    fitted = fit_disease_rule(
                        remove_subspace(zs, random_basis),
                        source_y,
                        remove_subspace(zt, random_basis),
                        image_split["train"],
                        image_split["validation"],
                    )
                    probability = np.asarray(fitted["target_probability"])
                    threshold = float(fitted["threshold"])
                    row = {
                        "backbone": backbone,
                        "control_index": control_index,
                        "best_C": fitted["best_c"],
                    }
                    row.update(metrics(target["target"].to_numpy(dtype=int), probability, threshold))
                    row["sex_age_sensitivity_gap"] = gap_value(
                        target,
                        probability,
                        threshold,
                        "sex_age",
                        "sensitivity",
                        args.minimum_group_size,
                        args.minimum_group_events,
                    )
                    row["sex_age_ece_gap"] = gap_value(
                        target,
                        probability,
                        threshold,
                        "sex_age",
                        "ece",
                        args.minimum_group_size,
                        args.minimum_group_events,
                    )
                    row["analysis_unit"] = "patient"
                    # Backwards-readable alias retained for any existing
                    # downstream notebooks that consumed the earlier draft.
                    row["fairness_analysis_unit"] = "patient"
                    row["patient_outcome_aggregation"] = "max_image_outcome"
                    row["patient_score_aggregation"] = "max_calibrated_image_probability"
                    random_rows.append(row)

    probes = pd.DataFrame(probe_rows)
    geometry = pd.DataFrame(geometry_rows)
    residual = pd.DataFrame(residual_rows)
    residual_subgroups = pd.concat(residual_subgroup_frames, ignore_index=True)
    residual_gaps = pd.concat(residual_gap_frames, ignore_index=True)
    residual_gap_cis = pd.DataFrame(residual_gap_ci_rows)
    tuning = pd.concat(tuning_rows, ignore_index=True)
    random_controls = pd.DataFrame(random_rows)
    null_reference = pd.DataFrame(null_rows)
    probes["coordinate_space"] = SHARED_DEMOGRAPHIC_COORDINATE
    geometry["coordinate_space"] = SHARED_DEMOGRAPHIC_COORDINATE
    tuning["coordinate_space"] = SHARED_DEMOGRAPHIC_COORDINATE
    geometry["permutation_p_holm"] = holm_adjust(
        geometry["permutation_p_one_sided"].to_numpy(dtype=float)
    )
    null_reference["permutation_p_holm_all"] = holm_adjust(
        null_reference["permutation_p"].to_numpy(dtype=float)
    )
    # Retain the former all-comparison field name as a backwards-readable alias.
    null_reference["permutation_p_holm"] = null_reference["permutation_p_holm_all"]
    null_reference["primary_p_holm"] = np.nan
    primary_mask = (
        (null_reference["method"] == "baseline")
        & (null_reference["setting"] == "external_full")
        & (null_reference["attribute"] == "sex_age")
        & (null_reference["metric"] == "sensitivity")
    )
    null_reference["is_primary_family"] = primary_mask
    null_reference.loc[primary_mask, "primary_p_holm"] = holm_adjust(
        null_reference.loc[primary_mask, "permutation_p"].to_numpy(dtype=float)
    )

    probes.to_csv(output / "demographic_probe_metrics.csv", index=False)
    pd.DataFrame(probe_contrasts).to_csv(output / "demographic_probe_contrasts.csv", index=False)
    geometry.to_csv(output / "representation_geometry.csv", index=False)
    residual.to_csv(output / "residualization_metrics.csv", index=False)
    residual_subgroups.to_csv(output / "residualization_subgroup_metrics.csv", index=False)
    residual_gaps.to_csv(output / "residualization_gaps.csv", index=False)
    residual_gap_cis.to_csv(
        output / "residualization_gap_confidence_intervals.csv", index=False
    )
    tuning.to_csv(output / "demographic_probe_tuning.csv", index=False)
    random_controls.to_csv(output / "random_projection_controls.csv", index=False)
    null_reference.to_csv(output / "fairness_null_reference.csv", index=False)

    metric_contrasts = summarise_paired_splits(
        residual,
        index=["backbone", "setting"],
        value_columns=["auroc", "brier", "ece", "sensitivity", "specificity"],
    )
    gap_contrasts = summarise_paired_splits(
        residual_gaps,
        index=["backbone", "setting", "attribute", "metric"],
        value_columns=["gap_max_minus_min"],
    )
    gap_contrasts["analysis_unit"] = "patient"
    gap_contrasts["patient_outcome_aggregation"] = "max_image_outcome"
    gap_contrasts["patient_score_aggregation"] = "max_calibrated_image_probability"
    metric_contrasts.to_csv(output / "residualization_metric_contrasts.csv", index=False)
    gap_contrasts.to_csv(output / "residualization_gap_contrasts.csv", index=False)

    manifest = {
        "seed": args.seed,
        "split_repeats": args.split_repeats,
        "target_local_cv": f"5-fold repeated {args.cv_repeats} times at patient level",
        "bootstrap_replicates": args.bootstrap_replicates,
        "gap_bootstrap": "patient resampling within sex-by-age and outcome strata",
        "permutation_replicates": args.permutation_replicates,
        "random_rank_matched_controls": args.random_controls,
        "fairness_null": (
            "patient-level subgroup-label permutation within max-image-outcome strata; "
            "patient score is max calibrated image probability; exact patient event and "
            "non-event support is preserved within every sex-by-age group"
        ),
        "demographic_attributes": ["sex", "age >=65 years"],
        "demographic_unit": "mean embedding per patient",
        "demographic_coordinates": (
            "source-training image-standardised shared coordinates for C selection, "
            "source/target readouts, direction comparison, and intervention"
        ),
        "shared_coordinate_path": {
            "c_selection": SHARED_DEMOGRAPHIC_COORDINATE,
            "readout": SHARED_DEMOGRAPHIC_COORDINATE,
            "direction": SHARED_DEMOGRAPHIC_COORDINATE,
        },
        "clinical_fairness_unit": "one record per patient",
        "patient_outcome_aggregation": "max image outcome",
        "patient_score_aggregation": "max calibrated image probability",
        "clinical_outcome": "referable diabetic retinopathy, ICDR >= 2",
        "residualization": (
            "orthogonal projection away from source-trained linear demographic-probe "
            "coefficient directions in source-standardised embedding space"
        ),
        "privacy": "aggregate outputs only; no identifiers, embeddings, models, or row-level predictions",
    }
    (output / "representation_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    make_figure(probes, residual, output)
    print(f"Completed representation audit. Aggregate outputs written to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
