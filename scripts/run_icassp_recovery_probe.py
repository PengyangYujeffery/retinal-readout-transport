#!/usr/bin/env python3
"""Ceiling probe: how much of the readout-transport penalty can be recovered?

This proposes no new method. Before any adaptation module is designed, it
measures how much of the penalty two standard strategies recover.

1. Label-free self-training. Starting from the source readout, all target
   patients are pseudo-labelled at the source training class proportion and
   the readout is refitted on them for a fixed number of rounds. No target
   label is used; evaluation is on every labelled target patient
   (transductive).
2. Few-shot recovery. k labelled target patients (class-stratified) refit the
   readout either from scratch or anchored to the source weights by a
   prior-centred L2 penalty (L2-SP; Li, Grandvalet and Davoine, ICML 2018).
   Evaluation is on the remaining target patients, beside the unchanged
   source readout and the out-of-fold target-local readout.

Any affine target alignment followed by the fixed source probe is itself a
linear readout of the target vectors, so the target-local probe is the
empirical reference ceiling for the whole family of affine alignments. It is
a reference, not a bound: it is an L2 logistic fit, not an AUROC optimum.

Outputs are aggregate only: no identifiers, embeddings, models, or
row-level predictions.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_experiments import BACKBONES, load_dataset  # noqa: E402
from run_representation_audit import (  # noqa: E402
    ATTRIBUTES,
    SHARED_DEMOGRAPHIC_COORDINATE,
    attribute_arrays,
    choose_attribute_c,
    fit_probe,
    patient_table,
    repeated_local_predictions,
)
from run_icassp_extensions import (  # noqa: E402
    DIRECTIONS,
    aggregate_split_results,
    bootstrap_statistics,
    labelled_source,
    prepare_context,
)


# "source_readout", not "source": the frames already carry a "source" dataset column.
FEW_SHOT_METHODS = ("source_readout", "scratch", "anchored", "target_local")
# The second self-training scheme keeps the most confident half of each pseudo-class.
# Fixed before the first run, never tuned against a result.
CONFIDENCE_FRACTION = 0.5
ZERO_PRIOR_COEF_TOLERANCE = 1e-3  # relative L2 difference of coefficient vectors
STRONG_PRIOR_C = 1e-12
STRONG_PRIOR_AUROC_TOLERANCE = 1e-6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=20260902)
    parser.add_argument("--split-repeats", type=int, default=5)
    parser.add_argument("--cv-repeats", type=int, default=5)
    parser.add_argument("--self-training-rounds", type=int, default=10)
    parser.add_argument("--shots", default="10,20,50,100,200")
    parser.add_argument("--draws", type=int, default=20)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--minimum-cv-class", type=int, default=5)
    return parser.parse_args()


def auroc(y: np.ndarray, score: np.ndarray) -> float:
    return float(roc_auc_score(y, score))


def linear_parameters(model) -> Tuple[np.ndarray, float]:
    classifier = model.named_steps["classifier"]
    return classifier.coef_[0].astype(np.float64), float(classifier.intercept_[0])


def fit_anchored_logistic(
    x: np.ndarray,
    y: np.ndarray,
    prior_coef: np.ndarray,
    prior_intercept: float,
    c_value: float,
) -> Tuple[np.ndarray, float]:
    """L2 logistic regression shrunk towards ``prior_coef`` (L2-SP).

    Minimises ``C * sum(logloss) + 0.5 * ||w - prior_coef||^2`` with an
    unpenalised intercept, the scikit-learn lbfgs objective with its origin
    moved to the prior. A zero prior therefore reproduces that estimator.
    """
    x = np.asarray(x, dtype=np.float64)
    sign = 2.0 * np.asarray(y, dtype=np.float64) - 1.0
    base = x @ prior_coef
    dimension = x.shape[1]

    def objective(parameters: np.ndarray) -> Tuple[float, np.ndarray]:
        delta, intercept = parameters[:dimension], parameters[dimension]
        margin = -sign * (base + x @ delta + intercept)
        gradient_z = -sign * expit(margin)
        value = c_value * float(np.logaddexp(0.0, margin).sum()) + 0.5 * float(delta @ delta)
        gradient = np.concatenate(
            [c_value * (x.T @ gradient_z) + delta, [c_value * float(gradient_z.sum())]]
        )
        return value, gradient

    start = np.concatenate([np.zeros(dimension), [prior_intercept]])
    # scipy's default ftol (~2.2e-9 relative decrease) stops early on the flat
    # objectives of small, nearly separable k-shot sets; use the tolerances
    # scikit-learn's lbfgs path uses (ftol = 64 eps) so optima are comparable.
    result = minimize(
        objective,
        start,
        jac=True,
        method="L-BFGS-B",
        options={"maxiter": 50000, "maxls": 50, "gtol": 1e-10, "ftol": 64 * np.finfo(float).eps},
    )
    return prior_coef + result.x[:dimension], float(result.x[dimension])


def self_training(
    source_coef: np.ndarray,
    source_intercept: float,
    target_all: np.ndarray,
    target_labelled: np.ndarray,
    target_y: np.ndarray,
    prevalence: float,
    c_value: float,
    rounds: int,
    confidence_fraction: float = 1.0,
) -> Tuple[List[float], np.ndarray]:
    """Refit the readout on its own pseudo-labels; round 0 is the source readout.

    ``confidence_fraction`` < 1 keeps only that share of each pseudo-class,
    the patients furthest from the decision threshold. It is the second
    self-training scheme, fixed before the first run: the standard objection
    to the first is that pseudo-labelling every patient propagates the source
    probe's mistakes, so the confident variant is the fairer test.
    """
    coef, intercept = source_coef, source_intercept
    x_all = np.asarray(target_all, dtype=np.float64)
    x_labelled = np.asarray(target_labelled, dtype=np.float64)
    trajectory = [auroc(target_y, x_labelled @ coef)]
    for _ in range(rounds):
        scores = x_all @ coef + intercept
        threshold = np.quantile(scores, 1.0 - prevalence)
        pseudo = (scores >= threshold).astype(int)
        if pseudo.min() == pseudo.max():
            raise RuntimeError("self-training pseudo-labels collapsed to one class")
        x_round, y_round = target_all, pseudo
        if confidence_fraction < 1.0:
            margin = np.abs(scores - threshold)
            keep = np.zeros(len(scores), dtype=bool)
            for value in (0, 1):
                members = np.flatnonzero(pseudo == value)
                ordered = members[np.argsort(-margin[members])]
                keep[ordered[: max(1, int(round(confidence_fraction * len(members))))]] = True
            x_round, y_round = target_all[keep], pseudo[keep]
            if y_round.min() == y_round.max():
                raise RuntimeError("confident self-training kept only one class")
        coef, intercept = linear_parameters(fit_probe(x_round, y_round, c_value, scale=False))
        trajectory.append(auroc(target_y, x_labelled @ coef))
    return trajectory, x_labelled @ coef


def draw_seed(*parts: int) -> int:
    return int(np.random.SeedSequence([int(part) for part in parts]).generate_state(1)[0])


def main() -> int:
    args = parse_args()
    shots = sorted({int(value) for value in args.shots.split(",") if value.strip()})
    raw_dir = args.data_root.expanduser().resolve() / "raw"
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    trajectory_rows: List[Dict[str, object]] = []
    few_shot_rows: List[Dict[str, object]] = []
    sanity_rows: List[Dict[str, object]] = []
    interval_rows: List[Dict[str, object]] = []

    for backbone_index, (backbone, files) in enumerate(BACKBONES.items()):
        frames: Dict[str, pd.DataFrame] = {}
        features: Dict[str, np.ndarray] = {}
        metadata: Dict[str, pd.DataFrame] = {}
        for name in ("brset", "mbrset"):
            frames[name], features[name] = load_dataset(raw_dir, name, files[name])
            metadata[name], _ = patient_table(frames[name], features[name])
        # store[(direction, attribute)] -> per-split arrays, kept in memory only.
        store: Dict[Tuple[str, str], List[Dict[str, np.ndarray]]] = {}

        for split_index in range(args.split_repeats):
            split_seed = args.seed + 101 * split_index
            for direction_index, (direction, source, target) in enumerate(DIRECTIONS):
                ctx = prepare_context(frames, features, metadata, source, target, split_seed)
                target_all = ctx["target_pz"]["unaligned"]
                for attribute_index, attribute in enumerate(ATTRIBUTES):
                    labels = {
                        "backbone": backbone,
                        "direction": direction,
                        "source": source,
                        "target": target,
                        "attribute": attribute,
                        "split_index": split_index,
                    }
                    source_valid, source_y = labelled_source(ctx, attribute)
                    split = ctx["split"]
                    train = split["train"] & source_valid
                    validation = split["validation"] & source_valid
                    best_c, _ = choose_attribute_c(
                        ctx["source_pz"], ctx["source_meta"], attribute, train, validation, scale=False
                    )
                    model = fit_probe(ctx["source_pz"][train], source_y[train], best_c, scale=False)
                    source_coef, source_intercept = linear_parameters(model)
                    target_valid, target_y = attribute_arrays(ctx["target_meta"], attribute)
                    if min(int(target_y.sum()), int(len(target_y) - target_y.sum())) < args.minimum_cv_class:
                        raise ValueError(f"{direction} {attribute}: target class support too small")
                    x_labelled = target_all[target_valid]
                    # Same seed as the extension run, so the ceiling is identical.
                    cv_seed = split_seed + 1000 * backbone_index + 100 * attribute_index
                    local = repeated_local_predictions(
                        x_labelled, target_y, best_c, cv_seed, args.cv_repeats, scale=False
                    )
                    local_auroc = auroc(target_y, local)
                    source_score = x_labelled.astype(np.float64) @ source_coef

                    scored = {}
                    for scheme, fraction in (
                        ("self_training", 1.0),
                        ("confident_self_training", CONFIDENCE_FRACTION),
                    ):
                        trajectory, scored[scheme] = self_training(
                            source_coef,
                            source_intercept,
                            target_all,
                            x_labelled,
                            target_y,
                            float(source_y[train].mean()),
                            best_c,
                            args.self_training_rounds,
                            confidence_fraction=fraction,
                        )
                        for round_index, value in enumerate(trajectory):
                            trajectory_rows.append(
                                {
                                    **labels,
                                    "scheme": scheme,
                                    "confidence_fraction": fraction,
                                    "round": round_index,
                                    "auroc": value,
                                    "source_auroc": trajectory[0],
                                    "target_local_auroc": local_auroc,
                                    "best_C": best_c,
                                    "n_target_unlabelled": int(len(target_all)),
                                    "n_target_labelled": int(len(target_y)),
                                }
                            )
                    store.setdefault((direction, attribute), []).append(
                        {
                            "y": target_y,
                            "local": local,
                            "source": source_score,
                            "self_trained": scored["self_training"],
                            "self_trained_confident": scored["confident_self_training"],
                        }
                    )

                    index = np.arange(len(target_y))
                    for k in shots:
                        if k >= len(target_y) - 2 * args.minimum_cv_class:
                            continue
                        for draw in range(args.draws):
                            labelled, heldout = train_test_split(
                                index,
                                train_size=k,
                                stratify=target_y,
                                random_state=draw_seed(
                                    args.seed, backbone_index, direction_index,
                                    attribute_index, split_index, k, draw,
                                ),
                            )
                            x_k = x_labelled[labelled].astype(np.float64)
                            y_k = target_y[labelled]
                            x_h = x_labelled[heldout].astype(np.float64)
                            y_h = target_y[heldout]
                            scratch_coef, _ = linear_parameters(
                                fit_probe(x_labelled[labelled], y_k, best_c, scale=False)
                            )
                            anchored_coef, _ = fit_anchored_logistic(
                                x_k, y_k, source_coef, source_intercept, best_c
                            )
                            values = {
                                "source_readout": auroc(y_h, x_h @ source_coef),
                                "scratch": auroc(y_h, x_h @ scratch_coef),
                                "anchored": auroc(y_h, x_h @ anchored_coef),
                                "target_local": auroc(y_h, local[heldout]),
                            }
                            for method, value in values.items():
                                few_shot_rows.append(
                                    {
                                        **labels,
                                        "k": k,
                                        "draw": draw,
                                        "method": method,
                                        "auroc": value,
                                        "best_C": best_c,
                                        "n_heldout": int(len(y_h)),
                                    }
                                )

                            if split_index == 0 and draw == 0 and k == shots[-1]:
                                # Invariant 1: a zero prior reproduces the scikit-learn
                                # optimum. The objective is strictly convex, so both
                                # solvers, run to tight tolerance, must agree.
                                zero_coef, _ = fit_anchored_logistic(
                                    x_k, y_k, np.zeros_like(source_coef), 0.0, best_c
                                )
                                reference = LogisticRegression(
                                    C=best_c, penalty="l2", solver="lbfgs", max_iter=50000, tol=1e-10
                                ).fit(x_k, y_k)
                                reference_coef = reference.coef_[0]
                                zero_gap = float(
                                    np.linalg.norm(zero_coef - reference_coef)
                                    / max(np.linalg.norm(reference_coef), 1e-12)
                                )
                                # Invariant 2: an overwhelming prior returns the source readout.
                                strong_coef, _ = fit_anchored_logistic(
                                    x_k, y_k, source_coef, source_intercept, STRONG_PRIOR_C
                                )
                                strong_gap = abs(auroc(y_h, x_h @ strong_coef) - values["source_readout"])
                                for check, value, tolerance in (
                                    ("zero_prior_matches_sklearn_coef", zero_gap, ZERO_PRIOR_COEF_TOLERANCE),
                                    ("strong_prior_matches_source_auroc", strong_gap, STRONG_PRIOR_AUROC_TOLERANCE),
                                ):
                                    sanity_rows.append(
                                        {
                                            **labels,
                                            "k": k,
                                            "check": check,
                                            "value": value,
                                            "tolerance": tolerance,
                                            "passed": bool(value <= tolerance),
                                        }
                                    )

        for attribute_index, attribute in enumerate(ATTRIBUTES):
            for direction_index, (direction, source, target) in enumerate(DIRECTIONS):
                splits = store[(direction, attribute)]
                y = splits[0]["y"]
                vectors = {}
                for s, arrays in enumerate(splits):
                    if not np.array_equal(arrays["y"], y):
                        raise AssertionError("target patients differ across splits")
                    for kind in ("local", "source", "self_trained", "self_trained_confident"):
                        vectors[(direction, kind, s)] = arrays[kind]
                mean = 1.0 / len(splits)

                def weight(kind: str, sign: float = 1.0) -> Dict[Tuple[str, str, int], float]:
                    return {(direction, kind, s): sign * mean for s in range(len(splits))}

                label = {
                    "backbone": backbone,
                    "direction": direction,
                    "source": source,
                    "target": target,
                    "attribute": attribute,
                    "n_splits": len(splits),
                }
                statistics = [
                    ({**label, "statistic": "source_auroc"}, weight("source")),
                    ({**label, "statistic": "self_trained_auroc"}, weight("self_trained")),
                    ({**label, "statistic": "target_local_auroc"}, weight("local")),
                    (
                        {**label, "statistic": "self_trained_minus_source_auroc"},
                        {**weight("self_trained"), **weight("source", -1.0)},
                    ),
                    (
                        {**label, "statistic": "remaining_penalty_after_self_training"},
                        {**weight("local"), **weight("self_trained", -1.0)},
                    ),
                    (
                        {**label, "statistic": "self_trained_confident_auroc"},
                        weight("self_trained_confident"),
                    ),
                    (
                        {**label, "statistic": "confident_self_trained_minus_source_auroc"},
                        {**weight("self_trained_confident"), **weight("source", -1.0)},
                    ),
                    (
                        {**label, "statistic": "remaining_penalty_after_confident_self_training"},
                        {**weight("local"), **weight("self_trained_confident", -1.0)},
                    ),
                ]
                interval_rows.extend(
                    bootstrap_statistics(
                        y,
                        vectors,
                        statistics,
                        args.bootstrap_replicates,
                        args.seed + 800000 + 1000 * backbone_index + 100 * attribute_index + 10 * direction_index,
                    )
                )

    trajectories = pd.DataFrame(trajectory_rows)
    few_shot = pd.DataFrame(few_shot_rows)
    sanity = pd.DataFrame(sanity_rows)
    intervals = pd.DataFrame(interval_rows)
    for frame in (trajectories, few_shot, intervals):
        frame["coordinate_space"] = SHARED_DEMOGRAPHIC_COORDINATE

    keys = ["backbone", "direction", "source", "target", "attribute"]
    trajectories.to_csv(output / "self_training_by_split.csv", index=False)
    aggregate_split_results(trajectories, keys + ["round"], "auroc").to_csv(
        output / "self_training_summary.csv", index=False
    )
    intervals.to_csv(output / "self_training_intervals.csv", index=False)
    few_shot.to_csv(output / "few_shot_by_draw.csv", index=False)
    sanity.to_csv(output / "recovery_sanity_checks.csv", index=False)

    # Paired per draw: gain over the source readout and share of the
    # source-to-ceiling gap recovered (ratio of means, stable when small).
    wide = few_shot.pivot_table(
        index=keys + ["split_index", "k", "draw"], columns="method", values="auroc"
    ).reset_index()
    summary_rows = []
    for group, subset in wide.groupby(keys + ["k"], sort=False):
        ceiling_gap = subset["target_local"] - subset["source_readout"]
        for method in ("scratch", "anchored"):
            gain = subset[method] - subset["source_readout"]
            summary_rows.append(
                {
                    **dict(zip(keys + ["k"], group)),
                    "method": method,
                    "n_draws": int(len(subset)),
                    "mean_auroc": float(subset[method].mean()),
                    "mean_source_auroc": float(subset["source_readout"].mean()),
                    "mean_target_local_auroc": float(subset["target_local"].mean()),
                    "mean_gain_over_source": float(gain.mean()),
                    "gain_draw_p2_5": float(np.quantile(gain, 0.025)),
                    "gain_draw_p97_5": float(np.quantile(gain, 0.975)),
                    "share_of_ceiling_gap_recovered": float(gain.mean() / ceiling_gap.mean()),
                    "anchored_minus_scratch": float((subset["anchored"] - subset["scratch"]).mean())
                    if method == "anchored"
                    else float("nan"),
                }
            )
    pd.DataFrame(summary_rows).to_csv(output / "few_shot_summary.csv", index=False)

    manifest = {
        "purpose": "ceiling probe before any adaptation module; no new method",
        "seed": args.seed,
        "split_seed_scheme": "seed + 101 * split_index (as the extension run)",
        "split_repeats": args.split_repeats,
        "target_local_cv": f"5-fold repeated {args.cv_repeats} times; same seeds as the extension run",
        "self_training": (
            f"{args.self_training_rounds} rounds; all target patients pseudo-labelled at the "
            "source training class proportion; L2 logistic refit with the source-selected C; "
            "no target labels used; evaluated on all labelled target patients"
        ),
        "few_shot": (
            f"k in {shots}, {args.draws} class-stratified draws per split; scratch = L2 logistic "
            "on k patients; anchored = L2-SP towards the source weights with the same C; "
            "evaluated on the remaining target patients"
        ),
        "few_shot_interval": "2.5/97.5 percentiles over draws and splits (variability, not a CI)",
        "bootstrap": (
            f"{args.bootstrap_replicates} target-patient resamples shared across splits; "
            "95% percentile intervals of split means, conditional on fitted readouts"
        ),
        "coordinate_space": SHARED_DEMOGRAPHIC_COORDINATE,
        "privacy": "aggregate outputs only; no identifiers, embeddings, models, or row-level predictions",
    }
    (output / "recovery_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if not bool(sanity["passed"].all()):
        print("SANITY FAILURE in the anchored estimator", file=sys.stderr)
        print(sanity[~sanity["passed"]].to_string(), file=sys.stderr)
        return 3
    print(f"Completed recovery probe. Aggregate outputs written to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
