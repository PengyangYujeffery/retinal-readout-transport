#!/usr/bin/env python3
"""Validate the aggregate contract of the ICASSP extension experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


BACKBONES = {"vits16", "convnext_tiny"}
DIRECTIONS = {"brset_to_mbrset", "mbrset_to_brset", "brset_to_mbrset_sizematched"}
ATTRIBUTES = {"sex", "age65"}
METHODS = {"unaligned", "mean_variance", "coral", "subspace", "ot_gaussian", "ot_entropic"}
# Per backbone and attribute and arm: target-local, then external AUROC and penalty for each of the
# six methods, then the change against unaligned for each of the five repairs; plus 2 size-matched
# contrasts per cell.
BOOTSTRAP_ROWS_PER_CELL = 3 * (1 + 2 * len(METHODS) + (len(METHODS) - 1)) + 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", required=True, type=Path)
    parser.add_argument("--expected-splits", type=int, default=5)
    parser.add_argument("--expected-bootstrap", type=int, default=2000)
    parser.add_argument("--require-reproduction-gate", action="store_true")
    return parser.parse_args()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def require_columns(frame: pd.DataFrame, filename: str, columns: set) -> None:
    missing = columns - set(frame.columns)
    require(not missing, f"{filename} lacks required columns: {sorted(missing)}")


def require_unit_interval(frame: pd.DataFrame, column: str, filename: str) -> None:
    values = pd.to_numeric(frame[column], errors="coerce")
    require(bool(np.isfinite(values).all()), f"{filename}.{column} contains non-finite values")
    require(bool(values.between(0, 1).all()), f"{filename}.{column} leaves [0, 1]")


def main() -> int:
    args = parse_args()
    root = args.results_dir.expanduser().resolve()
    required_files = {
        "transport_metrics_by_split.csv",
        "transport_metrics_summary.csv",
        "transport_penalties_by_split.csv",
        "transport_summary.csv",
        "direction_geometry_by_split.csv",
        "direction_geometry_summary.csv",
        "alignment_moment_diagnostics.csv",
        "alignment_effects.csv",
        "outcome_stratified_transport.csv",
        "outcome_stratified_summary.csv",
        "sanity_checks.csv",
        "bootstrap_intervals.csv",
        "icassp_extension_manifest.json",
    }
    if args.require_reproduction_gate:
        required_files.add("reproduction_gate.csv")
    missing_files = [name for name in sorted(required_files) if not (root / name).is_file()]
    require(not missing_files, f"missing ICASSP outputs: {missing_files}")

    metrics = pd.read_csv(root / "transport_metrics_by_split.csv")
    penalties = pd.read_csv(root / "transport_penalties_by_split.csv")
    geometry = pd.read_csv(root / "direction_geometry_by_split.csv")
    moments = pd.read_csv(root / "alignment_moment_diagnostics.csv")
    stratified = pd.read_csv(root / "outcome_stratified_transport.csv")
    sanity = pd.read_csv(root / "sanity_checks.csv")
    intervals = pd.read_csv(root / "bootstrap_intervals.csv")

    common = {"backbone", "direction", "source", "target", "split_index"}
    require_columns(
        metrics,
        "transport_metrics_by_split.csv",
        common | {"attribute", "evaluation", "method", "auroc", "best_C", "coordinate_space"},
    )
    require_columns(
        penalties,
        "transport_penalties_by_split.csv",
        common
        | {
            "attribute",
            "method",
            "target_local_auroc",
            "external_auroc",
            "transport_penalty",
            "source_train_n",
            "size_capped",
        },
    )
    require_columns(
        geometry,
        "direction_geometry_by_split.csv",
        common | {"attribute", "probe_coefficient_cosine", "centroid_direction_cosine"},
    )
    require_columns(
        moments,
        "alignment_moment_diagnostics.csv",
        common
        | {"method", "mean_l2_distance", "covariance_frobenius_distance", "covariance_relative_distance"},
    )
    require_columns(
        stratified,
        "outcome_stratified_transport.csv",
        common | {"attribute", "outcome", "method", "status", "transport_penalty"},
    )
    require_columns(
        intervals,
        "bootstrap_intervals.csv",
        {"backbone", "attribute", "target", "direction", "statistic", "method", "estimate",
         "ci_low", "ci_high", "valid_replicates", "n_splits"},
    )

    # Coverage of the full design.
    require(set(penalties["backbone"]) == BACKBONES, "penalties do not cover both backbones")
    require(set(penalties["direction"]) == DIRECTIONS, "penalties do not cover all three arms")
    require(set(penalties["attribute"]) == ATTRIBUTES, "penalties do not cover both attributes")
    require(set(penalties["method"]) == METHODS, "penalties do not cover all alignment methods")
    require(
        penalties["split_index"].nunique() == args.expected_splits,
        "unexpected number of source splits",
    )
    expected_rows = len(BACKBONES) * len(DIRECTIONS) * len(ATTRIBUTES) * len(METHODS) * args.expected_splits
    require(
        len(penalties) == expected_rows,
        f"expected {expected_rows} transport contrasts, found {len(penalties)}",
    )
    require(
        not penalties.duplicated(["backbone", "direction", "attribute", "method", "split_index"]).any(),
        "duplicate transport contrasts",
    )
    require_unit_interval(metrics, "auroc", "transport_metrics_by_split.csv")
    require_unit_interval(penalties, "target_local_auroc", "transport_penalties_by_split.csv")
    require_unit_interval(penalties, "external_auroc", "transport_penalties_by_split.csv")
    require(
        bool(np.isfinite(geometry[["probe_coefficient_cosine", "centroid_direction_cosine"]].to_numpy(float)).all()),
        "direction geometry contains non-finite values",
    )
    require(
        bool(np.isfinite(moments[["mean_l2_distance", "covariance_frobenius_distance",
                                  "covariance_relative_distance"]].to_numpy(float)).all()),
        "moment diagnostics contain non-finite values",
    )

    # The size-matched probe must use exactly as many labelled source patients
    # as the reverse-direction probe of the same split, and only it is capped.
    base = penalties[penalties["method"] == "unaligned"].set_index(["backbone", "split_index", "attribute"])
    matched = base[base["direction"] == "brset_to_mbrset_sizematched"]["source_train_n"]
    reverse = base[base["direction"] == "mbrset_to_brset"]["source_train_n"]
    full = base[base["direction"] == "brset_to_mbrset"]["source_train_n"]
    require(matched.sort_index().equals(reverse.sort_index()), "size-matched n differs from reverse n")
    require(bool((full.sort_index() > matched.sort_index()).all()), "size-matched arm is not smaller")
    capped = penalties.groupby("direction")["size_capped"].agg(lambda s: set(s.astype(str)))
    require(capped["brset_to_mbrset_sizematched"] == {"True"}, "size-matched rows are not capped")
    require(
        capped["brset_to_mbrset"] == {"False"} and capped["mbrset_to_brset"] == {"False"},
        "uncapped arms are marked capped",
    )

    require(bool(sanity["passed"].astype(str).eq("True").all()), "a translation sanity check failed")

    # Bootstrap: complete, finite, adequately replicated, and consistent with
    # the independently computed per-split penalties.
    expected_interval_rows = len(BACKBONES) * len(ATTRIBUTES) * BOOTSTRAP_ROWS_PER_CELL
    require(
        len(intervals) == expected_interval_rows,
        f"expected {expected_interval_rows} bootstrap rows, found {len(intervals)}",
    )
    require(
        bool(np.isfinite(intervals[["estimate", "ci_low", "ci_high"]].to_numpy(float)).all()),
        "bootstrap intervals contain non-finite values",
    )
    require(bool((intervals["ci_low"] <= intervals["ci_high"]).all()), "inverted bootstrap interval")
    require(
        bool((intervals["valid_replicates"] >= 0.9 * args.expected_bootstrap).all()),
        "too few valid bootstrap replicates",
    )
    split_mean = penalties.groupby(["backbone", "direction", "attribute", "method"])["transport_penalty"].mean()
    boot_penalty = intervals[intervals["statistic"] == "transport_penalty"].set_index(
        ["backbone", "direction", "attribute", "method"]
    )["estimate"]
    joined = pd.concat([split_mean.rename("split"), boot_penalty.rename("boot")], axis=1)
    require(not joined.isna().any().any(), "bootstrap penalties do not match split penalties")
    require(
        bool(np.allclose(joined["split"], joined["boot"], atol=1e-9, rtol=0)),
        "bootstrap point estimates disagree with the per-split penalties",
    )

    # Aggregate files may never expose row-level identity or predictions.
    prohibited = {"patient_key", "image_key", "probability", "prediction", "embedding"}
    for filename in sorted(name for name in required_files if name.endswith(".csv")):
        columns = {column.casefold() for column in pd.read_csv(root / filename, nrows=1).columns}
        overlap = columns & prohibited
        require(not overlap, f"{filename} exposes prohibited columns: {sorted(overlap)}")

    manifest = json.loads((root / "icassp_extension_manifest.json").read_text(encoding="utf-8"))
    require(manifest.get("split_repeats") == args.expected_splits, "manifest split count mismatch")
    require(
        "without target demographic labels" in manifest.get("alignment_contract", ""),
        "manifest does not state the label-free alignment contract",
    )
    if args.require_reproduction_gate:
        gate = pd.read_csv(root / "reproduction_gate.csv")
        require(len(gate) == 12, f"reproduction gate should cover 12 reference rows, found {len(gate)}")
        require(bool(gate["passed"].astype(str).eq("True").all()), "reproduction gate failed")
        require(manifest.get("reproduction_gate") == "passed", "manifest does not record a passed gate")
    print("ICASSP extension aggregate contract: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
