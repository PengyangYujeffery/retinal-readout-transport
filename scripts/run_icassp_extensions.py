#!/usr/bin/env python3
"""ICASSP extension experiments for demographic-readout transport.

The script uses only the released, frozen embeddings. It writes aggregate
metrics and diagnostics; it never writes identifiers, embeddings, fitted
models, or row-level predictions.

Prespecified analyses
---------------------
1. Repeat the complete readout-transport audit over independent patient-level
   source splits. Split ``s`` uses seed ``seed + 101 * s``, the same scheme as
   the earlier intervention loop, so split 0 is the reference primary split. When
   ``--reference-probe-metrics`` is given, split 0 of BRSET -> mBRSET must
   reproduce the reference demographic-probe AUROCs or the run fails.
2. Evaluate transport in both directions (BRSET -> mBRSET, mBRSET -> BRSET)
   and in a size-matched BRSET -> mBRSET arm whose demographic probe is tuned
   and fitted on as many labelled source patients as the mBRSET -> BRSET probe
   (class proportions preserved). The arm keeps the coordinate, target
   patients and alignment maps of the full BRSET -> mBRSET arm, so source
   label count is the only change. It separates sample size from setting when
   the two directions differ.
3. Compare the unchanged source readout with five label-free target-to-source
   alignments of the target patient vectors: diagonal mean/variance matching,
   Ledoit--Wolf CORAL, subspace alignment, the Gaussian optimal-transport map
   and an entropic optimal-transport barycentric map. The first four are
   affine; the last is not, which tests the reach of the affine argument. None
   uses target demographics or outcomes.
4. Secondary: demographic probes tuned, fitted and evaluated within
   referable-DR-negative and referable-DR-positive patients.

Uncertainty
-----------
Split-level results are summarised by mean, SD and range. Split-mean
estimates carry 95% percentile intervals from a target-patient bootstrap
conditional on the fitted probes and alignment maps. A direction's target
cohort is identical in every split, so each replicate resamples target
patients once and reuses them for every split, method and arm that shares the
target. With five splits an exact sign-flip test cannot fall below p = 0.0625,
so no split-level p-value is reported.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.special import logsumexp
from sklearn.covariance import LedoitWolf
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_experiments import BACKBONES, load_dataset, split_source_patients  # noqa: E402
from run_representation_audit import (  # noqa: E402
    ATTRIBUTES,
    SHARED_DEMOGRAPHIC_COORDINATE,
    attribute_arrays,
    centroid_direction,
    choose_attribute_c,
    cosine,
    fit_probe,
    patient_masks,
    patient_table,
    probe_direction_on_common_scale,
    repeated_local_predictions,
)


FORWARD = "brset_to_mbrset"
REVERSE = "mbrset_to_brset"
SIZE_MATCHED = "brset_to_mbrset_sizematched"
DIRECTIONS: Tuple[Tuple[str, str, str], ...] = (
    (FORWARD, "brset", "mbrset"),
    (REVERSE, "mbrset", "brset"),
)
ARMS = (FORWARD, REVERSE, SIZE_MATCHED)
ARM_DATASETS = {FORWARD: ("brset", "mbrset"), REVERSE: ("mbrset", "brset"), SIZE_MATCHED: ("brset", "mbrset")}
ALIGNMENT_METHODS = (
    "unaligned",
    "mean_variance",
    "coral",
    "subspace",
    "ot_gaussian",
    "ot_entropic",
)
TRANSLATION_TOLERANCE = 1e-9

# Fixed before the first run so that no hyper-parameter is chosen after seeing a result.
# Subspace alignment and the Gaussian OT map are
# affine, so Eq. (2) of the paper covers them; the entropic-OT barycentric map is not, which is
# exactly why it is included.
SUBSPACE_VARIANCE = 0.95
# Amended before any AUROC was computed, on a geometry-only diagnostic: at a fixed scale
# of 0.05 the Sinkhorn plan is so diffuse that each target patient is spread over about 1,400 source
# patients and the mapped cloud keeps 0.06% of the source variance -- every patient lands on almost
# the same point, which would make this repair fail for a numerical reason. The scale is therefore
# chosen per split by an outcome-blind rule: the smallest in the grid whose iteration converges,
# i.e. the sharpest transport plan that can be computed reliably. No label or AUROC enters the choice.
OT_REGULARISATION_GRID = (0.001, 0.002, 0.005, 0.01, 0.02, 0.05)
# Iteration budget per scale, not a convergence target: a smaller scale needs more iterations (585 at
# 0.001 and 161 at 0.002 in the diagnostic), so a budget of 300 selects the sharpest plan that is
# affordable 20 times over and bounds the worst case at 6 x 300 iterations per split. Compute only;
# no label, AUROC or penalty enters it.
OT_ITERATIONS = 300
OT_TOLERANCE = 1e-7
OT_MINIMUM_VARIANCE_KEPT = 0.01
OT_SOURCE_CAP = 3000
OT_SEED_OFFSET = 77

# reference demographic_probe_metrics.csv evaluation -> (evaluation, method) here.
REFERENCE_EVALUATIONS = {
    "source_internal": ("source_internal", "unaligned"),
    "target_transported": ("target_external", "unaligned"),
    "target_local_cv": ("target_local_cv", "target_local"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--seed",
        type=int,
        default=20260902,
        help="base seed; split s uses seed + 101*s, so split 0 is the reference primary split",
    )
    parser.add_argument("--split-repeats", type=int, default=5)
    parser.add_argument("--cv-repeats", type=int, default=5)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--minimum-cv-class", type=int, default=5)
    parser.add_argument(
        "--reference-probe-metrics",
        type=Path,
        default=None,
        help="reference demographic_probe_metrics.csv; split 0 must reproduce it",
    )
    parser.add_argument("--reproduction-tolerance", type=float, default=1e-6)
    return parser.parse_args()


def class_support(y: np.ndarray) -> Dict[str, int]:
    y = np.asarray(y, dtype=int)
    return {
        "n": int(len(y)),
        "n_negative": int(np.sum(y == 0)),
        "n_positive": int(np.sum(y == 1)),
    }


def has_two_classes(y: np.ndarray, minimum_each: int = 1) -> bool:
    support = class_support(y)
    return min(support["n_negative"], support["n_positive"]) >= minimum_each


def patient_means(frame: pd.DataFrame, x: np.ndarray, metadata: pd.DataFrame) -> np.ndarray:
    """Patient-mean vectors in exactly the row order of ``patient_table``.

    ``patient_table`` recomputes per-patient metadata with a Python loop; the
    metadata does not depend on the feature transform, so it is built once
    per dataset and only the means are recomputed per split.
    """
    codes, keys = pd.factorize(frame["patient_key"], sort=False)
    if [str(key) for key in keys] != metadata["patient_key"].astype(str).tolist():
        raise AssertionError("patient order differs from the patient metadata table")
    counts = np.bincount(codes).astype(np.float64)
    sums = np.zeros((len(keys), x.shape[1]), dtype=np.float64)
    np.add.at(sums, codes, x)
    return (sums / counts[:, None]).astype(np.float32)


def symmetric_matrix_power(matrix: np.ndarray, power: float) -> np.ndarray:
    """Stable power of a symmetric positive-semidefinite matrix."""
    matrix = np.asarray(matrix, dtype=np.float64)
    matrix = 0.5 * (matrix + matrix.T)
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    scale = max(float(np.max(eigenvalues)), 1.0)
    eigenvalues = np.clip(eigenvalues, scale * 1e-10, None)
    return (eigenvectors * np.power(eigenvalues, power)) @ eigenvectors.T


def mean_variance_target_to_source(
    source_reference: np.ndarray, target: np.ndarray
) -> np.ndarray:
    """Match each target feature's first two marginal moments to the source."""
    source_reference = np.asarray(source_reference, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    source_sd = np.maximum(source_reference.std(axis=0, ddof=0), 1e-6)
    target_sd = np.maximum(target.std(axis=0, ddof=0), 1e-6)
    aligned = (target - target.mean(axis=0)) / target_sd * source_sd + source_reference.mean(axis=0)
    if not np.isfinite(aligned).all():
        raise FloatingPointError("mean/variance alignment produced non-finite values")
    return aligned.astype(np.float32)


def coral_target_to_source(
    source_reference: np.ndarray, target: np.ndarray
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Map unlabelled target vectors onto the source mean and covariance.

    Target vectors are whitened with the target covariance and re-coloured
    with the source covariance, so the source readout can be applied
    unchanged. Ledoit--Wolf shrinkage keeps both factors well conditioned when
    the dimension is large relative to the cohort.
    """
    source_reference = np.asarray(source_reference, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    source_fit = LedoitWolf().fit(source_reference)
    target_fit = LedoitWolf().fit(target)
    transform = symmetric_matrix_power(target_fit.covariance_, -0.5) @ symmetric_matrix_power(
        source_fit.covariance_, 0.5
    )
    aligned = (target - target.mean(axis=0)) @ transform + source_reference.mean(axis=0)
    if not np.isfinite(aligned).all():
        raise FloatingPointError("CORAL alignment produced non-finite values")
    diagnostic = {
        "source_shrinkage": float(source_fit.shrinkage_),
        "target_shrinkage": float(target_fit.shrinkage_),
    }
    return aligned.astype(np.float32), diagnostic


def subspace_alignment_target_to_source(
    source_reference: np.ndarray, target: np.ndarray
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Subspace alignment (Fernando et al., ICCV 2013) as a target-to-source map.

    Target vectors are expressed in the leading target principal directions,
    those coordinates are read in the matching source directions, and the
    result is recentred on the source mean, so the fixed source probe applies
    unchanged. The number of components is the smallest that explains
    ``SUBSPACE_VARIANCE`` of the source-training variance, which uses no target
    labels. The map is affine.
    """
    source_reference = np.asarray(source_reference, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    source_mean = source_reference.mean(axis=0)
    target_mean = target.mean(axis=0)
    _, source_values, source_vt = np.linalg.svd(source_reference - source_mean, full_matrices=False)
    _, _, target_vt = np.linalg.svd(target - target_mean, full_matrices=False)
    spectrum = np.square(source_values)
    explained = np.cumsum(spectrum) / max(float(spectrum.sum()), 1e-12)
    components = int(min(np.searchsorted(explained, SUBSPACE_VARIANCE) + 1, len(source_vt), len(target_vt)))
    aligned = (target - target_mean) @ (target_vt[:components].T @ source_vt[:components]) + source_mean
    if not np.isfinite(aligned).all():
        raise FloatingPointError("subspace alignment produced non-finite values")
    return aligned.astype(np.float32), {
        "subspace_components": float(components),
        "subspace_source_variance": float(explained[components - 1]),
    }


def gaussian_ot_target_to_source(source_reference: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Monge map between Gaussians fitted to the two clouds; affine.

    ``A = S_t^{-1/2} (S_t^{1/2} S_s S_t^{1/2})^{1/2} S_t^{-1/2}`` is the
    squared-cost optimal map between the Ledoit--Wolf Gaussians. CORAL maps the
    same moments with a different, arbitrary rotation, so the pair separates
    "matching the moments" from "matching them optimally".
    """
    source_reference = np.asarray(source_reference, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    source_covariance = LedoitWolf().fit(source_reference).covariance_
    target_covariance = LedoitWolf().fit(target).covariance_
    root = symmetric_matrix_power(target_covariance, 0.5)
    inverse_root = symmetric_matrix_power(target_covariance, -0.5)
    middle = symmetric_matrix_power(root @ source_covariance @ root, 0.5)
    transform = inverse_root @ middle @ inverse_root
    aligned = (target - target.mean(axis=0)) @ transform + source_reference.mean(axis=0)
    if not np.isfinite(aligned).all():
        raise FloatingPointError("Gaussian OT alignment produced non-finite values")
    return aligned.astype(np.float32)


def entropic_ot_target_to_source(
    source_reference: np.ndarray, target: np.ndarray, seed: int
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Entropic OT with a barycentric map: the one repair that is not affine.

    Log-domain Sinkhorn on squared Euclidean cost with uniform marginals; each
    target patient is moved to the barycentre of the source-training patients
    under its row of the transport plan. The source side is subsampled to at
    most ``OT_SOURCE_CAP`` patients with a fixed seed to bound the plan. The
    regularisation is the smallest in ``OT_REGULARISATION_GRID`` whose
    iteration converges, which is the sharpest plan that can be computed
    reliably; a plan that collapses the cloud is refused rather than used. No
    target labels are used anywhere in the choice.
    """
    source_reference = np.asarray(source_reference, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if len(source_reference) > OT_SOURCE_CAP:
        rng = np.random.default_rng(seed + OT_SEED_OFFSET)
        support = source_reference[np.sort(rng.choice(len(source_reference), OT_SOURCE_CAP, replace=False))]
    else:
        support = source_reference
    cost = (
        np.square(target).sum(axis=1)[:, None]
        + np.square(support).sum(axis=1)[None, :]
        - 2.0 * (target @ support.T)
    )
    np.maximum(cost, 0.0, out=cost)
    median_cost = max(float(np.median(cost)), 1e-12)
    log_a = -math.log(len(target))
    log_b = -math.log(len(support))
    source_variance = max(float(np.asarray(source_reference).var(axis=0).sum()), 1e-12)
    for scale in OT_REGULARISATION_GRID:
        regularisation = scale * median_cost
        potential_target = np.zeros(len(target), dtype=np.float64)
        potential_source = np.zeros(len(support), dtype=np.float64)
        converged = False
        iterations = 0
        for iterations in range(1, OT_ITERATIONS + 1):
            previous = potential_target
            potential_target = -regularisation * logsumexp(
                (potential_source[None, :] - cost) / regularisation + log_b, axis=1
            )
            potential_source = -regularisation * logsumexp(
                (potential_target[:, None] - cost) / regularisation + log_a, axis=0
            )
            if float(np.max(np.abs(potential_target - previous))) <= OT_TOLERANCE * regularisation:
                converged = True
                break
        if not converged:
            continue
        plan = np.exp(
            (potential_target[:, None] + potential_source[None, :] - cost) / regularisation
        )
        row_mass = plan.sum(axis=1, keepdims=True)
        if float(row_mass.min()) <= 0.0:
            continue
        aligned = (plan / row_mass) @ support
        if not np.isfinite(aligned).all():
            continue
        kept = float(np.asarray(aligned).var(axis=0).sum() / source_variance)
        if kept < OT_MINIMUM_VARIANCE_KEPT:
            # A plan this diffuse maps every patient onto nearly one point; report the failure
            # instead of scoring a collapsed cloud.
            raise FloatingPointError(
                f"entropic OT collapsed the target cloud at scale {scale} (variance kept {kept:.4g})"
            )
        return aligned.astype(np.float32), {
            "ot_regularisation": float(regularisation),
            "ot_regularisation_scale": float(scale),
            "ot_iterations": float(iterations),
            "ot_variance_kept": kept,
            "ot_support_patients": float(len(support)),
        }
    raise FloatingPointError("entropic OT did not converge at any regularisation in the grid")


def moment_diagnostics(
    source_reference: np.ndarray, target_by_method: Mapping[str, np.ndarray]
) -> List[Dict[str, object]]:
    source_reference = np.asarray(source_reference, dtype=np.float64)
    source_mean = source_reference.mean(axis=0)
    source_covariance = np.cov(source_reference, rowvar=False, ddof=1)
    source_norm = max(float(np.linalg.norm(source_covariance, ord="fro")), 1e-12)
    rows: List[Dict[str, object]] = []
    for method, values in target_by_method.items():
        values = np.asarray(values, dtype=np.float64)
        covariance_distance = float(
            np.linalg.norm(np.cov(values, rowvar=False, ddof=1) - source_covariance, ord="fro")
        )
        rows.append(
            {
                "method": method,
                "mean_l2_distance": float(np.linalg.norm(values.mean(axis=0) - source_mean)),
                "covariance_frobenius_distance": covariance_distance,
                "covariance_relative_distance": covariance_distance / source_norm,
            }
        )
    return rows


def prepare_context(
    frames: Mapping[str, pd.DataFrame],
    features: Mapping[str, np.ndarray],
    metadata: Mapping[str, pd.DataFrame],
    source_name: str,
    target_name: str,
    split_seed: int,
) -> Dict[str, object]:
    """Source split, shared coordinate and alignment maps for one direction.

    Mirrors the reference transport analysis: features are standardised with
    source-training-image statistics, both cohorts are transformed and then
    averaged by patient. Alignment is estimated from source-training patients
    and all target patients, without target labels.
    """
    source_frame = frames[source_name]
    image_split = split_source_patients(source_frame, split_seed)
    scaler = StandardScaler().fit(features[source_name][image_split["train"]])
    source_pz = patient_means(
        source_frame,
        scaler.transform(features[source_name]).astype(np.float32),
        metadata[source_name],
    )
    target_pz = patient_means(
        frames[target_name],
        scaler.transform(features[target_name]).astype(np.float32),
        metadata[target_name],
    )
    split = patient_masks(metadata[source_name], source_frame, split_seed)
    reference = source_pz[split["train"]]
    target_coral, shrinkage = coral_target_to_source(reference, target_pz)
    target_subspace, subspace_diagnostic = subspace_alignment_target_to_source(reference, target_pz)
    target_ot_entropic, entropic_diagnostic = entropic_ot_target_to_source(
        reference, target_pz, split_seed
    )
    return {
        "source": source_name,
        "target": target_name,
        "source_meta": metadata[source_name],
        "target_meta": metadata[target_name],
        "split": split,
        "source_pz": source_pz,
        "reference": reference,
        "target_pz": {
            "unaligned": target_pz,
            "mean_variance": mean_variance_target_to_source(reference, target_pz),
            "coral": target_coral,
            "subspace": target_subspace,
            "ot_gaussian": gaussian_ot_target_to_source(reference, target_pz),
            "ot_entropic": target_ot_entropic,
        },
        "mean_shift": (reference.mean(axis=0) - target_pz.mean(axis=0)).astype(np.float32),
        "shrinkage": {**shrinkage, **subspace_diagnostic, **entropic_diagnostic},
    }


def labelled_source(ctx: Mapping[str, object], attribute: str) -> Tuple[np.ndarray, np.ndarray]:
    valid, y_valid = attribute_arrays(ctx["source_meta"], attribute)
    y = np.full(len(valid), -1, dtype=int)
    y[valid] = y_valid
    return valid, y


def cap_mask(mask: np.ndarray, labels: np.ndarray, size: int, seed: int) -> np.ndarray:
    """Class-stratified subsample of ``mask`` with exactly ``size`` members."""
    index = np.flatnonzero(mask)
    if size >= len(index):
        raise ValueError(f"size-matched cap {size} is not below the available {len(index)}")
    chosen, _ = train_test_split(
        index, train_size=int(size), random_state=seed, stratify=labels[index]
    )
    capped = np.zeros_like(mask)
    capped[chosen] = True
    return capped


def labelled_counts(ctx: Mapping[str, object], attribute: str) -> Tuple[int, int]:
    valid, _ = labelled_source(ctx, attribute)
    split = ctx["split"]
    return int((split["train"] & valid).sum()), int((split["validation"] & valid).sum())


def evaluate_readout(
    *,
    ctx: Mapping[str, object],
    labels: Mapping[str, object],
    attribute: str,
    cv_seed: int,
    cv_repeats: int,
    minimum_cv_class: int,
    cap: Optional[Tuple[int, int]] = None,
    cap_seed: int = 0,
) -> Dict[str, object]:
    source_valid, source_y = labelled_source(ctx, attribute)
    target_valid, target_y = attribute_arrays(ctx["target_meta"], attribute)
    split = ctx["split"]
    train = split["train"] & source_valid
    validation = split["validation"] & source_valid
    internal = split["internal_test"] & source_valid
    if cap is not None:
        train = cap_mask(train, source_y, cap[0], cap_seed)
        validation = cap_mask(validation, source_y, cap[1], cap_seed + 1)
    for name, mask in (("source train", train), ("source validation", validation)):
        if not has_two_classes(source_y[mask]):
            raise ValueError(f"{labels['direction']} {attribute}: {name} lacks two classes")
    if not has_two_classes(target_y, minimum_cv_class):
        raise ValueError(
            f"{labels['direction']} {attribute}: target has fewer than "
            f"{minimum_cv_class} patients per class"
        )

    source_x = ctx["source_pz"]
    best_c, _ = choose_attribute_c(
        source_x, ctx["source_meta"], attribute, train, validation, scale=False
    )
    model = fit_probe(source_x[train], source_y[train], best_c, scale=False)
    target_x = {method: ctx["target_pz"][method][target_valid] for method in ALIGNMENT_METHODS}
    local = repeated_local_predictions(
        target_x["unaligned"], target_y, best_c, cv_seed, cv_repeats, scale=False
    )
    external = {
        method: model.predict_proba(target_x[method])[:, 1] for method in ALIGNMENT_METHODS
    }
    local_auroc = float(roc_auc_score(target_y, local))
    external_auroc = {
        method: float(roc_auc_score(target_y, probability))
        for method, probability in external.items()
    }

    # A pure translation changes a linear score by a constant, so it must
    # leave AUROC unchanged; any other outcome means the pipeline is broken.
    # Decision scores in float64 avoid sigmoid saturation and float32 rounding.
    unshifted = target_x["unaligned"].astype(np.float64)
    shifted = unshifted + ctx["mean_shift"].astype(np.float64)
    translation_difference = abs(
        float(roc_auc_score(target_y, model.decision_function(shifted)))
        - float(roc_auc_score(target_y, model.decision_function(unshifted)))
    )

    train_support = class_support(source_y[train])
    common = {
        **labels,
        "attribute": attribute,
        "best_C": best_c,
        "source_train_n": train_support["n"],
        "source_train_positive": train_support["n_positive"],
        "source_validation_n": int(validation.sum()),
        "size_capped": cap is not None,
    }
    metric_rows: List[Dict[str, object]] = []
    if has_two_classes(source_y[internal]):
        metric_rows.append(
            {
                **common,
                "evaluation": "source_internal",
                "method": "unaligned",
                "auroc": float(
                    roc_auc_score(source_y[internal], model.predict_proba(source_x[internal])[:, 1])
                ),
                **class_support(source_y[internal]),
            }
        )
    metric_rows.append(
        {
            **common,
            "evaluation": "target_local_cv",
            "method": "target_local",
            "auroc": local_auroc,
            **class_support(target_y),
        }
    )
    penalty_rows: List[Dict[str, object]] = []
    for method in ALIGNMENT_METHODS:
        metric_rows.append(
            {
                **common,
                "evaluation": "target_external",
                "method": method,
                "auroc": external_auroc[method],
                **class_support(target_y),
            }
        )
        penalty_rows.append(
            {
                **common,
                "method": method,
                "target_local_auroc": local_auroc,
                "external_auroc": external_auroc[method],
                "transport_penalty": local_auroc - external_auroc[method],
                **class_support(target_y),
            }
        )

    geometry = {
        **common,
        "probe_coefficient_cosine": cosine(
            probe_direction_on_common_scale(source_x[train], source_y[train], best_c),
            probe_direction_on_common_scale(target_x["unaligned"], target_y, best_c),
        ),
        "centroid_direction_cosine": cosine(
            centroid_direction(source_x[train], source_y[train]),
            centroid_direction(target_x["unaligned"], target_y),
        ),
    }
    sanity = {
        **labels,
        "attribute": attribute,
        "check": "translation_invariance_external_auroc",
        "value": translation_difference,
        "tolerance": TRANSLATION_TOLERANCE,
        "passed": bool(translation_difference <= TRANSLATION_TOLERANCE),
    }
    arrays = {"y": target_y, "local": local, **external}
    return {
        "metrics": metric_rows,
        "penalties": penalty_rows,
        "geometry": geometry,
        "sanity": sanity,
        "arrays": arrays,
    }


def evaluate_outcome_strata(
    *,
    ctx: Mapping[str, object],
    labels: Mapping[str, object],
    attribute: str,
    cv_seed: int,
    cv_repeats: int,
    minimum_cv_class: int,
) -> List[Dict[str, object]]:
    source_valid, source_y = labelled_source(ctx, attribute)
    target_valid, target_valid_y = attribute_arrays(ctx["target_meta"], attribute)
    target_y = np.full(len(target_valid), -1, dtype=int)
    target_y[target_valid] = target_valid_y
    source_outcome = pd.to_numeric(ctx["source_meta"]["target"], errors="coerce").to_numpy()
    target_outcome = pd.to_numeric(ctx["target_meta"]["target"], errors="coerce").to_numpy()
    split = ctx["split"]
    source_x = ctx["source_pz"]
    rows: List[Dict[str, object]] = []
    for outcome in (0, 1):
        train = split["train"] & source_valid & (source_outcome == outcome)
        validation = split["validation"] & source_valid & (source_outcome == outcome)
        target_mask = target_valid & (target_outcome == outcome)
        common = {**labels, "attribute": attribute, "outcome": outcome}
        status = "complete"
        if not has_two_classes(source_y[train]) or not has_two_classes(source_y[validation]):
            status = "insufficient_source_class_support"
        elif not has_two_classes(target_y[target_mask], minimum_cv_class):
            status = "insufficient_target_class_support"
        if status != "complete":
            rows.append(
                {
                    **common,
                    "method": "not_evaluated",
                    "status": status,
                    "auroc": math.nan,
                    "target_local_auroc": math.nan,
                    "transport_penalty": math.nan,
                    **class_support(target_y[target_mask]),
                }
            )
            continue
        # C is selected strictly inside the source outcome stratum.
        best_c, _ = choose_attribute_c(
            source_x, ctx["source_meta"], attribute, train, validation, scale=False
        )
        model = fit_probe(source_x[train], source_y[train], best_c, scale=False)
        y = target_y[target_mask]
        local_auroc = float(
            roc_auc_score(
                y,
                repeated_local_predictions(
                    ctx["target_pz"]["unaligned"][target_mask],
                    y,
                    best_c,
                    cv_seed + outcome,
                    cv_repeats,
                    scale=False,
                ),
            )
        )
        for method in ALIGNMENT_METHODS:
            external_auroc = float(
                roc_auc_score(y, model.predict_proba(ctx["target_pz"][method][target_mask])[:, 1])
            )
            rows.append(
                {
                    **common,
                    "method": method,
                    "status": "complete",
                    "auroc": external_auroc,
                    "target_local_auroc": local_auroc,
                    "transport_penalty": local_auroc - external_auroc,
                    "best_C": best_c,
                    "source_train_n": int(train.sum()),
                    **class_support(y),
                }
            )
    return rows


def aggregate_split_results(
    frame: pd.DataFrame, group_columns: List[str], value_column: str
) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(
            columns=group_columns + ["value", "n_splits", "mean", "sd", "minimum", "maximum"]
        )
    records = []
    for keys, subset in frame.groupby(group_columns, dropna=False, sort=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        values = pd.to_numeric(subset[value_column], errors="coerce").dropna().to_numpy()
        record = dict(zip(group_columns, keys))
        record.update(
            {
                "value": value_column,
                "n_splits": int(len(values)),
                "mean": float(np.mean(values)) if len(values) else math.nan,
                "sd": float(np.std(values, ddof=1)) if len(values) > 1 else math.nan,
                "minimum": float(np.min(values)) if len(values) else math.nan,
                "maximum": float(np.max(values)) if len(values) else math.nan,
            }
        )
        records.append(record)
    return pd.DataFrame(records)


def alignment_effects(penalties: pd.DataFrame) -> pd.DataFrame:
    """Descriptive per-split change in external AUROC after alignment."""
    rows = []
    keys = ["backbone", "direction", "source", "target", "attribute"]
    for group_values, subset in penalties.groupby(keys, dropna=False, sort=False):
        pivot = subset.pivot(index="split_index", columns="method", values="external_auroc")
        for method in [name for name in ALIGNMENT_METHODS if name != "unaligned"]:
            difference = pivot[method] - pivot["unaligned"]
            record = dict(zip(keys, group_values))
            record.update(
                {
                    "method": method,
                    "contrast": "aligned_minus_unaligned_external_auroc",
                    "n_splits": int(difference.notna().sum()),
                    "mean_difference": float(difference.mean()),
                    "minimum_difference": float(difference.min()),
                    "maximum_difference": float(difference.max()),
                    "improved_splits": int((difference > 0).sum()),
                }
            )
            rows.append(record)
    return pd.DataFrame(rows)


def bootstrap_statistics(
    y: np.ndarray,
    vectors: Mapping[Tuple[str, str, int], np.ndarray],
    statistics: Sequence[Tuple[Dict[str, object], Mapping[Tuple[str, str, int], float]]],
    replicates: int,
    seed: int,
) -> List[Dict[str, object]]:
    """Paired target-patient bootstrap of linear combinations of AUROCs.

    Every statistic is a weighted sum of per-split AUROCs computed on the same
    target patients, so one resample is shared by all of them.
    """
    keys = list(vectors)
    position = {key: index for index, key in enumerate(keys)}
    matrix = np.vstack([vectors[key] for key in keys])
    weights = np.zeros((len(statistics), len(keys)))
    for row, (_, weight_map) in enumerate(statistics):
        for key, weight in weight_map.items():
            weights[row, position[key]] += weight
    observed = weights @ np.array([roc_auc_score(y, row) for row in matrix])
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(replicates):
        index = rng.integers(0, len(y), len(y))
        y_sample = y[index]
        if y_sample.min() == y_sample.max():
            continue
        draws.append(weights @ np.array([roc_auc_score(y_sample, row[index]) for row in matrix]))
    draws_array = np.asarray(draws, dtype=float).reshape(-1, len(statistics))
    rows = []
    for row, (label, _) in enumerate(statistics):
        if len(draws_array):
            low, high = np.quantile(draws_array[:, row], [0.025, 0.975])
            # One-sided bootstrap p-value against zero, for the Bonferroni adjustment over the
            # eight settings. (r + 1) / (B + 1) never returns an impossible zero.
            p_value = (int(np.sum(draws_array[:, row] <= 0.0)) + 1) / (len(draws_array) + 1)
        else:
            low, high, p_value = math.nan, math.nan, math.nan
        rows.append(
            {
                **label,
                "estimate": float(observed[row]),
                "ci_low": float(low),
                "ci_high": float(high),
                "p_value_above_zero": float(p_value),
                "valid_replicates": int(len(draws_array)),
            }
        )
    return rows


def bootstrap_group_statistics(
    arms: Mapping[str, List[Mapping[str, np.ndarray]]],
    label: Mapping[str, object],
) -> Tuple[np.ndarray, Dict[Tuple[str, str, int], np.ndarray], list]:
    """Assemble the vectors and weight maps for arms sharing one target."""
    y_reference: Optional[np.ndarray] = None
    vectors: Dict[Tuple[str, str, int], np.ndarray] = {}
    statistics = []
    for arm, splits in arms.items():
        n_splits = len(splits)
        for split_index, arrays in enumerate(splits):
            if y_reference is None:
                y_reference = arrays["y"]
            elif not np.array_equal(y_reference, arrays["y"]):
                raise AssertionError("arms sharing a target do not share target patients")
            for kind in ("local", *ALIGNMENT_METHODS):
                vectors[(arm, kind, split_index)] = arrays[kind]
        mean = 1.0 / n_splits

        def weight(kind: str, sign: float = 1.0, name: str = arm) -> Dict[Tuple[str, str, int], float]:
            return {(name, kind, s): sign * mean for s in range(n_splits)}

        base = {**label, "direction": arm, "n_splits": n_splits}
        statistics.append(({**base, "statistic": "target_local_auroc", "method": "target_local"}, weight("local")))
        for method in ALIGNMENT_METHODS:
            statistics.append(
                ({**base, "statistic": "external_auroc", "method": method}, weight(method))
            )
            statistics.append(
                (
                    {**base, "statistic": "transport_penalty", "method": method},
                    {**weight("local"), **weight(method, -1.0)},
                )
            )
        for method in [name for name in ALIGNMENT_METHODS if name != "unaligned"]:
            statistics.append(
                (
                    {**base, "statistic": "aligned_minus_unaligned_external_auroc", "method": method},
                    {**weight(method), **weight("unaligned", -1.0)},
                )
            )
    if FORWARD in arms and SIZE_MATCHED in arms:
        n_splits = len(arms[FORWARD])
        mean = 1.0 / n_splits
        base = {**label, "direction": SIZE_MATCHED, "n_splits": n_splits, "method": "unaligned"}
        external_difference = {}
        penalty_difference = {}
        for s in range(n_splits):
            external_difference[(SIZE_MATCHED, "unaligned", s)] = mean
            external_difference[(FORWARD, "unaligned", s)] = -mean
            penalty_difference[(SIZE_MATCHED, "local", s)] = mean
            penalty_difference[(SIZE_MATCHED, "unaligned", s)] = -mean
            penalty_difference[(FORWARD, "local", s)] = -mean
            penalty_difference[(FORWARD, "unaligned", s)] = mean
        statistics.append(
            ({**base, "statistic": "sizematched_minus_full_external_auroc"}, external_difference)
        )
        statistics.append(
            ({**base, "statistic": "sizematched_minus_full_transport_penalty"}, penalty_difference)
        )
    return y_reference, vectors, statistics


def reproduction_gate(
    metrics: pd.DataFrame, reference_path: Path, tolerance: float
) -> pd.DataFrame:
    reference = pd.read_csv(reference_path)
    rows = []
    for record in reference.to_dict("records"):
        evaluation, method = REFERENCE_EVALUATIONS[record["evaluation"]]
        match = metrics[
            (metrics["backbone"] == record["backbone"])
            & (metrics["attribute"] == record["attribute"])
            & (metrics["direction"] == FORWARD)
            & (metrics["split_index"] == 0)
            & (metrics["evaluation"] == evaluation)
            & (metrics["method"] == method)
        ]
        if len(match) != 1:
            raise AssertionError(
                f"reproduction gate: {len(match)} rows for {record['backbone']} "
                f"{record['attribute']} {record['evaluation']}"
            )
        reproduced = match.iloc[0]
        difference = abs(float(reproduced["auroc"]) - float(record["auroc"]))
        same_c = math.isclose(float(reproduced["best_C"]), float(record["best_C"]))
        same_n = int(reproduced["n"]) == int(record["n_patients"])
        rows.append(
            {
                "backbone": record["backbone"],
                "attribute": record["attribute"],
                "reference_evaluation": record["evaluation"],
                "reference_auroc": float(record["auroc"]),
                "reproduced_auroc": float(reproduced["auroc"]),
                "absolute_difference": difference,
                "reference_best_C": float(record["best_C"]),
                "reproduced_best_C": float(reproduced["best_C"]),
                "reference_n": int(record["n_patients"]),
                "reproduced_n": int(reproduced["n"]),
                "tolerance": tolerance,
                "passed": bool(difference <= tolerance and same_c and same_n),
            }
        )
    return pd.DataFrame(rows)


def main() -> int:
    args = parse_args()
    raw_dir = args.data_root.expanduser().resolve() / "raw"
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    metric_rows: List[Dict[str, object]] = []
    penalty_rows: List[Dict[str, object]] = []
    geometry_rows: List[Dict[str, object]] = []
    moment_rows: List[Dict[str, object]] = []
    stratified_rows: List[Dict[str, object]] = []
    sanity_rows: List[Dict[str, object]] = []
    bootstrap_rows: List[Dict[str, object]] = []

    for backbone_index, (backbone, files) in enumerate(BACKBONES.items()):
        frames: Dict[str, pd.DataFrame] = {}
        features: Dict[str, np.ndarray] = {}
        metadata: Dict[str, pd.DataFrame] = {}
        for name in ("brset", "mbrset"):
            frames[name], features[name] = load_dataset(raw_dir, name, files[name])
            metadata[name], _ = patient_table(frames[name], features[name])
        # store[(arm, attribute)] -> per-split target arrays, kept in memory only.
        store: Dict[Tuple[str, str], List[Dict[str, np.ndarray]]] = {}

        for split_index in range(args.split_repeats):
            split_seed = args.seed + 101 * split_index
            contexts = {
                direction: prepare_context(frames, features, metadata, source, target, split_seed)
                for direction, source, target in DIRECTIONS
            }
            for direction, ctx in contexts.items():
                for row in moment_diagnostics(ctx["reference"], ctx["target_pz"]):
                    row.update(
                        {
                            "backbone": backbone,
                            "direction": direction,
                            "source": ctx["source"],
                            "target": ctx["target"],
                            "split_index": split_index,
                            "source_reference_n": int(len(ctx["reference"])),
                            "target_n": int(len(ctx["target_pz"]["unaligned"])),
                            **ctx["shrinkage"],
                        }
                    )
                    moment_rows.append(row)

            for attribute_index, attribute in enumerate(ATTRIBUTES):
                # Split 0 reproduces the reference target-local seed exactly.
                cv_seed = split_seed + 1000 * backbone_index + 100 * attribute_index
                cap = labelled_counts(contexts[REVERSE], attribute)
                for arm in ARMS:
                    ctx = contexts[FORWARD if arm == SIZE_MATCHED else arm]
                    labels = {
                        "backbone": backbone,
                        "direction": arm,
                        "source": ARM_DATASETS[arm][0],
                        "target": ARM_DATASETS[arm][1],
                        "split_index": split_index,
                    }
                    result = evaluate_readout(
                        ctx=ctx,
                        labels=labels,
                        attribute=attribute,
                        cv_seed=cv_seed,
                        cv_repeats=args.cv_repeats,
                        minimum_cv_class=args.minimum_cv_class,
                        cap=cap if arm == SIZE_MATCHED else None,
                        cap_seed=split_seed + 7919 + 13 * attribute_index,
                    )
                    metric_rows.extend(result["metrics"])
                    penalty_rows.extend(result["penalties"])
                    geometry_rows.append(result["geometry"])
                    sanity_rows.append(result["sanity"])
                    store.setdefault((arm, attribute), []).append(result["arrays"])
                    if arm != SIZE_MATCHED:
                        stratified_rows.extend(
                            evaluate_outcome_strata(
                                ctx=ctx,
                                labels=labels,
                                attribute=attribute,
                                cv_seed=cv_seed + 2000000,
                                cv_repeats=args.cv_repeats,
                                minimum_cv_class=args.minimum_cv_class,
                            )
                        )

        for attribute_index, attribute in enumerate(ATTRIBUTES):
            for target_index, (target_name, arms) in enumerate(
                (("mbrset", (FORWARD, SIZE_MATCHED)), ("brset", (REVERSE,)))
            ):
                y, vectors, statistics = bootstrap_group_statistics(
                    {arm: store[(arm, attribute)] for arm in arms},
                    {"backbone": backbone, "attribute": attribute, "target": target_name},
                )
                bootstrap_rows.extend(
                    bootstrap_statistics(
                        y,
                        vectors,
                        statistics,
                        args.bootstrap_replicates,
                        args.seed + 700000 + 1000 * backbone_index + 100 * attribute_index + 10 * target_index,
                    )
                )

    metrics_frame = pd.DataFrame(metric_rows)
    penalties = pd.DataFrame(penalty_rows)
    geometry = pd.DataFrame(geometry_rows)
    moments = pd.DataFrame(moment_rows)
    stratified = pd.DataFrame(stratified_rows)
    sanity = pd.DataFrame(sanity_rows)
    intervals = pd.DataFrame(bootstrap_rows)
    for frame in (metrics_frame, penalties, geometry, moments, stratified, intervals):
        frame["coordinate_space"] = SHARED_DEMOGRAPHIC_COORDINATE

    arm_keys = ["backbone", "direction", "source", "target", "attribute"]
    metrics_frame.to_csv(output / "transport_metrics_by_split.csv", index=False)
    penalties.to_csv(output / "transport_penalties_by_split.csv", index=False)
    geometry.to_csv(output / "direction_geometry_by_split.csv", index=False)
    moments.to_csv(output / "alignment_moment_diagnostics.csv", index=False)
    stratified.to_csv(output / "outcome_stratified_transport.csv", index=False)
    sanity.to_csv(output / "sanity_checks.csv", index=False)
    intervals.to_csv(output / "bootstrap_intervals.csv", index=False)
    aggregate_split_results(
        metrics_frame, arm_keys + ["evaluation", "method"], "auroc"
    ).to_csv(output / "transport_metrics_summary.csv", index=False)
    aggregate_split_results(
        penalties, arm_keys + ["method"], "transport_penalty"
    ).to_csv(output / "transport_summary.csv", index=False)
    pd.concat(
        [
            aggregate_split_results(geometry, arm_keys, value)
            for value in ("probe_coefficient_cosine", "centroid_direction_cosine")
        ],
        ignore_index=True,
    ).to_csv(output / "direction_geometry_summary.csv", index=False)
    complete = stratified[stratified["status"] == "complete"]
    aggregate_split_results(
        complete, arm_keys + ["outcome", "method"], "transport_penalty"
    ).to_csv(output / "outcome_stratified_summary.csv", index=False)
    alignment_effects(penalties).to_csv(output / "alignment_effects.csv", index=False)

    gate_status = "not_requested"
    if args.reference_probe_metrics is not None:
        gate = reproduction_gate(
            metrics_frame, args.reference_probe_metrics, args.reproduction_tolerance
        )
        gate.to_csv(output / "reproduction_gate.csv", index=False)
        gate_status = "passed" if bool(gate["passed"].all()) else "failed"

    manifest = {
        "seed": args.seed,
        "split_seed_scheme": "seed + 101 * split_index; split 0 is the reference primary split",
        "split_repeats": args.split_repeats,
        "target_local_cv": f"5-fold repeated {args.cv_repeats} times at patient level",
        "target_local_seed_scheme": "split_seed + 1000 * backbone_index + 100 * attribute_index",
        "arms": list(ARMS),
        "size_matched_arm": (
            "BRSET -> mBRSET with the demographic probe tuned and fitted on a class-stratified "
            "subsample of labelled BRSET training and validation patients equal in number to "
            "the labelled mBRSET training and validation patients of the same split; "
            "coordinate, target patients and alignment maps are those of the full arm"
        ),
        "alignment_methods": list(ALIGNMENT_METHODS),
        "alignment_contract": (
            "target patient embeddings are mapped to source-training-patient first/second "
            "moments without target demographic labels or clinical outcomes"
        ),
        "coral_covariance": "Ledoit-Wolf shrinkage covariance",
        "target_local_reference": "target-local probe fitted on unaligned target vectors",
        "bootstrap": (
            f"{args.bootstrap_replicates} target-patient resamples shared across splits, "
            "methods and arms with the same target; conditional on fitted probes and "
            "alignment maps; 95% percentile intervals of split-mean statistics"
        ),
        "outcome_stratification": (
            "demographic probes are independently tuned and fitted within source patient "
            "referable-DR strata and evaluated within the corresponding target stratum"
        ),
        "analysis_unit": "patient mean of image embeddings",
        "coordinate_space": SHARED_DEMOGRAPHIC_COORDINATE,
        "reproduction_gate": gate_status,
        "privacy": (
            "aggregate outputs only; no identifiers, embeddings, models, or row-level predictions"
        ),
    }
    (output / "icassp_extension_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    if not bool(sanity["passed"].all()):
        print("SANITY FAILURE: translation changed an external AUROC", file=sys.stderr)
        return 3
    if gate_status == "failed":
        print("REPRODUCTION GATE FAILED: split 0 does not reproduce reference", file=sys.stderr)
        return 4
    print(f"Completed ICASSP extensions (reproduction gate: {gate_status}). Outputs: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
