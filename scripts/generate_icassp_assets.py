#!/usr/bin/env python3
"""Generate every number, the main table and both figures of the ICASSP paper.

Reads aggregate outputs only. Every qualitative statement the manuscript makes about these
numbers is asserted here as a named claim; if a regeneration falsifies one, the assets are still
written for inspection but the script exits 5 and names the failed claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


FWD, REV, SM = "brset_to_mbrset", "mbrset_to_brset", "brset_to_mbrset_sizematched"
BACKBONES = (("vits16", "ViT-S/16", "ViT"), ("convnext_tiny", "ConvNeXt-T", "CNX"))
ATTRIBUTES = (("sex", "Sex"), ("age65", "Age"))
DIRECTIONS = ((FWD, r"B$\to$mB", "B→mB"), (REV, r"mB$\to$B", "mB→B"))
TARGET_OF = {FWD: "mbrset", REV: "brset"}
NEUTRAL_PDF_METADATA = {"Creator": None, "Producer": None, "CreationDate": None}
COLOURS = {
    "unaligned": "#7f7f7f",
    "mean_variance": "#E69F00",
    "coral": "#0072B2",
    "target_local": "#000000",
    "scratch": "#D55E00",
    "anchored": "#009E73",
    "self_training": "#CC79A7",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--extension-dir", required=True, type=Path)
    parser.add_argument("--recovery-dir", required=True, type=Path)
    parser.add_argument("--cohort-summary", required=True, type=Path)
    parser.add_argument(
        "--aics-dir",
        required=True,
        type=Path,
        help="AICS representation-audit outputs (the gated submission run) for the removal analysis",
    )
    parser.add_argument("--paper-dir", required=True, type=Path)
    return parser.parse_args()


def f3(value: float) -> str:
    return f"{value:.3f}"


def signed(value: float) -> str:
    # A non-zero value that rounds to 0.000 would hide on which side of zero it lies.
    magnitude = abs(value)
    digits = 4 if 0 < magnitude < 0.0005 else 3
    return r"\ensuremath{%s%.*f}" % ("+" if value >= 0 else "-", digits, magnitude)


def signed4(value: float) -> str:
    return r"\ensuremath{%s%.4f}" % ("+" if value >= 0 else "-", abs(value))


def percent(value: float) -> str:
    return f"{100.0 * value:.0f}"


def thousands(value: int) -> str:
    return f"{int(value):,}".replace(",", "{,}")


class Claims:
    def __init__(self) -> None:
        self.rows: List[Dict[str, object]] = []

    def add(self, name: str, passed: bool, detail: str) -> None:
        self.rows.append({"claim": name, "passed": bool(passed), "detail": detail})

    @property
    def failed(self) -> List[Dict[str, object]]:
        return [row for row in self.rows if not row["passed"]]


def md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def main() -> int:
    args = parse_args()
    ext, rec, paper = args.extension_dir, args.recovery_dir, args.paper_dir
    (paper / "figures").mkdir(parents=True, exist_ok=True)
    claims = Claims()
    macros: Dict[str, str] = {}

    intervals = pd.read_csv(ext / "bootstrap_intervals.csv")
    moments = pd.read_csv(ext / "alignment_moment_diagnostics.csv")
    geometry = pd.read_csv(ext / "direction_geometry_summary.csv")
    strata = pd.read_csv(ext / "outcome_stratified_summary.csv")
    gate = pd.read_csv(ext / "reproduction_gate.csv")
    ext_manifest = json.loads((ext / "icassp_extension_manifest.json").read_text(encoding="utf-8"))
    self_training = pd.read_csv(rec / "self_training_intervals.csv")
    few_summary = pd.read_csv(rec / "few_shot_summary.csv")
    few_draws = pd.read_csv(rec / "few_shot_by_draw.csv")
    rec_sanity = pd.read_csv(rec / "recovery_sanity_checks.csv")
    rec_manifest = json.loads((rec / "recovery_manifest.json").read_text(encoding="utf-8"))
    cohort = pd.read_csv(args.cohort_summary).set_index("dataset")
    aics = args.aics_dir
    aics_manifest = json.loads((aics / "representation_manifest.json").read_text(encoding="utf-8"))
    metric_contrasts = pd.read_csv(aics / "residualization_metric_contrasts.csv")
    gap_contrasts = pd.read_csv(aics / "residualization_gap_contrasts.csv")
    residual_gaps = pd.read_csv(aics / "residualization_gaps.csv")
    random_controls = pd.read_csv(aics / "random_projection_controls.csv")

    claims.add("reproduction gate passed (12/12)", len(gate) == 12 and gate["passed"].astype(str).eq("True").all()
               and ext_manifest.get("reproduction_gate") == "passed", f"{int(gate['passed'].astype(str).eq('True').sum())}/12")
    claims.add("recovery invariants all pass", rec_sanity["passed"].astype(str).eq("True").all(),
               f"{int(rec_sanity['passed'].astype(str).eq('True').sum())}/{len(rec_sanity)}")

    def iv(backbone: str, attribute: str, direction: str, statistic: str, method: str) -> pd.Series:
        selected = intervals[
            (intervals["backbone"] == backbone) & (intervals["attribute"] == attribute)
            & (intervals["direction"] == direction) & (intervals["statistic"] == statistic)
            & (intervals["method"] == method)
        ]
        if len(selected) != 1:
            raise AssertionError(f"{len(selected)} rows for {backbone} {attribute} {direction} {statistic} {method}")
        return selected.iloc[0]

    cells = [(b, a, d) for b, _, _ in BACKBONES for a, _ in ATTRIBUTES for d, _, _ in DIRECTIONS]
    penalty = {c: iv(*c, "transport_penalty", "unaligned") for c in cells}
    local = {c: iv(*c, "target_local_auroc", "target_local") for c in cells}
    transported = {c: iv(*c, "external_auroc", "unaligned") for c in cells}
    coral_delta = {c: iv(*c, "aligned_minus_unaligned_external_auroc", "coral") for c in cells}
    mv_delta = {c: iv(*c, "aligned_minus_unaligned_external_auroc", "mean_variance") for c in cells}
    coral_penalty = {c: iv(*c, "transport_penalty", "coral") for c in cells}
    coral_external = {c: iv(*c, "external_auroc", "coral") for c in cells}

    # Design constants.
    macros["icNSplits"] = str(int(ext_manifest["split_repeats"]))
    macros["icNCvRepeats"] = re.search(r"repeated (\d+) times", ext_manifest["target_local_cv"]).group(1)
    macros["icNBoot"] = thousands(int(re.match(r"(\d+)", ext_manifest["bootstrap"]).group(1)))
    macros["icStRounds"] = re.match(r"(\d+) rounds", rec_manifest["self_training"]).group(1)
    macros["icNCells"] = str(len(cells))
    for name, dataset in (("Brset", "BRSET"), ("Mbrset", "mBRSET")):
        macros[f"ic{name}Images"] = thousands(cohort.loc[dataset, "n_images_analyzed"])
        macros[f"ic{name}Patients"] = thousands(cohort.loc[dataset, "n_patients"])

    # 3.1 Transport fails in both directions.
    values = [penalty[c]["estimate"] for c in cells]
    macros["icPenMin"], macros["icPenMax"] = f3(min(values)), f3(max(values))
    significant = sum(penalty[c]["ci_low"] > 0 for c in cells)
    macros["icPenNSig"] = str(significant)
    claims.add("every penalty interval excludes zero", significant == len(cells), f"{significant}/{len(cells)}")
    macros["icLocalMin"] = f3(min(local[c]["estimate"] for c in cells))
    macros["icLocalMax"] = f3(max(local[c]["estimate"] for c in cells))
    macros["icExtMin"] = f3(min(transported[c]["estimate"] for c in cells))
    macros["icExtMax"] = f3(max(transported[c]["estimate"] for c in cells))
    claims.add("targets encode both attributes (target-local AUROC > 0.7 everywhere)",
               min(local[c]["estimate"] for c in cells) > 0.7, macros["icLocalMin"])

    stratum = strata[(strata["method"] == "unaligned") & strata["direction"].isin([FWD, REV])]
    macros["icStratMin"], macros["icStratMax"] = f3(stratum["mean"].min()), f3(stratum["mean"].max())
    macros["icStratNPos"] = str(int((stratum["mean"] > 0).sum()))
    claims.add("penalty positive in every referable-DR stratum", len(stratum) == 16 and (stratum["mean"] > 0).all(),
               f"{int((stratum['mean'] > 0).sum())}/{len(stratum)}")

    size_rows = {(b, a): iv(b, a, SM, "sizematched_minus_full_transport_penalty", "unaligned")
                 for b, _, _ in BACKBONES for a, _ in ATTRIBUTES}
    non_null = {key for key, row in size_rows.items() if row["ci_low"] > 0 or row["ci_high"] < 0}
    macros["icSmNCells"] = str(len(size_rows))
    macros["icSmNNull"] = str(len(size_rows) - len(non_null))
    vit_sex = size_rows[("vits16", "sex")]
    macros["icSmVitSex"], macros["icSmVitSexLo"], macros["icSmVitSexHi"] = (
        f3(vit_sex["estimate"]), f3(vit_sex["ci_low"]), f3(vit_sex["ci_high"]))
    claims.add("size matching changes the penalty only for ViT sex, upwards",
               non_null == {("vits16", "sex")} and vit_sex["ci_low"] > 0, str(sorted(non_null)))

    reverse_larger = all(penalty[(b, a, REV)]["estimate"] > penalty[(b, a, FWD)]["estimate"]
                         for b, _, _ in BACKBONES for a, _ in ATTRIBUTES)
    claims.add("reverse penalty larger in every setting", reverse_larger, "")
    brset_higher = all(local[(b, a, REV)]["estimate"] > local[(b, a, FWD)]["estimate"]
                       for b, _, _ in BACKBONES for a, _ in ATTRIBUTES)
    claims.add("BRSET target-local AUROC higher in every setting", brset_higher, "")

    probe = geometry[(geometry["value"] == "probe_coefficient_cosine") & geometry["direction"].isin([FWD, REV])]
    macros["icCosProbeMin"], macros["icCosProbeMax"] = f3(probe["mean"].min()), f3(probe["mean"].max())
    claims.add("readout coefficients weakly aligned (cosine < 0.3)", probe["mean"].max() < 0.3, macros["icCosProbeMax"])

    # 3.2 Matching moments does not restore transport.
    covariance = moments.groupby(["backbone", "direction", "method"])["covariance_relative_distance"].mean().unstack()
    coral_reduction = 1.0 - covariance["coral"] / covariance["unaligned"]
    mv_reduction = 1.0 - covariance["mean_variance"] / covariance["unaligned"]
    macros["icCovRedMin"], macros["icCovRedMax"] = percent(coral_reduction.min()), percent(coral_reduction.max())
    macros["icMvRedMin"], macros["icMvRedMax"] = percent(mv_reduction.min()), percent(mv_reduction.max())
    claims.add("CORAL removes at least 90% of the covariance mismatch", coral_reduction.min() >= 0.9,
               f"{coral_reduction.min():.3f}")
    claims.add("diagonal matching removes less than CORAL everywhere", mv_reduction.max() < coral_reduction.min(),
               f"{mv_reduction.max():.3f} < {coral_reduction.min():.3f}")
    coral_mean = moments[moments["method"] == "coral"]["mean_l2_distance"].max()
    claims.add("CORAL mean difference is numerically zero (< 1e-6)", coral_mean < 1e-6, f"{coral_mean:.2e}")

    deltas = [coral_delta[c]["estimate"] for c in cells]
    macros["icCoralDeltaMin"], macros["icCoralDeltaMax"] = signed(min(deltas)), signed(max(deltas))
    better = sum(coral_delta[c]["ci_low"] > 0 for c in cells)
    worse = sum(coral_delta[c]["ci_high"] < 0 for c in cells)
    macros["icCoralNBetter"], macros["icCoralNWorse"] = str(better), str(worse)
    macros["icCoralNNull"] = str(len(cells) - better - worse)
    claims.add("CORAL changes transported AUROC by less than 0.05", max(abs(x) for x in deltas) < 0.05,
               f"{max(abs(x) for x in deltas):.3f}")
    macros["icPenCoralMin"] = f3(min(coral_penalty[c]["estimate"] for c in cells))
    macros["icPenCoralMax"] = f3(max(coral_penalty[c]["estimate"] for c in cells))
    claims.add("penalty after CORAL excludes zero everywhere", all(coral_penalty[c]["ci_low"] > 0 for c in cells), "")
    mv_values = [mv_delta[c]["estimate"] for c in cells]
    macros["icMvDeltaMin"], macros["icMvDeltaMax"] = signed(min(mv_values)), signed(max(mv_values))

    # 3.3 Recovery.
    def st(backbone: str, attribute: str, direction: str, statistic: str) -> pd.Series:
        selected = self_training[
            (self_training["backbone"] == backbone) & (self_training["attribute"] == attribute)
            & (self_training["direction"] == direction) & (self_training["statistic"] == statistic)
        ]
        if len(selected) != 1:
            raise AssertionError(f"{len(selected)} self-training rows for {backbone} {attribute} {direction} {statistic}")
        return selected.iloc[0]

    st_delta = {c: st(*c, "self_trained_minus_source_auroc") for c in cells}
    st_remaining = {c: st(*c, "remaining_penalty_after_self_training") for c in cells}
    agreement = max(abs(st(*c, "source_auroc")["estimate"] - transported[c]["estimate"]) for c in cells)
    claims.add("recovery and extension runs agree on the transported readout (< 1e-3)", agreement < 1e-3,
               f"{agreement:.2e}")
    values = [st_delta[c]["estimate"] for c in cells]
    macros["icStDeltaMin"], macros["icStDeltaMax"] = signed(min(values)), signed(max(values))
    claims.add("self-training changes transported AUROC by less than 0.03", max(abs(x) for x in values) < 0.03,
               f"{max(abs(x) for x in values):.3f}")
    macros["icStRemMin"] = f3(min(st_remaining[c]["estimate"] for c in cells))
    macros["icStRemMax"] = f3(max(st_remaining[c]["estimate"] for c in cells))
    claims.add("penalty after self-training excludes zero everywhere",
               all(st_remaining[c]["ci_low"] > 0 for c in cells), "")

    shots = sorted(int(k) for k in few_summary["k"].unique())
    n_splits = int(ext_manifest["split_repeats"])
    draws_per_split = int(few_summary["n_draws"].iloc[0]) // n_splits
    claims.add("few-shot draws divide evenly into splits", int(few_summary["n_draws"].iloc[0]) % n_splits == 0, "")
    mid, low, high = shots[len(shots) // 2], shots[0], shots[-1]
    macros["icShotList"] = ",".join(str(k) for k in shots)
    macros["icShotMin"], macros["icShotMax"], macros["icFsMidK"] = str(low), str(high), str(mid)
    macros["icNDraws"] = str(draws_per_split)

    real = few_summary[few_summary["direction"].isin([FWD, REV])]

    def share(attribute: str, k: int) -> pd.Series:
        rows = real[(real["attribute"] == attribute) & (real["k"] == k)]
        if len(rows) != 8:
            raise AssertionError(f"{len(rows)} few-shot summary rows for {attribute} k={k}")
        return rows["share_of_ceiling_gap_recovered"]

    for tag, attribute, k in (("AgeMid", "age65", mid), ("AgeMax", "age65", high), ("SexMax", "sex", high)):
        values = share(attribute, k)
        macros[f"icFs{tag}Min"], macros[f"icFs{tag}Max"] = percent(values.min()), percent(values.max())
    claims.add("age: the middle k recovers at least 40% of the gap", share("age65", mid).min() >= 0.4,
               macros["icFsAgeMidMin"])
    claims.add("sex recovers less than age at the largest k", share("sex", high).max() < share("age65", high).min(),
               f"{macros['icFsSexMaxMax']} < {macros['icFsAgeMaxMin']}")

    draws = few_draws[few_draws["direction"].isin([FWD, REV])]
    wide = draws.pivot_table(index=["backbone", "direction", "attribute", "split_index", "k", "draw"],
                             columns="method", values="auroc").reset_index()
    wide["gain_scratch"] = wide["scratch"] - wide["source_readout"]
    lower = wide.groupby(["backbone", "direction", "attribute", "k"])["gain_scratch"].quantile(0.025).reset_index()
    beat_k = None
    for k in shots:
        later = lower[(lower["attribute"] == "age65") & (lower["k"] >= k)]
        if len(later) == 4 * len([s for s in shots if s >= k]) and (later["gain_scratch"] > 0).all():
            beat_k = k
            break
    claims.add("age: some k beats the transported readout in >= 97.5% of draws in every setting", beat_k is not None,
               str(beat_k))
    macros["icFsAgeBeatK"] = str(beat_k if beat_k is not None else "NA")

    scratch_rows = real[(real["method"] == "scratch") & (real["attribute"] == "sex")]
    below_k = None
    for k in shots:
        rows = scratch_rows[scratch_rows["k"] <= k]
        if (rows["mean_gain_over_source"] < 0).all():
            below_k = k
        else:
            break
    claims.add("sex: scratch is below the transported readout on average at the smallest k", below_k is not None,
               str(below_k))
    macros["icFsSexScratchBelowK"] = str(below_k if below_k is not None else "NA")

    anchored = real[(real["method"] == "anchored") & (real["attribute"] == "sex") & (real["k"] == low)]
    macros["icFsAnchorSexMin"] = f3(anchored["anchored_minus_scratch"].min())
    macros["icFsAnchorSexMax"] = f3(anchored["anchored_minus_scratch"].max())
    claims.add("sex: L2-SP beats scratch at the smallest k in every setting",
               len(anchored) == 4 and (anchored["anchored_minus_scratch"] > 0).all(), macros["icFsAnchorSexMin"])

    # 3.4 Removal of source-defined directions (AICS audit outputs, BRSET -> mBRSET only).
    removal_auc = metric_contrasts[
        (metric_contrasts["setting"] == "external_full") & (metric_contrasts["method"] == "joint_erased")
        & (metric_contrasts["metric"] == "auroc")
    ]
    removal_gap = gap_contrasts[
        (gap_contrasts["setting"] == "external_full") & (gap_contrasts["method"] == "joint_erased")
        & (gap_contrasts["attribute"] == "sex_age") & (gap_contrasts["metric"] == "sensitivity")
    ]
    claims.add("removal contrasts cover both backbones", len(removal_auc) == 2 and len(removal_gap) == 2,
               f"{len(removal_auc)}/{len(removal_gap)}")
    macros["icRmSplits"] = str(int(aics_manifest["split_repeats"]))
    macros["icRmRandN"] = str(int(aics_manifest["random_rank_matched_controls"]))
    low_auc = float(removal_auc["min_method_minus_baseline"].min())
    high_auc = float(removal_auc["max_method_minus_baseline"].max())
    macros["icRmAucMin"], macros["icRmAucMax"] = signed4(low_auc), signed4(high_auc)
    claims.add("removal changes external AUROC by less than 0.005", max(abs(low_auc), abs(high_auc)) < 0.005,
               f"{max(abs(low_auc), abs(high_auc)):.4f}")
    decreased = int(removal_gap["improved_splits"].sum())
    pairs = int(removal_gap["n_splits"].sum())
    macros["icRmGapDecreased"], macros["icRmGapPairs"] = str(decreased), str(pairs)
    claims.add("removal reduces the sensitivity gap in fewer than half of backbone-split pairs",
               decreased < pairs / 2, f"{decreased}/{pairs}")
    for backbone, tag in (("vits16", "Vit"), ("convnext_tiny", "Cnx")):
        observed = residual_gaps[
            (residual_gaps["backbone"] == backbone) & (residual_gaps["method"] == "joint_erased")
            & (residual_gaps["split_index"] == 0) & (residual_gaps["setting"] == "external_full")
            & (residual_gaps["attribute"] == "sex_age") & (residual_gaps["metric"] == "sensitivity")
        ]["gap_max_minus_min"]
        if len(observed) != 1:
            raise AssertionError(f"{len(observed)} joint-removal gap rows for {backbone}")
        controls_gap = random_controls[random_controls["backbone"] == backbone]["sex_age_sensitivity_gap"]
        at_most = int((controls_gap <= float(observed.iloc[0]) + 1e-12).sum())
        macros[f"icRmRandLe{tag}"] = str(at_most)
        claims.add(f"{backbone}: random-projection count matches the manifest",
                   len(controls_gap) == int(aics_manifest["random_rank_matched_controls"]), str(len(controls_gap)))
        claims.add(f"{backbone}: source-defined removal is no better than 95% of random projections",
                   at_most >= 0.05 * len(controls_gap), f"{at_most}/{len(controls_gap)}")

    # numbers.tex
    for name in macros:
        if not re.fullmatch(r"ic[A-Za-z]+", name):
            raise ValueError(f"illegal macro name {name}")
    lines = ["% Generated by scripts/generate_icassp_assets.py. Do not edit by hand."]
    lines += [r"\newcommand{\%s}{%s}" % (name, value) for name, value in sorted(macros.items())]
    (paper / "numbers.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # table_main.tex
    def ci(row: pd.Series, signed_values: bool = False) -> str:
        fmt = signed if signed_values else f3
        return f"{fmt(row['estimate'])} [{fmt(row['ci_low'])}, {fmt(row['ci_high'])}]"

    # ICASSP asks for at least 9 pt type throughout, so the table stays at body size and is kept
    # narrow by content: mean-variance changes are reported in the text, self-training without CI.
    table = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Transport of demographic readouts. Split means over \icNSplits{} source splits; brackets: "
        r"95\% target-patient bootstrap intervals. ViT: DINOv3 ViT-S/16; CNX: ConvNeXt-Tiny; B: BRSET; "
        r"mB: mBRSET. $\Delta$: change in transported AUROC after CORAL and after self-training (ST).}",
        r"\label{tab:main}",
        r"\setlength{\tabcolsep}{3.5pt}",
        r"\begin{tabular}{lllccccc}",
        r"\toprule",
        r"Enc. & Attr. & Dir. & Target-local & Transported & Penalty $P$ & $\Delta$CORAL & $\Delta$ST \\",
        r"\midrule",
    ]
    for b, _, b_short in BACKBONES:
        for a, a_label in ATTRIBUTES:
            for d, d_label, _ in DIRECTIONS:
                c = (b, a, d)
                table.append(
                    f"{b_short} & {a_label} & {d_label} & {f3(local[c]['estimate'])} & "
                    f"{f3(transported[c]['estimate'])} & {ci(penalty[c])} & "
                    f"{ci(coral_delta[c], True)} & {signed(st_delta[c]['estimate'])} \\\\"
                )
        if b != BACKBONES[-1][0]:
            table.append(r"\midrule")
    table += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    (paper / "table_main.tex").write_text("\n".join(table) + "\n", encoding="utf-8")

    # table_controls.tex: the controls behind Section 3.1, one row per setting.
    stratum_mean = strata[strata["method"] == "unaligned"].assign(
        outcome=lambda frame: frame["outcome"].astype(int)
    ).set_index(["backbone", "direction", "attribute", "outcome"])["mean"]
    geometry_mean = geometry.set_index(["backbone", "direction", "attribute", "value"])["mean"]
    controls = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Controls. $P$ with readouts refitted and evaluated within referable-DR-negative (DR$-$) and "
        r"-positive (DR$+$) patients; cosine between source and target readout coefficients and between "
        r"class-centroid directions; and the change in $P$ when the BRSET readout is fitted on as many labelled "
        r"patients as the mBRSET one (95\% bootstrap interval; B$\to$mB only). Split means over \icNSplits{} splits.}",
        r"\label{tab:controls}",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{lllccccc}",
        r"\toprule",
        r"Enc. & Attr. & Dir. & $P$ (DR$-$) & $P$ (DR$+$) & cos (coef.) & cos (centroid) & "
        r"$\Delta P$, size-matched \\",
        r"\midrule",
    ]
    for b, _, b_short in BACKBONES:
        for a, a_label in ATTRIBUTES:
            for d, d_label, _ in DIRECTIONS:
                size_cell = ci(size_rows[(b, a)], True) if d == FWD else "---"
                controls.append(
                    f"{b_short} & {a_label} & {d_label} & {f3(stratum_mean[(b, d, a, 0)])} & "
                    f"{f3(stratum_mean[(b, d, a, 1)])} & "
                    f"{f3(geometry_mean[(b, d, a, 'probe_coefficient_cosine')])} & "
                    f"{f3(geometry_mean[(b, d, a, 'centroid_direction_cosine')])} & {size_cell} \\\\"
                )
        if b != BACKBONES[-1][0]:
            controls.append(r"\midrule")
    controls += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    (paper / "table_controls.tex").write_text("\n".join(controls) + "\n", encoding="utf-8")

    # Figures.
    plt.rcParams.update({
        "font.size": 7.5, "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
        "pdf.fonttype": 42, "ps.fonttype": 42, "axes.linewidth": 0.6,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6, "legend.frameon": False,
    })

    # Stacked panels, each a full column wide, so that every label can be set at >= 6.5 pt.
    figure, (ax_a, ax_b) = plt.subplots(2, 1, figsize=(3.39, 3.4), gridspec_kw={"height_ratios": [1.0, 1.45]})
    groups = [(b, d) for b, _, _ in BACKBONES for d, _, _ in DIRECTIONS]
    width = 0.26
    for offset, (method, label) in zip((-width, 0.0, width), (("unaligned", "Unaligned"),
                                                             ("mean_variance", "Mean–var."), ("coral", "CORAL"))):
        ax_a.bar([i + offset for i in range(len(groups))], [covariance.loc[(b, d), method] for b, d in groups],
                 width=width, color=COLOURS[method], label=label)
    ax_a.set_yscale("log")
    ax_a.set_xticks(range(len(groups)))
    short = {b: s for b, _, s in BACKBONES}
    arrow = {d: s for d, _, s in DIRECTIONS}
    ax_a.set_xticklabels([f"{short[b]} {arrow[d]}" for b, d in groups], fontsize=7)
    ax_a.set_ylabel("Relative cov. distance")
    ax_a.set_title("(a)", loc="left", fontsize=8)
    for i, c in enumerate(cells):
        for shift, (series, method, marker) in zip((-0.2, 0.0, 0.2), (
                (transported, "unaligned", "o"), (coral_external, "coral", "D"), (local, "target_local", "^"))):
            row = series[c]
            ax_b.errorbar(i + shift, row["estimate"], yerr=[[row["estimate"] - row["ci_low"]],
                                                            [row["ci_high"] - row["estimate"]]],
                          fmt=marker, ms=3.4, lw=0.8, color=COLOURS[method], capsize=0)
    ax_b.set_xticks(range(len(cells)))
    ax_b.set_xticklabels([f"{short[b]}\n{dict(ATTRIBUTES)[a]}\n{arrow[d]}" for b, a, d in cells], fontsize=6.5)
    ax_b.set_ylabel("AUROC")
    ax_b.axhline(0.5, color="#bbbbbb", lw=0.5, ls=":")
    ax_b.set_title("(b)", loc="left", fontsize=8)
    ax_b.set_ylim(0.45, 1.0)
    figure.tight_layout(pad=0.3)
    # One legend for both panels, above them: the colours mean the same thing in (a) and (b).
    handles = [plt.Line2D([], [], marker=m, ls="", color=COLOURS[k], ms=3.2, label=l) for k, m, l in (
        ("unaligned", "o", "Unaligned (transported)"), ("mean_variance", "s", "Mean–var."),
        ("coral", "D", "CORAL"), ("target_local", "^", "Target-local"))]
    figure.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=7,
                  handletextpad=0.3, columnspacing=0.8)
    # Neutral PDF metadata: no creator/producer strings in the figure files.
    figure.savefig(paper / "figures" / "moments.pdf", bbox_inches="tight", metadata=NEUTRAL_PDF_METADATA)
    plt.close(figure)

    figure, axes = plt.subplots(2, 4, figsize=(7.0, 3.0), sharex=True)
    columns = [(b, d) for b, _, _ in BACKBONES for d, _, _ in DIRECTIONS]
    for row_index, (a, a_label) in enumerate(ATTRIBUTES):
        for col_index, (b, d) in enumerate(columns):
            ax = axes[row_index, col_index]
            subset = draws[(draws["backbone"] == b) & (draws["direction"] == d) & (draws["attribute"] == a)]
            for method, label in (("scratch", "Scratch"), ("anchored", "L2-SP")):
                stats = subset[subset["method"] == method].groupby("k")["auroc"]
                mean, lo, hi = stats.mean(), stats.quantile(0.025), stats.quantile(0.975)
                ax.plot(mean.index, mean.values, color=COLOURS[method], lw=1.0, marker="o", ms=2.0, label=label)
                ax.fill_between(mean.index, lo.values, hi.values, color=COLOURS[method], alpha=0.15, lw=0)
            source_level = subset[subset["method"] == "source_readout"]["auroc"].mean()
            local_level = subset[subset["method"] == "target_local"]["auroc"].mean()
            st_level = st(b, a, d, "self_trained_auroc")["estimate"]
            ax.axhline(source_level, color=COLOURS["unaligned"], lw=0.8, ls="--", label="Transported")
            ax.axhline(local_level, color=COLOURS["target_local"], lw=0.8, ls=":", label="Target-local")
            ax.axhline(st_level, color=COLOURS["self_training"], lw=0.8, ls="-.", label="Self-training")
            ax.set_xscale("log")
            ax.set_xticks(shots)
            ax.set_xticklabels([str(k) for k in shots], fontsize=7)
            ax.minorticks_off()
            if row_index == 0:
                ax.set_title(f"{dict((x, y) for x, y, _ in BACKBONES)[b]}, {arrow[d]}", fontsize=7.5)
            if col_index == 0:
                ax.set_ylabel(f"{a_label} AUROC")
            if row_index == 1:
                ax.set_xlabel("Labelled target patients $k$")
    figure.tight_layout(pad=0.3)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    # Anchored by its lower edge above the panels, so it cannot cover the column titles.
    figure.legend(handles, labels, loc="lower center", ncol=5, fontsize=7.5, bbox_to_anchor=(0.5, 1.0))
    figure.savefig(paper / "figures" / "fewshot.pdf", bbox_inches="tight", metadata=NEUTRAL_PDF_METADATA)
    plt.close(figure)

    inputs = sorted(list(ext.glob("*.csv")) + list(ext.glob("*.json")) + list(rec.glob("*.csv"))
                    + list(rec.glob("*.json")) + [args.cohort_summary]
                    + [aics / "representation_manifest.json", aics / "residualization_metric_contrasts.csv",
                       aics / "residualization_gap_contrasts.csv", aics / "residualization_gaps.csv",
                       aics / "random_projection_controls.csv"])
    (paper / "numbers_provenance.json").write_text(json.dumps(
        {"inputs": {str(path): md5(path) for path in inputs}, "n_macros": len(macros)}, indent=2), encoding="utf-8")
    pd.DataFrame(claims.rows).to_csv(paper / "claims_check.csv", index=False)
    print(f"{len(macros)} macros, {len(claims.rows)} claims, {len(claims.failed)} failed")
    for row in claims.failed:
        print(f"CLAIM FAILED: {row['claim']} ({row['detail']})", file=sys.stderr)
    return 5 if claims.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
