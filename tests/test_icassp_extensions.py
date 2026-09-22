#!/usr/bin/env python3
"""Synthetic end-to-end tests for the ICASSP extension pipeline.

The reproduction test runs the reference representation audit and the ICASSP
extension on the same toy data: split 0 of BRSET -> mBRSET must reproduce the
audit's demographic-probe AUROCs. A second run with a perturbed reference
proves that the gate can fail.
"""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SEED = "31"


def run(command, environment):
    return subprocess.run(
        command,
        cwd=ROOT,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=1800,
        check=False,
    )


class IcasspExtensionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.environment = os.environ.copy()
        cls.environment.update(
            {
                "OMP_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
                "NUMEXPR_NUM_THREADS": "1",
                "MPLBACKEND": "Agg",
            }
        )
        cls.temporary = tempfile.TemporaryDirectory(prefix="icassp-extension-")
        root = Path(cls.temporary.name)
        cls.data_root = root / "data"
        cls.reference = root / "audit_reference"
        cls.output = root / "icassp"
        steps = (
            [sys.executable, str(ROOT / "tests" / "make_synthetic_data.py"), str(cls.data_root)],
            [
                sys.executable,
                str(ROOT / "scripts" / "run_representation_audit.py"),
                "--data-root", str(cls.data_root),
                "--output-dir", str(cls.reference),
                "--split-repeats", "1",
                "--cv-repeats", "1",
                "--bootstrap-replicates", "6",
                "--permutation-replicates", "7",
                "--random-controls", "1",
                "--minimum-group-size", "2",
                "--minimum-group-events", "1",
                "--seed", SEED,
            ],
            [
                sys.executable,
                str(ROOT / "scripts" / "run_icassp_extensions.py"),
                "--data-root", str(cls.data_root),
                "--output-dir", str(cls.output),
                "--split-repeats", "2",
                "--cv-repeats", "1",
                "--bootstrap-replicates", "25",
                "--minimum-cv-class", "2",
                "--seed", SEED,
                "--reference-probe-metrics", str(cls.reference / "demographic_probe_metrics.csv"),
            ],
            [
                sys.executable,
                str(ROOT / "scripts" / "validate_icassp_outputs.py"),
                "--results-dir", str(cls.output),
                "--expected-splits", "2",
                "--expected-bootstrap", "25",
                "--require-reproduction-gate",
            ],
        )
        for command in steps:
            completed = run(command, cls.environment)
            if completed.returncode != 0:
                raise AssertionError(
                    f"{Path(command[1]).name} exited {completed.returncode}\n"
                    + completed.stdout[-12000:]
                )

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temporary.cleanup()

    def test_split_zero_reproduces_reference_audit(self) -> None:
        gate = pd.read_csv(self.output / "reproduction_gate.csv")
        self.assertEqual(len(gate), 12)
        self.assertTrue(gate["passed"].all(), gate.to_string())
        # Identical computation; the only residue is one ulp from pandas' default
        # (not round-trip) CSV float parser reading the reference back.
        self.assertLess(float(gate["absolute_difference"].max()), 1e-12)

    def test_design_coverage(self) -> None:
        penalties = pd.read_csv(self.output / "transport_penalties_by_split.csv")
        self.assertEqual(
            set(penalties["direction"]),
            {"brset_to_mbrset", "mbrset_to_brset", "brset_to_mbrset_sizematched"},
        )
        self.assertEqual(
            set(penalties["method"]),
            {"unaligned", "mean_variance", "coral", "subspace", "ot_gaussian", "ot_entropic"},
        )
        self.assertEqual(penalties["split_index"].nunique(), 2)

    def test_alignment_matches_the_source_mean(self) -> None:
        # Both maps place the target mean exactly on the source-reference mean.
        # (CORAL targets the shrunk source covariance, not the sample one, so
        # a sample-covariance distance is a diagnostic, not an invariant.)
        moments = pd.read_csv(self.output / "alignment_moment_diagnostics.csv")
        pivot = moments.pivot_table(
            index=["backbone", "direction", "split_index"],
            columns="method",
            values="mean_l2_distance",
        )
        for method in ("coral", "mean_variance", "subspace", "ot_gaussian"):
            self.assertLess(float(pivot[method].max()), 1e-4, msg=method)
        self.assertGreater(float(pivot["unaligned"].min()), 1e-3)
        # The entropic-OT barycentric map is not affine and does not centre exactly, but it must
        # still move the target closer to the source mean than no alignment at all.
        self.assertLess(float(pivot["ot_entropic"].max()), float(pivot["unaligned"].min()))

        # A sample-covariance distance is a diagnostic, not an invariant, for any of the maps that
        # target a shrunk covariance, so the Gaussian OT map is checked directly instead, below.

    def test_gate_fails_on_a_perturbed_reference(self) -> None:
        reference = pd.read_csv(self.reference / "demographic_probe_metrics.csv")
        reference.loc[0, "auroc"] = float(reference.loc[0, "auroc"]) + 0.01
        perturbed = Path(self.temporary.name) / "perturbed.csv"
        reference.to_csv(perturbed, index=False)
        completed = run(
            [
                sys.executable,
                str(ROOT / "scripts" / "run_icassp_extensions.py"),
                "--data-root", str(self.data_root),
                "--output-dir", str(Path(self.temporary.name) / "perturbed_run"),
                "--split-repeats", "1",
                "--cv-repeats", "1",
                "--bootstrap-replicates", "5",
                "--minimum-cv-class", "2",
                "--seed", SEED,
                "--reference-probe-metrics", str(perturbed),
            ],
            self.environment,
        )
        self.assertEqual(completed.returncode, 4, completed.stdout[-4000:])


class LabelFreeRepairTest(unittest.TestCase):
    """Properties of the three added repairs, checked on their own.

    They are asserted here rather than on the pipeline's synthetic cohorts because those are small
    enough that Ledoit--Wolf shrinkage dominates, which makes a sample-covariance distance a
    diagnostic rather than an invariant.
    """

    @classmethod
    def setUpClass(cls) -> None:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
        rng = np.random.default_rng(20260918)
        dimension = 16
        cls.source = rng.normal(size=(1200, dimension)) * np.linspace(1.0, 3.0, dimension) + 4.0
        cls.target = rng.normal(size=(900, dimension)) * np.linspace(3.0, 1.0, dimension) - 2.0

    @staticmethod
    def relative_covariance_distance(values: np.ndarray, reference: np.ndarray) -> float:
        difference = np.cov(values, rowvar=False) - np.cov(reference, rowvar=False)
        return float(
            np.linalg.norm(difference) / np.linalg.norm(np.cov(reference, rowvar=False))
        )

    def test_gaussian_ot_map_matches_the_source_covariance(self) -> None:
        from run_icassp_extensions import gaussian_ot_target_to_source

        aligned = gaussian_ot_target_to_source(self.source, self.target)
        before = self.relative_covariance_distance(self.target, self.source)
        after = self.relative_covariance_distance(aligned, self.source)
        # The map matches the Ledoit--Wolf covariances, not the sample ones, so a residual of about
        # a tenth against the sample covariance is the shrinkage, not a defect. What must hold is
        # that the mismatch is reduced by a large factor.
        self.assertLess(after, 0.25)
        self.assertLess(after, before / 5.0)

    def test_subspace_alignment_is_affine_and_centred(self) -> None:
        from run_icassp_extensions import subspace_alignment_target_to_source

        aligned, diagnostic = subspace_alignment_target_to_source(self.source, self.target)
        self.assertLess(float(np.linalg.norm(aligned.mean(axis=0) - self.source.mean(axis=0))), 1e-3)
        self.assertGreaterEqual(diagnostic["subspace_source_variance"], 0.95)
        # Affine: the map of a convex combination is that combination of the maps.
        pair = self.target[:2]
        mixed, _ = subspace_alignment_target_to_source(
            self.source, np.vstack([pair, 0.5 * pair[0] + 0.5 * pair[1]])
        )
        # The fitted map depends on the cloud, so compare directions rather than absolute points.
        self.assertLess(
            float(np.linalg.norm(mixed[2] - 0.5 * (mixed[0] + mixed[1]))),
            1e-3 * float(np.linalg.norm(mixed[2])),
        )

    def test_entropic_ot_does_not_collapse_the_cloud(self) -> None:
        from run_icassp_extensions import OT_MINIMUM_VARIANCE_KEPT, entropic_ot_target_to_source

        aligned, diagnostic = entropic_ot_target_to_source(self.source, self.target, 3)
        self.assertGreater(diagnostic["ot_variance_kept"], OT_MINIMUM_VARIANCE_KEPT)
        self.assertLess(
            float(np.linalg.norm(aligned.mean(axis=0) - self.source.mean(axis=0))),
            float(np.linalg.norm(self.target.mean(axis=0) - self.source.mean(axis=0))),
        )
        # The map must move patients to different places, or the probe sees one constant score.
        self.assertGreater(float(np.median(np.std(aligned, axis=0))), 0.0)


if __name__ == "__main__":
    unittest.main()
