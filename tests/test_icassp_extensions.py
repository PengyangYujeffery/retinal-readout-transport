#!/usr/bin/env python3
"""Synthetic end-to-end tests for the ICASSP extension pipeline.

The reproduction test runs the AICS representation audit and the ICASSP
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
        cls.reference = root / "aics_reference"
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

    def test_split_zero_reproduces_aics_audit(self) -> None:
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
        self.assertEqual(set(penalties["method"]), {"unaligned", "mean_variance", "coral"})
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
        self.assertLess(float(pivot["coral"].max()), 1e-4)
        self.assertLess(float(pivot["mean_variance"].max()), 1e-4)
        self.assertGreater(float(pivot["unaligned"].min()), 1e-3)

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


if __name__ == "__main__":
    unittest.main()
