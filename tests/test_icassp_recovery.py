#!/usr/bin/env python3
"""Synthetic end-to-end test for the readout-recovery ceiling probe."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


class RecoveryProbeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        environment = os.environ.copy()
        environment.update(
            {
                "OMP_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
                "NUMEXPR_NUM_THREADS": "1",
                "MPLBACKEND": "Agg",
            }
        )
        cls.temporary = tempfile.TemporaryDirectory(prefix="icassp-recovery-")
        root = Path(cls.temporary.name)
        cls.output = root / "recovery"
        steps = (
            [sys.executable, str(ROOT / "tests" / "make_synthetic_data.py"), str(root / "data")],
            [
                sys.executable,
                str(ROOT / "scripts" / "run_icassp_recovery_probe.py"),
                "--data-root", str(root / "data"),
                "--output-dir", str(cls.output),
                "--split-repeats", "2",
                "--cv-repeats", "1",
                "--self-training-rounds", "3",
                "--shots", "10,20",
                "--draws", "3",
                "--bootstrap-replicates", "20",
                "--minimum-cv-class", "2",
                "--seed", "31",
            ],
        )
        for command in steps:
            completed = subprocess.run(
                command, cwd=ROOT, env=environment, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, text=True, timeout=1800, check=False,
            )
            if completed.returncode != 0:
                raise AssertionError(
                    f"{Path(command[1]).name} exited {completed.returncode}\n" + completed.stdout[-12000:]
                )

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temporary.cleanup()

    def test_invariants_pass(self) -> None:
        sanity = pd.read_csv(self.output / "recovery_sanity_checks.csv")
        self.assertEqual(
            set(sanity["check"]),
            {"zero_prior_matches_sklearn_coef", "strong_prior_matches_source_auroc"},
        )
        # 2 backbones x 2 directions x 2 attributes x 2 checks
        self.assertEqual(len(sanity), 16)
        self.assertTrue(sanity["passed"].all(), sanity.to_string())

    def test_coverage_and_ranges(self) -> None:
        few_shot = pd.read_csv(self.output / "few_shot_by_draw.csv")
        self.assertEqual(set(few_shot["direction"]), {"brset_to_mbrset", "mbrset_to_brset"})
        self.assertEqual(
            set(few_shot["method"]), {"source_readout", "scratch", "anchored", "target_local"}
        )
        self.assertEqual(set(few_shot["k"]), {10, 20})
        self.assertTrue(few_shot["auroc"].between(0, 1).all())
        trajectories = pd.read_csv(self.output / "self_training_by_split.csv")
        self.assertEqual(set(trajectories["round"]), {0, 1, 2, 3})
        self.assertEqual(set(trajectories["scheme"]), {"self_training", "confident_self_training"})
        intervals = pd.read_csv(self.output / "self_training_intervals.csv")
        # Eight statistics per backbone, attribute and direction: source, target-local and the two
        # self-training schemes, each with its change against the source probe and its remaining penalty.
        self.assertEqual(len(intervals), 2 * 2 * 2 * 8)
        self.assertTrue(np.isfinite(intervals[["estimate", "ci_low", "ci_high"]].to_numpy(float)).all())

    def test_round_zero_is_the_source_readout(self) -> None:
        trajectories = pd.read_csv(self.output / "self_training_by_split.csv")
        few_shot = pd.read_csv(self.output / "few_shot_by_draw.csv")
        round_zero = trajectories[trajectories["round"] == 0]
        self.assertTrue(np.allclose(round_zero["auroc"], round_zero["source_auroc"], atol=0, rtol=0))
        # The bootstrap point estimate of the source AUROC is the split mean of round 0.
        intervals = pd.read_csv(self.output / "self_training_intervals.csv")
        source = intervals[intervals["statistic"] == "source_auroc"].set_index(
            ["backbone", "direction", "attribute"]
        )["estimate"]
        split_mean = round_zero.groupby(["backbone", "direction", "attribute"])["auroc"].mean()
        joined = pd.concat([source, split_mean], axis=1).dropna()
        self.assertEqual(len(joined), 8)
        self.assertTrue(np.allclose(joined.iloc[:, 0], joined.iloc[:, 1], atol=1e-12, rtol=0))
        self.assertFalse(few_shot.empty)

    def test_manifest_and_privacy(self) -> None:
        manifest = json.loads((self.output / "recovery_manifest.json").read_text(encoding="utf-8"))
        self.assertIn("no new method", manifest["purpose"])
        prohibited = {"patient_key", "image_key", "probability", "prediction", "embedding"}
        for path in self.output.glob("*.csv"):
            columns = {column.casefold() for column in pd.read_csv(path, nrows=1).columns}
            self.assertFalse(columns & prohibited, path.name)


if __name__ == "__main__":
    unittest.main()
