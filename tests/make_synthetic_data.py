#!/usr/bin/env python3
"""Create non-sensitive toy data for a pipeline smoke test."""

from pathlib import Path
import sys

import numpy as np
import pandas as pd


def cohort(rng, name, patients, dimensions, shift):
    patient_ids = np.repeat(np.arange(patients), 2)
    count = len(patient_ids)
    sex = rng.integers(0, 2, size=patients)[patient_ids]
    age = np.clip(rng.normal(60 + shift, 13, size=patients), 25, 90)[patient_ids]
    patient_risk = rng.normal(size=patients)[patient_ids] + 0.3 * sex + 0.015 * (age - 60)
    icdr = np.digitize(patient_risk + rng.normal(0, 0.5, count), [-0.5, 0.2, 0.8, 1.4])
    identifiers = [f"{name}_{index:05d}.jpg" for index in range(count)]
    latent = rng.normal(size=(count, dimensions)).astype(np.float32)
    latent[:, 0] += (icdr >= 2) * 1.5 + shift / 10
    latent[:, 1] += sex * 0.4
    return patient_ids, sex, age, icdr, identifiers, latent


def write_embedding(path, identifier_name, identifiers, values):
    frame = pd.DataFrame(values, columns=[str(index) for index in range(values.shape[1])])
    frame.insert(0, identifier_name, identifiers)
    frame.to_csv(path, index=False)


def main():
    raw = Path(sys.argv[1]).resolve() / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(7)
    source = cohort(rng, "src", 300, 24, 0)
    target = cohort(rng, "tgt", 120, 24, 4)

    source_meta = pd.DataFrame(
        {
            "image_id": source[4],
            "patient_id": source[0],
            "patient_age": source[2],
            "patient_sex": np.where(source[1] == 1, 1, 2),
            "DR_ICDR": source[3],
        }
    )
    source_meta.to_csv(raw / "labels_brset.csv", index=False)
    target_meta = pd.DataFrame(
        {
            "file": target[4],
            "patient": target[0],
            "age": target[2],
            "sex": target[1],
            "final_icdr": target[3],
            "insurance": rng.integers(0, 2, size=len(target[0])),
            "educational_level": rng.integers(1, 8, size=len(target[0])),
        }
    )
    target_meta.to_csv(raw / "labels_mbrset.csv", index=False)

    for backbone, noise in (("vits16", 0.0), ("convnext_tiny", 0.2)):
        source_values = source[5] + rng.normal(0, noise, source[5].shape)
        target_values = target[5] + rng.normal(0, noise, target[5].shape)
        write_embedding(
            raw / f"Embeddings_brset_dinov3_{backbone}.csv",
            "image_id",
            source[4],
            source_values,
        )
        write_embedding(
            raw / f"Embeddings_mbrset_dinov3_{backbone}.csv",
            "file",
            target[4],
            target_values,
        )


if __name__ == "__main__":
    main()
