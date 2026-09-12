#!/usr/bin/env python3
"""Validate the minimal BRSET/mBRSET download without printing records."""

import argparse
import csv
from pathlib import Path
import sys
from typing import Dict, List, Set, Tuple


FILES = {
    "Embeddings_brset_dinov3_vits16.csv": {
        "id": "image_id",
        "columns": 385,
        "dataset": "brset",
    },
    "Embeddings_mbrset_dinov3_vits16.csv": {
        "id": "file",
        "columns": 385,
        "dataset": "mbrset",
    },
    "Embeddings_brset_dinov3_convnext_tiny.csv": {
        "id": "image_id",
        "columns": 769,
        "dataset": "brset",
    },
    "Embeddings_mbrset_dinov3_convnext_tiny.csv": {
        "id": "file",
        "columns": 769,
        "dataset": "mbrset",
    },
}

METADATA = {
    "labels_brset.csv": {
        "id": "image_id",
        "required": {"image_id", "patient_id", "patient_age", "patient_sex", "DR_ICDR"},
        "dataset": "brset",
    },
    "labels_mbrset.csv": {
        "id": "file",
        "required": {"file", "patient", "age", "sex", "final_icdr"},
        "dataset": "mbrset",
    },
}


def normalized_id(value: str) -> str:
    """Normalize an image identifier without revealing it in output."""
    return Path(value.strip()).stem.casefold()


def inspect_csv(path: Path, id_column: str) -> Tuple[List[str], Set[str], int]:
    ids = set()  # type: Set[str]
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("missing CSV header")
        if id_column not in reader.fieldnames:
            raise ValueError(f"missing identifier column {id_column!r}")
        rows = 0
        for row in reader:
            rows += 1
            value = row.get(id_column, "")
            if value:
                ids.add(normalized_id(value))
    return list(reader.fieldnames), ids, rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True, type=Path)
    args = parser.parse_args()
    raw_dir = args.data_root.expanduser().resolve() / "raw"

    if not raw_dir.is_dir():
        print(f"ERROR: missing directory: {raw_dir}", file=sys.stderr)
        return 2

    dataset_ids = {}  # type: Dict[str, Set[str]]
    failures = []  # type: List[str]

    for filename, spec in {**FILES, **METADATA}.items():
        path = raw_dir / filename
        if not path.is_file() or path.stat().st_size == 0:
            failures.append(f"{filename}: missing or empty")
            continue
        try:
            header, ids, rows = inspect_csv(path, str(spec["id"]))
        except (OSError, csv.Error, ValueError) as exc:
            failures.append(f"{filename}: {exc}")
            continue

        if filename in FILES and len(header) != spec["columns"]:
            failures.append(
                f"{filename}: expected {spec['columns']} columns, found {len(header)}"
            )
        if filename in METADATA:
            missing = set(spec["required"]) - set(header)
            if missing:
                failures.append(
                    f"{filename}: missing required columns {', '.join(sorted(missing))}"
                )
            dataset_ids[str(spec["dataset"])] = ids

        mib = path.stat().st_size / (1024 * 1024)
        print(f"OK {filename}: rows={rows:,}, columns={len(header):,}, size={mib:.1f} MiB")

    for filename, spec in FILES.items():
        path = raw_dir / filename
        metadata_ids = dataset_ids.get(str(spec["dataset"]))
        if not path.is_file() or metadata_ids is None:
            continue
        try:
            _, embedding_ids, _ = inspect_csv(path, str(spec["id"]))
        except (OSError, csv.Error, ValueError):
            continue
        matched = len(embedding_ids & metadata_ids)
        unmatched = len(embedding_ids - metadata_ids)
        print(f"MATCH {filename}: matched={matched:,}, embedding_only={unmatched:,}")
        if matched == 0:
            failures.append(f"{filename}: no identifiers match its metadata")

    if failures:
        print("\nValidation failed:", file=sys.stderr)
        for failure in failures:
            print(f"- {failure}", file=sys.stderr)
        return 1

    print("\nValidation passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
