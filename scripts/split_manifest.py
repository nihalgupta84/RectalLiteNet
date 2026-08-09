"""Create patient-disjoint training and validation manifests."""

from __future__ import annotations

import argparse
import csv
import random
from collections import defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split a CARE manifest by patient.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--train-output", required=True)
    parser.add_argument("--val-output", required=True)
    parser.add_argument("--val-fraction", type=float, default=0.125)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def write_rows(path: str | Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    if not 0 < args.val_fraction < 1:
        raise ValueError("--val-fraction must be between 0 and 1")

    with Path(args.manifest).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    if not rows or "patient_id" not in fieldnames:
        raise ValueError("Manifest is empty or has no patient_id column")

    rows_by_patient: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        rows_by_patient[row["patient_id"]].append(row)

    patients = sorted(rows_by_patient)
    if len(patients) < 2:
        raise ValueError("At least two patients are needed for a split")
    random.Random(args.seed).shuffle(patients)
    validation_count = max(1, round(len(patients) * args.val_fraction))
    validation_count = min(validation_count, len(patients) - 1)
    validation_patients = set(patients[:validation_count])

    train_rows = [row for row in rows if row["patient_id"] not in validation_patients]
    val_rows = [row for row in rows if row["patient_id"] in validation_patients]
    write_rows(args.train_output, fieldnames, train_rows)
    write_rows(args.val_output, fieldnames, val_rows)
    print(
        f"Training: {len(patients) - validation_count} patients, {len(train_rows)} slices\n"
        f"Validation: {validation_count} patients, {len(val_rows)} slices"
    )


if __name__ == "__main__":
    main()
