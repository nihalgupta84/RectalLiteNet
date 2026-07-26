from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import numpy as np


PATTERN = re.compile(r"^(?P<patient>case\d+)_slice(?P<slice>\d+)$")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a CARE NPZ manifest.")
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--relative-to",
        default=None,
        help="Write paths relative to this data root.",
    )
    args = parser.parse_args()
    input_dir = Path(args.input_dir).resolve()
    relative_to = Path(args.relative_to).resolve() if args.relative_to else None
    rows = []
    for path in sorted(input_dir.glob("*.npz")):
        match = PATTERN.match(path.stem)
        if match is None:
            raise ValueError(f"Unexpected CARE filename: {path.name}")
        with np.load(path) as sample:
            if "image" not in sample or "label" not in sample:
                raise ValueError(f"{path} must contain image and label arrays")
            height, width = np.squeeze(sample["label"]).shape
        output_path = path.relative_to(relative_to) if relative_to else path
        rows.append(
            {
                "path": str(output_path),
                "patient_id": match.group("patient"),
                "slice_id": match.group("slice"),
                "height": height,
                "width": width,
            }
        )
    if not rows:
        raise FileNotFoundError(f"No NPZ files in {input_dir}")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} records to {output}")


if __name__ == "__main__":
    main()
