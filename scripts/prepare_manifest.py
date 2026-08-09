"""Create a CSV manifest from a directory of CARE NPZ files."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import numpy as np

SAMPLE_PATTERN = re.compile(r"^(?P<patient>case\d+)_slice(?P<slice>\d+)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a CARE data manifest.")
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--relative-to",
        default=None,
        help="Store paths relative to this directory (normally --data-root).",
    )
    parser.add_argument(
        "--allow-unlabeled",
        action="store_true",
        help="Do not require a label array (useful for inference data).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir).resolve()
    relative_to = Path(args.relative_to).resolve() if args.relative_to else None
    rows = []

    for path in sorted(input_dir.glob("*.npz")):
        match = SAMPLE_PATTERN.match(path.stem)
        if match is None:
            raise ValueError(
                f"Unexpected filename {path.name!r}; expected case<number>_slice<number>.npz"
            )
        with np.load(path) as sample:
            if "image" not in sample:
                raise ValueError(f"{path} has no 'image' array")
            if not args.allow_unlabeled and "label" not in sample:
                raise ValueError(f"{path} has no 'label' array")
            image = np.squeeze(sample["image"])
            if image.ndim != 2:
                raise ValueError(f"Expected a 2D image in {path}, got {image.shape}")

        if relative_to is not None:
            try:
                stored_path = path.relative_to(relative_to)
            except ValueError as error:
                raise ValueError(f"{path} is not inside {relative_to}") from error
        else:
            stored_path = path
        rows.append(
            {
                "path": str(stored_path),
                "patient_id": match.group("patient"),
                "slice_id": int(match.group("slice")),
                "height": int(image.shape[0]),
                "width": int(image.shape[1]),
            }
        )

    if not rows:
        raise FileNotFoundError(f"No NPZ files found in {input_dir}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} slices to {output_path}")


if __name__ == "__main__":
    main()
