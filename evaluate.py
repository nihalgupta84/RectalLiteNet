from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.amp import autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from rectallitenet import CAREDataset, load_checkpoint, records_from_manifest
from rectallitenet.data import FIXED_CROP
from rectallitenet.metrics import CLASS_NAMES, overlap_counts, summarize_counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate RectalLiteNet on CARE.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--checkpoint", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--tta", choices=["none", "flips"], default="flips")
    parser.add_argument(
        "--reference-index-csv",
        action="append",
        default=[],
        help="Optional source CSV containing a pid column; repeat as needed.",
    )
    parser.add_argument("--amp", action="store_true")
    return parser.parse_args()


def reference_ids(paths: list[str]) -> set[str] | None:
    if not paths:
        return None
    sample_ids = set()
    for path in paths:
        with Path(path).open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if "pid" not in (reader.fieldnames or []):
                raise ValueError(f"{path} has no pid column")
            sample_ids.update(row["pid"] for row in reader)
    return sample_ids


def predict(model: torch.nn.Module, image: torch.Tensor, use_flips: bool) -> torch.Tensor:
    transforms = [(False, False)]
    if use_flips:
        transforms.extend([(True, False), (False, True), (True, True)])
    probabilities = []
    for horizontal, vertical in transforms:
        current = image
        dimensions = []
        if horizontal:
            dimensions.append(-1)
        if vertical:
            dimensions.append(-2)
        if dimensions:
            current = torch.flip(current, dimensions)
        output = torch.softmax(model(current).float(), dim=1)
        if dimensions:
            output = torch.flip(output, dimensions)
        probabilities.append(output)
    return torch.stack(probabilities).mean(dim=0)


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    records = records_from_manifest(args.manifest, args.data_root)
    dataset = CAREDataset(records, training=False)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.workers > 0,
        prefetch_factor=2 if args.workers > 0 else None,
    )
    models = [load_checkpoint(path, device)[0] for path in args.checkpoint]
    selected_ids = reference_ids(args.reference_index_csv)
    global_counts = {class_id: [0, 0, 0] for class_id in CLASS_NAMES}
    patient_counts = defaultdict(
        lambda: {class_id: [0, 0, 0] for class_id in CLASS_NAMES}
    )
    processed = 0
    reference_processed = 0
    x1, y1, x2, y2 = FIXED_CROP

    with torch.inference_mode():
        for batch in tqdm(loader, desc="evaluate"):
            image = batch["image"].to(device, non_blocking=True)
            with autocast(
                device_type=device.type,
                enabled=args.amp and device.type == "cuda",
            ):
                crop_probability = torch.stack(
                    [predict(model, image, args.tta == "flips") for model in models]
                ).mean(dim=0)
            full_probability = torch.zeros(
                image.shape[0], 3, 512, 512, device=device
            )
            full_probability[:, 0] = 1.0
            full_probability[:, :, y1:y2, x1:x2] = F.interpolate(
                crop_probability,
                size=(y2 - y1, x2 - x1),
                mode="bilinear",
                align_corners=False,
            )
            reference_probability = F.interpolate(
                full_probability, size=(224, 224), mode="bilinear", align_corners=False
            )
            prediction = reference_probability.argmax(dim=1).cpu().numpy()
            target = batch["reference_label"].numpy()
            original_prediction = full_probability.argmax(dim=1).cpu().numpy()
            original_target = batch["original_label"].numpy()
            for index, sample_id in enumerate(batch["sample_id"]):
                patient_id = batch["patient_id"][index]
                for class_id in CLASS_NAMES:
                    patient_values = overlap_counts(
                        original_prediction[index], original_target[index], class_id
                    )
                    for position, value in enumerate(patient_values):
                        patient_counts[patient_id][class_id][position] += value
                processed += 1
                if selected_ids is not None and sample_id not in selected_ids:
                    continue
                for class_id in CLASS_NAMES:
                    values = overlap_counts(prediction[index], target[index], class_id)
                    for position, value in enumerate(values):
                        global_counts[class_id][position] += value
                reference_processed += 1

    per_patient = []
    for patient_id in sorted(patient_counts):
        per_patient.append(
            {
                "patient_id": patient_id,
                **summarize_counts(patient_counts[patient_id]),
            }
        )
    result = {
        "schema_version": 1,
        "model": "RectalLiteNet",
        "checkpoints": args.checkpoint,
        "tta": args.tta,
        "processed_slices": processed,
        "reference_slices": reference_processed,
        "patients": len(patient_counts),
        "reference_profile": summarize_counts(global_counts),
        "patient_profile_grid": "original_512x512_all_manifest_slices",
        "per_patient": per_patient,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["reference_profile"], indent=2))


if __name__ == "__main__":
    main()
