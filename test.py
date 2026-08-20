"""Evaluate one checkpoint or an ensemble on a labeled CARE manifest."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from torch.amp import autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets import CAREDataset, records_from_manifest
from utils import (
    MetricAccumulator,
    load_checkpoint,
    load_config,
    predict_probabilities,
    restore_probability,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test RectalLiteNet on labeled CARE data."
    )
    parser.add_argument("--config", default="configs/default.json")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--checkpoint", action="append", required=True)
    parser.add_argument(
        "--output",
        required=True,
        help="New JSON result path; must not already exist.",
    )
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--tta", choices=["none", "flips"], default=None)
    parser.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    return parser.parse_args()


def _data_settings(payload: dict[str, Any]) -> dict[str, Any]:
    config = payload.get("model_config", {})
    return {
        "image_size": int(config.get("image_size", 384)),
        "crop_box": config.get("fixed_crop_box", [88, 54, 440, 406]),
        "mean": float(config.get("mean", 0.1364736)),
        "std": float(config.get("std", 0.23238614)),
    }


def main() -> None:
    args = parse_args()
    output_path = Path(args.output)
    if output_path.exists():
        raise FileExistsError(
            f"Output file already exists: {output_path}. Choose a new result path."
        )
    inference_config = load_config(args.config)["inference"]
    batch_size = (
        args.batch_size
        if args.batch_size is not None
        else int(inference_config["batch_size"])
    )
    workers = (
        args.workers
        if args.workers is not None
        else int(inference_config["workers"])
    )
    tta = args.tta if args.tta is not None else str(inference_config["tta"])
    amp_requested = (
        args.amp if args.amp is not None else bool(inference_config["amp"])
    )
    if batch_size <= 0 or workers < 0:
        raise ValueError("batch-size must be positive and workers cannot be negative")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaded = [load_checkpoint(path, device) for path in args.checkpoint]
    models = [model for model, _ in loaded]
    settings = _data_settings(loaded[0][1])
    for _, payload in loaded[1:]:
        if _data_settings(payload) != settings:
            raise ValueError("All ensemble checkpoints must use identical preprocessing")

    records = records_from_manifest(args.manifest, args.data_root)
    dataset = CAREDataset(
        records,
        training=False,
        require_label=True,
        return_original_label=True,
        **settings,
    )
    loader_options: dict[str, Any] = {
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": workers > 0,
    }
    if workers > 0:
        loader_options["prefetch_factor"] = 2
    loader = DataLoader(dataset, **loader_options)

    global_metrics = MetricAccumulator()
    patient_metrics: dict[str, MetricAccumulator] = defaultdict(MetricAccumulator)
    use_amp = amp_requested and device.type == "cuda"
    processed_slices = 0

    with torch.inference_mode():
        for batch in tqdm(loader, desc="test"):
            image = batch["image"].to(device, non_blocking=True)
            with autocast(device_type=device.type, enabled=use_amp):
                crop_probability = predict_probabilities(models, image, tta)

            heights = batch["height"].tolist()
            widths = batch["width"].tolist()
            if len(set(zip(heights, widths))) != 1:
                raise ValueError("All images in a batch must have the same original shape")
            full_probability = restore_probability(
                crop_probability,
                (heights[0], widths[0]),
                settings["crop_box"],
            )
            predictions = full_probability.argmax(dim=1).cpu().numpy()
            targets = batch["original_label"].numpy()

            for index, prediction in enumerate(predictions):
                target = targets[index]
                patient_id = batch["patient_id"][index]
                global_metrics.update(prediction, target)
                patient_metrics[patient_id].update(prediction, target)
                processed_slices += 1

    per_patient = [
        {"patient_id": patient_id, **patient_metrics[patient_id].compute()}
        for patient_id in sorted(patient_metrics)
    ]
    result = {
        "schema_version": 1,
        "model": "RectalLiteNet",
        "checkpoints": args.checkpoint,
        "tta": tta,
        "processed_slices": processed_slices,
        "patients": len(patient_metrics),
        "global_metrics": global_metrics.compute(),
        "per_patient": per_patient,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")
    print(json.dumps(result["global_metrics"], indent=2))
    print(f"Saved detailed results to {output_path}")


if __name__ == "__main__":
    main()
