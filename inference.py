"""Run RectalLiteNet on an unlabeled or labeled directory of CARE NPZ files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch.amp import autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets import CAREDataset, records_from_directory
from utils import (
    load_checkpoint,
    load_config,
    predict_probabilities,
    restore_probability,
)

PALETTE = np.array(
    [
        [0, 0, 0],  # background: black
        [0, 200, 0],  # normal rectum: green
        [0, 0, 255],  # tumor: red (OpenCV BGR)
    ],
    dtype=np.uint8,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run RectalLiteNet inference.")
    parser.add_argument("--config", default="configs/default.json")
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Directory containing CARE NPZ files.",
    )
    parser.add_argument("--checkpoint", action="append", required=True)
    parser.add_argument(
        "--output-dir",
        required=True,
        help="New prediction directory; must not already exist.",
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
    output_dir = Path(args.output_dir)
    if output_dir.exists():
        raise FileExistsError(
            f"Output directory already exists: {output_dir}. "
            "Choose a new prediction directory."
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

    records = records_from_directory(args.input_dir)
    dataset = CAREDataset(
        records,
        training=False,
        require_label=False,
        return_original_label=False,
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

    output_dir.mkdir(parents=True, exist_ok=False)
    use_amp = amp_requested and device.type == "cuda"
    output_files = []

    with torch.inference_mode():
        for batch in tqdm(loader, desc="inference"):
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
            predictions = full_probability.argmax(dim=1).cpu().numpy().astype(np.uint8)

            for index, prediction in enumerate(predictions):
                sample_id = batch["sample_id"][index]
                numpy_path = output_dir / f"{sample_id}.npy"
                mask_path = output_dir / f"{sample_id}.png"
                color_path = output_dir / f"{sample_id}_color.png"
                np.save(numpy_path, prediction)
                cv2.imwrite(str(mask_path), prediction)
                cv2.imwrite(str(color_path), PALETTE[prediction])
                output_files.append(
                    {
                        "sample_id": sample_id,
                        "mask": str(mask_path),
                        "color_mask": str(color_path),
                        "array": str(numpy_path),
                    }
                )

    summary = {
        "model": "RectalLiteNet",
        "checkpoints": args.checkpoint,
        "tta": tta,
        "predictions": len(output_files),
        "label_values": {"0": "background", "1": "normal_rectum", "2": "tumor"},
        "files": output_files,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Saved {len(output_files)} predictions to {output_dir}")


if __name__ == "__main__":
    main()
