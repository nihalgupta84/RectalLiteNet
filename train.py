"""Train RectalLiteNet.

Example:
    python train.py --train-manifest data/manifests/train.csv \
        --val-manifest data/manifests/val.csv --data-root data/CARE
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets import CAREDataset, records_from_manifest
from models import RectalLiteNet
from utils import (
    MetricAccumulator,
    SegmentationLoss,
    load_config,
    restore_probability,
    save_checkpoint,
    seed_everything,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train RectalLiteNet on CARE data.")
    parser.add_argument("--config", default="configs/default.json")
    parser.add_argument("--train-manifest", required=True)
    parser.add_argument("--val-manifest", default=None)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-dir", default="logs/training")
    parser.add_argument("--epochs", type=int, default=None, help="Override config value.")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override config value.",
    )
    parser.add_argument(
        "--accumulation-steps",
        type=int,
        default=None,
        help="Override gradient_accumulation_steps from the config.",
    )
    parser.add_argument("--workers", type=int, default=None, help="Override config value.")
    parser.add_argument("--seed", type=int, default=None, help="Override config value.")
    parser.add_argument(
        "--pretrained",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable or disable ImageNet encoder initialization.",
    )
    parser.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable or disable automatic mixed precision.",
    )
    return parser.parse_args()


def _loader(
    dataset: CAREDataset,
    batch_size: int,
    workers: int,
    shuffle: bool,
    device: torch.device,
) -> DataLoader:
    options: dict[str, Any] = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": workers > 0,
        "drop_last": shuffle and len(dataset) >= batch_size,
    }
    if workers > 0:
        options["prefetch_factor"] = 2
    return DataLoader(dataset, **options)


def learning_rate_factor(epoch: int, epochs: int, warmup_epochs: int) -> float:
    """Linear warmup followed by cosine decay."""
    if epoch < warmup_epochs:
        return (epoch + 1) / max(1, warmup_epochs)
    progress = (epoch - warmup_epochs) / max(1, epochs - warmup_epochs - 1)
    progress = min(max(progress, 0.0), 1.0)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def validate(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: SegmentationLoss,
    device: torch.device,
    use_amp: bool,
    crop_box: list[int],
) -> dict[str, float]:
    """Evaluate loss and overlap restored to the original image grid."""
    model.eval()
    metrics = MetricAccumulator()
    total_loss = 0.0
    with torch.inference_mode():
        for batch in tqdm(loader, desc="validate", leave=False):
            image = batch["image"].to(device, non_blocking=True)
            target = batch["label"].to(device, non_blocking=True)
            with autocast(device_type=device.type, enabled=use_amp):
                logits = model(image)
                loss = criterion(logits, target)
            total_loss += float(loss)
            heights = batch["height"].tolist()
            widths = batch["width"].tolist()
            if len(set(zip(heights, widths))) != 1:
                raise ValueError("All images in a batch must have the same original shape")
            full_probability = restore_probability(
                torch.softmax(logits.float(), dim=1),
                (heights[0], widths[0]),
                crop_box,
            )
            prediction = full_probability.argmax(dim=1).cpu().numpy()
            target_array = batch["original_label"].numpy()
            for predicted_mask, target_mask in zip(prediction, target_array):
                metrics.update(predicted_mask, target_mask)
    summary = metrics.compute()
    return {
        "loss": total_loss / len(loader),
        "macro_dice": float(summary["macro_dice"]),
        "normal_dice": float(summary["classes"]["normal_rectum"]["dice"]),
        "tumor_dice": float(summary["classes"]["tumor"]["dice"]),
    }


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    data_config = config["data"]
    model_config = config["model"]
    training_config = config["training"]

    epochs = args.epochs if args.epochs is not None else int(training_config["epochs"])
    batch_size = (
        args.batch_size
        if args.batch_size is not None
        else int(training_config["batch_size"])
    )
    accumulation_steps = (
        args.accumulation_steps
        if args.accumulation_steps is not None
        else int(training_config["gradient_accumulation_steps"])
    )
    workers = args.workers if args.workers is not None else int(training_config["workers"])
    seed = args.seed if args.seed is not None else int(training_config["seed"])
    pretrained = (
        args.pretrained
        if args.pretrained is not None
        else bool(model_config["pretrained"])
    )
    amp_requested = args.amp if args.amp is not None else bool(training_config["amp"])

    if epochs <= 0 or batch_size <= 0 or accumulation_steps <= 0 or workers < 0:
        raise ValueError(
            "epochs, batch-size, and accumulation-steps must be positive; "
            "workers cannot be negative"
        )

    seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = amp_requested and device.type == "cuda"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_options = {
        "image_size": int(data_config["image_size"]),
        "crop_box": data_config["fixed_crop_box"],
        "mean": float(data_config["mean"]),
        "std": float(data_config["std"]),
    }
    train_records = records_from_manifest(args.train_manifest, args.data_root)
    train_dataset = CAREDataset(train_records, training=True, **dataset_options)
    train_loader = _loader(train_dataset, batch_size, workers, True, device)

    val_loader = None
    val_records = []
    if args.val_manifest:
        val_records = records_from_manifest(args.val_manifest, args.data_root)
        train_patients = {record.patient_id for record in train_records}
        val_patients = {record.patient_id for record in val_records}
        overlap = train_patients & val_patients
        if overlap:
            raise ValueError(f"Patient leakage between train and val: {sorted(overlap)[:5]}")
        val_dataset = CAREDataset(
            val_records,
            training=False,
            return_original_label=True,
            **dataset_options,
        )
        val_loader = _loader(val_dataset, batch_size, workers, False, device)

    model = RectalLiteNet(
        context_slices=int(data_config["context_slices"]),
        encoder_name=str(model_config["encoder_name"]),
        pretrained=pretrained,
    ).to(device)
    criterion = SegmentationLoss(
        cross_entropy_weight=float(training_config["cross_entropy_weight"]),
        dice_weight=float(training_config["dice_weight"]),
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training_config["learning_rate"]),
        weight_decay=float(training_config["weight_decay"]),
    )
    warmup_epochs = int(training_config["warmup_epochs"])
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda epoch: learning_rate_factor(epoch, epochs, warmup_epochs),
    )
    scaler = GradScaler(device.type, enabled=use_amp)

    stored_model_config = {
        "name": "RectalLiteNet",
        "context_slices": int(data_config["context_slices"]),
        "encoder_name": str(model_config["encoder_name"]),
        "image_size": int(data_config["image_size"]),
        "normalization": "CARE",
        "mean": float(data_config["mean"]),
        "std": float(data_config["std"]),
        "fixed_crop_box": list(data_config["fixed_crop_box"]),
    }
    history: list[dict[str, Any]] = []
    best_score = -np.inf

    print(
        json.dumps(
            {
                "device": str(device),
                "amp": use_amp,
                "training_slices": len(train_records),
                "validation_slices": len(val_records),
                "effective_batch_size": batch_size * accumulation_steps,
            }
        )
    )

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        progress = tqdm(train_loader, desc=f"epoch {epoch}/{epochs}")
        optimizer.zero_grad(set_to_none=True)
        for batch_index, batch in enumerate(progress, start=1):
            image = batch["image"].to(device, non_blocking=True)
            target = batch["label"].to(device, non_blocking=True)
            with autocast(device_type=device.type, enabled=use_amp):
                batch_loss = criterion(model(image), target)
                group_start = ((batch_index - 1) // accumulation_steps) * accumulation_steps
                group_size = min(accumulation_steps, len(train_loader) - group_start)
                scaled_loss = batch_loss / group_size
            if not torch.isfinite(batch_loss):
                raise FloatingPointError(f"Non-finite loss at epoch {epoch}")
            scaler.scale(scaled_loss).backward()

            should_step = (
                batch_index % accumulation_steps == 0
                or batch_index == len(train_loader)
            )
            if should_step:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            total_loss += float(batch_loss.detach())
            progress.set_postfix(loss=f"{float(batch_loss.detach()):.4f}")

        scheduler.step()
        row: dict[str, Any] = {
            "epoch": epoch,
            "train_loss": total_loss / len(train_loader),
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        if val_loader is not None:
            row["validation"] = validate(
                model,
                val_loader,
                criterion,
                device,
                use_amp,
                list(data_config["fixed_crop_box"]),
            )
            score = row["validation"]["macro_dice"]
        else:
            score = -row["train_loss"]
        history.append(row)
        print(json.dumps(row))

        checkpoint_payload = {
            "format_version": 1,
            "model_state": model.state_dict(),
            "model_config": stored_model_config,
            "metadata": {
                "seed": seed,
                "epoch": epoch,
                "training_patients": len({record.patient_id for record in train_records}),
                "validation_patients": len({record.patient_id for record in val_records}),
                "physical_batch_size": batch_size,
                "gradient_accumulation_steps": accumulation_steps,
                "effective_batch_size": batch_size * accumulation_steps,
            },
            "history": history,
        }
        save_checkpoint(checkpoint_payload, output_dir / "last.pt")
        if score > best_score:
            best_score = score
            save_checkpoint(checkpoint_payload, output_dir / "best.pt")

        (output_dir / "history.json").write_text(
            json.dumps(history, indent=2) + "\n",
            encoding="utf-8",
        )

    print(json.dumps({"status": "completed", "output_dir": str(output_dir)}))


if __name__ == "__main__":
    main()
