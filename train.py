from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from rectallitenet import CAREDataset, RectalLiteNet, SegmentationLoss
from rectallitenet.data import FIXED_CROP, records_from_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train RectalLiteNet on CARE.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", default="rectallitenet")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def learning_rate_factor(epoch: int, epochs: int, warmup_epochs: int) -> float:
    if epoch < warmup_epochs:
        return (epoch + 1) / max(1, warmup_epochs)
    progress = (epoch - warmup_epochs) / max(1, epochs - warmup_epochs - 1)
    return 0.5 * (1 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    records = records_from_manifest(args.manifest, args.data_root)
    dataset = CAREDataset(records, training=True)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.workers > 0,
        prefetch_factor=2 if args.workers > 0 else None,
        drop_last=True,
    )
    model = RectalLiteNet(pretrained=args.pretrained).to(device)
    criterion = SegmentationLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda epoch: learning_rate_factor(
            epoch, args.epochs, args.warmup_epochs
        ),
    )
    scaler = GradScaler(device.type, enabled=args.amp and device.type == "cuda")

    run = None
    if args.wandb:
        import wandb

        run = wandb.init(project=args.wandb_project, config=vars(args))

    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        progress = tqdm(loader, desc=f"epoch {epoch}/{args.epochs}", leave=False)
        for batch in progress:
            image = batch["image"].to(device, non_blocking=True)
            target = batch["label"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast(
                device_type=device.type,
                enabled=args.amp and device.type == "cuda",
            ):
                loss = criterion(model(image), target)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss at epoch {epoch}")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            total_loss += float(loss.detach())
            progress.set_postfix(loss=f"{float(loss):.4f}")
        scheduler.step()
        row = {
            "epoch": epoch,
            "loss": total_loss / len(loader),
            "lr": optimizer.param_groups[0]["lr"],
        }
        history.append(row)
        print(json.dumps(row))
        if run is not None:
            run.log(row, step=epoch)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format_version": 1,
            "model_state": model.state_dict(),
            "model_config": {
                "name": "RectalLiteNet",
                "context_slices": 3,
                "encoder_name": "convnext_tiny",
                "image_size": 384,
                "normalization": "CARE",
                "fixed_crop_box": list(FIXED_CROP),
            },
            "metadata": {
                "seed": args.seed,
                "epoch": args.epochs,
                "training_patients": len({record.patient_id for record in records}),
                "training_slices": len(records),
            },
            "history": history,
        },
        output,
    )
    if run is not None:
        run.finish()
    print(json.dumps({"checkpoint": str(output), "status": "completed"}))


if __name__ == "__main__":
    main()
