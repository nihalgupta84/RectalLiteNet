"""Checkpoint save and load functions."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch

from models import RectalLiteNet


def save_checkpoint(payload: dict[str, Any], path: str | Path) -> None:
    """Write a checkpoint atomically so interrupted saves do not corrupt it."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    torch.save(payload, temporary_path)
    os.replace(temporary_path, output_path)


def load_checkpoint(
    path: str | Path,
    device: str | torch.device = "cpu",
) -> tuple[RectalLiteNet, dict[str, Any]]:
    """Build RectalLiteNet and load either a project or plain state checkpoint."""
    checkpoint_path = Path(path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)

    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if not isinstance(payload, dict):
        raise TypeError(f"Checkpoint {checkpoint_path} must contain a dictionary")

    state = payload.get("model_state", payload)
    model_config = payload.get("model_config", {})
    model = RectalLiteNet(
        context_slices=int(model_config.get("context_slices", 3)),
        encoder_name=str(model_config.get("encoder_name", "convnext_tiny")),
        pretrained=False,
    )
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    return model, payload
