"""Shared prediction functions for testing and inference."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F


def predict_probabilities(
    models: Sequence[torch.nn.Module],
    image: torch.Tensor,
    tta: str = "flips",
) -> torch.Tensor:
    """Average class probabilities over models and optional flip transforms."""
    if not models:
        raise ValueError("At least one model is required")
    if tta not in {"none", "flips"}:
        raise ValueError("tta must be 'none' or 'flips'")

    flip_dimensions: list[tuple[int, ...]] = [()]
    if tta == "flips":
        flip_dimensions.extend([(-1,), (-2,), (-2, -1)])

    model_probabilities = []
    for model in models:
        transformed_probabilities = []
        for dimensions in flip_dimensions:
            transformed_image = torch.flip(image, dimensions) if dimensions else image
            probability = torch.softmax(model(transformed_image).float(), dim=1)
            if dimensions:
                probability = torch.flip(probability, dimensions)
            transformed_probabilities.append(probability)
        model_probabilities.append(torch.stack(transformed_probabilities).mean(dim=0))
    return torch.stack(model_probabilities).mean(dim=0)


def restore_probability(
    crop_probability: torch.Tensor,
    output_size: tuple[int, int],
    crop_box: Sequence[int],
) -> torch.Tensor:
    """Place crop probabilities back into the original full image."""
    if crop_probability.ndim != 4:
        raise ValueError("crop_probability must have shape [batch, classes, height, width]")
    x1, y1, x2, y2 = (int(value) for value in crop_box)
    output_height, output_width = output_size
    if not (0 <= x1 < x2 <= output_width and 0 <= y1 < y2 <= output_height):
        raise ValueError(f"Crop {tuple(crop_box)} is outside output size {output_size}")

    full_probability = torch.zeros(
        crop_probability.shape[0],
        crop_probability.shape[1],
        output_height,
        output_width,
        dtype=crop_probability.dtype,
        device=crop_probability.device,
    )
    full_probability[:, 0] = 1.0
    full_probability[:, :, y1:y2, x1:x2] = F.interpolate(
        crop_probability,
        size=(y2 - y1, x2 - x1),
        mode="bilinear",
        align_corners=False,
    )
    return full_probability
