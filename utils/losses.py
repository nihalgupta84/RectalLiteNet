"""Loss functions used to train RectalLiteNet."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SegmentationLoss(nn.Module):
    """Weighted cross-entropy and foreground Dice loss."""

    def __init__(
        self,
        cross_entropy_weight: float = 0.4,
        dice_weight: float = 0.6,
        smooth: float = 1e-6,
    ) -> None:
        super().__init__()
        self.cross_entropy_weight = cross_entropy_weight
        self.dice_weight = dice_weight
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        cross_entropy = F.cross_entropy(logits, target)

        probabilities = torch.softmax(logits, dim=1)[:, 1:]
        one_hot_target = (
            F.one_hot(target, num_classes=3)
            .permute(0, 3, 1, 2)
            .float()[:, 1:]
        )
        intersection = (probabilities * one_hot_target).sum(dim=(0, 2, 3))
        denominator = probabilities.sum(dim=(0, 2, 3)) + one_hot_target.sum(
            dim=(0, 2, 3)
        )
        valid_classes = denominator > 0
        dice = (2 * intersection + self.smooth) / (denominator + self.smooth)
        dice_loss = 1 - dice[valid_classes].mean()

        return (
            self.cross_entropy_weight * cross_entropy
            + self.dice_weight * dice_loss
        )
