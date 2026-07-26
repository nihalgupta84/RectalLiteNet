from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SegmentationLoss(nn.Module):
    def __init__(self, smooth: float = 1e-6) -> None:
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        cross_entropy = F.cross_entropy(logits, target)
        probability = torch.softmax(logits, dim=1)[:, 1:]
        one_hot = F.one_hot(target, num_classes=3).permute(0, 3, 1, 2).float()[:, 1:]
        intersection = (probability * one_hot).sum(dim=(0, 2, 3))
        denominator = probability.sum(dim=(0, 2, 3)) + one_hot.sum(dim=(0, 2, 3))
        valid = denominator > 0
        dice = (2 * intersection + self.smooth) / (denominator + self.smooth)
        dice_loss = 1 - dice[valid].mean()
        return 0.4 * cross_entropy + 0.6 * dice_loss
