"""RectalLiteNet model definition.

The network receives three neighboring axial CT slices as three input channels
and predicts one of three classes for every pixel in the center slice.
"""

from __future__ import annotations

from pathlib import Path

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F


def _normalization_groups(channels: int, maximum: int = 16) -> int:
    """Return the largest valid GroupNorm group count up to ``maximum``."""
    return max(group for group in range(1, min(maximum, channels) + 1) if channels % group == 0)


def convolution_block(in_channels: int, out_channels: int) -> nn.Sequential:
    """Two convolution, GroupNorm, and GELU layers."""
    groups = _normalization_groups(out_channels)
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.GroupNorm(groups, out_channels),
        nn.GELU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.GroupNorm(groups, out_channels),
        nn.GELU(),
    )


class SCSE(nn.Module):
    """Concurrent channel and spatial attention."""

    def __init__(self, channels: int, reduction: int = 8) -> None:
        super().__init__()
        hidden_channels = max(1, channels // reduction)
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden_channels, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, channels, kernel_size=1),
            nn.Sigmoid(),
        )
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(channels, 1, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        channel_attention = features * self.channel_gate(features)
        spatial_attention = features * self.spatial_gate(features)
        return channel_attention + spatial_attention


class DecoderBlock(nn.Module):
    """Upsample, combine an encoder skip connection, and refine features."""

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = convolution_block(in_channels + skip_channels, out_channels)
        self.scse = SCSE(out_channels)

    def forward(
        self,
        features: torch.Tensor,
        skip: torch.Tensor | None = None,
    ) -> torch.Tensor:
        features = F.interpolate(
            features,
            scale_factor=2,
            mode="bilinear",
            align_corners=False,
        )
        if skip is not None:
            if features.shape[-2:] != skip.shape[-2:]:
                features = F.interpolate(
                    features,
                    size=skip.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
            features = torch.cat([features, skip], dim=1)
        return self.scse(self.block(features))


class RectalLiteNet(nn.Module):
    """Compact 2.5D model for normal-rectum and rectal-tumor segmentation."""

    def __init__(
        self,
        context_slices: int = 3,
        encoder_name: str = "convnext_tiny",
        pretrained: bool = False,
        encoder_checkpoint: str | None = None,
    ) -> None:
        super().__init__()

        pretrained_overlay = None
        if encoder_checkpoint is not None:
            checkpoint_path = Path(encoder_checkpoint)
            if not checkpoint_path.is_file():
                raise FileNotFoundError(checkpoint_path)
            if not pretrained:
                raise ValueError("encoder_checkpoint requires pretrained=True")
            pretrained_overlay = {
                "file": str(checkpoint_path),
                "hf_hub_id": None,
                "url": "",
            }

        self.encoder = timm.create_model(
            encoder_name,
            pretrained=pretrained,
            features_only=True,
            in_chans=context_slices,
            drop_path_rate=0.0,
            pretrained_cfg_overlay=pretrained_overlay,
        )
        stage_0, stage_1, stage_2, stage_3 = self.encoder.feature_info.channels()

        self.bottleneck = convolution_block(stage_3, 256)
        self.bridge = nn.Identity()
        self.dec3 = DecoderBlock(256, stage_2, 128)
        self.dec2 = DecoderBlock(128, stage_1, 64)
        self.dec1 = DecoderBlock(64, stage_0, 32)
        self.dec0 = DecoderBlock(32, 0, 32)
        self.classifier = nn.Conv2d(32, 3, kernel_size=1)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Return logits with shape ``[batch, 3, height, width]``."""
        input_size = image.shape[-2:]
        stage_0, stage_1, stage_2, stage_3 = self.encoder(image)
        features = self.bridge(self.bottleneck(stage_3))
        features = self.dec3(features, stage_2)
        features = self.dec2(features, stage_1)
        features = self.dec1(features, stage_0)
        features = self.dec0(features)
        features = F.interpolate(
            features,
            size=input_size,
            mode="bilinear",
            align_corners=False,
        )
        return self.classifier(features)
