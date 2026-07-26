from __future__ import annotations

from pathlib import Path

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F


def convolution_block(in_channels: int, out_channels: int) -> nn.Sequential:
    groups = max(
        value
        for value in range(1, min(16, out_channels) + 1)
        if out_channels % value == 0
    )
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
        nn.GroupNorm(groups, out_channels),
        nn.GELU(),
        nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
        nn.GroupNorm(groups, out_channels),
        nn.GELU(),
    )


class SCSE(nn.Module):
    def __init__(self, channels: int, reduction: int = 8) -> None:
        super().__init__()
        hidden = max(1, channels // reduction)
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, 1),
            nn.GELU(),
            nn.Conv2d(hidden, channels, 1),
            nn.Sigmoid(),
        )
        self.spatial_gate = nn.Sequential(nn.Conv2d(channels, 1, 1), nn.Sigmoid())

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return features * self.channel_gate(features) + features * self.spatial_gate(
            features
        )


class DecoderBlock(nn.Module):
    def __init__(
        self, in_channels: int, skip_channels: int, out_channels: int
    ) -> None:
        super().__init__()
        self.block = convolution_block(in_channels + skip_channels, out_channels)
        self.scse = SCSE(out_channels)
        # Retained for strict compatibility with the evaluated computation graph.
        self.coordinate_gate = nn.Identity()

    def forward(
        self, features: torch.Tensor, skip: torch.Tensor | None = None
    ) -> torch.Tensor:
        features = F.interpolate(
            features, scale_factor=2, mode="bilinear", align_corners=False
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
        return self.coordinate_gate(self.scse(self.block(features)))


class RectalLiteNet(nn.Module):
    """Compact 2.5D ConvNeXt-SCSE network evaluated in the CARE study."""

    def __init__(
        self,
        context_slices: int = 3,
        encoder_name: str = "convnext_tiny",
        pretrained: bool = False,
        encoder_checkpoint: str | None = None,
    ) -> None:
        super().__init__()
        overlay = None
        if encoder_checkpoint is not None:
            checkpoint = Path(encoder_checkpoint)
            if not checkpoint.is_file():
                raise FileNotFoundError(checkpoint)
            if not pretrained:
                raise ValueError("encoder_checkpoint requires pretrained=True")
            overlay = {"file": str(checkpoint), "hf_hub_id": None, "url": ""}
        self.encoder = timm.create_model(
            encoder_name,
            pretrained=pretrained,
            features_only=True,
            in_chans=context_slices,
            drop_path_rate=0.0,
            pretrained_cfg_overlay=overlay,
        )
        c0, c1, c2, c3 = self.encoder.feature_info.channels()
        self.bottleneck = convolution_block(c3, 256)
        self.bridge = nn.Identity()
        self.dec3 = DecoderBlock(256, c2, 128)
        self.dec2 = DecoderBlock(128, c1, 64)
        self.dec1 = DecoderBlock(64, c0, 32)
        self.dec0 = DecoderBlock(32, 0, 32)
        self.classifier = nn.Conv2d(32, 3, 1)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        input_size = image.shape[-2:]
        f0, f1, f2, f3 = self.encoder(image)
        features = self.bridge(self.bottleneck(f3))
        features = self.dec3(features, f2)
        features = self.dec2(features, f1)
        features = self.dec1(features, f0)
        features = self.dec0(features)
        features = F.interpolate(
            features, size=input_size, mode="bilinear", align_corners=False
        )
        return self.classifier(features)


def load_checkpoint(
    path: str | Path, device: str | torch.device = "cpu"
) -> tuple[RectalLiteNet, dict[str, object]]:
    payload = torch.load(path, map_location=device, weights_only=False)
    state = payload.get("model_state", payload)
    config = payload.get("model_config", {})
    model = RectalLiteNet(
        context_slices=int(config.get("context_slices", 3)),
        encoder_name=str(config.get("encoder_name", "convnext_tiny")),
        pretrained=False,
    )
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    return model, payload
