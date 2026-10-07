"""3D ResNet-34 + Block-3 decoder, following MedIA Figure 2."""

from __future__ import annotations

from typing import Sequence, cast

import torch
from torch import nn
from torch.nn import functional as F

from training_framework.contracts.types import Batch
from training_framework.plugins.qcresunet.representation import (
    BRATS_REGIONS,
    pack,
    region_masks,
    validate_mask,
    validate_regions,
)


def norm(channels: int) -> nn.Module:
    return nn.InstanceNorm3d(channels, affine=True)


class ResidualBlock(nn.Module):
    def __init__(self, incoming: int, channels: int, stride: tuple[int, int, int], dropout: float) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv3d(incoming, channels, 3, stride=stride, padding=1, bias=False),
            norm(channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(channels, channels, 3, padding=1, bias=False),
            norm(channels),
            nn.Dropout3d(dropout),
        )
        self.shortcut = (
            nn.Sequential(nn.Conv3d(incoming, channels, 1, stride=stride, bias=False), norm(channels))
            if incoming != channels or stride != (1, 1, 1)
            else nn.Identity()
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return F.relu(self.body(value) + self.shortcut(value))


class DecoderBlock(nn.Module):
    def __init__(self, incoming: int, skip: int, channels: int) -> None:
        super().__init__()
        self.project = nn.Conv3d(incoming, channels, 1, bias=False)
        self.body = nn.Sequential(
            nn.Conv3d(channels + skip, channels, 3, padding=1, bias=False),
            norm(channels),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv3d(channels, channels, 3, padding=1, bias=False),
            norm(channels),
            nn.LeakyReLU(0.01, inplace=True),
        )

    def forward(self, value: torch.Tensor, shape: Sequence[int], skip: torch.Tensor | None = None) -> torch.Tensor:
        value = self.project(F.interpolate(value, size=list(shape), mode="nearest"))
        if skip is not None:
            value = torch.cat((value, skip), dim=1)
        return self.body(value)


class EfficientChannelAttention(nn.Module):
    def __init__(self, kernel_size: int = 3) -> None:
        super().__init__()
        if kernel_size < 1 or kernel_size % 2 != 1:
            raise ValueError("ECA kernel_size must be a positive odd integer")
        self.conv = nn.Conv1d(1, 1, kernel_size, padding=kernel_size // 2, bias=False)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        weights = self.conv(value.mean(dim=(2, 3, 4)).unsqueeze(1)).sigmoid()
        return value * weights.squeeze(1)[:, :, None, None, None]


class QCResUNet(nn.Module):
    """Defaults are the paper model. Reduced widths/depths are for smoke tests.

    signal contains M MRI channels and ONE raw integer query-label channel.
    Neither label nor metadata is read. Forward returns a packed tensor; use
    forward_heads(signal) for the native [B,2], [B,C,D,H,W] outputs.
    """

    def __init__(
        self,
        num_classes: int = 3,
        modalities: int = 4,
        base_channels: int = 64,
        blocks: Sequence[int] = (3, 4, 6, 3),
        dropout: float = 0.3,
        regions: Sequence[Sequence[int]] = BRATS_REGIONS,
        downsample_strides: Sequence[Sequence[int]] = ((2, 2, 2),) * 4,
        stem_stride: Sequence[int] = (2, 2, 2),
        eca_kernel_size: int = 3,
        attention: bool = True,
    ) -> None:
        super().__init__()
        self.regions = validate_regions(regions, num_classes)
        self.modalities, self.num_classes = modalities, num_classes
        if modalities < 1 or base_channels < 2 or base_channels % 2 or len(blocks) != 4 or min(blocks) < 1:
            raise ValueError("Positive modalities, an even base_channels >= 2 and four positive block counts required")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0,1)")
        strides = [cast(tuple[int, int, int], tuple(int(v) for v in item)) for item in downsample_strides]
        stem = cast(tuple[int, int, int], tuple(int(v) for v in stem_stride))
        if len(strides) != 4 or any(len(s) != 3 or min(s) < 1 for s in [*strides, stem]):
            raise ValueError("Four 3D downsample strides and a 3D stem stride are required")
        self.stem = nn.Sequential(
            nn.Conv3d(modalities + 1, base_channels, 7, stride=stem, padding=3, bias=False),
            norm(base_channels),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.MaxPool3d(kernel_size=strides[0], stride=strides[0])
        stages: list[nn.Module] = []
        incoming = base_channels
        for index, count in enumerate(blocks):
            channels = base_channels * 2**index
            stride = (1, 1, 1) if index == 0 else strides[index]
            stages.append(
                nn.Sequential(
                    ResidualBlock(incoming, channels, stride, dropout),
                    *[ResidualBlock(channels, channels, (1, 1, 1), dropout) for _ in range(count - 1)],
                )
            )
            incoming = channels
        self.stages = nn.ModuleList(stages)
        self.quality_head = nn.Linear(base_channels * 8, 2)
        self.decoder = nn.ModuleList(
            [
                DecoderBlock(base_channels * 4, base_channels * 2, base_channels * 2),
                DecoderBlock(base_channels * 2, base_channels, base_channels),
                DecoderBlock(base_channels, base_channels, base_channels),
                DecoderBlock(base_channels, 0, base_channels // 2),
            ]
        )
        self.error_project = nn.Conv3d(base_channels // 2, num_classes, 1)
        self.attention = EfficientChannelAttention(eca_kernel_size) if attention else nn.Identity()
        self.error_head = nn.Conv3d(num_classes * 2, num_classes, 1)

    def forward_heads(self, signal: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if signal.ndim != 5 or signal.shape[1] != self.modalities + 1:
            raise ValueError("Expected MRI plus raw query mask [B,M+1,D,H,W]")
        query = signal[:, -1]
        validate_mask(query, self.regions)
        stem = self.stem(signal)
        block1 = self.stages[0](self.pool(stem))
        block2 = self.stages[1](block1)
        block3 = self.stages[2](block2)
        block4 = self.stages[3](block3)
        quality = self.quality_head(block4.mean(dim=(2, 3, 4)))
        value = block3
        for decoder, skip in zip(self.decoder[:3], (block2, block1, stem)):
            value = decoder(value, skip.shape[2:], skip)
        value = self.decoder[3](value, signal.shape[2:])
        value = torch.cat((self.error_project(value), region_masks(query, self.regions).to(value.dtype)), dim=1)
        return quality, self.error_head(self.attention(value))

    def forward(self, batch: Batch) -> torch.Tensor:
        return pack(*self.forward_heads(batch["signal"]))
