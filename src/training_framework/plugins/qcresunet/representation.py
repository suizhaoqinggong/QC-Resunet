"""QC regions and the tensor-only transport used by the framework contracts."""

from __future__ import annotations

import math
from typing import Sequence

import torch

BRATS_REGIONS = ((1, 2, 3), (1, 3), (3,))


def validate_regions(regions: Sequence[Sequence[int]], num_classes: int) -> tuple[tuple[int, ...], ...]:
    result = tuple(tuple(int(label) for label in region) for region in regions)
    if len(result) != num_classes or any(not region for region in result):
        raise ValueError("One nonempty label group is required per QC tissue class")
    if any(label < 1 for region in result for label in region):
        raise ValueError("QC regions must contain positive labels; background is zero")
    return result


def validate_mask(mask: torch.Tensor, regions: Sequence[Sequence[int]]) -> None:
    allowed = {0, *(label for region in regions for label in region)}
    if not torch.isfinite(mask).all() or not torch.equal(mask, mask.round()):
        raise ValueError("Query/reference masks must contain finite integer labels")
    valid = torch.zeros_like(mask, dtype=torch.bool)
    for value in allowed:
        valid |= mask == value
    if not valid.all():
        raise ValueError(f"Mask labels must belong to {sorted(allowed)}; remap BraTS label 4 to 3")


def region_masks(mask: torch.Tensor, regions: Sequence[Sequence[int]]) -> torch.Tensor:
    """Convert [B,D,H,W] integer masks to overlapping [B,C,D,H,W] masks."""
    return torch.stack([torch.stack([mask == label for label in region]).any(0) for region in regions], dim=1)


def pack(quality: torch.Tensor, errors: torch.Tensor) -> torch.Tensor:
    """[B,2] DSC/NSD followed by C spatial maps in channel-major order."""
    return torch.cat((quality, errors.flatten(1)), dim=1)


def unpack(
    value: torch.Tensor, num_classes: int, spatial_shape: Sequence[int] | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    if value.ndim != 2 or value.shape[1] <= 2 or (value.shape[1] - 2) % num_classes:
        raise ValueError("Expected packed QC tensor [B, 2 + C*D*H*W]")
    errors = value[:, 2:].reshape(value.shape[0], num_classes, -1)
    if spatial_shape is not None:
        if math.prod(spatial_shape) != errors.shape[-1]:
            raise ValueError("QC spatial shape does not match the packed tensor")
        errors = errors.reshape(value.shape[0], num_classes, *spatial_shape)
    return value[:, :2], errors
