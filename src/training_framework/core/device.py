"""Device resolution, AMP, and batch movement utilities."""

from typing import Optional

import torch
from torch.amp import GradScaler

from training_framework.contracts.types import Batch


def resolve_device(device_spec: Optional[str]) -> torch.device:
    """Resolve device string to torch.device."""
    if device_spec is None or device_spec == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(device_spec)

    if device.type == "cuda" and not torch.backends.cudnn.deterministic:
        torch.backends.cudnn.benchmark = True
    return device


def build_grad_scaler(enabled: bool) -> Optional[GradScaler]:
    """Build a GradScaler if AMP is enabled."""
    if enabled:
        return GradScaler()
    return None


def move_batch_to_device(batch: Batch, device: torch.device) -> Batch:
    """Move all tensors in a batch to the given device."""
    return Batch(
        signal=batch["signal"].to(device, non_blocking=True),
        label=batch["label"].to(device, non_blocking=True),
        id=batch["id"],
        meta=batch["meta"],
    )
