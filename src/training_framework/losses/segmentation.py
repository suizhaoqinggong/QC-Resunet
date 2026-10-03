"""Generic multiclass losses for 2D or 3D segmentation."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class SoftDiceLoss(nn.Module):
    def __init__(self, smooth: float = 1e-6, include_background: bool = True) -> None:
        super().__init__()
        self.smooth, self.include_background = smooth, include_background

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        num_classes = logits.shape[1]
        if not self.include_background and num_classes < 2:
            raise ValueError("Excluding background requires at least two classes")
        probabilities = logits.softmax(dim=1)
        one_hot = F.one_hot(targets.long(), num_classes).movedim(-1, 1).to(probabilities.dtype)
        dims = tuple(range(2, logits.ndim))
        intersection = (probabilities * one_hot).sum(dim=dims)
        denominator = probabilities.sum(dim=dims) + one_hot.sum(dim=dims)
        dice = (2 * intersection + self.smooth) / (denominator + self.smooth)
        if not self.include_background:
            dice = dice[:, 1:]
        return 1 - dice.mean()


class DiceCrossEntropyLoss(nn.Module):
    def __init__(self, include_background: bool = True) -> None:
        super().__init__()
        self.dice = SoftDiceLoss(include_background=include_background)
        self._last_components: dict[str, torch.Tensor] = {}

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        dice, ce = self.dice(logits, targets), F.cross_entropy(logits, targets.long())
        self._last_components = {"loss_dice": dice.detach(), "loss_ce": ce.detach()}
        return dice + ce

    def get_last_components(self) -> dict[str, torch.Tensor]:
        return dict(self._last_components)


class SoftmaxFocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0) -> None:
        super().__init__()
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, targets.long(), reduction="none")
        return ((1 - torch.exp(-ce)) ** self.gamma * ce).mean()
