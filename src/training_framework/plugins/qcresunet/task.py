"""Subject regression and independent binary error-map supervision."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from training_framework.contracts.task import ProblemType
from training_framework.contracts.types import Batch
from training_framework.plugins.qcresunet.representation import pack, unpack


class QCLoss(nn.Module):
    def __init__(self, num_classes: int, balance: float = 1.0, smooth: float = 1e-5) -> None:
        super().__init__()
        if balance < 0 or smooth <= 0:
            raise ValueError("balance must be nonnegative and smooth must be positive")
        self.num_classes, self.balance, self.smooth = num_classes, balance, smooth
        self.components: dict[str, torch.Tensor] = {}

    def forward(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if outputs.shape != targets.shape:
            raise ValueError("QC outputs and targets must have the same packed shape")
        quality, logits = unpack(outputs.float(), self.num_classes)
        scores, errors = unpack(targets.float(), self.num_classes)
        mae = (quality - scores).abs().sum(dim=1).mean()
        probabilities = logits.sigmoid()
        intersection = (probabilities * errors).sum(dim=(0, 2))
        denominator = (probabilities + errors).sum(dim=(0, 2))
        # Batch Dice, averaged across tissue classes. Negative Dice as in paper.
        dice = -((2 * intersection + self.smooth) / (denominator + self.smooth)).mean()
        bce = F.binary_cross_entropy_with_logits(logits, errors)
        self.components = {"loss_mae": mae.detach(), "loss_dice": dice.detach(), "loss_bce": bce.detach()}
        return mae + self.balance * (dice + bce)

    def get_last_components(self) -> dict[str, torch.Tensor]:
        return self.components


class QCTask:
    def __init__(self, num_classes: int, threshold: float = 0.5, balance: float = 1.0, smooth: float = 1e-5) -> None:
        self.num_classes = num_classes
        self.loss_fn = QCLoss(num_classes, balance, smooth)

    def compute_loss(self, outputs: torch.Tensor, batch: Batch) -> torch.Tensor:
        return self.loss_fn(outputs, batch["label"])

    def extract_targets(self, batch: Batch) -> torch.Tensor:
        return batch["label"]

    def postprocess_outputs(self, outputs: torch.Tensor) -> torch.Tensor:
        quality, logits = unpack(outputs, self.num_classes)
        return pack(quality, logits.sigmoid())

    def infer_problem_type(self) -> ProblemType:
        return "multilabel"
