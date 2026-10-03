"""Generic multiclass semantic segmentation task."""

from __future__ import annotations

import torch
from torch import nn

from training_framework.contracts.task import ProblemType
from training_framework.contracts.types import Batch, LossValue, ModelOutput, Predictions, Targets
from training_framework.losses.segmentation import DiceCrossEntropyLoss, SoftDiceLoss, SoftmaxFocalLoss


class SegmentationTask:
    """Integer labels [B, ...] and logits [B, C, ...], with C >= 2."""

    def __init__(
        self,
        num_classes: int,
        threshold: float = 0.5,
        loss: str = "ce",
        loss_gamma: float = 2.0,
        include_background: bool = True,
    ) -> None:
        if num_classes < 2:
            raise ValueError("Multiclass segmentation requires at least two classes")
        self.num_classes, self.threshold, self.loss_name = num_classes, threshold, loss
        self.loss_fn: nn.Module
        if loss == "ce":
            self.loss_fn = nn.CrossEntropyLoss()
        elif loss == "dice":
            self.loss_fn = SoftDiceLoss(include_background=include_background)
        elif loss in {"dice_ce", "dc_and_ce"}:
            self.loss_fn = DiceCrossEntropyLoss(include_background=include_background)
        elif loss == "focal":
            self.loss_fn = SoftmaxFocalLoss(gamma=loss_gamma)
        else:
            raise ValueError(f"Unknown segmentation loss '{loss}'; choose ce, dice, dice_ce or focal")

    def compute_loss(self, outputs: ModelOutput, batch: Batch) -> LossValue:
        if outputs.shape[1] != self.num_classes:
            raise ValueError(f"Expected {self.num_classes} logits channels, got {outputs.shape[1]}")
        return self.loss_fn(outputs, self.extract_targets(batch))

    def extract_targets(self, batch: Batch) -> Targets:
        return batch["label"].long()

    def postprocess_outputs(self, outputs: ModelOutput) -> Predictions:
        return torch.softmax(outputs, dim=1)

    def infer_problem_type(self) -> ProblemType:
        return "multiclass"
