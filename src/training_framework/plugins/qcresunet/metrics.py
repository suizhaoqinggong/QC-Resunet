"""Case-level MAE/Pearson and tissue-specific SEM Dice."""

from __future__ import annotations

from typing import Any

import torch

from training_framework.contracts.types import MetricResults
from training_framework.plugins.qcresunet.representation import unpack


class QCMetrics:
    name = "qc"
    higher_is_better = False

    def __init__(
        self, num_classes: int, threshold: float = 0.5, class_names: list[str] | None = None, **kwargs: Any
    ) -> None:
        self.num_classes, self.threshold = num_classes, threshold
        self.class_names = class_names or [str(i) for i in range(num_classes)]
        self.reset()

    def reset(self) -> None:
        self.scores: list[torch.Tensor] = []
        self.targets: list[torch.Tensor] = []
        self.dice: list[torch.Tensor] = []

    def update(self, preds: torch.Tensor, targets: torch.Tensor) -> None:
        quality, errors = unpack(preds.detach().float().cpu(), self.num_classes)
        scores, truth = unpack(targets.detach().float().cpu(), self.num_classes)
        predicted, actual = errors >= self.threshold, truth.bool()
        numerator = 2 * (predicted & actual).sum(dim=2)
        denominator = predicted.sum(dim=2) + actual.sum(dim=2)
        dice = torch.where(denominator > 0, numerator / denominator.clamp_min(1), 1.0)
        self.scores.append(quality)
        self.targets.append(scores)
        self.dice.append(dice)

    def compute(self) -> MetricResults:
        if not self.scores:
            return {}
        quality, scores, dice = torch.cat(self.scores), torch.cat(self.targets), torch.cat(self.dice)
        results: MetricResults = {}
        for index, name in enumerate(("dsc", "nsd")):
            absolute_errors = (quality[:, index] - scores[:, index]).abs()
            results[f"{name}_mae"] = absolute_errors.mean().item()
            results[f"{name}_mae_std"] = absolute_errors.std(unbiased=False).item()
            x = quality[:, index].double() - quality[:, index].double().mean()
            y = scores[:, index].double() - scores[:, index].double().mean()
            denominator = torch.linalg.vector_norm(x) * torch.linalg.vector_norm(y)
            results[f"{name}_pearson"] = (x @ y / denominator).item() if denominator > 0 else float("nan")
        results["mae"] = (quality - scores).abs().mean().item()
        results["sem_dice"] = dice.mean().item()
        results["sem_dice_std"] = dice.mean(1).std(unbiased=False).item()
        for index, name in enumerate(self.class_names):
            results[f"sem_dice_{name}"] = dice[:, index].mean().item()
        return results
