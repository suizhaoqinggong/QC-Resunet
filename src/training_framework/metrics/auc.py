"""AUC metrics: AUROC and AUPRC."""

from typing import Any

import torch

from training_framework.contracts.types import MetricResults, Predictions, Targets
from training_framework.metrics._helpers import prepare_metric_inputs


class AUCMetric:
    """Multilabel AUROC and step-integrated AUPRC (average precision)."""

    name = "auc"
    higher_is_better = True

    def __init__(self, num_classes: int, **kwargs: Any) -> None:
        self.num_classes = num_classes
        self._probs: list[torch.Tensor] = []
        self._targets: list[torch.Tensor] = []

    def update(self, preds: Predictions, targets: Targets) -> None:
        preds, targets = prepare_metric_inputs(preds, targets)
        valid_mask = ~torch.isnan(preds).any(dim=1)
        if not valid_mask.any():
            return
        preds = preds[valid_mask]
        targets = targets[valid_mask]
        self._probs.append(preds)
        self._targets.append(targets)

    def compute(self) -> MetricResults:
        if not self._probs:
            return {"auroc": 0.0, "auprc": 0.0}

        probs = torch.cat(self._probs, dim=0)
        targets = torch.cat(self._targets, dim=0)

        auroc_vals = []
        auprc_vals = []
        for i in range(self.num_classes):
            y_true = targets[:, i]
            y_score = probs[:, i]
            if y_true.sum() == 0 or y_true.sum() == len(y_true):
                continue  # skip classes with all same label
            order = torch.argsort(y_score, descending=True)
            y_sorted = y_true[order]
            scores_sorted = y_score[order]
            # Equal scores form a single threshold, so ties are order-independent.
            boundaries = torch.nonzero(scores_sorted[1:] != scores_sorted[:-1]).flatten()
            indices = torch.cat([boundaries, torch.tensor([len(y_sorted) - 1])])
            tps = torch.cumsum(y_sorted, dim=0)[indices]
            fps = (indices + 1).to(tps.dtype) - tps
            recall = tps / tps[-1]
            zero = torch.zeros(1, dtype=recall.dtype)
            tpr = torch.cat([zero, recall])
            fpr = torch.cat([zero, fps / fps[-1]])
            auroc = torch.trapz(tpr, fpr).item()
            precision = tps / (tps + fps)
            previous_recall = torch.cat([zero, recall[:-1]])
            auprc = ((recall - previous_recall) * precision).sum().item()
            auroc_vals.append(auroc)
            auprc_vals.append(auprc)

        return {
            "auroc": sum(auroc_vals) / len(auroc_vals) if auroc_vals else 0.0,
            "auprc": sum(auprc_vals) / len(auprc_vals) if auprc_vals else 0.0,
        }

    def reset(self) -> None:
        self._probs.clear()
        self._targets.clear()
