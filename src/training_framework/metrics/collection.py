"""Metric collection for grouping and managing metrics per split."""

from __future__ import annotations

from typing import Any

from training_framework.contracts.metric import Metric
from training_framework.contracts.types import MetricResults, Predictions, Targets
from training_framework.registry.registry import Registry


class MetricCollection:
    """A collection of metrics with a common prefix."""

    def __init__(self, metrics: list[Metric], prefix: str = "") -> None:
        self.metrics = metrics
        self.prefix = prefix

    def update(self, preds: Predictions, targets: Targets) -> None:
        for metric in self.metrics:
            metric.update(preds, targets)

    def compute(self) -> MetricResults:
        results: MetricResults = {}
        for metric in self.metrics:
            for k, v in metric.compute().items():
                key = f"{self.prefix}{metric.name}_{k}" if self.prefix else f"{metric.name}_{k}"
                results[key] = v
        return results

    def reset(self) -> None:
        for metric in self.metrics:
            metric.reset()


class SplitMetricCollections:
    """Metric collections for train/val/test splits."""

    def __init__(
        self,
        train: MetricCollection,
        val: MetricCollection,
        test: MetricCollection,
    ) -> None:
        self.train = train
        self.val = val
        self.test = test


def build_split_metric_collections(
    metric_params: dict[str, Any],
    metric_registry: Registry[Metric],
    common_kwargs: dict[str, Any],
) -> SplitMetricCollections:
    """Build train/val/test metric collections from config."""

    def build_collection() -> MetricCollection:
        metrics: list[Metric] = []
        for metric_name, params in metric_params.items():
            if not isinstance(params, dict):
                raise ValueError(f"Metric '{metric_name}' parameters must be a table")
            kwargs = {**common_kwargs, **params}
            metrics.append(metric_registry.create(metric_name, **kwargs))
        return MetricCollection(metrics)

    return SplitMetricCollections(
        train=build_collection(),
        val=build_collection(),
        test=build_collection(),
    )
