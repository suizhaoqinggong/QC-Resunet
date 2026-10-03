"""Built-in registries with lazy, optional BraTS support."""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any

from training_framework.contracts.data import DataAdapter
from training_framework.contracts.metric import Metric
from training_framework.contracts.model import ModelAdapter
from training_framework.contracts.task import Task

from .registry import Registry


@dataclass(frozen=True)
class RegistryBundle:
    data_adapters: Registry[DataAdapter]
    models: Registry[ModelAdapter]
    tasks: Registry[Task]
    metrics: Registry[Metric]


def _create_brats_adapter(**kwargs: Any) -> DataAdapter:
    from training_framework.data_adapters.brats_adapter import BraTSDataAdapter

    return BraTSDataAdapter(**kwargs)


def _create_brats_2d_adapter(**kwargs: Any) -> DataAdapter:
    from training_framework.data_adapters.brats_adapter import BraTS2DDataAdapter

    return BraTS2DDataAdapter(**kwargs)


def _create_brats_dice(**kwargs: Any) -> Metric:
    from training_framework.metrics.brats import BraTSRegionDiceMetric

    return BraTSRegionDiceMetric(**kwargs)


def _create_brats_hd(**kwargs: Any) -> Metric:
    from training_framework.metrics.brats import BraTSRegionHausdorffMetric

    return BraTSRegionHausdorffMetric(**kwargs)


def create_default_registries(plugin_modules: list[str] | None = None) -> RegistryBundle:
    """Create fresh registries, then call register(registries) on application plugins."""
    from training_framework.data_adapters.dummy_adapter import DummyDataAdapter
    from training_framework.data_adapters.dummy_segmentation import DummySegmentationAdapter
    from training_framework.metrics.accuracy import AccuracyMetric
    from training_framework.metrics.auc import AUCMetric
    from training_framework.metrics.dice import DiceMetric
    from training_framework.metrics.iou import IoUMetric
    from training_framework.metrics.precision_recall_f1 import PrecisionRecallF1Metric
    from training_framework.models.simple_net import SimpleNet
    from training_framework.models.tiny_seg_net import TinySegNet
    from training_framework.tasks.classification import ClassificationTask
    from training_framework.tasks.segmentation import SegmentationTask

    registries = RegistryBundle(Registry("data_adapter"), Registry("model"), Registry("task"), Registry("metric"))
    registries.data_adapters.register("dummy", DummyDataAdapter)
    registries.data_adapters.register("dummy_segmentation", DummySegmentationAdapter)
    registries.data_adapters.register("brats", _create_brats_adapter)
    registries.data_adapters.register("brats_2d", _create_brats_2d_adapter)
    registries.models.register("simple_net", SimpleNet)
    registries.models.register("tiny_seg_net", TinySegNet)
    registries.tasks.register("classification", ClassificationTask)
    registries.tasks.register("segmentation", SegmentationTask)
    registries.metrics.register("accuracy", AccuracyMetric)
    registries.metrics.register("auc", AUCMetric)
    registries.metrics.register("precision_recall_f1", PrecisionRecallF1Metric)
    registries.metrics.register("dice", DiceMetric)
    registries.metrics.register("iou", IoUMetric)
    registries.metrics.register("brats_dice", _create_brats_dice)
    registries.metrics.register("brats_hd", _create_brats_hd)
    for module_name in plugin_modules or []:
        plugin = importlib.import_module(module_name)
        register = getattr(plugin, "register", None)
        if not callable(register):
            raise ValueError(f"Plugin '{module_name}' must export a callable register(registries)")
        register(registries)
    return registries
