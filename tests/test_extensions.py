import sys
import types

import pytest

from training_framework.metrics.collection import build_split_metric_collections
from training_framework.models.simple_net import SimpleNet
from training_framework.registry.defaults import create_default_registries
from training_framework.registry.registry import DuplicateRegistrationError, UnknownRegistrationError


def test_registry_reports_duplicates_and_unknown_names():
    registry = create_default_registries().models
    with pytest.raises(DuplicateRegistrationError):
        registry.register("simple_net", SimpleNet)
    with pytest.raises(UnknownRegistrationError, match="Available"):
        registry.create("missing")


def test_plugin_registers_project_model_without_mutating_defaults(monkeypatch):
    plugin = types.ModuleType("application_components")

    def register(registries):
        registries.models.register("my_classifier", SimpleNet)

    plugin.register = register
    monkeypatch.setitem(sys.modules, plugin.__name__, plugin)
    registries = create_default_registries([plugin.__name__])
    assert isinstance(registries.models.create("my_classifier", num_classes=2), SimpleNet)
    assert "my_classifier" not in create_default_registries().models


def test_invalid_plugin_reports_contract_error(monkeypatch):
    plugin = types.ModuleType("invalid_plugin")
    monkeypatch.setitem(sys.modules, plugin.__name__, plugin)
    with pytest.raises(ValueError, match="register"):
        create_default_registries([plugin.__name__])


def test_metric_parameters_and_instances_are_independent_for_each_split():
    collections = build_split_metric_collections(
        {"precision_recall_f1": {"average": "micro", "threshold": 0.9}},
        create_default_registries().metrics,
        {"num_classes": 3, "threshold": 0.5},
    )
    metrics = [collection.metrics[0] for collection in (collections.train, collections.val, collections.test)]
    assert all(metric.average == "micro" and metric.threshold == 0.9 for metric in metrics)
    assert len({id(metric) for metric in metrics}) == 3
