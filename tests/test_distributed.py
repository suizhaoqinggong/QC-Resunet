import torch

from training_framework.registry.defaults import create_default_registries
from training_framework.registry.factories import build_component_bundle


def test_distributed_context_detects_torchrun_environment(monkeypatch):
    """torchrun environment variables should enable DDP context detection."""
    from training_framework.core.distributed import infer_distributed_context

    monkeypatch.setenv("RANK", "1")
    monkeypatch.setenv("LOCAL_RANK", "1")
    monkeypatch.setenv("WORLD_SIZE", "2")

    context = infer_distributed_context()

    assert context.enabled
    assert context.rank == 1
    assert context.local_rank == 1
    assert context.world_size == 2
    assert not context.is_main_process


def test_component_bundle_uses_distributed_sampler_for_training_only():
    """DDP should shard training data while leaving validation/test loaders complete."""
    from torch.utils.data.distributed import DistributedSampler

    from training_framework.core.distributed import DistributedContext

    config = {
        "experiment": {"name": "ddp-test", "seed": 7, "class_names": ["a", "b"]},
        "data": {
            "adapter": "dummy",
            "num_samples": 20,
            "input_dim": 8,
            "batch_size": 2,
            "num_workers": 0,
        },
        "model": {"name": "simple_net", "input_dim": 8},
        "task": {"name": "classification"},
        "optimizer": {"name": "adam", "lr": 1e-3},
        "metrics": {"params": {}},
    }
    context = DistributedContext(enabled=True, rank=1, local_rank=1, world_size=2, device=torch.device("cpu"))

    bundle = build_component_bundle(config, create_default_registries(), distributed_context=context)

    assert isinstance(bundle.train_loader.sampler, DistributedSampler)
    assert bundle.train_loader.sampler.rank == 1
    assert bundle.train_loader.sampler.num_replicas == 2
    assert not isinstance(bundle.val_loader.sampler, DistributedSampler)
    assert not isinstance(bundle.test_loader.sampler, DistributedSampler)


def test_missing_experiment_seed_uses_broadcast_sampler_seed_in_ddp(monkeypatch):
    """DDP sampler shuffling should stay consistent across ranks without a training seed."""
    from torch.utils.data.distributed import DistributedSampler

    import training_framework.registry.factories as factories
    from training_framework.core.distributed import DistributedContext

    monkeypatch.setattr(factories.random, "randrange", lambda upper: 9876)
    monkeypatch.setattr(factories, "broadcast_object", lambda value, context: value)

    config = {
        "experiment": {"name": "ddp-test", "class_names": ["a", "b"]},
        "data": {
            "adapter": "dummy",
            "seed": 37,
            "num_samples": 20,
            "input_dim": 8,
            "batch_size": 2,
            "num_workers": 0,
        },
        "model": {"name": "simple_net", "input_dim": 8},
        "task": {"name": "classification"},
        "optimizer": {"name": "adam", "lr": 1e-3},
        "metrics": {"params": {}},
    }
    context = DistributedContext(enabled=True, rank=0, local_rank=0, world_size=2, device=torch.device("cpu"))

    bundle = build_component_bundle(config, create_default_registries(), distributed_context=context)

    assert bundle.data_adapter.seed == 37
    assert isinstance(bundle.train_loader.sampler, DistributedSampler)
    assert bundle.train_loader.sampler.seed == 9876


def test_average_metric_dict_returns_averaged_metrics(monkeypatch):
    """DDP metric averaging should return a metrics dict, not mutate and drop it."""
    import training_framework.core.distributed as distributed
    from training_framework.core.distributed import DistributedContext, average_metric_dict

    def fake_all_reduce(values, op=None):
        values.mul_(2)

    monkeypatch.setattr(distributed.dist, "all_reduce", fake_all_reduce)
    context = DistributedContext(enabled=True, rank=0, local_rank=0, world_size=2, device=torch.device("cpu"))

    averaged = average_metric_dict({"loss": 0.5, "accuracy": 0.25}, context)

    assert averaged == {"loss": 0.5, "accuracy": 0.25}
