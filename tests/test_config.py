from training_framework.registry.defaults import create_default_registries
from training_framework.registry.factories import build_component_bundle, load_merged_experiment_config


def test_experiment_seed_controls_data_adapter_when_data_seed_is_missing():
    """The experiment seed should drive data splitting unless [data].seed overrides it."""
    config = {
        "experiment": {
            "name": "seed-test",
            "seed": 123,
            "class_names": ["negative", "positive"],
        },
        "data": {
            "adapter": "dummy",
            "num_samples": 20,
            "input_dim": 8,
            "batch_size": 2,
            "num_workers": 0,
        },
        "model": {
            "name": "simple_net",
            "input_dim": 8,
        },
        "task": {
            "name": "classification",
        },
        "optimizer": {
            "name": "adam",
            "lr": 1e-3,
        },
        "metrics": {
            "params": {},
        },
    }

    bundle = build_component_bundle(config, create_default_registries())

    assert bundle.data_adapter.seed == 123


def test_data_seed_controls_data_adapter_when_experiment_seed_is_missing():
    """A fixed [data].seed should keep splits stable while training seed is unset."""
    config = {
        "experiment": {
            "name": "seed-test",
            "class_names": ["negative", "positive"],
        },
        "data": {
            "adapter": "dummy",
            "seed": 37,
            "num_samples": 20,
            "input_dim": 8,
            "batch_size": 2,
            "num_workers": 0,
        },
        "model": {
            "name": "simple_net",
            "input_dim": 8,
        },
        "task": {
            "name": "classification",
        },
        "optimizer": {
            "name": "adam",
            "lr": 1e-3,
        },
        "metrics": {
            "params": {},
        },
    }

    bundle = build_component_bundle(config, create_default_registries())

    assert bundle.data_adapter.seed == 37


def test_config_loading():
    """Test loading merged config."""
    import tempfile
    from pathlib import Path

    exp_toml = Path(tempfile.gettempdir()) / "test_exp.toml"
    model_toml = Path(tempfile.gettempdir()) / "test_model.toml"

    exp_toml.write_text("""
[experiment]
name = "test"
class_names = ["A", "B"]

[data]
adapter = "dummy"
batch_size = 16
""")
    model_toml.write_text("""
[model]
name = "simple_net"
input_dim = 64
""")

    config = load_merged_experiment_config(exp_toml, model_toml)
    assert config["experiment"]["name"] == "test"
    assert config["model"]["name"] == "simple_net"
    assert config["data"]["batch_size"] == 16
    assert config["model"]["input_dim"] == 64

    exp_toml.unlink()
    model_toml.unlink()


def test_dataloader_performance_options_are_configurable():
    """DataLoader throughput options from config are applied."""
    from training_framework.registry.defaults import create_default_registries
    from training_framework.registry.factories import build_component_bundle

    config = {
        "experiment": {
            "name": "test",
            "class_names": ["A", "B"],
        },
        "data": {
            "adapter": "dummy",
            "num_samples": 16,
            "input_dim": 8,
            "batch_size": 4,
            "num_workers": 2,
            "pin_memory": True,
            "drop_last": True,
            "persistent_workers": True,
            "prefetch_factor": 4,
        },
        "model": {
            "name": "simple_net",
            "input_dim": 8,
        },
        "task": {
            "name": "classification",
        },
        "metrics": {
            "params": {
                "accuracy": {},
            },
        },
    }

    bundle = build_component_bundle(config, create_default_registries())

    assert bundle.train_loader.pin_memory is True
    assert bundle.train_loader.drop_last is True
    assert bundle.train_loader.persistent_workers is True
    assert bundle.train_loader.prefetch_factor == 4
    assert bundle.val_loader.pin_memory is True
    assert bundle.val_loader.drop_last is False
    assert bundle.val_loader.persistent_workers is True
    assert bundle.test_loader.prefetch_factor == 4


def test_snapshot_split_ids_writes_dataset_case_ids(tmp_path):
    """Run directories should contain exact split membership for reproducibility."""
    from training_framework.core.run_layout import snapshot_split_ids

    class DatasetWithIds:
        def __init__(self, case_ids):
            self.case_ids = case_ids

    class AdapterWithSplits:
        def get_splits(self):
            return (
                DatasetWithIds(["train-a", "train-b"]),
                DatasetWithIds(["val-a"]),
                DatasetWithIds([]),
            )

    snapshot_split_ids(AdapterWithSplits(), tmp_path)

    assert (tmp_path / "splits" / "train_ids.txt").read_text().splitlines() == ["train-a", "train-b"]
    assert (tmp_path / "splits" / "val_ids.txt").read_text().splitlines() == ["val-a"]
    assert (tmp_path / "splits" / "test_ids.txt").read_text().splitlines() == []
