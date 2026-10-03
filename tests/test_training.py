import torch

from training_framework.contracts.types import Batch
from training_framework.core.checkpoint import CheckpointManager
from training_framework.core.trainer import Trainer, TrainerSettings
from training_framework.data_adapters.dummy_adapter import DummyDataAdapter
from training_framework.logging.structured_logger import NullExperimentLogger
from training_framework.metrics.collection import MetricCollection
from training_framework.models.simple_net import SimpleNet
from training_framework.tasks.classification import ClassificationTask


def test_model_forward():
    """Test SimpleNet forward pass."""
    model = SimpleNet(num_classes=5, input_dim=128)
    batch = Batch(
        signal=torch.randn(4, 128),
        label=torch.randint(0, 2, (4, 5)).float(),
        id=["s1", "s2", "s3", "s4"],
        meta=[{}, {}, {}, {}],
    )
    outputs = model(batch)
    assert outputs.shape == (4, 5)


def test_data_adapter():
    """Test DummyDataAdapter produces valid data."""
    adapter = DummyDataAdapter(num_samples=100, num_classes=3, input_dim=64)
    adapter.prepare()
    train_ds, val_ds, test_ds = adapter.get_splits()
    assert len(train_ds) > 0
    assert len(val_ds) > 0
    assert len(test_ds) > 0

    sample = train_ds[0]
    assert sample["signal"].shape == (64,)
    assert sample["label"].shape == (3,)

    batch = adapter.collate_fn([train_ds[0], train_ds[1]])
    assert batch["signal"].shape == (2, 64)
    assert batch["label"].shape == (2, 3)


def test_classification_task():
    """Test ClassificationTask computes loss correctly."""
    task = ClassificationTask(num_classes=5, loss="bce")
    logits = torch.randn(4, 5)
    labels = torch.randint(0, 2, (4, 5)).float()
    batch = Batch(
        signal=torch.randn(4, 128),
        label=labels,
        id=["s1", "s2", "s3", "s4"],
        meta=[{}, {}, {}, {}],
    )
    loss = task.compute_loss(logits, batch)
    assert loss.item() >= 0
    assert task.infer_problem_type() == "multilabel"


def test_focal_loss():
    """Test FocalLoss runs without errors."""
    from training_framework.losses.focal import FocalLoss

    loss_fn = FocalLoss(gamma=2.0)
    logits = torch.randn(4, 5)
    targets = torch.randint(0, 2, (4, 5)).float()
    loss = loss_fn(logits, targets)
    assert loss.item() >= 0


def test_tensorboard_compat_installs_pkg_resources_shim():
    """TensorBoard compat entry should not require setuptools pkg_resources."""
    import sys

    from training_framework.cli.tensorboard_compat import install_pkg_resources_shim

    previous = sys.modules.pop("pkg_resources", None)
    try:
        install_pkg_resources_shim(force=True)
        shim = sys.modules["pkg_resources"]

        assert list(shim.iter_entry_points("tensorboard_plugins")) == []
        assert shim.parse_version("2.0") > shim.parse_version("1.0")
    finally:
        if previous is None:
            sys.modules.pop("pkg_resources", None)
        else:
            sys.modules["pkg_resources"] = previous


def test_trainer_seed_defaults_to_unset():
    """Training should not set global RNG state when [experiment].seed is omitted."""
    assert TrainerSettings().seed is None


def test_checkpoint_manager(tmp_path):
    """Test checkpoint save and load."""
    ckpt_dir = tmp_path / "checkpoints"
    manager = CheckpointManager(ckpt_dir, monitor="loss", mode="min")

    state = {"weight": torch.tensor([1.0])}
    metrics = {"loss": 0.5}
    manager.save(state, epoch=1, metrics=metrics, is_best=True)

    assert (ckpt_dir / "last.pt").exists()
    assert (ckpt_dir / "best.pt").exists()

    payload = manager.load()
    assert payload["epoch"] == 1
    assert payload["metrics"]["loss"] == 0.5


def test_checkpoint_manager_tracks_explicit_monitor_value(tmp_path):
    """Best checkpoint tracking should work when the saved metrics use an unprefixed key."""
    ckpt_dir = tmp_path / "checkpoints"
    manager = CheckpointManager(ckpt_dir, monitor="val_score", mode="max")

    state = {"weight": torch.tensor([1.0])}
    first_is_best = manager.is_improved(0.5)
    manager.save(state, epoch=1, metrics={"score": 0.5}, is_best=first_is_best, monitor_value=0.5)

    second_is_best = manager.is_improved(0.4)
    manager.save(state, epoch=2, metrics={"score": 0.4}, is_best=second_is_best, monitor_value=0.4)

    assert manager.load()["epoch"] == 1
    assert manager.load(ckpt_dir / "last.pt")["epoch"] == 2


def test_trainer_runs_one_epoch(tmp_path):
    """Test that Trainer can run for one epoch without crashing."""
    model = SimpleNet(num_classes=3, input_dim=64)
    task = ClassificationTask(num_classes=3, loss="bce")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    adapter = DummyDataAdapter(num_samples=32, num_classes=3, input_dim=64)
    adapter.prepare()
    train_ds, val_ds, test_ds = adapter.get_splits()

    from torch.utils.data import DataLoader

    train_loader = DataLoader(train_ds, batch_size=8, collate_fn=adapter.collate_fn)
    val_loader = DataLoader(val_ds, batch_size=8, collate_fn=adapter.collate_fn)
    test_loader = DataLoader(test_ds, batch_size=8, collate_fn=adapter.collate_fn)

    from training_framework.metrics.accuracy import AccuracyMetric

    metrics = [AccuracyMetric(threshold=0.5)]
    train_mc = MetricCollection(metrics)
    val_mc = MetricCollection([AccuracyMetric(threshold=0.5)])
    test_mc = MetricCollection([AccuracyMetric(threshold=0.5)])

    settings = TrainerSettings(epochs=1, patience=5, device="cpu")
    ckpt = CheckpointManager(tmp_path / "ckpt", monitor="loss", mode="min")

    trainer = Trainer(
        model=model,
        task=task,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        train_metrics=train_mc,
        val_metrics=val_mc,
        test_metrics=test_mc,
        settings=settings,
        checkpoint_manager=ckpt,
        logger=NullExperimentLogger(),
        run_dir=tmp_path,
    )

    metrics = trainer.train()
    assert "accuracy_accuracy" in metrics
