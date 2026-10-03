import math
from pathlib import Path

import pytest
import torch

from training_framework.cli.main import build_trainer
from training_framework.contracts.types import Batch
from training_framework.losses.segmentation import SoftDiceLoss, SoftmaxFocalLoss
from training_framework.registry.defaults import create_default_registries
from training_framework.registry.factories import load_merged_experiment_config
from training_framework.tasks.segmentation import SegmentationTask

PROJECT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("shape", [(1, 2, 3, 3), (1, 2, 2, 3, 3)])
def test_dice_is_a_real_dice_objective_with_gradients(shape):
    logits = torch.zeros(shape, requires_grad=True)
    targets = torch.arange(logits[:, 0].numel()).reshape(logits[:, 0].shape) % 2
    batch = Batch(signal=torch.empty(0), label=targets, id=[], meta=[])
    task = SegmentationTask(num_classes=2, loss="dice")
    assert isinstance(task.loss_fn, SoftDiceLoss)
    loss = task.compute_loss(logits, batch)
    assert not torch.isclose(loss, torch.tensor(math.log(2)))
    loss.backward()
    assert torch.isfinite(logits.grad).all()
    assert logits.grad.abs().sum() > 0


def test_focal_gamma_zero_matches_cross_entropy():
    logits = torch.randn(2, 3, 4, 4, requires_grad=True)
    labels = torch.randint(3, (2, 4, 4))
    assert torch.allclose(SoftmaxFocalLoss(gamma=0)(logits, labels), torch.nn.functional.cross_entropy(logits, labels))


def test_unknown_task_parameters_are_rejected():
    with pytest.raises(TypeError):
        SegmentationTask(num_classes=2, loss_lamda=1)


@pytest.mark.parametrize("kind,model_name", [("classification", "simple_net"), ("segmentation", "tiny_seg_net")])
def test_fit_evaluates_best_weights_and_writes_snapshots(tmp_path, kind, model_name):
    config = load_merged_experiment_config(
        PROJECT / f"configs/experiment.{kind}.toml", PROJECT / f"configs/model.{model_name}.toml"
    )
    config["output"]["root_dir"] = str(tmp_path)
    trainer, layout = build_trainer(config, create_default_registries())
    try:
        results = trainer.fit()
    finally:
        trainer.logger.end_run()
    assert "loss" in results and math.isfinite(results["loss"])
    assert trainer.checkpoint_manager.best_path.exists()
    assert (layout.run_dir / "experiment.snapshot.toml").exists()
    assert (layout.run_dir / "metrics.jsonl").read_text().count("epoch") == 2
    train_ids = set((layout.run_dir / "splits/train_ids.txt").read_text().splitlines())
    val_ids = set((layout.run_dir / "splits/val_ids.txt").read_text().splitlines())
    assert train_ids and val_ids and train_ids.isdisjoint(val_ids)
    if kind == "segmentation":
        assert "loss_ce" in results and "loss_dice" in results
        assert "dice_mean" in results
