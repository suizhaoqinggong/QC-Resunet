"""Synthetic evidence for architecture, targets and the complete QC workflow."""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from training_framework.cli.main import build_trainer
from training_framework.contracts.types import Batch
from training_framework.plugins.qcresunet.augmentation import augment_qc, warp_masks
from training_framework.plugins.qcresunet.data import (
    BalancedQCDataset,
    QCDataAdapter,
    QCDataset,
    SyntheticQCDataAdapter,
)
from training_framework.plugins.qcresunet.metrics import QCMetrics
from training_framework.plugins.qcresunet.model import QCResUNet
from training_framework.plugins.qcresunet.predict import QCEnsemble, ensemble_predict
from training_framework.plugins.qcresunet.prepare import assign_folds, prepare_manifest, preprocess
from training_framework.plugins.qcresunet.quality import quality_targets
from training_framework.plugins.qcresunet.representation import BRATS_REGIONS, pack, unpack
from training_framework.plugins.qcresunet.task import QCLoss, QCTask
from training_framework.registry.defaults import create_default_registries
from training_framework.registry.factories import load_merged_experiment_config

ROOT = Path(__file__).resolve().parents[1]
PLUGIN = "training_framework.plugins.qcresunet.components"


def tiny_model(**kwargs):
    return QCResUNet(base_channels=2, blocks=(1, 1, 1, 1), dropout=0, **kwargs)


def test_plugin_is_opt_in_and_paper_defaults():
    assert "qcresunet" not in create_default_registries().models
    assert "qcresunet" in create_default_registries([PLUGIN]).models
    # Inspect defaults without allocating the full ~66M-parameter network.
    import inspect

    defaults = inspect.signature(QCResUNet).parameters
    assert defaults["blocks"].default == (3, 4, 6, 3)
    assert defaults["base_channels"].default == 64
    assert defaults["dropout"].default == 0.3


def test_model_ignores_reference_and_decoder_starts_at_block3():
    adapter = SyntheticQCDataAdapter(split_sizes=(2, 2, 2))
    dataset = adapter.get_splits()[0]
    batch = adapter.collate_fn([dataset[0], dataset[1]])
    model = tiny_model().eval()
    original = model(batch)
    altered = Batch(signal=batch["signal"], label=torch.randn_like(batch["label"]), id=batch["id"], meta=[{}, {}])
    assert torch.equal(original, model(altered))
    quality, errors = unpack(original, 3, (32, 32, 64))
    assert quality.shape == (2, 2)
    assert errors.shape == (2, 3, 32, 32, 64)
    # SEM loss must reach Block 3 and ECA, and bypass regression-only Block 4.
    errors.square().mean().backward()
    assert next(model.stages[2].parameters()).grad is not None
    assert next(model.stages[3].parameters()).grad.abs().sum() == 0
    assert model.attention.conv.weight.grad.abs().sum() > 0
    model.zero_grad(set_to_none=True)
    model(batch)[:, :2].sum().backward()
    assert next(model.stages[3].parameters()).grad is not None
    assert model.quality_head.weight.grad.abs().sum() > 0


def test_anisotropic_cardiac_and_odd_sizes():
    regions = ((1,), (2,), (3,))
    model = tiny_model(
        modalities=1,
        regions=regions,
        stem_stride=(1, 2, 2),
        downsample_strides=((1, 2, 2), (1, 2, 2), (1, 2, 2), (2, 2, 2)),
    ).eval()
    signal = torch.randn(1, 2, 16, 33, 35)
    signal[:, -1] = torch.randint(0, 4, (1, 16, 33, 35)).float()
    scores, errors = model.forward_heads(signal)
    assert scores.shape == (1, 2)
    assert errors.shape == (1, 3, 16, 33, 35)


def test_targets_region_xor_empty_conventions_and_nsd_spacing():
    mask = np.zeros((8, 8, 8), dtype=np.int64)
    mask[2:4, 2:4, 2:4] = 3
    quality, errors = quality_targets(mask, mask, (1, 1, 1), 1)
    np.testing.assert_array_equal(quality, [1, 1])
    assert not errors.any()
    quality, errors = quality_targets(mask, np.zeros_like(mask), (1, 1, 1), 1)
    np.testing.assert_array_equal(quality, [0, 0])
    assert errors[:, 2:4, 2:4, 2:4].all()
    np.testing.assert_array_equal(quality_targets(np.zeros_like(mask), np.zeros_like(mask), (1, 1, 1), 1)[0], [1, 1])
    shifted = np.roll(mask, 2, axis=0)
    isotropic = quality_targets(shifted, mask, (1, 1, 1), 1)[0]
    anisotropic = quality_targets(shifted, mask, (4, 1, 1), 1)[0]
    assert isotropic[0] == anisotropic[0] == 0
    assert anisotropic[1] < isotropic[1]
    _, errors = quality_targets(np.ones_like(mask), np.full_like(mask, 2), (1, 1, 1), 1)
    assert not errors[0].any()  # NCR -> edema keeps WT but changes TC.
    assert errors[1].all()
    assert not errors[2].any()
    with pytest.raises(ValueError, match="remap"):
        quality_targets(np.full_like(mask, 4), mask, (1, 1, 1), 1)


def test_loss_equation_and_gradient_for_every_head():
    scores = torch.tensor([[0.2, 0.7], [0.8, 0.1]])
    errors = torch.tensor([[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], [[0.0, 1.0], [1.0, 0.0], [0.0, 0.0]]])
    targets = pack(scores, errors)
    outputs = torch.zeros_like(targets, requires_grad=True)
    objective = QCLoss(3, balance=2)
    value = objective(outputs, targets)
    # MAE sums two subject errors, unlike torch L1Loss's default factor 1/2.
    expected_mae = scores.sum(1).mean()
    probability = torch.full_like(errors, 0.5)
    expected_dice = -(
        (2 * (probability * errors).sum((0, 2)) + 1e-5) / ((probability + errors).sum((0, 2)) + 1e-5)
    ).mean()
    expected_bce = torch.tensor(2.0).log()
    torch.testing.assert_close(value, expected_mae + 2 * (expected_dice + expected_bce))
    value.backward()
    assert outputs.grad[:, :2].abs().sum() > 0
    assert outputs.grad[:, 2:].abs().sum() > 0
    assert set(objective.get_last_components()) == {"loss_mae", "loss_dice", "loss_bce"}


def test_metrics_case_average_and_correlations():
    metric = QCMetrics(3, class_names=["WT", "TC", "ET"])
    quality = torch.tensor([[0.2, 0.3], [0.8, 0.9]])
    truth = torch.tensor([[0.1, 0.2], [0.7, 0.8]])
    actual = torch.tensor([[[1.0, 0.0], [1.0, 0.0], [0.0, 0.0]], [[0.0, 1.0], [0.0, 1.0], [1.0, 1.0]]])
    predicted = actual.clone()
    predicted[0, 0] = 0
    metric.update(pack(quality[:1], predicted[:1]), pack(truth[:1], actual[:1]))
    metric.update(pack(quality[1:], predicted[1:]), pack(truth[1:], actual[1:]))
    results = metric.compute()
    assert results["dsc_mae"] == pytest.approx(0.1)
    assert results["nsd_mae"] == pytest.approx(0.1)
    assert results["dsc_pearson"] == pytest.approx(1)
    assert results["sem_dice_WT"] == 0.5
    assert results["sem_dice_ET"] == 1
    assert results["sem_dice"] == pytest.approx(5 / 6)
    metric.reset()
    assert metric.compute() == {}


class FixedHeads(nn.Module):
    def __init__(self, quality, probabilities):
        super().__init__()
        self.output = pack(torch.tensor([quality]), torch.logit(torch.tensor([[probabilities]], dtype=torch.float32)))

    def forward(self, batch):
        return self.output


def test_ensemble_uses_majority_not_mean_probability():
    models = [FixedHeads([0.2, 0.4], [0.51]), FixedHeads([0.4, 0.6], [0.51]), FixedHeads([0.6, 0.8], [0.01])]
    batch = Batch(signal=torch.zeros(1, 1, 1, 1, 1), label=torch.zeros(1), id=["case"], meta=[{}])
    scores, probabilities, vote = ensemble_predict(models, batch, 1)
    torch.testing.assert_close(scores, torch.tensor([[0.4, 0.6]]))
    assert probabilities.item() < 0.5
    assert vote.item()
    _, predictions = unpack(QCTask(1).postprocess_outputs(QCEnsemble(models, 1)(batch)), 1)
    assert predictions.item() > 0.99
    with pytest.raises(ValueError, match="odd"):
        ensemble_predict(models[:2], batch, 1)


def make_manifest(tmp_path):
    records = []
    image = np.zeros((4, 12, 12, 12), dtype=np.float32)
    mask = np.zeros((12, 12, 12), dtype=np.int64)
    mask[2:5, 2:5, 2:5] = 3
    for split in ("train", "val", "test"):
        path = f"{split}.npz"
        np.savez(tmp_path / path, image=image, query=mask, reference=mask, spacing=np.ones(3))
        records.append({"id": split, "subject_id": split, "split": split, "path": path, "dsc": 1.0, "nsd": 1.0})
    manifest = tmp_path / "manifest.json"
    content = {"regions": [list(r) for r in BRATS_REGIONS], "tolerance_mm": 1.0, "samples": records}
    manifest.write_text(json.dumps(content))
    return manifest, content


def test_manifest_rejects_subject_leakage_paths_tolerance_and_deepmedic(tmp_path):
    manifest, content = make_manifest(tmp_path)
    adapter = QCDataAdapter(str(manifest), balance_train=False, balance_eval=False)
    adapter.prepare()
    sample = adapter.get_splits()[0][0]
    assert sample["signal"].shape == (5, 12, 12, 12)
    scores, errors = unpack(sample["label"][None], 3)
    assert scores.tolist() == [[1.0, 1.0]]
    assert not errors.any()
    for field, value, message in [
        ("subject_id", "train", "leakage"),
        ("path", "../train.npz", "relative"),
        ("method", "deepmedic", "DeepMedic"),
    ]:
        bad = json.loads(json.dumps(content))
        bad["samples"][1][field] = value
        manifest.write_text(json.dumps(bad))
        with pytest.raises(ValueError, match=message):
            QCDataAdapter(str(manifest), balance_train=False, balance_eval=False).prepare()
    manifest.write_text(json.dumps(content))
    with pytest.raises(ValueError, match="tolerance"):
        QCDataAdapter(str(manifest), tolerance_mm=2).prepare()


def test_balancing_and_epoch_selection(tmp_path):
    records = [
        {"id": f"{bin_index}-{index}", "dsc": (bin_index + 0.5) / 10}
        for bin_index in range(10)
        for index in range(4 if bin_index else 2)
    ]
    dataset = QCDataset(records, tmp_path, 4, BRATS_REGIONS, 1)
    balanced = BalancedQCDataset(dataset, 42)
    assert len(balanced) == 20
    first = balanced._selection().copy()
    assert len(set(first)) == 20
    assert np.bincount([int(records[index]["dsc"] * 10) for index in first]).tolist() == [2] * 10
    assert first == BalancedQCDataset(dataset, 42)._selection()
    balanced.set_epoch(1)
    assert first != balanced._selection()
    with pytest.raises(ValueError, match="every bin"):
        BalancedQCDataset(QCDataset(records[2:], tmp_path, 4, BRATS_REGIONS, 1), 42)


def test_fold_assignment_has_fixed_test_and_unique_validation():
    folds = assign_folds([f"subject-{i}" for i in range(13)], 42, 4)
    tests = [{subject for subject, split in fold.items() if split == "test"} for fold in folds]
    assert tests[0] == tests[1] == tests[2]
    for subject in folds[0]:
        assignments = [fold[subject] for fold in folds]
        if subject not in tests[0]:
            assert assignments.count("val") == 1
            assert assignments.count("train") == 2


def test_preparation_normalization_padding_and_end_to_end(tmp_path):
    image = np.zeros((4, 12, 12, 12), dtype=np.float32)
    image[:, 1:10, 1:10, 1:10] = np.arange(9**3).reshape(9, 9, 9) + 1
    reference = np.zeros((12, 12, 12), dtype=np.int16)
    reference[3:6, 3:6, 3:6] = 4
    query = reference.copy()
    query[11, 11, 11] = 4  # Bad segmentation outside MRI must survive cropping.
    normalized, truth, queries = preprocess(image, reference, [query], (12, 12, 12))
    assert normalized.shape == image.shape
    assert normalized.mean() == pytest.approx(0, abs=1e-6)
    assert np.count_nonzero(queries[0]) == np.count_nonzero(query)
    assert np.count_nonzero(truth) == np.count_nonzero(reference)
    with pytest.raises(ValueError, match="exceeds"):
        preprocess(image, reference, [query], (8, 8, 8))
    cases = []
    for split in ("train", "val", "test"):
        np.savez(tmp_path / f"{split}.npz", image=image, reference=reference, spacing=np.ones(3))
        np.save(tmp_path / f"{split}.npy", reference)
        cases.append(
            {
                "subject_id": split,
                "split": split,
                "volume": f"{split}.npz",
                "queries": [{"path": f"{split}.npy", "method": "nnunet"}],
            }
        )
    source = tmp_path / "source.json"
    source.write_text(json.dumps({"subjects": cases}))
    manifest = prepare_manifest(source, tmp_path / "prepared", 1, shape=(12, 12, 12), seggen=0)
    adapter = QCDataAdapter(str(manifest), balance_train=False, balance_eval=False)
    adapter.prepare()
    sample = adapter.get_splits()[0][0]
    assert sample["signal"][-1].max() == 3
    assert sample["label"][:2].tolist() == [1, 1]
    with pytest.raises(FileExistsError):
        prepare_manifest(source, tmp_path / "prepared", 1, shape=(12, 12, 12), seggen=0)


def test_seggen_and_training_augmentation_preserve_grid_and_labels():
    reference = np.zeros((12, 12, 12), dtype=np.int16)
    reference[2:8, 2:8, 2:8] = 3
    first, _ = warp_masks([reference], np.random.default_rng(42), seggen=True)
    second, _ = warp_masks([reference], np.random.default_rng(42), seggen=True)
    np.testing.assert_array_equal(first[0], second[0])
    assert set(np.unique(first[0])) <= {0, 3}
    image = np.repeat(reference[None], 4, 0).astype(np.float32)
    np.random.seed(42)
    image, query, truth = augment_qc(image, reference, reference)
    assert image.shape == (4, 12, 12, 12)
    assert np.isfinite(image).all()
    np.testing.assert_array_equal(query, truth)
    assert set(np.unique(query)) <= {0, 3}


def test_external_test_only_manifest(tmp_path):
    manifest, content = make_manifest(tmp_path)
    content["samples"] = [content["samples"][2]]
    manifest.write_text(json.dumps(content))
    with pytest.raises(ValueError, match="train is empty"):
        QCDataAdapter(str(manifest), balance_train=False, balance_eval=False).prepare()
    adapter = QCDataAdapter(str(manifest), balance_train=False, balance_eval=False, test_only=True)
    adapter.prepare()
    assert [len(dataset) for dataset in adapter.get_splits()] == [0, 0, 1]


def test_plugin_imports_and_runs_without_optional_dependencies(tmp_path):
    code = """
import importlib.abc
import sys
class BlockOptional(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'scipy', 'nibabel', 'surface_distance'}:
            raise ImportError('Optional dependency was eagerly imported: ' + fullname)
sys.meta_path.insert(0, BlockOptional())
import torch
torch.set_num_threads(1)
from training_framework.registry.defaults import create_default_registries
registries = create_default_registries(['training_framework.plugins.qcresunet.components'])
adapter = registries.data_adapters.create('qc_synthetic', split_sizes=(1,1,1))
batch = adapter.collate_fn([adapter.get_splits()[0][0]])
model = registries.models.create('qcresunet', base_channels=2, blocks=(1,1,1,1))
task = registries.tasks.create('segmentation_qc', num_classes=3)
assert torch.isfinite(task.compute_loss(model(batch), batch))
"""
    result = subprocess.run(
        [sys.executable, "-I", "-c", code], cwd=tmp_path, capture_output=True, text=True, timeout=60
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_fit_checkpoint_reload_loss_components_and_exponential_floor(tmp_path):
    config = load_merged_experiment_config(
        ROOT / "configs/experiment.qcresunet.smoke.toml", ROOT / "configs/model.qcresunet.smoke.toml"
    )
    config["output"]["root_dir"] = str(tmp_path)
    config["optimizer"]["lr"] = 0.0000011
    config["trainer"]["epochs"] = 2
    trainer, layout = build_trainer(config, create_default_registries([PLUGIN]))
    try:
        metrics = trainer.fit()
        assert np.isfinite(metrics["qc_dsc_mae"])
        assert 0 <= metrics["qc_sem_dice"] <= 1
        assert all(f"loss_{component}" in metrics for component in ("mae", "dice", "bce"))
        assert trainer.optimizer.param_groups[0]["lr"] == pytest.approx(0.000001)
        batch = next(iter(trainer.test_loader))
        original = trainer.model(batch)
        model = tiny_model().eval()
        model.load_state_dict(trainer.checkpoint_manager.load()["state_dict"])
        torch.testing.assert_close(original, model(batch))
        assert (layout.run_dir / "experiment.snapshot.toml").is_file()
        assert (layout.run_dir / "splits/test_ids.txt").read_text().count("synthetic") == 2
    finally:
        trainer.logger.end_run()
