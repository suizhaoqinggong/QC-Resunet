import pytest

pytest.importorskip("nibabel")
pytest.importorskip("scipy")
import torch


def test_brats_split_keeps_related_suffixes_together_and_allows_empty_test(tmp_path):
    """BraTS split should keep -000/-001 variants of the same subject in one split."""
    from training_framework.data_adapters.brats_adapter import BraTSDataAdapter, brats_subject_id

    case_ids = [
        "BraTS-GLI-00001-000",
        "BraTS-GLI-00001-001",
        "BraTS-GLI-00002-000",
        "BraTS-GLI-00002-001",
        "BraTS-GLI-00003-000",
        "BraTS-GLI-00003-001",
        "BraTS-GLI-00004-000",
        "BraTS-GLI-00004-001",
    ]
    for case_id in case_ids:
        (tmp_path / case_id).mkdir()

    adapter = BraTSDataAdapter(
        root_dir=str(tmp_path),
        val_ratio=0.5,
        test_ratio=0.0,
        seed=0,
        use_preprocessed=True,
        preprocessed_dir=str(tmp_path),
    )
    adapter.prepare()
    train_ds, val_ds, test_ds = adapter.get_splits()

    train_subjects = {brats_subject_id(case_id) for case_id in train_ds.case_ids}
    val_subjects = {brats_subject_id(case_id) for case_id in val_ds.case_ids}

    assert len(test_ds) == 0
    assert train_subjects.isdisjoint(val_subjects)
    assert train_subjects | val_subjects == {
        "BraTS-GLI-00001",
        "BraTS-GLI-00002",
        "BraTS-GLI-00003",
        "BraTS-GLI-00004",
    }


def test_brats_split_uses_fixed_validation_ids_file(tmp_path):
    from training_framework.data_adapters.brats_adapter import BraTSDataAdapter

    case_ids = [
        "BraTS20_Training_001",
        "BraTS20_Training_002",
        "BraTS20_Training_003",
        "BraTS20_Training_004",
    ]
    for case_id in case_ids:
        (tmp_path / case_id).mkdir()

    val_ids_file = tmp_path / "val_ids.txt"
    val_ids_file.write_text("BraTS20_Training_002\nBraTS20_Training_004\n")

    adapter = BraTSDataAdapter(
        root_dir=str(tmp_path),
        use_preprocessed=True,
        preprocessed_dir=str(tmp_path),
        val_ratio=0.0,
        test_ratio=0.0,
        val_ids_file=str(val_ids_file),
    )
    adapter.prepare()
    train_ds, val_ds, test_ds = adapter.get_splits()

    assert train_ds.case_ids == ["BraTS20_Training_001", "BraTS20_Training_003"]
    assert val_ds.case_ids == ["BraTS20_Training_002", "BraTS20_Training_004"]
    assert test_ds.case_ids == []


def test_brats_region_dice_uses_wt_tc_et_regions():
    """BraTS Dice should report WT, TC, and ET composite regions."""
    from training_framework.metrics.brats import BraTSRegionDiceMetric

    targets = torch.tensor([[[[0, 1], [2, 3]], [[0, 1], [2, 3]]]])
    preds = torch.nn.functional.one_hot(targets, num_classes=4).permute(0, 4, 1, 2, 3).float()

    metric = BraTSRegionDiceMetric(num_classes=4)
    metric.update(preds, targets)
    results = metric.compute()

    assert results["wt"] == 1.0
    assert results["tc"] == 1.0
    assert results["et"] == 1.0
    assert results["mean"] == 1.0


def test_brats_region_hausdorff_is_zero_for_exact_match():
    """BraTS HD should be zero when WT, TC, and ET masks match exactly."""
    from training_framework.metrics.brats import BraTSRegionHausdorffMetric

    targets = torch.tensor([[[[0, 1], [2, 3]], [[0, 1], [2, 3]]]])
    preds = torch.nn.functional.one_hot(targets, num_classes=4).permute(0, 4, 1, 2, 3).float()

    metric = BraTSRegionHausdorffMetric(num_classes=4)
    metric.update(preds, targets)
    results = metric.compute()

    assert results["wt"] == 0.0
    assert results["tc"] == 0.0
    assert results["et"] == 0.0
    assert results["mean"] == 0.0


@pytest.mark.parametrize("data_format", ["npy", "npz", "raw"])
def test_synthetic_brats_can_train_and_test_without_external_data(tmp_path, data_format):
    from pathlib import Path

    import nibabel as nib
    import numpy as np

    from training_framework.cli.main import build_trainer
    from training_framework.registry.defaults import create_default_registries
    from training_framework.registry.factories import load_merged_experiment_config

    data_root = tmp_path / "data"
    data_root.mkdir()
    rng = np.random.default_rng(42)
    for index in range(6):
        case_id = f"case_{index}"
        case_dir = data_root / case_id
        case_dir.mkdir()
        labels = rng.integers(0, 4, size=(4, 4, 4), dtype=np.int16)
        arrays = {name: rng.normal(size=(4, 4, 4)).astype(np.float32) for name in ("t1", "t1ce", "t2", "flair")}
        if data_format == "npy":
            for name, array in arrays.items():
                np.save(case_dir / f"{name}.npy", array)
            np.save(case_dir / "seg.npy", labels)
        elif data_format == "npz":
            np.savez(data_root / f"{case_id}.npz", **arrays, seg=labels)
        else:
            for source_name, raw_name in zip(arrays, ("t1n", "t1c", "t2w", "t2f")):
                nib.save(nib.Nifti1Image(arrays[source_name], np.eye(4)), case_dir / f"{case_id}-{raw_name}.nii.gz")
            raw_labels = np.where(labels == 3, 4, labels).astype(np.int16)
            nib.save(nib.Nifti1Image(raw_labels, np.eye(4)), case_dir / f"{case_id}-seg.nii.gz")
    project = Path(__file__).resolve().parents[1]
    config = load_merged_experiment_config(
        project / "configs/experiment.brats.example.toml", project / "configs/model.tiny_seg_net.toml"
    )
    config["data"].update(
        root_dir=str(data_root),
        preprocessed_dir=str(data_root),
        data_format=data_format,
        use_preprocessed=data_format != "raw",
        augment_train=False,
        normalize=False,
        crop_size=[4, 4, 4],
    )
    config["trainer"].update(device="cpu", epochs=1)
    config["output"]["root_dir"] = str(tmp_path / "runs")
    trainer, layout = build_trainer(config, create_default_registries())
    try:
        results = trainer.fit()
    finally:
        trainer.logger.end_run()
    assert 0 <= results["brats_dice_mean"] <= 1
    assert trainer.checkpoint_manager.best_path.is_file()
    assert (layout.run_dir / "splits/test_ids.txt").read_text().strip()
