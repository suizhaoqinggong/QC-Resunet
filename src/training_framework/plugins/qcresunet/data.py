"""Manifest-driven, subject-isolated QC samples and CPU smoke data."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from training_framework.contracts.types import Batch, Sample
from training_framework.plugins.qcresunet.quality import quality_targets
from training_framework.plugins.qcresunet.representation import (
    BRATS_REGIONS,
    pack,
    region_masks,
    validate_mask,
    validate_regions,
)


def collate_samples(samples: list[Sample]) -> Batch:
    if len({tuple(sample["signal"].shape) for sample in samples}) != 1:
        raise ValueError("QC requires common whole-volume shapes; preprocess before batching")
    return Batch(
        signal=torch.stack([sample["signal"] for sample in samples]),
        label=torch.stack([sample["label"] for sample in samples]),
        id=[sample["id"] for sample in samples],
        meta=[sample["meta"] for sample in samples],
    )


def load_arrays(
    path: Path, modalities: int, volume_path: Path | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with np.load(volume_path or path, allow_pickle=False) as content:
        image = content["image"].astype(np.float32)
        query = np.load(path, allow_pickle=False) if volume_path is not None else content["query"]
        reference = content["reference"]
        spacing = content["spacing"].astype(np.float64)
    if (
        image.ndim != 4
        or image.shape[0] != modalities
        or query.shape != image.shape[1:]
        or reference.shape != query.shape
    ):
        raise ValueError(f"{path}: expected image [M,D,H,W], query/reference [D,H,W]")
    if not np.isfinite(image).all():
        raise ValueError(f"{path}: non-finite MRI intensities")
    if spacing.shape != (3,) or not np.isfinite(spacing).all() or min(spacing) <= 0:
        raise ValueError(f"{path}: three positive finite voxel spacings are required")
    return image, query, reference, spacing


class QCDataset(Dataset):
    def __init__(
        self,
        records: list[dict[str, Any]],
        root: Path,
        modalities: int,
        regions: tuple[tuple[int, ...], ...],
        tolerance_mm: float,
        augment: bool = False,
    ) -> None:
        self.records, self.root = records, root
        self.modalities, self.regions, self.tolerance_mm, self.augment = modalities, regions, tolerance_mm, augment
        self.ids = [record["id"] for record in records]

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Sample:
        record = self.records[index]
        volume = self.root / record["volume"] if "volume" in record else None
        image, query, reference, spacing = load_arrays(self.root / record["path"], self.modalities, volume)
        if self.augment:
            from training_framework.plugins.qcresunet.augmentation import augment_qc

            image, query, reference = augment_qc(image, query, reference)
            scores, errors = quality_targets(query, reference, spacing.tolist(), self.tolerance_mm, self.regions)
        else:
            predicted, actual = torch.from_numpy(query.copy()), torch.from_numpy(reference.copy())
            validate_mask(predicted, self.regions)
            validate_mask(actual, self.regions)
            errors = (
                (region_masks(predicted[None], self.regions) ^ region_masks(actual[None], self.regions))[0]
                .numpy()
                .astype(np.float32)
            )
            scores = np.asarray([record["dsc"], record["nsd"]], dtype=np.float32)
        signal = torch.from_numpy(np.concatenate((image, query[None].astype(np.float32))))
        label = pack(torch.from_numpy(scores)[None], torch.from_numpy(errors)[None])[0]
        return Sample(
            signal=signal,
            label=label,
            id=record["id"],
            meta={"subject_id": record["subject_id"], "shape": query.shape, "spacing": spacing.tolist()},
        )


class BalancedQCDataset(Dataset):
    """Equal counts in each DSC bin, without replacement within each epoch.

    The shared epoch tensor also updates persistent DataLoader workers.
    Validation/test retain the deterministic epoch-zero sample subset.
    """

    def __init__(self, dataset: QCDataset, seed: int, bins: int = 10) -> None:
        self.dataset, self.seed = dataset, seed
        if bins < 1:
            raise ValueError("bins must be positive")
        self.groups: list[list[int]] = [[] for _ in range(bins)]
        for index, record in enumerate(dataset.records):
            score = float(record["dsc"])
            if not np.isfinite(score) or not 0 <= score <= 1:
                raise ValueError("Manifest DSC must be finite and in [0,1]")
            self.groups[min(int(score * bins), bins - 1)].append(index)
        self.count = min(map(len, self.groups))
        if not self.count:
            raise ValueError("DSC balancing requires samples in every bin; supply diverse queries or disable balancing")
        self.epoch = torch.zeros((), dtype=torch.int64).share_memory_()
        self._cached_epoch = -1
        self._indices: list[int] = []

    def set_epoch(self, epoch: int) -> None:
        self.epoch.fill_(epoch)

    def _selection(self) -> list[int]:
        epoch = int(self.epoch)
        if epoch != self._cached_epoch:
            rng = np.random.default_rng(self.seed + epoch)
            self._indices = [
                int(index) for group in self.groups for index in rng.choice(group, self.count, replace=False)
            ]
            self._cached_epoch = epoch
        return self._indices

    @property
    def ids(self) -> list[str]:
        return [self.dataset.ids[index] for index in self._selection()]

    def __len__(self) -> int:
        return len(self.groups) * self.count

    def __getitem__(self, index: int) -> Sample:
        return self.dataset[self._selection()[index]]


class QCDataAdapter:
    collate_fn = staticmethod(collate_samples)

    def __init__(
        self,
        manifest: str,
        seed: int = 42,
        num_classes: int = 3,
        modalities: int = 4,
        regions: Sequence[Sequence[int]] = BRATS_REGIONS,
        tolerance_mm: float = 1.0,
        augment_train: bool = False,
        balance_train: bool = True,
        balance_eval: bool = True,
        bins: int = 10,
        test_only: bool = False,
    ) -> None:
        self.manifest = Path(manifest)
        self.seed, self.modalities = seed, modalities
        self.regions = validate_regions(regions, num_classes)
        self.tolerance_mm, self.augment_train = tolerance_mm, augment_train
        self.balance_train, self.balance_eval, self.bins = balance_train, balance_eval, bins
        self.test_only = test_only
        self.datasets: tuple[Dataset, Dataset, Dataset] | None = None

    def prepare(self) -> None:
        content = json.loads(self.manifest.read_text())
        records = content["samples"]
        if (
            content.get("regions") != [list(region) for region in self.regions]
            or content.get("tolerance_mm") != self.tolerance_mm
        ):
            raise ValueError("Manifest regions/NSD tolerance must match the experiment")
        subjects: dict[str, str] = {}
        ids: set[str] = set()
        paths: set[Path] = set()
        volume_subjects: dict[Path, str] = {}
        splits: dict[str, list[dict[str, Any]]] = {name: [] for name in ("train", "val", "test")}
        root = self.manifest.parent
        for record in records:
            subject, split, sample_id = record["subject_id"], record["split"], record["id"]
            if (
                not all(isinstance(value, str) and value for value in (subject, split, sample_id))
                or split not in splits
            ):
                raise ValueError("Nonempty id, subject_id and train/val/test split are required")
            if sample_id in ids or (subject in subjects and subjects[subject] != split):
                raise ValueError("Duplicate sample ID or subject leakage across QC splits")
            path = Path(record["path"])
            resolved = (root / path).resolve()
            if path.is_absolute() or not resolved.is_relative_to(root.resolve()) or resolved in paths:
                raise ValueError("Each sample must have a unique relative path inside the manifest directory")
            if not resolved.is_file():
                raise FileNotFoundError(resolved)
            paths.add(resolved)
            if "volume" in record:
                volume = Path(record["volume"])
                resolved_volume = (root / volume).resolve()
                if volume.is_absolute() or not resolved_volume.is_relative_to(root.resolve()):
                    raise ValueError("Shared volume must use a relative path inside the manifest directory")
                if resolved_volume in volume_subjects and volume_subjects[resolved_volume] != subject:
                    raise ValueError("A shared volume cannot belong to different subjects")
                if not resolved_volume.is_file():
                    raise FileNotFoundError(resolved_volume)
                volume_subjects[resolved_volume] = subject
            for metric_name in ("dsc", "nsd"):
                score = float(record[metric_name])
                if not np.isfinite(score) or not 0 <= score <= 1:
                    raise ValueError(f"Manifest {metric_name} must be finite and in [0,1]")
            ids.add(sample_id)
            subjects[subject] = split
            if split in ("train", "val") and record.get("method", "").lower() == "deepmedic":
                raise ValueError("Paper protocol reserves DeepMedic queries for testing")
            splits[split].append(record)
        datasets: list[Dataset] = []
        for split, entries in splits.items():
            if not entries and not (self.test_only and split != "test"):
                raise ValueError(f"QC split {split} is empty")
            dataset = QCDataset(
                entries, root, self.modalities, self.regions, self.tolerance_mm, split == "train" and self.augment_train
            )
            balance = self.balance_train if split == "train" else self.balance_eval
            if entries and balance:
                datasets.append(BalancedQCDataset(dataset, self.seed, self.bins))
            else:
                datasets.append(dataset)
        self.datasets = (datasets[0], datasets[1], datasets[2])

    def get_splits(self) -> tuple[Dataset, Dataset, Dataset]:
        if self.datasets is None:
            raise RuntimeError("Call prepare before get_splits")
        return self.datasets


class SyntheticQCDataset(Dataset):
    def __init__(
        self, count: int, seed: int, shape: Sequence[int], modalities: int, regions: tuple[tuple[int, ...], ...]
    ) -> None:
        self.count, self.seed, self.shape, self.modalities, self.regions = (
            count,
            seed,
            tuple(shape),
            modalities,
            regions,
        )
        self.ids = [f"synthetic-{seed}-{index}" for index in range(count)]

    def __len__(self) -> int:
        return self.count

    def __getitem__(self, index: int) -> Sample:
        rng = np.random.default_rng(self.seed + index)
        reference = np.zeros(self.shape, dtype=np.int64)
        labels = sorted({label for region in self.regions for label in region})
        for offset, label in enumerate(labels):
            reference[2 + offset * 2 : 4 + offset * 2, 3:8, 3:8] = label
        # Perfect and empty queries give exact DSC/NSD without optional scipy.
        query = reference.copy() if index % 2 == 0 else np.zeros_like(reference)
        image = rng.normal(0, 0.1, (self.modalities, *self.shape)).astype(np.float32) + (reference > 0)[None]
        scores, errors = quality_targets(query, reference, (1, 1, 1), 1.0, self.regions)
        return Sample(
            signal=torch.from_numpy(np.concatenate((image, query[None].astype(np.float32)))),
            label=pack(torch.from_numpy(scores)[None], torch.from_numpy(errors)[None])[0],
            id=self.ids[index],
            meta={"synthetic": True, "shape": self.shape},
        )


class SyntheticQCDataAdapter:
    collate_fn = staticmethod(collate_samples)

    def __init__(
        self,
        seed: int = 42,
        num_classes: int = 3,
        modalities: int = 4,
        shape: Sequence[int] = (32, 32, 64),
        split_sizes: Sequence[int] = (4, 2, 2),
        regions: Sequence[Sequence[int]] = BRATS_REGIONS,
    ) -> None:
        self.regions = validate_regions(regions, num_classes)
        if len(shape) != 3 or min(shape) < 12 or len(split_sizes) != 3 or min(split_sizes) < 1:
            raise ValueError("Three spatial dimensions >=12 and three nonempty split sizes are required")
        self.datasets = tuple(
            SyntheticQCDataset(count, seed + split * 10000, shape, modalities, self.regions)
            for split, count in enumerate(split_sizes)
        )

    def prepare(self) -> None:
        pass

    def get_splits(self) -> tuple[Dataset, Dataset, Dataset]:
        return self.datasets[0], self.datasets[1], self.datasets[2]
