"""Synthetic volumes for CPU-safe segmentation examples."""

from __future__ import annotations

import torch
from torch.utils.data import Dataset

from training_framework.contracts.types import Batch, Sample


class SyntheticVolumes(Dataset):
    def __init__(self, count: int, num_classes: int, num_modalities: int, volume_size: tuple[int, int, int], seed: int):
        rng = torch.Generator().manual_seed(seed)
        self.signals = torch.randn(count, num_modalities, *volume_size, generator=rng)
        self.labels = torch.randint(num_classes, (count, *volume_size), generator=rng)
        self.ids = [f"synthetic-{seed}-{index}" for index in range(count)]

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, index: int) -> Sample:
        return Sample(signal=self.signals[index], label=self.labels[index], id=self.ids[index], meta={})


class DummySegmentationAdapter:
    def __init__(
        self,
        num_samples: int = 20,
        num_classes: int = 4,
        num_modalities: int = 4,
        volume_size: tuple[int, int, int] = (8, 8, 8),
        val_ratio: float = 0.2,
        test_ratio: float = 0.2,
        seed: int = 42,
    ) -> None:
        if num_samples < 1 or val_ratio < 0 or test_ratio < 0 or val_ratio + test_ratio >= 1:
            raise ValueError("Require positive sample count and non-negative split ratios summing to less than 1")
        if len(volume_size) != 3 or any(size < 1 for size in volume_size):
            raise ValueError("volume_size must contain three positive dimensions")
        self.num_samples, self.num_classes, self.num_modalities = num_samples, num_classes, num_modalities
        self.volume_size = (volume_size[0], volume_size[1], volume_size[2])
        self.val_ratio, self.test_ratio, self.seed = val_ratio, test_ratio, seed
        self._datasets: tuple[SyntheticVolumes, ...] | None = None

    def prepare(self) -> None:
        n_val, n_test = int(self.num_samples * self.val_ratio), int(self.num_samples * self.test_ratio)
        counts = (self.num_samples - n_val - n_test, n_val, n_test)
        self._datasets = tuple(
            SyntheticVolumes(count, self.num_classes, self.num_modalities, self.volume_size, self.seed + index)
            for index, count in enumerate(counts, start=1)
        )

    def get_splits(self) -> tuple[Dataset, Dataset, Dataset]:
        if self._datasets is None:
            raise RuntimeError("Call prepare() before get_splits()")
        return self._datasets[0], self._datasets[1], self._datasets[2]

    def collate_fn(self, batch: list[Sample]) -> Batch:
        return Batch(
            signal=torch.stack([sample["signal"] for sample in batch]),
            label=torch.stack([sample["label"] for sample in batch]),
            id=[sample["id"] for sample in batch],
            meta=[sample["meta"] for sample in batch],
        )
