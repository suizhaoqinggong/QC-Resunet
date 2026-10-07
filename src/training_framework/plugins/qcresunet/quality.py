"""Supervision from query/reference masks; optional area-weighted surface Dice."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch

from training_framework.plugins.qcresunet.representation import BRATS_REGIONS, region_masks, validate_mask


def quality_targets(
    query: np.ndarray,
    reference: np.ndarray,
    spacing: Sequence[float],
    tolerance_mm: float,
    regions: Sequence[Sequence[int]] = BRATS_REGIONS,
) -> tuple[np.ndarray, np.ndarray]:
    """Macro-average region DSC/NSD and region-wise XOR error maps.

    Both-empty regions score 1; exactly-one-empty regions score 0.
    Spacing follows the array's D,H,W axes. NSD weights surfel areas.
    """
    if query.ndim != 3 or reference.shape != query.shape:
        raise ValueError("Query and reference must have matching [D,H,W] shapes")
    if len(spacing) != 3 or not np.isfinite(spacing).all() or min(spacing) <= 0:
        raise ValueError("Three positive finite voxel spacings are required")
    if not np.isfinite(tolerance_mm) or tolerance_mm <= 0:
        raise ValueError("NSD tolerance_mm must be finite and positive")
    query_tensor, truth_tensor = torch.from_numpy(query.copy()), torch.from_numpy(reference.copy())
    validate_mask(query_tensor, regions)
    validate_mask(truth_tensor, regions)
    predicted = region_masks(query_tensor[None], regions)[0].numpy()
    actual = region_masks(truth_tensor[None], regions)[0].numpy()
    dsc, nsd = [], []
    for prediction, truth in zip(predicted, actual):
        total = int(prediction.sum() + truth.sum())
        dsc.append(2 * np.count_nonzero(prediction & truth) / total if total else 1.0)
        if not prediction.any() or not truth.any():
            nsd.append(1.0 if not total else 0.0)
        elif np.array_equal(prediction, truth):
            nsd.append(1.0)
        else:
            # No scipy/nibabel/surface_distance imports on plugin registration.
            try:
                from surface_distance import metrics
            except ImportError as error:
                raise ImportError("NSD target generation requires uv sync --extra brats --extra dev") from error
            surfaces = metrics.compute_surface_distances(truth, prediction, tuple(spacing))
            nsd.append(float(metrics.compute_surface_dice_at_tolerance(surfaces, tolerance_mm)))
    return np.asarray([np.mean(dsc), np.mean(nsd)], dtype=np.float32), (predicted ^ actual).astype(np.float32)
