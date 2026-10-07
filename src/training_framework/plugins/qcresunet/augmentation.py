"""Joint QC augmentation and SegGen, with nearest-neighbor label sampling."""

from __future__ import annotations

import math

import numpy as np


def warp_masks(
    masks: list[np.ndarray],
    rng: np.random.Generator,
    image: np.ndarray | None = None,
    seggen: bool = False,
) -> tuple[list[np.ndarray], np.ndarray | None]:
    """Each SegGen transform independently has probability 0.5 (paper Table 2).

    A smooth dense elastic field is an explicit reproduction choice; the
    paper specifies displacement but not control-grid size or smoothing.
    """
    from scipy.ndimage import gaussian_filter, map_coordinates

    shape = masks[0].shape
    matrix = np.eye(3)
    if rng.random() < 0.5:
        for axis, angle in enumerate(rng.uniform(-15, 15, 3) * math.pi / 180):
            a, b = [i for i in range(3) if i != axis]
            rotation = np.eye(3)
            rotation[a, a] = rotation[b, b] = math.cos(angle)
            rotation[a, b], rotation[b, a] = -math.sin(angle), math.sin(angle)
            matrix = rotation @ matrix
    if rng.random() < 0.5:
        matrix *= rng.uniform(0.85, 1.25)
    center = (np.asarray(shape) - 1) / 2
    grid = np.indices(shape, dtype=np.float32).reshape(3, -1)
    coordinates = np.linalg.inv(matrix) @ (grid - center[:, None]) + center[:, None]
    if seggen and rng.random() < 0.5:
        coordinates -= rng.uniform(-20, 20, 3)[:, None]
    coordinates = coordinates.reshape(3, *shape)
    if seggen and rng.random() < 0.5:
        for axis in range(3):
            field = gaussian_filter(rng.normal(size=shape).astype(np.float32), sigma=max(1, min(shape) / 16))
            field /= max(float(np.abs(field).max()), 1e-6)
            coordinates[axis] += field * rng.uniform(0, 20)
    warped = [map_coordinates(mask, coordinates, order=0, mode="constant", cval=0, prefilter=False) for mask in masks]
    warped_image = None
    if image is not None:
        warped_image = np.stack(
            [map_coordinates(channel, coordinates, order=1, mode="constant", cval=0) for channel in image]
        )
    return warped, warped_image


def augment_qc(
    image: np.ndarray, query: np.ndarray, reference: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(int(np.random.randint(0, 2**31)))
    masks, warped_image = warp_masks([query, reference], rng, image=image)
    assert warped_image is not None
    query, reference = masks
    image = warped_image
    for axis in range(3):
        if rng.random() < 0.5:
            image = np.flip(image, axis + 1)
            query, reference = np.flip(query, axis), np.flip(reference, axis)
    if rng.random() < 0.5:
        image = image + rng.normal(0, rng.uniform(0, 0.1), image.shape).astype(np.float32)
    if rng.random() < 0.5:
        original_mean = image.mean(axis=(1, 2, 3), keepdims=True)
        original_std = image.std(axis=(1, 2, 3), keepdims=True)
        low = image.min(axis=(1, 2, 3), keepdims=True)
        scale = image.max(axis=(1, 2, 3), keepdims=True) - low
        image = ((image - low) / np.maximum(scale, 1e-6)) ** rng.uniform(0.7, 1.5)
        image = (image - image.mean(axis=(1, 2, 3), keepdims=True)) / np.maximum(
            image.std(axis=(1, 2, 3), keepdims=True), 1e-6
        )
        image = image * original_std + original_mean
    return image.astype(np.float32), query.copy(), reference.copy()
