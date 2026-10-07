"""Prepare portable QC NPZ samples and subject-level cross-validation folds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np

from training_framework.plugins.qcresunet.quality import quality_targets
from training_framework.plugins.qcresunet.representation import BRATS_REGIONS, validate_regions


def preprocess(
    image: np.ndarray, reference: np.ndarray, queries: list[np.ndarray], shape: Sequence[int]
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    """Nonzero per-modality z-score, common foreground crop and zero padding.

    Inputs must already be skull-stripped, registered and resampled. Never
    center-crop a volume: dropping tumor voxels changes subject QC labels.
    Crop includes reference AND queries so segmentation failures are retained.
    """
    if (
        image.ndim != 4
        or reference.shape != image.shape[1:]
        or any(query.shape != reference.shape for query in queries)
    ):
        raise ValueError("All images/masks must share an already registered grid")
    if len(shape) != 3 or min(shape) < 1 or not np.isfinite(image).all():
        raise ValueError("Invalid output shape or non-finite MRI")
    image = image.astype(np.float32).copy()
    foreground = (image != 0).any(0) | (reference != 0)
    for query in queries:
        foreground |= query != 0
    positions = np.where(foreground)
    if not positions[0].size:
        raise ValueError("Cannot preprocess an entirely empty volume")
    crop = tuple(slice(int(axis.min()), int(axis.max()) + 1) for axis in positions)
    for channel in image:
        valid = channel != 0
        if valid.any():
            values = channel[valid]
            channel[valid] = (values - values.mean()) / max(float(values.std()), 1e-6)
    image = image[(slice(None), *crop)]
    masks = [mask[crop] for mask in [reference, *queries]]
    if any(size > target for size, target in zip(masks[0].shape, shape)):
        raise ValueError("Foreground exceeds the requested shape; resample upstream or choose a larger shape")
    padding = [((target - size) // 2, (target - size + 1) // 2) for size, target in zip(masks[0].shape, shape)]
    image = np.pad(image, [(0, 0), *padding])
    masks = [np.pad(mask, padding) for mask in masks]
    return image, masks[0], masks[1:]


def assign_folds(subject_ids: Sequence[str], seed: int, test_subjects: int) -> list[dict[str, str]]:
    """A fixed held-out test group and three disjoint validation folds."""
    subjects = np.asarray(sorted(set(subject_ids)))
    if test_subjects < 1 or len(subjects) - test_subjects < 3:
        raise ValueError("Cross-validation requires test subjects and at least three remaining subjects")
    shuffled = np.random.default_rng(seed).permutation(subjects)
    test = set(shuffled[:test_subjects].tolist())
    validation_groups = [set(group.tolist()) for group in np.array_split(shuffled[test_subjects:], 3)]
    return [
        {
            str(subject): "test" if subject in test else "val" if subject in validation else "train"
            for subject in subjects
        }
        for validation in validation_groups
    ]


def prepare_manifest(
    source: Path,
    output: Path,
    tolerance_mm: float,
    shape: Sequence[int] = (160, 192, 160),
    regions: Sequence[Sequence[int]] = BRATS_REGIONS,
    modalities: int = 4,
    seggen: int = 3,
    seed: int = 42,
    fold: int | None = None,
    test_subjects: int = 251,
    test_only: bool = False,
) -> Path:
    """Source JSON describes each volume and its independently generated queries.

    volume NPZ: image [M,D,H,W], reference [D,H,W], spacing [3].
    queries: list of {path: NPY mask, method: name}. Explicit split or a fold
    assignment is mandatory. DeepMedic records are only admitted to testing.
    """
    regions = validate_regions(regions, len(regions))
    content = json.loads(source.read_text())
    cases = content["subjects"]
    if not cases or seggen < 0 or modalities < 1:
        raise ValueError("Need nonempty subjects, nonnegative SegGen count and positive modalities")
    ids = [case["subject_id"] for case in cases]
    if len(set(ids)) != len(ids) or any(not isinstance(value, str) or not value for value in ids):
        raise ValueError("Each source subject must appear exactly once with a nonempty ID")
    if fold is not None and fold not in (0, 1, 2):
        raise ValueError("fold must be 0, 1 or 2")
    assignment = (
        assign_folds(ids, seed, test_subjects)[fold]
        if fold is not None
        else {case["subject_id"]: case["split"] for case in cases}
    )
    if any(split not in ("train", "val", "test") for split in assignment.values()):
        raise ValueError("Source splits must be train, val or test")
    if test_only and (fold is not None or any(split != "test" for split in assignment.values())):
        raise ValueError("test_only requires only explicit test subjects and no fold")
    output.mkdir(parents=True, exist_ok=True)
    destination = output / "manifest.json"
    if destination.exists() or any(output.glob("sample-*")) or any(output.glob("subject-*.npz")):
        raise FileExistsError("Output already contains a QC dataset; use a new output directory")
    rng = np.random.default_rng(seed)
    records: list[dict[str, Any]] = []
    counts: dict[str, int] = {split: 0 for split in ("train", "val", "test")}
    for case_index, case in enumerate(cases):
        split = assignment[case["subject_id"]]
        with np.load(source.parent / case["volume"], allow_pickle=False) as volume:
            image = volume["image"].astype(np.float32)
            reference = volume["reference"]
            spacing = volume["spacing"].astype(np.float64)
        if image.ndim != 4 or image.shape[0] != modalities:
            raise ValueError("Volume MRI channels do not match modalities")
        if not np.allclose(spacing, (1, 1, 1)) and modalities == 4:
            raise ValueError("BraTS inputs must already be resampled to 1mm isotropic resolution")
        descriptors = [
            query
            for query in case.get("queries", [])
            if split == "test" or query.get("method", "").lower() != "deepmedic"
        ]
        queries = [np.load(source.parent / query["path"], allow_pickle=False, mmap_mode="r") for query in descriptors]
        # BraTS 2021 raw labels use 4 for ET. Canonicalize every mask together.
        if tuple(regions) == BRATS_REGIONS:
            reference = np.where(reference == 4, 3, reference)
        # Scan memory-mapped queries for common crop bounds, then process only
        # one query at a time. Hundreds of full-volume masks must not pile up.
        footprint = np.zeros(reference.shape, dtype=bool)
        for query in queries:
            if query.shape != reference.shape:
                raise ValueError("Query and reference grids must match")
            footprint |= query != 0
        foreground = (image != 0).any(0) | (reference != 0) | footprint
        positions = np.where(foreground)
        if not positions[0].size:
            raise ValueError("Cannot preprocess an entirely empty volume")
        crop = tuple(slice(int(axis.min()), int(axis.max()) + 1) for axis in positions)
        cropped_shape = tuple(item.stop - item.start for item in crop)
        image, reference, _ = preprocess(image, reference, [footprint], shape)
        padding = [((target - size) // 2, (target - size + 1) // 2) for size, target in zip(cropped_shape, shape)]
        volume_path = output / f"subject-{case_index:05d}.npz"
        np.savez_compressed(volume_path, image=image, reference=reference.astype(np.int16), spacing=spacing)

        def iter_queries() -> Iterator[tuple[np.ndarray, dict[str, Any]]]:
            for raw_query, descriptor in zip(queries, descriptors):
                query = raw_query[crop]
                if tuple(regions) == BRATS_REGIONS:
                    query = np.where(query == 4, 3, query)
                yield np.pad(query, padding), descriptor
            if seggen and split != "test":
                from training_framework.plugins.qcresunet.augmentation import warp_masks

                for _ in range(seggen):
                    generated, _ = warp_masks([reference], rng, seggen=True)
                    yield generated[0], {"method": "seggen"}

        if not queries and (not seggen or split == "test"):
            raise ValueError(f"No query segmentations for subject {case['subject_id']}")
        for query_index, (query, descriptor) in enumerate(iter_queries()):
            scores, _ = quality_targets(query, reference, spacing, tolerance_mm, regions)
            sample_id = f"sample-{case_index:05d}-{query_index:04d}"
            path = output / f"{sample_id}.npy"
            np.save(path, query.astype(np.int16))
            records.append(
                {
                    "id": sample_id,
                    "subject_id": case["subject_id"],
                    "split": split,
                    "path": path.name,
                    "volume": volume_path.name,
                    "method": descriptor.get("method", "unspecified"),
                    "dsc": float(scores[0]),
                    "nsd": float(scores[1]),
                }
            )
            counts[split] += 1
        print(f"Prepared {case_index + 1}/{len(cases)} subjects ({len(records)} queries)", flush=True)
    if any(count == 0 for count in counts.values()) and not test_only:
        raise ValueError("Every prepared split must be nonempty")
    manifest = {
        "version": 1,
        "regions": [list(region) for region in regions],
        "tolerance_mm": tolerance_mm,
        "seed": seed,
        "fold": fold,
        "samples": records,
    }
    destination.write_text(json.dumps(manifest, indent=2) + "\n")
    report = {
        "counts": counts,
        "subject_counts": {split: sum(value == split for value in assignment.values()) for split in counts},
        "dsc_histograms": {
            split: np.histogram(
                [record["dsc"] for record in records if record["split"] == split], bins=np.linspace(0, 1, 11)
            )[0].tolist()
            for split in counts
        },
    }
    (output / "preparation-report.json").write_text(json.dumps(report, indent=2) + "\n")
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--tolerance-mm", type=float, required=True, help="NSD tolerance is not specified in the supplied PDF"
    )
    parser.add_argument("--shape", type=int, nargs=3, default=(160, 192, 160))
    parser.add_argument(
        "--regions", type=json.loads, default=BRATS_REGIONS, help="JSON label groups; ACDC e.g. [[1],[2],[3]]"
    )
    parser.add_argument("--modalities", type=int, default=4)
    parser.add_argument("--seggen", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fold", type=int, choices=(0, 1, 2))
    parser.add_argument("--test-subjects", type=int, default=251)
    parser.add_argument("--test-only", action="store_true", help="Prepare an external test-only dataset")
    args = parser.parse_args()
    print(
        prepare_manifest(
            args.source,
            args.output,
            args.tolerance_mm,
            args.shape,
            args.regions,
            args.modalities,
            args.seggen,
            args.seed,
            args.fold,
            args.test_subjects,
            args.test_only,
        )
    )


if __name__ == "__main__":
    main()
