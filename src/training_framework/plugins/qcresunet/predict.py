"""Whole-volume QC inference and the paper's three-model ensemble."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch import nn

from training_framework.contracts.types import Batch
from training_framework.core.device import resolve_device
from training_framework.plugins.qcresunet.representation import pack, unpack
from training_framework.registry.defaults import create_default_registries
from training_framework.registry.factories import load_merged_experiment_config


@torch.no_grad()
def ensemble_predict(
    models: Sequence[nn.Module], batch: Batch, num_classes: int, threshold: float = 0.5
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Mean subject scores; strict majority voting for independent SEMs."""
    if not models or len(models) % 2 == 0:
        raise ValueError("Use one model or an odd-sized ensemble to avoid tied SEM votes")
    quality_sum: torch.Tensor | None = None
    probability_sum: torch.Tensor | None = None
    votes: torch.Tensor | None = None
    for model in models:
        model.eval()
        quality, logits = unpack(model(batch).float(), num_classes, batch["signal"].shape[2:])
        probabilities = logits.sigmoid()
        quality_sum = quality if quality_sum is None else quality_sum + quality
        probability_sum = probabilities if probability_sum is None else probability_sum + probabilities
        vote = (probabilities >= threshold).int()
        votes = vote if votes is None else votes + vote
    assert quality_sum is not None and probability_sum is not None and votes is not None
    return quality_sum / len(models), probability_sum / len(models), votes > len(models) // 2


class QCEnsemble(nn.Module):
    """Usable with the framework Evaluator for ensemble metrics."""

    def __init__(self, models: Sequence[nn.Module], num_classes: int, threshold: float = 0.5) -> None:
        super().__init__()
        self.models = nn.ModuleList(models)
        self.num_classes, self.threshold = num_classes, threshold

    def forward(self, batch: Batch) -> torch.Tensor:
        quality, _, errors = ensemble_predict(list(self.models), batch, self.num_classes, self.threshold)
        # Task postprocessing expects logits, so encode the majority vote.
        return pack(quality, torch.where(errors, 20.0, -20.0))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configs", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--checkpoint", nargs="+", required=True, type=Path)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument(
        "--input", type=Path, help="Preprocessed NPZ: image [M,D,H,W], query [D,H,W]; no reference needed"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    inputs.add_argument("--evaluate-manifest", help="Prepared manifest for ensemble test metrics")
    args = parser.parse_args()
    config = load_merged_experiment_config(args.configs, args.model)
    registries = create_default_registries(config.get("plugins", {}).get("modules", []))
    num_classes = len(config["experiment"]["class_names"])
    threshold = float(config["experiment"].get("threshold", 0.5))
    device = resolve_device(args.device)
    model_config = dict(config["model"])
    name = model_config.pop("name")
    model_config["num_classes"] = num_classes
    models: list[nn.Module] = []
    for checkpoint in args.checkpoint:
        model = registries.models.create(name, **model_config)
        if not isinstance(model, nn.Module):
            raise TypeError("QC model must be a torch module")
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        model.load_state_dict(payload["state_dict"])
        models.append(model.to(device))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.evaluate_manifest:
        import json

        from torch.utils.data import DataLoader

        from training_framework.core.evaluator import Evaluator
        from training_framework.metrics.collection import MetricCollection
        from training_framework.plugins.qcresunet.data import QCDataAdapter
        from training_framework.plugins.qcresunet.metrics import QCMetrics
        from training_framework.plugins.qcresunet.task import QCTask

        data_config = {
            key: value
            for key, value in config["data"].items()
            if key in ("modalities", "regions", "tolerance_mm", "balance_eval", "bins")
        }
        adapter = QCDataAdapter(
            args.evaluate_manifest,
            num_classes=num_classes,
            balance_train=False,
            test_only=True,
            seed=int(config["experiment"].get("seed", 42)),
            **data_config,
        )
        adapter.prepare()
        loader = DataLoader(adapter.get_splits()[2], batch_size=1, collate_fn=adapter.collate_fn)
        evaluator = Evaluator(
            QCEnsemble(models, num_classes, threshold).to(device),
            QCTask(num_classes),
            MetricCollection([QCMetrics(num_classes, threshold, config["experiment"]["class_names"])]),
            device,
        )
        _, metrics = evaluator.evaluate(loader)
        args.output.write_text(json.dumps(metrics, indent=2) + "\n")
        print(metrics)
        return
    with np.load(args.input, allow_pickle=False) as content:
        image, query = content["image"].astype(np.float32), content["query"]
        if query.shape != image.shape[1:]:
            raise ValueError("Image and query grids must match")
        if not np.isfinite(image).all():
            raise ValueError("MRI contains non-finite values")
        signal = torch.from_numpy(np.concatenate((image, query[None].astype(np.float32))))[None].to(device)
        spacing = content["spacing"] if "spacing" in content else np.ones(3)
    batch = Batch(signal=signal, label=torch.empty(0, device=device), id=[args.input.stem], meta=[{}])
    scores, probabilities, errors = ensemble_predict(models, batch, num_classes, threshold)
    with args.output.open("xb") as handle:
        np.savez_compressed(
            handle,
            quality=scores.cpu().numpy()[0],
            sem_probability=probabilities.cpu().numpy()[0],
            sem=errors.cpu().numpy()[0].astype(np.uint8),
            spacing=spacing,
        )
    print(f"DSC={scores[0, 0].item():.6f}, NSD={scores[0, 1].item():.6f}; saved {args.output}")


if __name__ == "__main__":
    main()
