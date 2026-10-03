"""Real CPU-DDP regression worker; executed by test_ddp_fit."""

import json
import sys
from pathlib import Path

import torch

from training_framework.cli.main import build_trainer
from training_framework.core.distributed import cleanup_distributed, setup_distributed
from training_framework.registry.defaults import create_default_registries


def main():
    torch.set_num_threads(1)
    destination = Path(sys.argv[1])
    context = setup_distributed("cpu")
    config = {
        "experiment": {"name": "ddp-regression", "seed": 42, "class_names": ["a", "b"]},
        "data": {"adapter": "dummy", "num_samples": 20, "input_dim": 4, "batch_size": 2},
        "model": {"name": "simple_net", "input_dim": 4, "hidden_dim": 4, "dropout": 0.0},
        "task": {"name": "classification"},
        "optimizer": {"name": "sgd", "lr": 0.0},
        "trainer": {"epochs": 4, "patience": 1, "device": "cpu"},
        "checkpoint": {"monitor": "val_loss", "mode": "min"},
        "output": {"root_dir": str(destination / "runs")},
    }
    trainer = None
    try:
        trainer, layout = build_trainer(config, create_default_registries(), distributed_context=context)
        metrics = trainer.fit()
        payload = trainer.checkpoint_manager.load()
        last = trainer.checkpoint_manager.load(layout.checkpoint_dir / "last.pt")
        result = {
            "best_epoch": payload["epoch"],
            "last_epoch": last["epoch"],
            "loss": metrics["loss"],
            "best_path": str(trainer.checkpoint_manager.best_path),
        }
        (destination / f"rank-{context.rank}.json").write_text(json.dumps(result))
    finally:
        if trainer is not None:
            trainer.logger.end_run()
        cleanup_distributed(context)


if __name__ == "__main__":
    main()
