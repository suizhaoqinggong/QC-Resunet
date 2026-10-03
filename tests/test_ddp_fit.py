import json
import os
import socket
import subprocess
import sys
from pathlib import Path

import pytest
import torch.distributed as dist


@pytest.mark.skipif(not dist.is_available() or not dist.is_gloo_available(), reason="CPU Gloo unavailable")
def test_cpu_ddp_early_stopping_and_best_checkpoint_agree(tmp_path):
    worker = Path(__file__).with_name("ddp_worker.py")
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--master_addr=127.0.0.1",
            f"--master_port={port}",
            "--nproc_per_node=2",
            str(worker),
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
        timeout=90,
        env={**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
    )
    assert result.returncode == 0, result.stdout + result.stderr
    first = json.loads((tmp_path / "rank-0.json").read_text())
    second = json.loads((tmp_path / "rank-1.json").read_text())
    assert first == second
    assert first["best_epoch"] == 1
    assert first["last_epoch"] == 2
