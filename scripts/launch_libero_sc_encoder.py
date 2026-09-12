#!/usr/bin/env python3
"""Closed-environment, owned-process-only launcher for explicitly authorized SC DP2 training/probes."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from duo_vla.gpu_preflight import require_idle_training_gpus  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--artifact-reference", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, choices=(8, 16, 32, 64, 72, 80), required=True)
    parser.add_argument("--probe-steps", type=int, default=0)
    parser.add_argument("--stop-after", type=int)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    args.control.mkdir(parents=True, exist_ok=False)
    gpu = require_idle_training_gpus((0, 1))
    cache = Path("/hdd2/hyunbin/vla/cache")
    env = {
        "HOME": "/root",
        "PATH": "/usr/bin:/bin",
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "TZ": "UTC",
        "HF_HOME": "/hdd2/hyunbin/vla/huggingface",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_HUB_DISABLE_PROGRESS_BARS": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        "CUDA_VISIBLE_DEVICES": "0,1",
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "RAYON_NUM_THREADS": "1",
        "PYTHONHASHSEED": "0",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONNOUSERSITE": "1",
        "DUO_VLA_CACHE_ROOT": str(cache),
        "PYTORCH_ALLOC_CONF": "expandable_segments:True",
        "TORCH_NCCL_ASYNC_ERROR_HANDLING": "1",
    }
    command = [
        str(cache / "venvs/train-single-gpu/bin/python"),
        "-B",
        "-u",
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nnodes=1",
        "--nproc_per_node=2",
        str(args.source / "scripts/train_libero_sc_encoder.py"),
        "--artifact-reference",
        str(args.artifact_reference),
        "--dataset",
        str(args.dataset),
        "--output",
        str(args.output),
        "--batch-size",
        str(args.batch_size),
        "--checkpoint-prefix",
    ]
    for flag, value in (
        ("--probe-steps", args.probe_steps),
        ("--stop-after", args.stop_after),
        ("--resume", args.resume),
    ):
        if value:
            command += [flag, str(value)]
    state = {
        "status": "starting",
        "pid": os.getpid(),
        "command": command,
        "gpu_preflight": gpu,
        "started_utc": datetime.now(UTC).isoformat(),
    }

    def write(**values):
        state.update(values, updated_utc=datetime.now(UTC).isoformat())
        temp = args.control / "supervisor.json.tmp"
        with temp.open("w") as handle:
            handle.write(json.dumps(state, indent=2) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, args.control / "supervisor.json")

    write()
    with (args.control / "train.log").open("x") as log:
        process = subprocess.Popen(
            command, cwd=args.source, env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT
        )

        def stop(signum, frame):
            write(status="stop_requested", signal=signum)
            if process.poll() is None:
                process.send_signal(signum)

        for signum in (signal.SIGTERM, signal.SIGINT):
            signal.signal(signum, stop)
        write(status="running", training_pid=process.pid)
        while process.poll() is None:
            write()
            time.sleep(5)
        code = process.wait()
    final = (
        json.loads((args.output / "progress.json").read_text()) if (args.output / "progress.json").exists() else None
    )
    if code == 0 and (final is None or final["status"] not in {"complete", "stopped", "probe_complete"}):
        write(status="failed", returncode=code, error="child exited without a valid terminal training state")
        raise RuntimeError("invalid terminal training state")
    write(status=final["status"] if code == 0 else "failed", returncode=code, final_progress=final)
    if code:
        raise SystemExit(code)


if __name__ == "__main__":
    main()
