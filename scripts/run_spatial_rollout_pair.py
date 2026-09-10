#!/usr/bin/env python3
"""Supervise the requested K=4/K=8 Spatial runs on disjoint pairs of GPUs."""

from __future__ import annotations

import argparse
import json
import os
import secrets
import subprocess
import sys
import tempfile
import time
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trace-routing", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    runner = Path(__file__).resolve().with_name("rollout_libero_spatial.py")
    environment = os.environ.copy()
    for key in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "PYTHONPATH", "LD_PRELOAD"):
        environment.pop(key, None)
    environment.update(
        {
            "HF_HOME": str(args.cache_root.parent / "huggingface"),
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_HUB_DISABLE_PROGRESS_BARS": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "PYTHONHASHSEED": "0",
            "PYTHONDONTWRITEBYTECODE": "1",
            "DUO_VLA_CACHE_ROOT": str(args.cache_root),
        }
    )
    children: list[subprocess.Popen] = []
    logs = []
    jobs = []
    with tempfile.TemporaryDirectory(prefix="dvla-spatial-") as private:
        private_path = Path(private)
        authkey = private_path / "authkey"
        authkey.write_bytes(secrets.token_bytes(32))
        authkey.chmod(0o600)
        try:
            for horizon, gpus in ((4, "4,5"), (8, "6,7")):
                socket_path = private_path / f"k{horizon}.sock"
                log = (args.output / f"k{horizon}-server.log").open("w")
                logs.append(log)
                common = ["--socket", str(socket_path), "--authkey", str(authkey), "--nfe", "2"]
                command = [
                    str(args.cache_root / "venvs/train/bin/python"),
                    "-u",
                    str(runner),
                    "server",
                    *common,
                    "--checkpoint",
                    str(args.checkpoint),
                    "--output",
                    str(args.output / f"k{horizon}-policy"),
                ]
                if args.trace_routing:
                    command.append("--trace-routing")
                server = subprocess.Popen(
                    command, env={**environment, "CUDA_VISIBLE_DEVICES": gpus}, stdout=log, stderr=subprocess.STDOUT
                )
                children.append(server)
                jobs.append(
                    {
                        "horizon": horizon,
                        "server": server,
                        "evaluator": None,
                        "socket": socket_path,
                        "common": common,
                        "render_gpu": gpus.split(",")[0],
                    }
                )
                print(json.dumps({"status": "loading", "k": horizon, "gpus": gpus, "pid": server.pid}), flush=True)
            while True:
                complete = 0
                for job in jobs:
                    horizon, server, evaluator = job["horizon"], job["server"], job["evaluator"]
                    if evaluator is None:
                        if server.poll() is not None:
                            raise RuntimeError(f"K={horizon} server failed with code {server.returncode}")
                        if job["socket"].exists():
                            log = (args.output / f"k{horizon}-evaluate.log").open("w")
                            logs.append(log)
                            eval_environment = {
                                **environment,
                                "LIBERO_CONFIG_PATH": str(args.cache_root / "simulators/libero/config"),
                                "MUJOCO_GL": "egl",
                                "PYOPENGL_PLATFORM": "egl",
                                "MUJOCO_EGL_DEVICE_ID": job["render_gpu"],
                                "__EGL_VENDOR_LIBRARY_FILENAMES": str(args.cache_root / "egl-nvidia.json"),
                            }
                            command = [
                                str(args.cache_root / "venvs/libero-eval/bin/python"),
                                "-u",
                                str(runner),
                                "evaluate",
                                *job["common"],
                                "--execution-horizon",
                                str(horizon),
                                "--evaluation-seed",
                                "0",
                                "--resets-per-task",
                                "10",
                                "--output",
                                str(args.output / f"k{horizon}-evaluation"),
                            ]
                            evaluator = subprocess.Popen(
                                command, env=eval_environment, stdout=log, stderr=subprocess.STDOUT
                            )
                            job["evaluator"] = evaluator
                            children.append(evaluator)
                            print(json.dumps({"status": "evaluating", "k": horizon, "pid": evaluator.pid}), flush=True)
                    else:
                        code = evaluator.poll()
                        if code is not None:
                            if code != 0:
                                raise RuntimeError(f"K={horizon} evaluation failed with code {code}")
                            complete += 1
                        elif server.poll() is not None:
                            raise RuntimeError(f"K={horizon} server exited during evaluation")
                if complete == 2:
                    break
                time.sleep(2)
            summaries = {}
            for job in jobs:
                horizon = job["horizon"]
                summary = json.loads((args.output / f"k{horizon}-evaluation/summary.json").read_text())
                if summary["status"] != "complete" or summary["completed"] != 100:
                    raise RuntimeError(f"K={horizon} did not complete exactly 100 episodes")
                summaries[f"k{horizon}"] = summary
            (args.output / "comparison.json").write_text(json.dumps(summaries, indent=2) + "\n")
            print(json.dumps({"status": "complete", "episodes": 200, "output": str(args.output)}), flush=True)
        finally:
            for process in children:
                if process.poll() is None:
                    process.terminate()
            for process in children:
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            for log in logs:
                log.close()


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"Spatial pair failed: {error}", file=sys.stderr, flush=True)
        raise
