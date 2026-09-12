#!/usr/bin/env python3
"""Wait on the live training process, authenticate completion, then run 200 episodes."""

from __future__ import annotations

import argparse
import ctypes
import fcntl
import json
import os
import platform
import secrets
import select
import signal
import subprocess
import sys
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rollout_libero_encoder_lora import read_stage_checkpoint  # noqa: E402
from rollout_libero_spatial import episode_matrix, sha256, write_json  # noqa: E402

CACHE = Path("/hdd2/hyunbin/vla/cache")
TRAINING = CACHE / "runs/libero-spatial15k-prefix-lora-dp2-b128-lr5e-5-1k-recompute-v1"
TRAIN_CONTROL = CACHE / "reports/spatial-prefix-lora-1k-v1/supervisor.json"
PYTHON = CACHE / "venvs/train-single-gpu/bin/python"
EVAL_PYTHON = CACHE / "venvs/libero-eval/bin/python"
CHILDREN = []


def utc():
    return datetime.now(UTC).isoformat()


def open_process_handle(pid):
    """Use Linux pidfds even when the pinned Python build omits os.pidfd_open."""
    if hasattr(os, "pidfd_open"):
        return os.pidfd_open(pid)
    libc = ctypes.CDLL(None, use_errno=True)
    if hasattr(libc, "pidfd_open"):
        libc.pidfd_open.argtypes = [ctypes.c_int, ctypes.c_uint]
        libc.pidfd_open.restype = ctypes.c_int
        result = libc.pidfd_open(pid, 0)
    elif platform.system() == "Linux" and platform.machine() == "x86_64":
        # __NR_pidfd_open from this host's asm/unistd_64.h. Other ABIs fail closed.
        libc.syscall.restype = ctypes.c_long
        result = libc.syscall(ctypes.c_long(434), ctypes.c_int(pid), ctypes.c_uint(0))
    else:
        raise RuntimeError("this platform has no supported pidfd interface")
    if result < 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))
    os.set_inheritable(result, False)
    return result


def interrupt(signum, frame):
    # Only this evaluation workflow is cancelled; never signal the training job.
    raise KeyboardInterrupt(f"received signal {signum}")


def stop_children():
    for child in CHILDREN:
        if child.poll() is None:
            child.terminate()
    for child in CHILDREN:
        try:
            child.wait(timeout=15)
        except subprocess.TimeoutExpired:
            child.kill()
            child.wait()


def render_devices():
    """Read-only EGL/PCI mapping; CUDA indices are not EGL indices on this host."""
    os.environ["__EGL_VENDOR_LIBRARY_FILENAMES"] = "/usr/share/glvnd/egl_vendor.d/10_nvidia.json"
    c = ctypes
    library = c.CDLL("libEGL.so.1")
    library.eglGetProcAddress.restype = c.c_void_p
    library.eglGetProcAddress.argtypes = [c.c_char_p]
    query = c.CFUNCTYPE(c.c_uint, c.c_int, c.POINTER(c.c_void_p), c.POINTER(c.c_int))(
        library.eglGetProcAddress(b"eglQueryDevicesEXT")
    )
    query_string = c.CFUNCTYPE(c.c_char_p, c.c_void_p, c.c_int)(library.eglGetProcAddress(b"eglQueryDeviceStringEXT"))
    devices, count = (c.c_void_p * 32)(), c.c_int()
    if not query(32, devices, c.byref(count)):
        raise RuntimeError("cannot enumerate EGL devices")
    selected = subprocess.check_output(
        [
            "nvidia-smi",
            "--id=0,1",
            "--query-gpu=index,pci.bus_id",
            "--format=csv,noheader",
        ],
        text=True,
    )
    result = {}
    for row in selected.strip().splitlines():
        index, pci = [field.strip().lower() for field in row.split(",")]
        matches = []
        for egl_index in range(count.value):
            drm = query_string(devices[egl_index], 0x3233)
            if drm is None:
                continue
            drm = drm.decode()
            bus = (Path("/sys/class/drm") / Path(drm).name / "device").resolve().name.lower()
            if bus.split(":", 1)[-1] == pci.split(":", 1)[-1]:
                matches.append(
                    {"physical_gpu": int(index), "egl_device_id": egl_index, "drm_device": drm, "pci_bus_id": pci}
                )
        if len(matches) != 1:
            raise RuntimeError(f"ambiguous EGL mapping for physical GPU {index}: {matches}")
        result[int(index)] = matches[0]
    if set(result) != {0, 1}:
        raise RuntimeError("both authorized GPUs must have an EGL mapping")
    return result


def check_training_completion():
    control = json.loads(TRAIN_CONTROL.read_text())
    progress = json.loads((TRAINING / "progress.json").read_text())
    if control["status"] != "complete" or control["returncode"] != 0:
        raise RuntimeError("training did not complete successfully; rollout is not authorized yet")
    if progress["status"] != "complete" or progress["stage_update"] != 1000 or progress["update"] != 16000:
        raise RuntimeError("training did not finish the requested additional 1000 updates")
    rows = [json.loads(line) for line in (TRAINING / "metrics.jsonl").read_text().splitlines()]
    if [row["stage_update"] for row in rows] != list(range(1, 1001)):
        raise RuntimeError("training update coverage has a gap or duplicate")
    for index, row in enumerate(rows, 1):
        if row["update"] != 15000 + index or row["global_batch_size"] != 128:
            raise RuntimeError("training progress/geometry changed")
        if any(row[key] != 5e-5 for key in ("encoder_learning_rate", "lora_learning_rate", "interface_learning_rate")):
            raise RuntimeError("training learning rate changed")
    checkpoints = {}
    for update in (15250, 15500, 15750, 16000):
        path = TRAINING / "checkpoints" / f"update-{update:06d}"
        digest = sha256(path / "manifest.json")
        manifest, _, _ = read_stage_checkpoint(path, digest, final=update == 16000)
        if manifest["absolute_update"] != update:
            raise RuntimeError("checkpoint update does not match its path")
        checkpoints[str(update)] = {"path": str(path), "manifest_sha256": digest}
    latest = json.loads((TRAINING / "latest_checkpoint.json").read_text())
    if (
        latest["path"] != checkpoints["16000"]["path"]
        or latest["manifest_sha256"] != checkpoints["16000"]["manifest_sha256"]
    ):
        raise RuntimeError("latest checkpoint is not the authenticated final checkpoint")
    return checkpoints


def validate_shard(directory, shard, checkpoint_hash, encoder_hash):
    evaluation = directory / "evaluation"
    summary = json.loads((evaluation / "summary.json").read_text())
    rows = [json.loads(line) for line in (evaluation / "episodes.jsonl").read_text().splitlines()]
    expected = set(episode_matrix(20, shard_index=shard, num_shards=2))
    identities = [(row["task_id"], row["reset_id"]) for row in rows]
    if len(rows) != 100 or set(identities) != expected or len(set(identities)) != len(rows):
        raise RuntimeError("missing, duplicated, or off-shard rollout episodes")
    for row in rows:
        if row["nfe"] != 4 or row["execution_horizon"] != 8 or row["shard_index"] != shard:
            raise RuntimeError("episode NFE/K/shard changed")
        if type(row["success"]) is not bool or not 0 <= row["steps"] <= 220 or not 1 <= row["calls"] <= 28:
            raise RuntimeError("invalid rollout outcome")
    if (
        summary["status"] != "complete"
        or summary["target"] != 100
        or summary["completed"] != 100
        or summary["successes"] != sum(row["success"] for row in rows)
        or summary["success_rate"] != sum(row["success"] for row in rows) / 100
        or summary["nfe"] != 4
        or summary["execution_horizon"] != 8
    ):
        raise RuntimeError("rollout summary does not match episode outcomes")
    serving = json.loads((directory / "policy/serving.json").read_text())
    if (
        serving["checkpoint_manifest_sha256"] != checkpoint_hash
        or serving["encoder_adapter_sha256"] != encoder_hash
        or serving["encoder_adapter_loaded"] is not True
        or serving["nfe"] != 4
        or serving["absolute_update"] != 16000
        or serving["stage_update"] != 1000
        or serving["cuda_visible_devices"] != str(shard)
        or serving["prefix_autocast"] is not False
    ):
        raise RuntimeError("server did not use the exact final encoder-adapted policy on its assigned GPU")
    parity = json.loads((evaluation / "inference_validation.json").read_text())
    if parity["repeat_max_abs_error"] != 0 or not 0 <= parity["permutation_max_abs_error"] <= 1e-5:
        raise RuntimeError("repeat/permutation inference qualification failed")
    if parity["singleton_max_abs_error"] is None or not 0 <= parity["singleton_max_abs_error"] <= 1e-5:
        raise RuntimeError("singleton inference qualification failed")
    resets = [json.loads(line) for line in (evaluation / "reset_audit.jsonl").read_text().splitlines()]
    if len(resets) != 100 or {(row["task_id"], row["reset_id"]) for row in resets} != expected:
        raise RuntimeError("reset audit does not cover the exact rollout matrix")
    return rows, summary, resets


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-pid", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    lock = (args.output / "workflow.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state = {"status": "starting", "pid": os.getpid(), "target_episodes": 200, "shards": {}}

    def update(**values):
        state.update(values, updated_utc=utc())
        state["completed_episodes"] = sum(shard.get("completed", 0) for shard in state["shards"].values())
        write_json(args.output / "progress.json", state)

    for signum in (signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, interrupt)
    logs = []
    try:
        proc = Path(f"/proc/{args.training_pid}")
        handle = open_process_handle(args.training_pid)
        try:
            command = (proc / "cmdline").read_bytes().replace(b"\0", b" ").decode()
            if "torch.distributed.run" not in command or str(TRAINING) not in command:
                raise RuntimeError("training PID does not belong to this experiment")
            update(
                status="waiting_for_training",
                training_pid=args.training_pid,
                training_process_stat=(proc / "stat").read_text(),
                training_command=command,
            )
            while not select.select([handle], [], [], 30)[0]:
                update(
                    training_progress=json.loads((TRAINING / "progress.json").read_text()),
                    training_handle_confirmed_live_at_utc=utc(),
                )
        finally:
            os.close(handle)
        # torchrun exits before its outer supervisor commits its final status.
        for _ in range(10):
            if json.loads(TRAIN_CONTROL.read_text())["status"] != "running":
                break
            time.sleep(1)
        checkpoints = check_training_completion()
        update(status="training_verified", training_checkpoints=checkpoints)
        from duo_vla.gpu_preflight import require_idle_training_gpus

        gpu = require_idle_training_gpus((0, 1))
        render = render_devices()
        checkpoint = Path(checkpoints["16000"]["path"])
        digest = checkpoints["16000"]["manifest_sha256"]
        manifest = json.loads((checkpoint / "manifest.json").read_text())
        protocol = {
            "schema": "duo-vla-prefix-lora-spatial-200-v1",
            "created_utc": utc(),
            "checkpoint": str(checkpoint),
            "checkpoint_manifest_sha256": digest,
            "encoder_adapter_sha256": manifest["artifacts"]["encoder_adapter"]["sha256"],
            "gpus": gpu,
            "render_devices": render,
            "nfe": 4,
            "execution_horizon": 8,
            "episodes": 200,
            "resets_per_task": 20,
            "evaluation_seed": 0,
            "environment_seed": 7,
            "settle_steps": 10,
            "max_policy_steps": 220,
            "source_snapshot": str(ROOT),
            "source_files_sha256": {
                str(p.relative_to(ROOT)): sha256(p)
                for folder in (ROOT / "src", ROOT / "scripts")
                for p in sorted(folder.rglob("*.py"))
            },
            "reset_source": "published official initial states, reset IDs 0-19 per task",
            "reset_warning_thresholds": {"displacement_m": 0.1, "linear_speed_m_s": 3.0},
            "reset_warning_handling": "report without filtering, replacement or extra settling",
            "scope": "experimental subset, not an official preregistered LIBERO result",
        }
        write_json(args.output / "protocol.json", protocol)
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
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "RAYON_NUM_THREADS": "1",
            "PYTHONHASHSEED": "0",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONNOUSERSITE": "1",
            "DUO_VLA_CACHE_ROOT": str(CACHE),
        }
        jobs = []
        with tempfile.TemporaryDirectory(prefix="dvla-encoder-eval-") as private:
            authkey = Path(private) / "authkey"
            authkey.write_bytes(secrets.token_bytes(32))
            authkey.chmod(0o600)
            for index in range(2):
                directory = args.output / f"gpu{index}"
                directory.mkdir()
                socket = Path(private) / f"gpu{index}.sock"
                common = ["--socket", str(socket), "--authkey", str(authkey), "--shard-index", str(index)]
                script = ROOT / "scripts/rollout_libero_encoder_lora.py"
                server_command = [
                    str(PYTHON),
                    "-u",
                    str(script),
                    "server",
                    *common,
                    "--checkpoint",
                    str(checkpoint),
                    "--expected-manifest-sha256",
                    digest,
                    "--output",
                    str(directory / "policy"),
                ]
                log = (directory / "server.log").open("x")
                logs.append(log)
                server = subprocess.Popen(
                    server_command,
                    cwd=ROOT,
                    env={**env, "CUDA_VISIBLE_DEVICES": str(index)},
                    stdin=subprocess.DEVNULL,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
                CHILDREN.append(server)
                jobs.append(
                    {
                        "index": index,
                        "server": server,
                        "evaluator": None,
                        "socket": socket,
                        "common": common,
                        "directory": directory,
                        "ready_deadline": time.monotonic() + 1800,
                    }
                )
                state["shards"][str(index)] = {"status": "loading", "server_pid": server.pid, "completed": 0}
            started = time.perf_counter()
            while True:
                complete = 0
                for job in jobs:
                    index, server, evaluator = job["index"], job["server"], job["evaluator"]
                    if evaluator is None:
                        if server.poll() is not None:
                            raise RuntimeError(f"GPU {index} server exited before becoming ready: {server.returncode}")
                        if time.monotonic() > job["ready_deadline"]:
                            raise TimeoutError(f"GPU {index} model loading timed out")
                        if not job["socket"].exists():
                            continue
                        eval_command = [
                            str(EVAL_PYTHON),
                            "-u",
                            str(script),
                            "evaluate",
                            *job["common"],
                            "--output",
                            str(job["directory"] / "evaluation"),
                        ]
                        log = (job["directory"] / "evaluate.log").open("x")
                        logs.append(log)
                        evaluator = subprocess.Popen(
                            eval_command,
                            cwd=ROOT,
                            env={
                                **env,
                                "CUDA_VISIBLE_DEVICES": "",
                                "LIBERO_CONFIG_PATH": str(CACHE / "simulators/libero/config"),
                                "MUJOCO_GL": "egl",
                                "PYOPENGL_PLATFORM": "egl",
                                "MUJOCO_EGL_DEVICE_ID": str(render[index]["egl_device_id"]),
                                "__EGL_VENDOR_LIBRARY_FILENAMES": "/usr/share/glvnd/egl_vendor.d/10_nvidia.json",
                            },
                            stdin=subprocess.DEVNULL,
                            stdout=log,
                            stderr=subprocess.STDOUT,
                        )
                        CHILDREN.append(evaluator)
                        job["evaluator"] = evaluator
                        state["shards"][str(index)].update(status="evaluating", evaluator_pid=evaluator.pid)
                    summary = job["directory"] / "evaluation/summary.json"
                    if summary.exists():
                        state["shards"][str(index)].update(json.loads(summary.read_text()))
                    if evaluator.poll() is not None:
                        if evaluator.returncode != 0:
                            raise RuntimeError(f"GPU {index} evaluator failed: {evaluator.returncode}")
                        server.wait(timeout=60)
                        if server.returncode != 0:
                            raise RuntimeError(f"GPU {index} server failed: {server.returncode}")
                        complete += 1
                    elif server.poll() is not None:
                        evaluator.wait(timeout=15)
                        if evaluator.returncode != 0 or server.returncode != 0:
                            raise RuntimeError(f"GPU {index} inference process failed")
                update(status="evaluating")
                if complete == 2:
                    break
                time.sleep(5)
            episodes, summaries, resets = [], [], []
            for job in jobs:
                rows, summary, audit = validate_shard(
                    job["directory"],
                    job["index"],
                    digest,
                    protocol["encoder_adapter_sha256"],
                )
                episodes.extend(rows)
                summaries.append(summary)
                resets.extend(audit)
            if len(episodes) != 200 or {(row["task_id"], row["reset_id"]) for row in episodes} != set(
                episode_matrix(20)
            ):
                raise RuntimeError("combined results do not cover the full 200-episode matrix")
            from evaluate_libero import wilson95

            successes = sum(row["success"] for row in episodes)
            result = {
                "status": "complete",
                "completed": 200,
                "successes": successes,
                "success_rate": successes / 200,
                "wilson95": wilson95(successes, 200),
                "nfe": 4,
                "execution_horizon": 8,
                "checkpoint_manifest_sha256": digest,
                "elapsed_seconds": time.perf_counter() - started,
                "reset_physics_warnings": sum(row["warning"] for row in resets),
                "shards": summaries,
                "tasks": [
                    {
                        "task_id": task,
                        "completed": 20,
                        "successes": sum(row["success"] for row in episodes if row["task_id"] == task),
                    }
                    for task in range(10)
                ],
            }
            write_json(args.output / "result.json", result)
            with (args.output / "episodes.jsonl").open("x") as log:
                for row in sorted(episodes, key=lambda row: (row["task_id"], row["reset_id"])):
                    log.write(json.dumps(row, allow_nan=False) + "\n")
            update(status="complete", result=result)
    except BaseException as exc:
        update(
            status="cancelled" if isinstance(exc, KeyboardInterrupt) else "failed", error=f"{type(exc).__name__}: {exc}"
        )
        raise
    finally:
        stop_children()
        for log in logs:
            log.close()
        lock.close()


if __name__ == "__main__":
    main()
