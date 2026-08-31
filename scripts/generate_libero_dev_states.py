#!/usr/bin/env python3
"""Materialize deterministic, non-official LIBERO development reset states."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from duo_vla.benchmarks.libero_dev_states import (
    CAPTURE_PHASE,
    DEV_STATE_SCHEMA,
    OPEN_GRIPPER_NOOP,
    SAMPLER_SEED_DOMAIN,
    SETTLE_STEPS,
    STATE_DTYPE,
    STATE_ENCODING,
    SUITES,
    VALIDATION_GATES,
    DevStateValidationError,
    InitialSuccessError,
    SettleSuccessError,
    artifact_record,
    canonical_official_state_hashes,
    deterministic_npy_bytes,
    deterministic_sampler_seed,
    finalize_manifest,
    sample_reproducible_pre_settle_state,
    sequence_root_sha256,
    state_sha256,
    task_artifact_path,
    validate_manifest_and_artifacts,
    validate_pre_settle_candidate,
    write_bank_exclusive,
)

EXPECTED_PACKAGES = {
    "cmake": "4.1.3",
    "hf-egl-probe": "1.0.2",
    "hf-libero": "0.1.4",
    "huggingface-hub": "1.29.0",
    "mujoco": "3.8.1",
    "numpy": "2.2.6",
    "robosuite": "1.4.0",
    "torch": "2.11.0+cpu",
    "torchvision": "0.26.0+cpu",
}
EXPECTED_SOURCE_REVISION = "8561c60eea2fb93096146f240194649df73d8b1e"
EXPECTED_ASSETS_REVISION = "0b3ea86be5fe169d0fd036ae63d1070ec09e90f6"
CAMERA_NAMES = ("agentview", "robot0_eye_in_hand")


@dataclass(frozen=True, slots=True)
class TaskInputs:
    suite: str
    task_id: int
    task: Any
    official_states: np.ndarray
    state_size: int
    official_hashes: tuple[str, ...]
    bddl_path: Path


@dataclass(frozen=True, slots=True)
class GeneratedTaskBank:
    states: np.ndarray
    entries: tuple[dict[str, Any], ...]
    rejections: dict[str, int]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def parse_selection(value: str, *, upper: int) -> tuple[int, ...]:
    if value == "all":
        return tuple(range(upper))
    selected: set[int] = set()
    try:
        for part in value.split(","):
            bounds = part.split("-")
            if len(bounds) == 1:
                selected.add(int(bounds[0]))
            elif len(bounds) == 2:
                start, end = map(int, bounds)
                require(start <= end, "task-id range must be increasing")
                selected.update(range(start, end + 1))
            else:
                raise ValueError
    except ValueError as exc:
        raise ValueError("task ids must be a comma/range selection such as 0,2-4 or all") from exc
    require(bool(selected) and min(selected) >= 0 and max(selected) < upper, f"task ids must be in [0, {upper})")
    return tuple(sorted(selected))


def generate_task_bank(
    environment: Any,
    *,
    suite: str,
    task_id: int,
    official_hashes: Sequence[str],
    state_size: int,
    base_seed: int,
    states_per_task: int,
    max_attempts_per_task: int,
    bank_hashes: set[str],
) -> GeneratedTaskBank:
    """Sample one task bank, skipping only explicitly declared contamination/validity failures."""

    require(states_per_task > 0, "states_per_task must be positive")
    require(max_attempts_per_task >= states_per_task, "max_attempts_per_task must cover the requested states")
    official = set(official_hashes)
    require(len(official) == 50, "official state hash set must contain 50 unique states")
    accepted: list[np.ndarray] = []
    entries: list[dict[str, Any]] = []
    rejections = {
        "bank_duplicate": 0,
        "initial_success": 0,
        "official_match": 0,
        "settle_success": 0,
    }
    for attempt_id in range(max_attempts_per_task):
        sampler_seed = deterministic_sampler_seed(base_seed, suite, task_id, attempt_id)
        state = sample_reproducible_pre_settle_state(
            environment,
            sampler_seed=sampler_seed,
            expected_size=state_size,
        )
        digest = state_sha256(state, expected_size=state_size)
        if digest in official:
            rejections["official_match"] += 1
            continue
        if digest in bank_hashes:
            rejections["bank_duplicate"] += 1
            continue
        try:
            validation = validate_pre_settle_candidate(
                environment,
                state,
                sampler_seed=sampler_seed,
                settle_steps=SETTLE_STEPS,
            )
        except InitialSuccessError:
            rejections["initial_success"] += 1
            continue
        except SettleSuccessError:
            rejections["settle_success"] += 1
            continue
        require(validation.state_size == state_size, "candidate validation changed the state size")
        require(validation.state_sha256 == digest, "candidate validation changed the pre-settle state hash")
        reset_id = len(accepted)
        accepted.append(state)
        bank_hashes.add(digest)
        entries.append(
            {
                "attempt_id": attempt_id,
                "reset_id": reset_id,
                "sampler_seed": sampler_seed,
                "settled_state_sha256": validation.settled_state_sha256,
                "state_sha256": digest,
            }
        )
        if len(accepted) == states_per_task:
            break
    require(
        len(accepted) == states_per_task,
        f"could not generate {states_per_task} clean states for {suite} task {task_id} "
        f"within {max_attempts_per_task} attempts: accepted={len(accepted)}, rejections={rejections}",
    )
    states = np.ascontiguousarray(np.stack(accepted), dtype=STATE_DTYPE)
    return GeneratedTaskBank(states=states, entries=tuple(entries), rejections=rejections)


def _validate_runtime() -> tuple[dict[str, Any], Any, Any, Any]:
    cache_root = Path(os.environ.get("DUO_VLA_CACHE_ROOT", "/root/.cache/duo-vla"))
    runtime_root = cache_root / "simulators/libero"
    manifest_path = runtime_root / "manifest.json"
    require(manifest_path.is_file(), f"missing simulator manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    require(manifest.get("schema") == "duo-vla-libero-simulator-v1", "unexpected simulator manifest schema")
    require(manifest.get("source", {}).get("revision") == EXPECTED_SOURCE_REVISION, "LIBERO source pin mismatch")
    require(manifest.get("assets", {}).get("revision") == EXPECTED_ASSETS_REVISION, "LIBERO asset pin mismatch")
    require(manifest.get("training_data_downloaded") is False, "simulator manifest includes training data")
    require(os.environ.get("MUJOCO_GL") == "egl", "MUJOCO_GL must be exactly 'egl'")
    require(os.environ.get("PYOPENGL_PLATFORM") == "egl", "PYOPENGL_PLATFORM must be exactly 'egl'")
    require(sys.version_info[:2] == (3, 12), f"unexpected Python version: {sys.version.split()[0]}")
    package_versions = {name: importlib.metadata.version(name) for name in EXPECTED_PACKAGES}
    require(package_versions == EXPECTED_PACKAGES, f"package pin mismatch: {package_versions}")
    require(manifest.get("environment", {}).get("python") == sys.version.split()[0], "simulator Python pin mismatch")

    import mujoco.gl_context
    import robosuite
    from libero.libero import benchmark, get_assets_path, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    require(mujoco.gl_context.GLContext.__module__ == "mujoco.egl", "MuJoCo did not select its EGL backend")
    require(robosuite.__version__ == "1.4.0", "robosuite version mismatch")
    require(
        Path(get_assets_path()).resolve() == Path(manifest["assets"]["path"]).resolve(),
        "LIBERO assets path does not match the pinned manifest",
    )
    dataset_path = Path(get_libero_path("datasets"))
    unexpected_data = [path for path in dataset_path.rglob("*") if path.is_file()]
    require(
        not unexpected_data,
        f"training/demo data unexpectedly exists under simulator datasets: {unexpected_data[:3]}",
    )
    return manifest, benchmark, get_libero_path, OffScreenRenderEnv


def _task_inputs(
    benchmark: Any,
    get_libero_path: Any,
    *,
    suites: tuple[str, ...],
    task_ids: tuple[int, ...],
) -> tuple[TaskInputs, ...]:
    benchmark_map = benchmark.get_benchmark_dict()
    result: list[TaskInputs] = []
    for suite_name in suites:
        suite = benchmark_map[suite_name]()
        require(suite.n_tasks == 10, f"{suite_name} task count changed")
        for task_id in task_ids:
            task = suite.get_task(task_id)
            official_states = np.asarray(suite.get_task_init_states(task_id))
            state_size, official_hashes = canonical_official_state_hashes(official_states)
            bddl_path = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
            require(bddl_path.is_file(), f"missing pinned BDDL file: {bddl_path}")
            result.append(
                TaskInputs(
                    suite=suite_name,
                    task_id=task_id,
                    task=task,
                    official_states=official_states,
                    state_size=state_size,
                    official_hashes=official_hashes,
                    bddl_path=bddl_path,
                )
            )
    return tuple(result)


def build_bank(
    *,
    simulator_manifest: dict[str, Any],
    task_inputs: Sequence[TaskInputs],
    environment_class: Any,
    base_seed: int,
    states_per_task: int,
    max_attempts_per_task: int,
    render_gpu_device_id: int,
) -> tuple[dict[str, Any], dict[str, bytes]]:
    require(task_inputs, "at least one LIBERO task must be selected")
    artifacts: dict[str, bytes] = {}
    task_records: list[dict[str, Any]] = []
    bank_hashes: set[str] = set()
    for inputs in task_inputs:
        environment = environment_class(
            bddl_file_name=str(inputs.bddl_path),
            camera_heights=256,
            camera_widths=256,
            camera_names=list(CAMERA_NAMES),
            control_freq=20,
            render_gpu_device_id=render_gpu_device_id,
        )
        try:
            generated = generate_task_bank(
                environment,
                suite=inputs.suite,
                task_id=inputs.task_id,
                official_hashes=inputs.official_hashes,
                state_size=inputs.state_size,
                base_seed=base_seed,
                states_per_task=states_per_task,
                max_attempts_per_task=max_attempts_per_task,
                bank_hashes=bank_hashes,
            )
        finally:
            environment.close()
        path = task_artifact_path(inputs.suite, inputs.task_id)
        data = deterministic_npy_bytes(generated.states)
        artifacts[path] = data
        task_records.append(
            {
                "artifact": artifact_record(path, data, generated.states.shape),
                "bddl": {
                    "file": inputs.task.bddl_file,
                    "problem_folder": inputs.task.problem_folder,
                    "sha256": sha256_file(inputs.bddl_path),
                },
                "entries": list(generated.entries),
                "instruction": inputs.task.language,
                "official_states": {
                    "count": len(inputs.official_hashes),
                    "root_sha256": sequence_root_sha256(inputs.official_hashes),
                    "sha256": list(inputs.official_hashes),
                    "state_size": inputs.state_size,
                },
                "rejections": generated.rejections,
                "suite": inputs.suite,
                "task_id": inputs.task_id,
                "task_name": inputs.task.name,
            }
        )
    payload = {
        "base_seed": base_seed,
        "generator": {
            "capture_phase": CAPTURE_PHASE,
            "max_attempts_per_task": max_attempts_per_task,
            "sampler_seed_domain": SAMPLER_SEED_DOMAIN,
            "settle_action": OPEN_GRIPPER_NOOP.tolist(),
            "settle_steps": SETTLE_STEPS,
            "state_encoding": STATE_ENCODING,
            "validation_gates": list(VALIDATION_GATES),
        },
        "schema": DEV_STATE_SCHEMA,
        "simulator": {
            "assets_revision": simulator_manifest["assets"]["revision"],
            "egl_device": str(render_gpu_device_id),
            "packages": simulator_manifest["environment"]["packages"],
            "python": simulator_manifest["environment"]["python"],
            "schema": simulator_manifest["schema"],
            "source_revision": simulator_manifest["source"]["revision"],
        },
        "states_per_task": states_per_task,
        "tasks": task_records,
    }
    manifest = finalize_manifest(payload)
    validate_manifest_and_artifacts(manifest, artifacts)
    return manifest, artifacts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path, help="new output directory; existing paths are rejected")
    parser.add_argument("--suite", choices=(*SUITES, "all"), required=True)
    parser.add_argument("--task-ids", default="all", help="sorted task selection such as 0,2-4 or all")
    parser.add_argument("--base-seed", type=int, required=True, help="predeclared development reset sampler seed")
    parser.add_argument("--states-per-task", type=int, required=True)
    parser.add_argument(
        "--max-attempts-per-task",
        type=int,
        help="default: 100 times --states-per-task",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    require(0 <= args.base_seed < 2**63, "base-seed must be in [0, 2^63)")
    require(args.states_per_task > 0, "states-per-task must be positive")
    max_attempts = args.max_attempts_per_task or 100 * args.states_per_task
    require(max_attempts >= args.states_per_task, "max-attempts-per-task must cover states-per-task")
    suites = SUITES if args.suite == "all" else (args.suite,)
    task_ids = parse_selection(args.task_ids, upper=10)
    render_gpu_device_id = int(os.environ.get("MUJOCO_EGL_DEVICE_ID", "0"))
    require(render_gpu_device_id >= 0, "MUJOCO_EGL_DEVICE_ID must be nonnegative")

    simulator_manifest, benchmark, get_libero_path, environment_class = _validate_runtime()
    inputs = _task_inputs(
        benchmark,
        get_libero_path,
        suites=suites,
        task_ids=task_ids,
    )
    manifest, artifacts = build_bank(
        simulator_manifest=simulator_manifest,
        task_inputs=inputs,
        environment_class=environment_class,
        base_seed=args.base_seed,
        states_per_task=args.states_per_task,
        max_attempts_per_task=max_attempts,
        render_gpu_device_id=render_gpu_device_id,
    )
    manifest_sha256 = write_bank_exclusive(args.output_dir, manifest, artifacts)
    print(
        json.dumps(
            {
                "artifacts": len(artifacts),
                "manifest_sha256": manifest_sha256,
                "output_dir": str(args.output_dir.resolve()),
                "root_sha256": manifest["root_sha256"],
                "states": len(inputs) * args.states_per_task,
                "status": "ok",
                "tasks": len(inputs),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    try:
        main()
    except DevStateValidationError as exc:
        raise RuntimeError(f"development reset bank validation failed: {exc}") from exc
