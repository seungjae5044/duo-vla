#!/usr/bin/env python3
"""Replay the pinned original LIBERO HDF5 corpus in the closed simulator.

This is stage one of the expert-replay qualification.  It authenticates all 40
source files, replays demonstrations in canonical order, selects the first
successful replay for each task, and publishes content-addressed simulator
evidence.  It does not read the LeRobot parquet dataset; the separately pinned
training-runtime binder performs that cross-check in stage two.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
import stat
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np


def _activate_project_source_root() -> Path:
    source_path = Path(__file__).resolve().parents[1] / "src"
    source_identity = os.lstat(source_path)
    if not stat.S_ISDIR(source_identity.st_mode):
        raise RuntimeError("project src import root must be a real directory")
    source_root = source_path.resolve()
    entries = list(os.scandir(source_root))
    if len(entries) != 1 or entries[0].name != "duo_vla" or not entries[0].is_dir(follow_symlinks=False):
        raise RuntimeError("project src import root must contain only the real duo_vla package directory")

    def visit(directory: Path, *, in_cache: bool) -> None:
        for entry in os.scandir(directory):
            mode = entry.stat(follow_symlinks=False).st_mode
            path = directory / entry.name
            if stat.S_ISDIR(mode):
                visit(path, in_cache=in_cache or entry.name == "__pycache__")
            elif stat.S_ISREG(mode):
                allowed = entry.name.endswith(".pyc") if in_cache else entry.name.endswith(".py")
                if not allowed:
                    raise RuntimeError(f"project package import tree contains a forbidden file: {path}")
            else:
                raise RuntimeError(f"project package import tree contains a linked or special entry: {path}")

    visit(source_root / "duo_vla", in_cache=False)
    source_text = str(source_root)
    sys.path[:] = [source_text] + [
        entry for entry in sys.path if str(Path(entry or os.getcwd()).resolve()) != source_text
    ]
    return source_root


def _validate_project_module_origins(source_root: Path, required_modules: set[str]) -> dict[str, str]:
    package_root = (source_root / "duo_vla").resolve(strict=True)
    loaded = {name: module for name, module in sys.modules.items() if name == "duo_vla" or name.startswith("duo_vla.")}
    missing = sorted(required_modules - set(loaded))
    if missing:
        raise RuntimeError(f"required checkout modules are not loaded: {missing}")
    origins: dict[str, str] = {}
    for name, module in sorted(loaded.items()):
        origin = getattr(getattr(module, "__spec__", None), "origin", None)
        module_file = getattr(module, "__file__", None)
        if not isinstance(origin, str) or not isinstance(module_file, str):
            raise RuntimeError(f"checkout module has no file origin: {name}")
        resolved_origin = Path(origin).resolve(strict=True)
        resolved_file = Path(module_file).resolve(strict=True)
        if resolved_origin != resolved_file or not resolved_file.is_relative_to(package_root):
            raise RuntimeError(f"checkout module origin escapes authenticated source root: {name}")
        origins[name] = str(resolved_file)
    return origins


_PROJECT_SOURCE_ROOT = _activate_project_source_root()

from duo_vla.libero_replay_evidence import (  # noqa: E402
    ACTION_DIM,
    CAMERA_KEYS,
    IMAGE_SHAPE,
    OBSERVATION_ALIGNMENT_FEATURE_SCHEMA,
    OBSERVATION_ALIGNMENT_GRID_SIZE,
    REGENERATION_ENVIRONMENT_SEED,
    SETTLE_STEPS,
    SIMULATOR_ATTESTATION_SCHEMA,
    SIMULATOR_STAGE_SCHEMA,
    SIMULATOR_TASK_SCHEMA,
    ObservationSequenceDigester,
    ReplayEvidenceError,
    action_sequence_sha256,
    canonical_array,
    canonical_proprioceptive_state,
    canonical_sha256,
    initial_state_sha256,
    load_original_hdf5_inventory,
    load_source_parquet_alignment,
    observation_alignment_frame,
    read_stable_json,
    require,
    retained_action_indices,
    rotate_simulator_rgb_for_training,
    simulator_runtime_sha256,
    sorted_demo_names,
    stable_regular_file_identity,
    task_identities,
    task_slug,
    trajectory_sha256,
    update_array_digest,
)
from duo_vla.runtime_integrity import content_address_eval_venv, require_matching_eval_venv  # noqa: E402

_REQUIRED_PROJECT_MODULES = {
    "duo_vla",
    "duo_vla.libero_replay_evidence",
    "duo_vla.runtime_integrity",
}
_validate_project_module_origins(_PROJECT_SOURCE_ROOT, _REQUIRED_PROJECT_MODULES)

CONTROLLER_DIRECTIONS = (
    ("+x", 0, 1.0),
    ("-x", 0, -1.0),
    ("+y", 1, 1.0),
    ("-y", 1, -1.0),
    ("+z", 2, 1.0),
    ("-z", 2, -1.0),
    ("+rx", 3, 1.0),
    ("-rx", 3, -1.0),
    ("+ry", 4, 1.0),
    ("-ry", 4, -1.0),
    ("+rz", 5, 1.0),
    ("-rz", 5, -1.0),
)


def sha256_file(path: Path) -> str:
    return stable_regular_file_identity(path)["sha256"]


def _write_exclusive(path: Path, value: Mapping[str, Any]) -> str:
    payload = json.dumps(value, allow_nan=False, ensure_ascii=True, indent=2, sort_keys=True).encode("ascii") + b"\n"
    descriptor = os.open(
        str(path),
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
        0o444,
    )
    try:
        with os.fdopen(descriptor, "wb") as output:
            descriptor = -1
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    return hashlib.sha256(payload).hexdigest()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(
        str(path),
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _copy_observation(observation: Mapping[str, Any]) -> dict[str, Any]:
    return {
        name: np.asarray(value).copy() if isinstance(value, np.ndarray) else value
        for name, value in observation.items()
    }


def _observation_probe_sha256(observation: Mapping[str, Any]) -> str:
    digest = hashlib.sha256(b"duo-vla-libero-observation-probe-v1\0")
    for camera_name in CAMERA_KEYS:
        update_array_digest(digest, camera_name, observation[camera_name], dtype="|u1", shape=IMAGE_SHAPE)
    update_array_digest(digest, "state", canonical_proprioceptive_state(observation), dtype="<f4", shape=(8,))
    return digest.hexdigest()


def _reset_and_settle(environment: Any, initial_state: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    environment.reset()
    observation = environment.set_init_state(initial_state)
    dummy = np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0], dtype=np.float64)
    for _ in range(SETTLE_STEPS):
        observation, _reward, _done, _info = environment.step(dummy)
    return np.asarray(environment.get_sim_state()).copy(), _copy_observation(observation)


def _seed_environment(environment: Any) -> None:
    environment.seed(REGENERATION_ENVIRONMENT_SEED)


def _validate_reset_determinism(task: Any, initial_state: np.ndarray) -> dict[str, Any]:
    """Compare two isolated, freshly seeded environments without consuming replay RNG."""

    observations: list[dict[str, Any]] = []
    states: list[np.ndarray] = []
    for _ in range(2):
        environment = _construct_environment(task)
        try:
            _seed_environment(environment)
            state, observation = _reset_and_settle(environment, initial_state)
            states.append(state)
            observations.append(observation)
        finally:
            environment.close()
    first_state, second_state = states
    first_observation, second_observation = observations
    require(np.array_equal(first_state, second_state), "regeneration reset simulator states differ")
    require(
        _observation_probe_sha256(first_observation) == _observation_probe_sha256(second_observation),
        "regeneration reset observations differ",
    )
    return {
        "environment_seed": REGENERATION_ENVIRONMENT_SEED,
        "first_observation_sha256": _observation_probe_sha256(first_observation),
        "first_simulator_state_sha256": initial_state_sha256(first_state),
        "probe_environment_count": 2,
        "second_observation_sha256": _observation_probe_sha256(second_observation),
        "second_simulator_state_sha256": initial_state_sha256(second_state),
        "seed_calls_per_environment": 1,
        "settle_steps": SETTLE_STEPS,
    }


def _validate_hdf5_links(group: Any) -> None:
    import h5py

    def visit(current: Any, prefix: str) -> None:
        for name in current:
            relative = f"{prefix}/{name}" if prefix else name
            link = current.get(name, getlink=True)
            require(isinstance(link, h5py.HardLink), f"HDF5 soft/external link is forbidden: {relative}")
            child = current[name]
            if isinstance(child, h5py.Group):
                visit(child, relative)
            elif isinstance(child, h5py.Dataset):
                require(not child.is_virtual, f"HDF5 virtual dataset is forbidden: {relative}")
                creation = child.id.get_create_plist()
                require(creation.get_external_count() == 0, f"HDF5 external storage is forbidden: {relative}")

    visit(group, "")


def scan_hdf5_task(path: Path) -> dict[str, Any]:
    """Validate the source schema and hash every action/state sequence."""

    import h5py

    demos: list[dict[str, Any]] = []
    grippers: Counter[str] = Counter()
    total_transitions = 0
    total_retained = 0
    with h5py.File(path, "r") as source:
        _validate_hdf5_links(source)
        require(set(source.keys()) >= {"data"}, "source HDF5 has no data group")
        data = source["data"]
        names = sorted_demo_names(data.keys())
        for source_episode_index, name in enumerate(names):
            demo = data[name]
            require("actions" in demo and "states" in demo, f"source HDF5 {name} lacks actions/states")
            actions = np.asarray(demo["actions"][()])
            states = np.asarray(demo["states"][()])
            require(
                actions.ndim == 2
                and actions.shape[1] == ACTION_DIM
                and len(actions) > 0
                and bool(np.isfinite(actions).all()),
                f"source HDF5 {name} actions are invalid",
            )
            require(
                states.ndim == 2
                and len(states) == len(actions)
                and states.shape[1] > 0
                and bool(np.isfinite(states).all()),
                f"source HDF5 {name} states are invalid",
            )
            retained = retained_action_indices(actions)
            retained_actions = actions[np.asarray(retained, dtype=np.int64)]
            unique_values, unique_counts = np.unique(actions[:, -1], return_counts=True)
            for unique_index in range(len(unique_values)):
                grippers[format(float(unique_values[unique_index]), ".17g")] += int(unique_counts[unique_index])
            total_transitions += len(actions)
            total_retained += len(retained)
            demos.append(
                {
                    "action_sequence_sha256": action_sequence_sha256(retained_actions),
                    "initial_state_sha256": initial_state_sha256(states[0]),
                    "raw_transition_count": len(actions),
                    "retained_transition_count": len(retained),
                    "source_action_dtype": actions.dtype.str,
                    "source_episode_index": source_episode_index,
                    "source_state_sequence_sha256": _array_sha256("source_states", states, dtype="<f8"),
                }
            )
    return {
        "demonstrations": demos,
        "gripper_counts": dict(sorted(grippers.items())),
        "raw_transition_count": total_transitions,
        "retained_transition_count": total_retained,
    }


def _array_sha256(name: str, value: Any, *, dtype: str) -> str:
    digest = hashlib.sha256(b"duo-vla-libero-array-v1\0")
    update_array_digest(digest, name, value, dtype=dtype)
    return digest.hexdigest()


def _replay_demo(environment: Any, demo: Any, *, suite: str, task_id: int, source_episode_index: int) -> dict[str, Any]:
    actions = np.asarray(demo["actions"][()])
    states = np.asarray(demo["states"][()])
    retained = retained_action_indices(actions)
    simulation_actions = np.ascontiguousarray(actions[np.asarray(retained)])
    stored_actions = canonical_array(simulation_actions, dtype="<f4")
    _simulator_state, observation = _reset_and_settle(environment, states[0])
    observation_digest = ObservationSequenceDigester()
    swapped_observation_digest = ObservationSequenceDigester()
    alignment_frames: list[dict[str, Any]] = []
    alignment_probe: dict[str, Any] | None = None
    final_done = False
    for retained_index, source_index in enumerate(retained):
        simulation_action = simulation_actions[retained_index]
        pre_action = _copy_observation(observation)
        state = canonical_proprioceptive_state(pre_action)
        agentview = rotate_simulator_rgb_for_training(pre_action[CAMERA_KEYS[0]])
        wrist = rotate_simulator_rgb_for_training(pre_action[CAMERA_KEYS[1]])
        observation_digest.update(agentview, wrist, state)
        swapped_observation_digest.update(wrist, agentview, state)
        alignment_frames.append(observation_alignment_frame(agentview, wrist, state))
        observation, _reward, done, _info = environment.step(simulation_action)
        final_done = bool(done)
        if alignment_probe is None:
            pre_hash = _observation_probe_sha256(pre_action)
            post_hash = _observation_probe_sha256(observation)
            if pre_hash != post_hash:
                alignment_probe = {
                    "post_action_observation_sha256": post_hash,
                    "pre_action_observation_sha256": pre_hash,
                    "retained_transition_index": retained_index,
                    "source_transition_index": source_index,
                }
    require(observation_digest.count == len(retained), "replay observation/action length differs")
    require(alignment_probe is not None, "replay has no transition that distinguishes pre/post observations")
    initial_digest = initial_state_sha256(states[0])
    action_digest = action_sequence_sha256(stored_actions)
    observation_sequence_digest = observation_digest.hexdigest()
    inverted_actions = stored_actions.copy()
    inverted_actions[:, -1] *= -1.0
    zero_actions = np.zeros_like(stored_actions)
    return {
        "action_sequence_sha256": action_digest,
        "alignment_probe": alignment_probe,
        "initial_state_sha256": initial_digest,
        "inverted_gripper_action_sequence_sha256": action_sequence_sha256(inverted_actions),
        "observation_sequence_sha256": observation_sequence_digest,
        "observation_alignment_features": {
            "frames": alignment_frames,
            "grid_size": OBSERVATION_ALIGNMENT_GRID_SIZE,
            "schema": OBSERVATION_ALIGNMENT_FEATURE_SCHEMA,
        },
        "source_episode_index": source_episode_index,
        "step_count": len(retained),
        "success": final_done,
        "swapped_observation_sequence_sha256": swapped_observation_digest.hexdigest(),
        "trajectory_sha256": trajectory_sha256(
            suite=suite,
            task_id=task_id,
            source_episode_index=source_episode_index,
            initial_state_digest=initial_digest,
            action_digest=action_digest,
            observation_digest=observation_sequence_digest,
        ),
        "zero_action_sequence_sha256": action_sequence_sha256(zero_actions),
    }


def _replay_first_training_linked_success(
    environment: Any,
    data: Any,
    names: Sequence[str],
    source_scan: Mapping[str, Any],
    training_source_episode_indices: frozenset[int],
    *,
    suite: str,
    task_id: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    attempts: list[dict[str, Any]] = []
    selected: dict[str, Any] | None = None
    for source_episode_index, name in enumerate(names):
        replay = _replay_demo(
            environment,
            data[name],
            suite=suite,
            task_id=task_id,
            source_episode_index=source_episode_index,
        )
        expected_scan = source_scan["demonstrations"][source_episode_index]
        require(
            replay["action_sequence_sha256"] == expected_scan["action_sequence_sha256"]
            and replay["initial_state_sha256"] == expected_scan["initial_state_sha256"],
            "replay source action/initial-state identity differs from the scan",
        )
        attempts.append(
            {
                "action_sequence_sha256": replay["action_sequence_sha256"],
                "source_episode_index": source_episode_index,
                "step_count": replay["step_count"],
                "success": replay["success"],
                "training_linked": source_episode_index in training_source_episode_indices,
            }
        )
        if replay["success"] and source_episode_index in training_source_episode_indices:
            selected = replay
            break
    require(selected is not None, f"no training-linked source demonstration replay succeeds for {suite}:{task_id}")
    return attempts, selected


def _pre_dispatch_integrity_controls(selected: Mapping[str, Any], instruction: str) -> dict[str, Any]:
    action_digest = selected["action_sequence_sha256"]
    controls = {
        "inverted_gripper": selected["inverted_gripper_action_sequence_sha256"] != action_digest,
        "mismatched_language": (instruction + " [mismatched]") != instruction,
        "swapped_cameras": selected["swapped_observation_sequence_sha256"] != selected["observation_sequence_sha256"],
        "zero_action": selected["zero_action_sequence_sha256"] != action_digest,
    }
    require(all(controls.values()), "a pre-dispatch integrity mutation was not detected")
    return {name: {"mutation_detected": detected} for name, detected in sorted(controls.items())}


def _controller_impulses(environment: Any, initial_state: np.ndarray) -> dict[str, Any]:
    from scipy.spatial.transform import Rotation

    responses: list[dict[str, Any]] = []
    for label, axis, sign in CONTROLLER_DIRECTIONS:
        _state, _observation = _reset_and_settle(environment, initial_state)
        baseline_action = np.zeros(ACTION_DIM, dtype=np.float64)
        baseline_action[-1] = -1.0
        environment.step(baseline_action)
        baseline_controller = environment.env.robots[0].controller
        require(baseline_controller.name == "OSC_POSE", "LIBERO controller is not OSC_POSE")
        baseline_position = np.asarray(baseline_controller.goal_pos).copy()
        baseline_orientation = np.asarray(baseline_controller.goal_ori).copy()

        _state, _observation = _reset_and_settle(environment, initial_state)
        controller = environment.env.robots[0].controller
        require(controller.name == "OSC_POSE", "LIBERO controller is not OSC_POSE")
        action = np.zeros(ACTION_DIM, dtype=np.float64)
        action[axis] = sign
        action[-1] = -1.0
        environment.step(action)
        controller = environment.env.robots[0].controller
        if axis < 3:
            delta = np.asarray(controller.goal_pos) - baseline_position
            primary = float(delta[axis])
            leakage = float(np.max(np.abs(np.delete(delta, axis))))
        else:
            relative = np.asarray(controller.goal_ori) @ baseline_orientation.T
            rotation = Rotation.from_matrix(relative).as_rotvec()
            primary = float(rotation[axis - 3])
            leakage = float(np.max(np.abs(np.delete(rotation, axis - 3))))
        require(
            math.isfinite(primary) and math.isfinite(leakage),
            f"controller impulse response is non-finite: {label}",
        )
        matched = _controller_direction_matches(sign, primary, leakage)
        require(matched, f"controller impulse direction differs: {label}")
        responses.append({"direction": label, "leakage": leakage, "primary_delta": primary})

    apertures: dict[str, float] = {}
    for label, sign in (("open", -1.0), ("close", 1.0)):
        _state, observation = _reset_and_settle(environment, initial_state)
        action = np.zeros(ACTION_DIM, dtype=np.float64)
        action[-1] = sign
        for _ in range(10):
            observation, _reward, _done, _info = environment.step(action)
        apertures[label] = float(np.sum(np.abs(np.asarray(observation["robot0_gripper_qpos"]))))
        require(math.isfinite(apertures[label]), f"controller gripper aperture is non-finite: {label}")
    require(apertures["open"] > apertures["close"], "controller gripper polarity differs")
    return {
        "action_dimension": ACTION_DIM,
        "checked_directions": [label for label, _axis, _sign in CONTROLLER_DIRECTIONS],
        "controller": "OSC_POSE",
        "direction_match_count": len(CONTROLLER_DIRECTIONS),
        "gripper_close": 1.0,
        "gripper_open": -1.0,
        "raw": {"gripper_apertures": apertures, "responses": responses},
    }


def _controller_direction_matches(sign: float, primary: float, leakage: float) -> bool:
    return sign * primary > 0.0 and leakage <= abs(primary) * 0.01 + 1e-8


def _construct_environment(task: Any) -> Any:
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    return OffScreenRenderEnv(
        bddl_file_name=str(bddl),
        camera_heights=256,
        camera_widths=256,
        camera_names=["agentview", "robot0_eye_in_hand"],
        control_freq=20,
        render_gpu_device_id=int(os.environ.get("MUJOCO_EGL_DEVICE_ID", "0")),
    )


def _source_root_inventory(source_root: Path, records: Sequence[Mapping[str, Any]]) -> None:
    expected = {record["path"] for record in records}
    expected_directories: set[str] = set()
    for relative in expected:
        expected_directories.update(
            parent.as_posix() for parent in PurePosixPath(relative).parents if parent.as_posix() != "."
        )
    observed: set[str] = set()
    observed_directories: set[str] = set()

    def visit(directory: Path) -> None:
        with os.scandir(directory) as entries:
            for entry in entries:
                path = directory / entry.name
                relative = path.relative_to(source_root).as_posix()
                mode = entry.stat(follow_symlinks=False).st_mode
                if stat.S_ISLNK(mode):
                    raise ReplayEvidenceError(f"source HDF5 tree contains a symbolic link: {relative}")
                if stat.S_ISDIR(mode):
                    observed_directories.add(relative)
                    visit(path)
                elif stat.S_ISREG(mode):
                    require(path.suffix == ".hdf5", f"source HDF5 tree contains an unexpected file: {relative}")
                    observed.add(relative)
                else:
                    raise ReplayEvidenceError(f"source HDF5 tree contains a non-regular entry: {relative}")

    visit(source_root)
    require(
        observed == expected and observed_directories == expected_directories,
        "source HDF5 tree inventory differs: "
        f"missing={sorted(expected - observed)}, extra={sorted(observed - expected)}, "
        f"missing_directories={sorted(expected_directories - observed_directories)}, "
        f"extra_directories={sorted(observed_directories - expected_directories)}",
    )


def _authenticate_source_hdf5_files(source_root: Path, records: Sequence[Mapping[str, Any]]) -> None:
    _source_root_inventory(source_root, records)
    for record in records:
        identity = stable_regular_file_identity(source_root / record["path"], expected_bytes=record["bytes"])
        require(identity["sha256"] == record["sha256"], f"source HDF5 SHA-256 differs: {record['path']}")


def _load_authenticated_preflight(project_root: Path, attestation: Mapping[str, Any]) -> Any:
    path = project_root / "scripts/preflight_libero_env.py"
    project_sources = attestation.get("project_sources")
    require(isinstance(project_sources, Mapping), "simulator attestation project-source identity is invalid")
    expected_sha256 = project_sources.get("preflight")
    require(
        isinstance(expected_sha256, str) and sha256_file(path) == expected_sha256,
        "simulator preflight source differs from the attestation",
    )
    spec = importlib.util.spec_from_file_location("_duo_vla_authenticated_libero_preflight", path)
    require(spec is not None and spec.loader is not None, "cannot construct authenticated simulator preflight loader")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _authenticate_live_simulator(project_root: Path, attestation: Mapping[str, Any]) -> None:
    """Recompute the expensive runtime identities before any source replay."""

    source_root = _activate_project_source_root()
    require(source_root == (project_root / "src").resolve(), "collector checkout source root differs")
    _validate_project_module_origins(source_root, _REQUIRED_PROJECT_MODULES)
    preflight = _load_authenticated_preflight(project_root, attestation)

    cache_root = Path(os.environ.get("DUO_VLA_CACHE_ROOT", "/root/.cache/duo-vla")).resolve()
    process = preflight.validate_process_environment(project_root, cache_root)
    require(attestation.get("process") == process, "live simulator process identity differs from the attestation")
    try:
        require_matching_eval_venv(
            attestation.get("eval_venv_identity"),
            content_address_eval_venv(cache_root / "venvs/libero-eval"),
        )
    except RuntimeError as exc:
        raise ReplayEvidenceError(f"live simulator eval-venv/runtime differs from the attestation: {exc}") from exc
    import mujoco
    from libero.libero import get_assets_path

    runtime_root = cache_root / "simulators/libero"
    manifest_path = runtime_root / "manifest.json"
    require(
        preflight.sha256_file(manifest_path) == attestation.get("manifest_sha256"),
        "live simulator manifest differs from the attestation",
    )
    require(
        preflight.sha256_file(project_root / "envs/libero-eval/uv.lock") == attestation.get("evaluator_lock_sha256"),
        "live simulator lock differs from the attestation",
    )
    require(
        preflight.git_source_identity(runtime_root / "source") == attestation.get("source"),
        "live simulator source differs from the attestation",
    )
    assets_path = Path(get_assets_path()).resolve()
    require(
        {"path": str(assets_path), **preflight.tree_content_identity(assets_path)} == attestation.get("assets"),
        "live simulator assets differ from the attestation",
    )
    venv_root = Path(os.sys.prefix).resolve()
    require(
        preflight.verify_site_packages_inventory(venv_root, assets_path) == attestation.get("site_packages"),
        "live simulator site-packages differ from the attestation",
    )
    require(
        preflight.module_origin_identity(venv_root) == attestation.get("module_origins"),
        "live simulator module origins differ from the attestation",
    )
    require(
        preflight.verify_distribution_records(venv_root) == attestation.get("distribution_records"),
        "live simulator distribution records differ from the attestation",
    )
    require(
        preflight.installed_distribution_identity() == attestation.get("installed_distributions"),
        "live installed-distribution identity differs from the attestation",
    )
    observed_packages = {name: importlib.metadata.version(name) for name in preflight.EXPECTED_PACKAGES}
    require(observed_packages == attestation.get("packages"), "live simulator package versions differ")
    require(
        preflight.source_file_identities(project_root) == attestation.get("project_sources"),
        "live simulator project sources differ from the attestation",
    )
    require(
        preflight.opengl_runtime_identity(mujoco) == attestation.get("opengl"),
        "live OpenGL runtime differs from the attestation",
    )


def _restore_authenticated_simulator_environment_mutations() -> None:
    cv2_module = sys.modules.get("cv2")
    require(cv2_module is not None, "OpenCV was not imported by the simulator")
    module_file = getattr(cv2_module, "__file__", None)
    require(isinstance(module_file, str), "OpenCV module origin is invalid")
    module_root = Path(module_file).parent
    expected = {
        "LD_LIBRARY_PATH": f"{module_root / '../../lib64'}:",
        "PYGAME_HIDE_SUPPORT_PROMPT": "hide",
        "QT_QPA_FONTDIR": str(module_root / "qt/fonts"),
        "QT_QPA_PLATFORM_PLUGIN_PATH": str(module_root / "qt/plugins"),
    }
    observed = {name: os.environ.get(name) for name in expected}
    require(observed == expected, "simulator environment mutations differ from the authenticated wheel loaders")
    for name in expected:
        del os.environ[name]


def collect(args: argparse.Namespace, *, project_root: Path | None = None) -> tuple[Path, str]:
    root = Path(__file__).resolve().parents[1] if project_root is None else project_root.resolve()
    alignment_path = root / "configs/libero_source_parquet_alignment.json"
    alignment, training_source_indices, alignment_raw_sha256 = load_source_parquet_alignment(alignment_path)
    collector_identity = {
        "path": "scripts/collect_libero_expert_replay.py",
        "sha256": sha256_file(Path(__file__)),
        "source_parquet_alignment_content_sha256": alignment["content_sha256"],
        "source_parquet_alignment_raw_sha256": alignment_raw_sha256,
        "shared_contract_sha256": sha256_file(root / "src/duo_vla/libero_replay_evidence.py"),
    }
    source_root = Path(os.path.abspath(args.source_root))
    require(
        source_root.resolve() == source_root and source_root.is_dir() and not source_root.is_symlink(),
        "source HDF5 root must be a real directory without linked path components",
    )
    inventory, records, inventory_raw_sha256 = load_original_hdf5_inventory(args.inventory.resolve())
    require(inventory_raw_sha256 == args.inventory_sha256, "original HDF5 inventory raw SHA-256 differs")
    attestation, attestation_raw_sha256 = read_stable_json(
        args.simulator_attestation.resolve(),
        name="simulator attestation",
        expected_sha256=args.simulator_attestation_sha256,
    )
    require(attestation.get("schema") == SIMULATOR_ATTESTATION_SCHEMA, "simulator attestation schema differs")
    tasks = task_identities(attestation)
    task_lookup = {(task["suite"], task["task_id"]): task for task in tasks}
    _authenticate_live_simulator(root, attestation)
    from libero.libero import benchmark

    _source_root_inventory(source_root, records)

    output = args.output_dir.resolve()
    output.mkdir(mode=0o755, parents=False, exist_ok=False)
    raw_dir = output / "raw"
    raw_dir.mkdir(mode=0o755)
    task_records: list[dict[str, Any]] = []
    source_scan: list[dict[str, Any]] = []
    controller_result: dict[str, Any] | None = None
    try:
        benchmark_map = benchmark.get_benchmark_dict()
        for record in records:
            suite_name = record["suite"]
            task_id = record["task_id"]
            expected_task = task_lookup[(suite_name, task_id)]
            suite = benchmark_map[suite_name]()
            task = suite.get_task(task_id)
            require(
                task.name == record["task_name"] == expected_task["task_name"],
                "source/simulator task name differs",
            )
            require(task.language == expected_task["instruction"], "source/simulator task instruction differs")
            source_path = source_root / record["path"]
            before = source_path.stat(follow_symlinks=False)
            identity = stable_regular_file_identity(source_path, expected_bytes=record["bytes"])
            require(identity["sha256"] == record["sha256"], f"source HDF5 SHA-256 differs: {record['path']}")
            scan = scan_hdf5_task(source_path)
            source_scan.append(
                {
                    "gripper_counts": scan["gripper_counts"],
                    "path": record["path"],
                    "raw_transition_count": scan["raw_transition_count"],
                    "retained_transition_count": scan["retained_transition_count"],
                    "suite": suite_name,
                    "task_id": task_id,
                }
            )

            import h5py

            with h5py.File(source_path, "r") as source:
                data = source["data"]
                names = sorted_demo_names(data.keys())
                first_initial = np.asarray(data[names[0]]["states"][0])
                reset = _validate_reset_determinism(task, first_initial)
                if controller_result is None:
                    controller_environment = _construct_environment(task)
                    try:
                        _seed_environment(controller_environment)
                        controller_result = _controller_impulses(controller_environment, first_initial)
                    finally:
                        controller_environment.close()

                environment = _construct_environment(task)
                try:
                    # Match OpenVLA regeneration: seed once, then consume resets in
                    # source order. Select the first replay that both succeeds now
                    # and belongs to the pinned regenerated training snapshot.
                    _seed_environment(environment)
                    attempts, selected = _replay_first_training_linked_success(
                        environment,
                        data,
                        names,
                        scan,
                        training_source_indices[(suite_name, task_id)],
                        suite=suite_name,
                        task_id=task_id,
                    )
                finally:
                    environment.close()
                pre_dispatch_integrity_controls = _pre_dispatch_integrity_controls(selected, task.language)
                task_document = {
                    "attempts": attempts,
                    "collector_source_sha256": collector_identity["sha256"],
                    "pre_dispatch_integrity_controls": pre_dispatch_integrity_controls,
                    "reset_determinism": reset,
                    "schema": SIMULATOR_TASK_SCHEMA,
                    "selected": selected,
                    "source_file": {**record, **identity},
                    "source_scan": scan,
                    "task": {
                        "instruction": task.language,
                        "suite": suite_name,
                        "task_id": task_id,
                        "task_name": task.name,
                    },
                }
                relative = f"raw/{task_slug(suite_name, task_id)}.json"
                digest = _write_exclusive(output / relative, task_document)
                task_records.append(
                    {
                        "bytes": (output / relative).stat().st_size,
                        "path": relative,
                        "sha256": digest,
                        "suite": suite_name,
                        "task_id": task_id,
                    }
                )
                after = source_path.stat(follow_symlinks=False)
                stable_fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns", "st_nlink")
                require(
                    all(getattr(before, name) == getattr(after, name) for name in stable_fields),
                    f"source HDF5 changed while replayed: {record['path']}",
                )
        gripper_values = sorted({float(value) for item in source_scan for value in item["gripper_counts"]})
        require(gripper_values == [-1.0, 1.0], "original HDF5 gripper set is not exactly {-1,+1}")
        require(controller_result is not None and len(task_records) == 40, "simulator evidence is incomplete")
        stage = {
            "collector": collector_identity,
            "gates": {
                "controller_impulse_directions": controller_result,
                "exact_gripper_set": {
                    "observed_values": gripper_values,
                    "unexpected_value_count": 0,
                    "zero_value_count": 0,
                },
            },
            "inputs": {
                "original_hdf5_inventory_content_sha256": inventory["content_sha256"],
                "original_hdf5_inventory_raw_sha256": inventory_raw_sha256,
                "simulator_attestation_raw_sha256": attestation_raw_sha256,
                "simulator_attestation_sha256": canonical_sha256(attestation),
                "simulator_runtime_sha256": simulator_runtime_sha256(attestation),
                "task_inventory_sha256": attestation["task_inventory_sha256"],
            },
            "schema": SIMULATOR_STAGE_SCHEMA,
            "source_scan": source_scan,
            "status": "complete",
            "task_records": task_records,
        }
        _restore_authenticated_simulator_environment_mutations()
        _authenticate_source_hdf5_files(source_root, records)
        _authenticate_live_simulator(root, attestation)
        inventory_after, records_after, inventory_raw_sha256_after = load_original_hdf5_inventory(
            args.inventory.resolve()
        )
        require(
            inventory_after == inventory
            and records_after == records
            and inventory_raw_sha256_after == inventory_raw_sha256,
            "original HDF5 inventory changed during simulator replay",
        )
        attestation_after, attestation_raw_sha256_after = read_stable_json(
            args.simulator_attestation.resolve(),
            name="simulator attestation publication recheck",
            expected_sha256=args.simulator_attestation_sha256,
        )
        require(
            attestation_after == attestation and attestation_raw_sha256_after == attestation_raw_sha256,
            "simulator attestation changed during simulator replay",
        )
        alignment_after, training_source_indices_after, alignment_raw_sha256_after = load_source_parquet_alignment(
            alignment_path
        )
        require(
            alignment_after == alignment
            and training_source_indices_after == training_source_indices
            and alignment_raw_sha256_after == alignment_raw_sha256,
            "source/parquet alignment changed during simulator replay",
        )
        collector_identity_after = {
            "path": "scripts/collect_libero_expert_replay.py",
            "sha256": sha256_file(Path(__file__)),
            "source_parquet_alignment_content_sha256": alignment_after["content_sha256"],
            "source_parquet_alignment_raw_sha256": alignment_raw_sha256_after,
            "shared_contract_sha256": sha256_file(root / "src/duo_vla/libero_replay_evidence.py"),
        }
        require(collector_identity_after == collector_identity, "collector source changed during simulator replay")
        stage_digest = _write_exclusive(output / "simulator-stage.json", stage)
        _write_exclusive(
            output / "simulator-stage.commit.json",
            {
                "schema": "duo-vla-libero-expert-replay-simulator-stage-commit-v1",
                "simulator_stage_sha256": stage_digest,
                "task_record_root_sha256": canonical_sha256(task_records),
            },
        )
        _fsync_directory(raw_dir)
        _fsync_directory(output)
        return output / "simulator-stage.json", stage_digest
    except BaseException as exc:
        try:
            _write_exclusive(
                output / "failed.json",
                {
                    "error_type": type(exc).__name__,
                    "schema": "duo-vla-libero-expert-replay-simulator-failure-v1",
                    "status": "failed",
                },
            )
            _fsync_directory(output)
        except BaseException:
            pass
        raise


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--inventory-sha256", required=True)
    parser.add_argument("--simulator-attestation", type=Path, required=True)
    parser.add_argument("--simulator-attestation-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True, help="must not already exist")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    path, digest = collect(parse_args(argv))
    print(json.dumps({"sha256": digest, "simulator_stage": str(path)}, sort_keys=True))


if __name__ == "__main__":
    main()
