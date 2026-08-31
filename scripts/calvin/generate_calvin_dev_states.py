#!/usr/bin/env python3
"""Replay a Python-3.11-exported bundle into held-out CALVIN reset states.

This script must run in the pinned Python 3.8 CALVIN simulator environment.  It
never imports the archive reader and never opens an episode path or archive.
It accepts only bundle members cross-bound to the current v4 manifest, v2
index, projected metadata, normalization artifact, and simulator sources.
"""

from __future__ import annotations

# ruff: noqa: UP006, UP035, UP045
import argparse
import json
import math
import os
import sys
import types
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_SOURCE_ROOT = _PROJECT_ROOT / "src"
if str(_SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(_SOURCE_ROOT))
# The modern package root eagerly imports dataclass(slots=True) model modules,
# which Python 3.8 cannot execute.  Expose only the package path in the pinned
# simulator process, then import the deliberately 3.8-compatible data module.
if sys.version_info < (3, 10) and "duo_vla" not in sys.modules:
    _legacy_package = types.ModuleType("duo_vla")
    _legacy_package.__path__ = [str(_SOURCE_ROOT / "duo_vla")]
    sys.modules["duo_vla"] = _legacy_package

from duo_vla.data.calvin_dev_states import (  # noqa: E402
    ABC_SCENES,
    BundledCalvinReplay,
    CalvinDevStateError,
    CalvinResetFrame,
    MaterializedCalvinReset,
    authenticate_dev_inputs,
    build_bank,
    candidate_rank_sha256,
    load_replay_bundle,
    rank_candidates,
    require_full_training_root,
    write_bank_exclusive,
)

EXPECTED_TABLE_ASSETS = {
    "calvin_scene_A": "calvin_table_A/urdf/calvin_table_A.urdf",
    "calvin_scene_B": "calvin_table_B/urdf/calvin_table_B.urdf",
    "calvin_scene_C": "calvin_table_C/urdf/calvin_table_C.urdf",
}
RESET_STATE_ATOL = 5e-3
RESET_EULER_SERIALIZATION_ATOL = 1e-12
# The authenticated A/B/C scene_obs layout is six scalar fixture states,
# followed by three movable-object poses encoded as [x, y, z, roll, pitch,
# yaw].  Euler coordinates are not uniquely defined at gimbal lock, so object
# orientations are checked as raw Bullet quaternions rather than coordinatewise.
SCENE_OBS_FIXTURE_SLICE = slice(0, 6)
SCENE_OBS_OBJECT_POSE_SLICES = (
    (slice(6, 9), slice(9, 12)),
    (slice(12, 15), slice(15, 18)),
    (slice(18, 21), slice(21, 24)),
)
DEFAULT_BASE_SEED = 20260829
DEFAULT_SMOKE_TASKS_PER_SCENE = 4


def require(condition: bool, message: str) -> None:
    if not condition:
        raise CalvinDevStateError(message)


def instantiate_abc_environment(
    training_root: Path,
    source_root: Path,
    scene: str,
    *,
    merged_config_bytes: bytes,
    scene_config_bytes: bytes,
) -> Any:
    """Instantiate one exact A/B/C scene without the pinned broken scene override."""

    require(scene in ABC_SCENES, "held-out development environment must be scene A/B/C")
    import calvin_env
    import hydra
    from omegaconf import OmegaConf

    require_full_training_root(training_root)
    imported_package = Path(calvin_env.__file__).resolve().parent
    expected_package = (source_root / "calvin_env" / "calvin_env").resolve()
    require(imported_package == expected_package, "imported calvin_env is not the authenticated pinned checkout")
    require(
        isinstance(merged_config_bytes, bytes) and bool(merged_config_bytes),
        "authenticated render config is missing",
    )
    require(isinstance(scene_config_bytes, bytes) and bool(scene_config_bytes), "authenticated scene config is missing")
    try:
        config = OmegaConf.create(merged_config_bytes.decode("utf-8"))
        scene_config = OmegaConf.create(scene_config_bytes.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise CalvinDevStateError("cannot decode authenticated CALVIN simulator config bytes") from exc
    scene_config["name"] = scene
    config.scene = scene_config
    require(config.scene.name == scene, "CALVIN scene replacement did not take effect")
    table_asset = str(config.scene.objects.fixed_objects.table.file)
    require(table_asset == EXPECTED_TABLE_ASSETS[scene], "CALVIN scene table asset mismatch")

    camera_names = set(config.cameras.keys())
    require({"static", "gripper"}.issubset(camera_names), "CALVIN render config lacks required cameras")
    for camera_name in list(camera_names - {"static", "gripper"}):
        del config.cameras[camera_name]
    require(set(config.cameras.keys()) == {"static", "gripper"}, "CALVIN camera pruning failed")

    environment = hydra.utils.instantiate(
        config.env,
        show_gui=False,
        use_vr=False,
        use_scene_info=True,
    )
    observed_table = str(environment.scene.object_cfg["fixed_objects"]["table"]["file"])
    require(
        observed_table == EXPECTED_TABLE_ASSETS[scene], "instantiated CALVIN scene is not the requested A/B/C scene"
    )
    expected_objects = tuple(config.scene.objects.movable_objects.keys())
    observed_objects = tuple(item.name for item in environment.scene.movable_objects)
    require(observed_objects == expected_objects, "CALVIN movable-object order differs from scene_obs encoding")
    return environment


def instantiate_task_oracle(source_root: Path, *, task_oracle_bytes: bytes) -> Any:
    import calvin_env
    import hydra
    from omegaconf import OmegaConf

    imported_package = Path(calvin_env.__file__).resolve().parent
    expected_package = (source_root / "calvin_env" / "calvin_env").resolve()
    require(imported_package == expected_package, "task oracle import is not from the authenticated pinned checkout")
    require(isinstance(task_oracle_bytes, bytes) and bool(task_oracle_bytes), "authenticated task oracle is missing")
    try:
        task_config = OmegaConf.create(task_oracle_bytes.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise CalvinDevStateError("cannot decode authenticated CALVIN task-oracle bytes") from exc
    return hydra.utils.instantiate(task_config)


def _unit_quaternion(values: Any, label: str) -> np.ndarray:
    quaternion = np.asarray(values, dtype=np.float64)
    require(quaternion.shape == (4,) and bool(np.isfinite(quaternion).all()), f"{label} quaternion is invalid")
    norm = float(np.linalg.norm(quaternion))
    require(math.isfinite(norm) and norm > np.finfo(np.float64).eps, f"{label} quaternion has zero norm")
    return quaternion / norm


def _quaternion_geodesic(first: Any, second: Any, label: str) -> float:
    first_unit = _unit_quaternion(first, label)
    second_unit = _unit_quaternion(second, label)
    # q and -q encode the same rotation.  The absolute dot product therefore
    # gives the shortest SO(3) arc, including at Euler branch cuts/gimbal lock.
    cosine_half_angle = float(np.clip(abs(np.dot(first_unit, second_unit)), 0.0, 1.0))
    return 2.0 * math.acos(cosine_half_angle)


def _validate_object_orientations(environment: Any, scene_obs: np.ndarray, frame: CalvinResetFrame) -> None:
    scene = getattr(environment, "scene", None)
    physics = getattr(environment, "p", None)
    cid = getattr(environment, "cid", None)
    require(
        scene is not None and physics is not None and isinstance(cid, (int, np.integer)),
        "CALVIN raw scene API is invalid",
    )
    require(
        getattr(scene, "p", None) is physics and getattr(scene, "cid", None) == cid,
        "CALVIN scene physics binding is invalid",
    )

    object_cfg = getattr(scene, "object_cfg", None)
    configured = object_cfg.get("movable_objects") if isinstance(object_cfg, Mapping) else None
    movable_objects = getattr(scene, "movable_objects", None)
    require(isinstance(configured, Mapping), "CALVIN movable-object config is invalid")
    require(isinstance(movable_objects, list), "CALVIN movable-object inventory is invalid")
    configured_names = tuple(configured.keys())
    observed_names = tuple(getattr(item, "name", None) for item in movable_objects)
    require(
        len(configured_names) == len(SCENE_OBS_OBJECT_POSE_SLICES) and observed_names == configured_names,
        "CALVIN movable-object order differs from the pinned scene_obs layout",
    )

    euler_to_quaternion = getattr(physics, "getQuaternionFromEuler", None)
    quaternion_to_euler = getattr(physics, "getEulerFromQuaternion", None)
    get_raw_pose = getattr(physics, "getBasePositionAndOrientation", None)
    require(
        callable(euler_to_quaternion) and callable(quaternion_to_euler) and callable(get_raw_pose),
        "CALVIN quaternion API is invalid",
    )
    for object_index, movable_object in enumerate(movable_objects):
        _position_slice, euler_slice = SCENE_OBS_OBJECT_POSE_SLICES[object_index]
        require(
            getattr(movable_object, "p", None) is physics and getattr(movable_object, "cid", None) == cid,
            "CALVIN movable object physics binding is invalid",
        )
        uid = getattr(movable_object, "uid", None)
        require(isinstance(uid, (int, np.integer)), "CALVIN movable object uid is invalid")
        raw_pose = get_raw_pose(uid, physicsClientId=int(cid))
        require(isinstance(raw_pose, (tuple, list)) and len(raw_pose) == 2, "CALVIN movable object pose is invalid")
        raw_quaternion = raw_pose[1]
        source_quaternion = euler_to_quaternion(frame.scene_obs[euler_slice])
        source_distance = _quaternion_geodesic(source_quaternion, raw_quaternion, "source/raw object")
        require(
            source_distance <= RESET_STATE_ATOL,
            "reset scene object orientation drifted beyond tolerance",
        )
        # Do not invert the serialized Euler observation at gimbal lock:
        # Bullet's forward quaternion->Euler projection intentionally loses a
        # small amount of rotational information there.  Re-run that exact
        # forward projection on the unchanged raw pose instead.
        serialized_euler = np.asarray(quaternion_to_euler(raw_quaternion), dtype=np.float64)
        require(
            serialized_euler.shape == (3,) and bool(np.isfinite(serialized_euler).all()),
            "CALVIN serialized object Euler orientation is invalid",
        )
        require(
            bool(
                np.allclose(
                    scene_obs[euler_slice],
                    serialized_euler,
                    rtol=0.0,
                    atol=RESET_EULER_SERIALIZATION_ATOL,
                )
            ),
            "reset scene Euler observation differs from Bullet serialization",
        )


def _validate_reset_observation(
    observation: Mapping[str, Any],
    frame: CalvinResetFrame,
    environment: Any,
) -> None:
    require(isinstance(observation, Mapping), "CALVIN reset did not return an observation object")
    robot_obs = np.asarray(observation.get("robot_obs"))
    scene_obs = np.asarray(observation.get("scene_obs"))
    require(robot_obs.shape == (15,) and bool(np.isfinite(robot_obs).all()), "reset robot_obs is invalid")
    require(scene_obs.shape == (24,) and bool(np.isfinite(scene_obs).all()), "reset scene_obs is invalid")
    require(robot_obs[14] == frame.robot_obs[14], "reset lost the previous gripper command")
    # The pinned robot reset reconstructs TCP pose from joint states and the env
    # advances one Bullet step.  Compare the directly restored coordinates, not
    # the source TCP coordinates that Robot.reset intentionally ignores.
    require(
        bool(np.allclose(robot_obs[6:14], frame.robot_obs[6:14], rtol=0.0, atol=RESET_STATE_ATOL)),
        "reset robot joints/gripper width drifted beyond tolerance",
    )
    linear_slices = (
        SCENE_OBS_FIXTURE_SLICE,
        *(position_slice for position_slice, _euler_slice in SCENE_OBS_OBJECT_POSE_SLICES),
    )
    require(
        all(
            bool(np.allclose(scene_obs[item], frame.scene_obs[item], rtol=0.0, atol=RESET_STATE_ATOL))
            for item in linear_slices
        ),
        "reset scene fixture/object position drifted beyond tolerance",
    )
    _validate_object_orientations(environment, scene_obs, frame)
    rgb_obs = observation.get("rgb_obs")
    require(isinstance(rgb_obs, Mapping), "CALVIN reset observation lacks rgb_obs")
    static = np.asarray(rgb_obs.get("rgb_static"))
    gripper = np.asarray(rgb_obs.get("rgb_gripper"))
    require(static.shape == (200, 200, 3) and static.dtype == np.uint8, "reset static RGB is invalid")
    require(gripper.shape == (84, 84, 3) and gripper.dtype == np.uint8, "reset gripper RGB is invalid")


def _restore_toggle_controller_state(environment: Any) -> None:
    """Restore controller-only state omitted by the pinned scene reset.

    Button and switch logical states are not part of ``scene_obs``.  Their
    attached light is, however, restored from that authenticated observation.
    Buttons additionally retain an edge-detector latch across reset.  Rebuild
    those hidden values only after the public reset observation has passed its
    source-state check.
    """

    scene = getattr(environment, "scene", None)
    require(scene is not None, "CALVIN environment has no raw scene interface")
    buttons = getattr(scene, "buttons", None)
    switches = getattr(scene, "switches", None)
    lights = getattr(scene, "lights", None)
    require(isinstance(buttons, list), "CALVIN scene buttons inventory is invalid")
    require(isinstance(switches, list), "CALVIN scene switches inventory is invalid")
    require(isinstance(lights, list), "CALVIN scene lights inventory is invalid")

    def restored_state(controller: Any, kind: str) -> Tuple[Any, int]:
        light = getattr(controller, "light", None)
        require(light is not None, f"CALVIN {kind} has no attached light")
        require(any(light is candidate for candidate in lights), f"CALVIN {kind} light is absent from the scene")
        get_state = getattr(light, "get_state", None)
        require(callable(get_state), f"CALVIN {kind} light has no state interface")
        light_state = get_state()
        require(
            isinstance(light_state, (int, np.integer)) and int(light_state) in (0, 1),
            f"CALVIN {kind} light state is not binary",
        )
        logical_state = int(light_state)
        current_state = getattr(controller, "state", None)
        require(current_state is not None and hasattr(current_state, "value"), f"CALVIN {kind} state is invalid")
        try:
            state = type(current_state)(logical_state)
        except (TypeError, ValueError) as exc:
            raise CalvinDevStateError(f"CALVIN {kind} state cannot represent its light state") from exc
        require(getattr(state, "value", None) == logical_state, f"CALVIN {kind} state conversion is invalid")
        return state, logical_state

    # Validate the complete inventory before mutating it so a source/API drift
    # cannot leave a partially synchronized environment.
    button_updates = []
    for button in buttons:
        state, _light_state = restored_state(button, "button")
        is_pressed = getattr(button, "_is_pressed", None)
        require(isinstance(is_pressed, (bool, np.bool_)), "CALVIN button pressed state is invalid")
        require(hasattr(button, "prev_is_pressed"), "CALVIN button edge-detector state is missing")
        button_updates.append((button, state, bool(is_pressed)))

    switch_updates = []
    for switch in switches:
        state, light_state = restored_state(switch, "switch")
        is_pressed = getattr(switch, "is_pressed", None)
        require(isinstance(is_pressed, (bool, np.bool_)), "CALVIN switch pressed state is invalid")
        require(
            int(bool(is_pressed)) == light_state,
            "CALVIN switch physical state disagrees with its authenticated light state",
        )
        switch_updates.append((switch, state))

    for button, state, is_pressed in button_updates:
        button.state = state
        button.prev_is_pressed = is_pressed
    for switch, state in switch_updates:
        switch.state = state


def strict_source_reset(environment: Any, frame: CalvinResetFrame) -> Mapping[str, Any]:
    """Restore source state plus hidden reset omissions in a fresh environment."""

    require(hasattr(environment, "robot"), "CALVIN environment has no raw robot interface")
    environment.robot.gripper_action = int(frame.robot_obs[14])
    observation = environment.reset(
        robot_obs=frame.robot_obs.copy(order="C"),
        scene_obs=frame.scene_obs.copy(order="C"),
    )
    _validate_reset_observation(observation, frame, environment)
    _restore_toggle_controller_state(environment)
    return observation


def replay_candidate(
    replay: BundledCalvinReplay,
    task_oracle: Any,
    environment_factory: Callable[[str], Any],
) -> Tuple[Optional[MaterializedCalvinReset], str]:
    """Require the recorded actions to solve the annotation from a fresh reset."""

    candidate = replay.candidate
    frame = replay.frame
    environment = environment_factory(candidate.scene)
    try:
        strict_source_reset(environment, frame)
        start_info = environment.get_info()
        initially_solved = task_oracle.get_task_info_for_set(start_info, start_info, {candidate.task})
        require(not initially_solved, "task oracle reports success before any action")
        for action_count, bundled_action in enumerate(replay.actions, start=1):
            action = np.asarray(bundled_action, dtype=np.float64).copy(order="C")
            original = action.copy()
            transition = environment.step(action.copy(order="C"))
            require(isinstance(transition, tuple) and len(transition) == 4, "CALVIN step must return a 4-tuple")
            _observation, reward, done, current_info = transition
            require(np.array_equal(action, original), "owned source action changed before environment.step")
            try:
                finite_reward = math.isfinite(float(reward))
            except (TypeError, ValueError) as exc:
                raise CalvinDevStateError("CALVIN replay reward is not numeric") from exc
            require(finite_reward, "CALVIN replay reward is non-finite")
            require(not bool(done), "CALVIN replay terminated unexpectedly")
            solved = task_oracle.get_task_info_for_set(start_info, current_info, {candidate.task})
            if solved:
                return (
                    MaterializedCalvinReset(
                        candidate=candidate,
                        frame=frame,
                        candidate_rank_sha256=candidate_rank_sha256(candidate, 0),
                        replay_actions=action_count,
                        replay_bundle_record_sha256=replay.record_sha256,
                    ),
                    "",
                )
        return None, "recorded annotation actions did not satisfy the pinned task oracle"
    except CalvinDevStateError as exc:
        return None, str(exc)
    finally:
        close = getattr(environment, "close", None)
        if callable(close):
            close()


def materialize_replay_valid_resets(
    replays: Sequence[BundledCalvinReplay],
    task_oracle: Any,
    environment_factory: Callable[[str], Any],
    base_seed: int,
) -> Tuple[Tuple[MaterializedCalvinReset, ...], Tuple[Dict[str, Any], ...]]:
    """Select exactly one replay-valid reset for every observed scene/task pair."""

    replay_by_candidate = {replay.candidate: replay for replay in replays}
    require(len(replay_by_candidate) == len(replays), "replay bundle contains duplicate candidates")
    ranked = rank_candidates(tuple(replay_by_candidate), base_seed)
    oracle_tasks = set(task_oracle.tasks)
    require(all(task in oracle_tasks for _scene, task in ranked), "annotation contains a task absent from the oracle")
    materialized = []
    rejections = []
    for (scene, task), values in sorted(ranked.items()):
        selected = None
        for candidate in values:
            rank_sha256 = candidate_rank_sha256(candidate, base_seed)
            replay = replay_by_candidate[candidate]
            candidate_result, reason = replay_candidate(
                replay,
                task_oracle,
                environment_factory,
            )
            if candidate_result is not None:
                selected = MaterializedCalvinReset(
                    candidate=candidate_result.candidate,
                    frame=candidate_result.frame,
                    candidate_rank_sha256=rank_sha256,
                    replay_actions=candidate_result.replay_actions,
                    replay_bundle_record_sha256=replay.record_sha256,
                )
                break
            rejections.append(
                {
                    "annotation_index": candidate.annotation_index,
                    "candidate_rank_sha256": rank_sha256,
                    "episode_index": candidate.episode_index,
                    "global_start": candidate.global_start,
                    "reason": reason,
                    "replay_bundle_record_sha256": replay.record_sha256,
                    "scene": scene,
                    "task": task,
                }
            )
        require(selected is not None, f"no replay-valid reset remains for {scene}/{task}")
        materialized.append(selected)
    return tuple(materialized), tuple(rejections)


def parse_args() -> argparse.Namespace:
    cache_root = Path(os.environ.get("DUO_VLA_CACHE_ROOT", "/root/.cache/duo-vla"))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-root", type=Path, default=cache_root / "data/calvin/task_ABC_D/training")
    parser.add_argument("--normalization", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, default=cache_root / "simulators/calvin")
    parser.add_argument("--revision-file", type=Path, default=_PROJECT_ROOT / "scripts/calvin/revisions.env")
    parser.add_argument("--replay-bundle", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-seed", type=int, default=DEFAULT_BASE_SEED)
    parser.add_argument("--smoke-tasks-per-scene", type=int, default=DEFAULT_SMOKE_TASKS_PER_SCENE)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    inputs = authenticate_dev_inputs(
        args.training_root,
        args.normalization,
        args.source_root,
        args.revision_file,
    )
    replay_manifest, replays = load_replay_bundle(args.replay_bundle, inputs)
    task_oracle = instantiate_task_oracle(
        args.source_root,
        task_oracle_bytes=inputs.source_files["task_oracle.yaml"],
    )

    def environment_factory(scene: str) -> Any:
        return instantiate_abc_environment(
            args.training_root,
            args.source_root,
            scene,
            merged_config_bytes=inputs.metadata["training/.hydra/merged_config.yaml"],
            scene_config_bytes=inputs.source_files["scene/" + scene + ".yaml"],
        )

    materialized, rejections = materialize_replay_valid_resets(
        replays,
        task_oracle,
        environment_factory,
        args.base_seed,
    )
    manifest, artifacts = build_bank(
        materialized,
        inputs.identity,
        inputs.split,
        replay_manifest,
        args.base_seed,
        smoke_tasks_per_scene=args.smoke_tasks_per_scene,
        rejections=rejections,
    )
    write_bank_exclusive(args.output_dir, manifest, artifacts)
    print(
        json.dumps(
            {
                "bank": str(args.output_dir.resolve()),
                "records": len(manifest["records"]),
                "rejections": len(manifest["rejections"]),
                "root_sha256": manifest["root_sha256"],
                "scenes": list(ABC_SCENES),
                "smoke_resets": len(manifest["selection"]["smoke_reset_indices"]),
                "status": "ok",
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
