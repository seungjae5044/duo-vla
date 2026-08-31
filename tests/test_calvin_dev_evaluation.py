from __future__ import annotations

import json
import sys
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "calvin"))

from evaluate_calvin_dev import (
    DevelopmentJournal,
    _close_fresh_environment,
    calvin_env_action,
    canonical_sha256,
    evaluate_reset_indices,
    rollout_reset,
    summarize_development,
    validate_live_policy,
)
from generate_calvin_dev_states import (
    RESET_STATE_ATOL,
    _quaternion_geodesic,
    _validate_reset_observation,
    replay_candidate,
    strict_source_reset,
)

from duo_vla.data.calvin_dev_states import (
    ABC_SCENES,
    AuthenticatedCalvinDevInputs,
    BundledCalvinReplay,
    CalvinDevCandidate,
    CalvinDevStateError,
    CalvinResetFrame,
)


class _Robot:
    def __init__(self) -> None:
        self.gripper_action = 1


def _euler_quaternion(euler: Any) -> np.ndarray:
    roll, pitch, yaw = np.asarray(euler, dtype=np.float64)
    cr, sr = np.cos(roll / 2), np.sin(roll / 2)
    cp, sp = np.cos(pitch / 2), np.sin(pitch / 2)
    cy, sy = np.cos(yaw / 2), np.sin(yaw / 2)
    return np.asarray(
        [
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
            cr * cp * cy + sr * sp * sy,
        ],
        dtype=np.float64,
    )


class _Physics:
    def __init__(self) -> None:
        self.poses: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        self.serialized_eulers: dict[bytes, np.ndarray] = {}

    @staticmethod
    def _quaternion_key(quaternion: Any) -> bytes:
        return np.asarray(quaternion, dtype=np.float64).tobytes(order="C")

    def set_pose(
        self,
        uid: int,
        position: Any,
        quaternion: Any,
        serialized_euler: Any,
    ) -> None:
        position_array = np.asarray(position, dtype=np.float64).copy()
        quaternion_array = np.asarray(quaternion, dtype=np.float64).copy()
        self.poses[uid] = (position_array, quaternion_array)
        self.serialized_eulers[self._quaternion_key(quaternion_array)] = np.asarray(
            serialized_euler,
            dtype=np.float64,
        ).copy()

    def getQuaternionFromEuler(self, euler: Any) -> np.ndarray:
        return _euler_quaternion(euler)

    def getEulerFromQuaternion(self, quaternion: Any) -> np.ndarray:
        return self.serialized_eulers[self._quaternion_key(quaternion)].copy()

    def getBasePositionAndOrientation(
        self,
        uid: int,
        *,
        physicsClientId: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        assert physicsClientId == 7
        return self.poses[uid]


class _MovableObject:
    def __init__(self, name: str, uid: int) -> None:
        self.name = name
        self.uid = uid
        self.p: _Physics | None = None
        self.cid: int | None = None


class _BinaryState(Enum):
    ON = 1
    OFF = 0


class _Light:
    def __init__(self, state: int) -> None:
        self.state = state

    def get_state(self) -> int:
        return self.state


class _Button:
    def __init__(self, light: _Light | None, *, is_pressed: bool) -> None:
        self.light = light
        self.state = _BinaryState.OFF
        self.prev_is_pressed = not is_pressed
        self._pressed = is_pressed

    @property
    def _is_pressed(self) -> bool:
        return self._pressed


class _Switch:
    def __init__(self, light: _Light | None, *, is_pressed: bool) -> None:
        self.light = light
        self.state = _BinaryState.OFF
        self._pressed = is_pressed

    @property
    def is_pressed(self) -> bool:
        return self._pressed


class _Scene:
    def __init__(
        self,
        *,
        buttons: list[_Button] | None = None,
        switches: list[_Switch] | None = None,
        lights: list[_Light] | None = None,
        movable_names: tuple[str, ...] = ("block_pink", "block_blue", "block_red"),
    ) -> None:
        self.buttons = [] if buttons is None else buttons
        self.switches = [] if switches is None else switches
        self.lights = [] if lights is None else lights
        self.object_cfg = {"movable_objects": {name: {} for name in movable_names}}
        self.movable_objects = [_MovableObject(name, uid=index + 20) for index, name in enumerate(movable_names)]
        self.p: _Physics | None = None
        self.cid: int | None = None

    def bind_physics(self, physics: _Physics, cid: int) -> None:
        self.p = physics
        self.cid = cid
        for movable_object in self.movable_objects:
            movable_object.p = physics
            movable_object.cid = cid


class _Environment:
    def __init__(self, scene: str, solve_after: int, *, raw_scene: _Scene | None = None) -> None:
        self.scene_name = scene
        self.p = _Physics()
        self.cid = 7
        self.scene = _Scene() if raw_scene is None else raw_scene
        self.scene.bind_physics(self.p, self.cid)
        self.solve_after = solve_after
        self.robot = _Robot()
        self.progress = 0
        self.closed = False
        self.reset_robot: np.ndarray | None = None
        self.reset_scene: np.ndarray | None = None
        self.actions: list[np.ndarray] = []

    def _observation(self) -> dict[str, Any]:
        assert self.reset_robot is not None and self.reset_scene is not None
        robot = self.reset_robot.copy()
        robot[14] = self.robot.gripper_action
        return {
            "rgb_obs": {
                "rgb_gripper": np.zeros((84, 84, 3), dtype=np.uint8),
                "rgb_static": np.zeros((200, 200, 3), dtype=np.uint8),
            },
            "robot_obs": robot,
            "scene_obs": self.reset_scene.copy(),
        }

    def reset(self, *, robot_obs: np.ndarray, scene_obs: np.ndarray) -> dict[str, Any]:
        assert robot_obs.flags.owndata and scene_obs.flags.owndata
        self.reset_robot = robot_obs.copy()
        self.reset_scene = scene_obs.copy()
        position_slices = (slice(6, 9), slice(12, 15), slice(18, 21))
        euler_slices = (slice(9, 12), slice(15, 18), slice(21, 24))
        for object_index, movable_object in enumerate(self.scene.movable_objects):
            position_slice = position_slices[object_index]
            euler_slice = euler_slices[object_index]
            self.p.set_pose(
                movable_object.uid,
                scene_obs[position_slice],
                self.p.getQuaternionFromEuler(scene_obs[euler_slice]),
                scene_obs[euler_slice],
            )
        self.progress = 0
        return self._observation()

    def get_info(self) -> dict[str, Any]:
        return {"progress": self.progress, "scene": self.scene_name}

    def step(self, action: np.ndarray) -> tuple[dict[str, Any], float, bool, dict[str, Any]]:
        self.actions.append(action.copy())
        # The pinned simulator scales the first six channels in place.  Tests
        # deliberately emulate that ownership hazard.
        action[:6] *= 0.02
        self.progress += 1
        return self._observation(), 0.0, False, self.get_info()

    def close(self) -> None:
        self.closed = True


class _PinnedLifecycleEnvironment:
    """Emulate the pinned simulator's non-idempotent destructor contract."""

    def __init__(self) -> None:
        self.cid = 0
        self.connected = True
        self.disconnects = 0
        self.ownsPhysicsClient = True

    def close(self) -> None:
        if not self.connected:
            raise RuntimeError("Not connected to physics server")
        self.connected = False
        self.disconnects += 1

    def __del__(self) -> None:
        if self.ownsPhysicsClient:
            self.close()


class _Oracle:
    def __init__(self, solve_after: int) -> None:
        self.tasks = {"fixture_task": object()}
        self.solve_after = solve_after
        self.calls: list[tuple[dict[str, Any], dict[str, Any], set[str]]] = []

    def get_task_info_for_set(
        self,
        start_info: dict[str, Any],
        end_info: dict[str, Any],
        task_filter: set[str],
    ) -> set[str]:
        self.calls.append((start_info, end_info, set(task_filter)))
        if end_info["progress"] - start_info["progress"] >= self.solve_after:
            return set(task_filter)
        return set()


class _Client:
    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []

    def predict(self, **fields: Any) -> tuple[np.ndarray, dict[str, Any]]:
        self.requests.append(fields)
        actions = np.zeros((8, 7), dtype=np.float32)
        actions[:, 0] = 2.0
        actions[:, 6] = 1.0
        return actions, {"fixture": True}


def _frame(gripper: float = -1.0) -> CalvinResetFrame:
    robot = np.linspace(0.01, 0.15, 15, dtype=np.float64)
    robot[14] = gripper
    scene = np.linspace(0.0, 0.23, 24, dtype=np.float64)
    return CalvinResetFrame(robot_obs=robot, scene_obs=scene, source_frame_sha256="f" * 64)


def _record(scene: str = "calvin_scene_A", index: int = 0) -> dict[str, Any]:
    return {
        "annotation_index": 9 + index,
        "episode_index": 3 + index,
        "global_start": 100 + index,
        "instruction": "perform the fixture task",
        "reset_id_sha256": f"{index + 1:064x}",
        "scene": scene,
        "source_frame_sha256": "f" * 64,
        "task": "fixture_task",
    }


def test_strict_reset_restores_robot_scene_and_previous_gripper() -> None:
    environment = _Environment("calvin_scene_A", solve_after=1)
    frame = _frame(-1.0)
    observation = strict_source_reset(environment, frame)

    assert environment.robot.gripper_action == -1
    assert np.array_equal(environment.reset_robot, frame.robot_obs)
    assert np.array_equal(environment.reset_scene, frame.scene_obs)
    assert observation["robot_obs"][14] == -1.0


def test_strict_reset_restores_button_toggle_and_edge_latch_from_authenticated_light() -> None:
    light = _Light(1)
    button = _Button(light, is_pressed=True)
    environment = _Environment(
        "calvin_scene_A",
        solve_after=1,
        raw_scene=_Scene(buttons=[button], lights=[light]),
    )

    strict_source_reset(environment, _frame(-1.0))

    assert button.state is _BinaryState.ON
    assert button.prev_is_pressed is True


@pytest.mark.parametrize("light_state,is_pressed", [(0, False), (1, True)])
def test_strict_reset_restores_switch_state_when_physics_and_authenticated_light_agree(
    light_state: int,
    is_pressed: bool,
) -> None:
    light = _Light(light_state)
    switch = _Switch(light, is_pressed=is_pressed)
    environment = _Environment(
        "calvin_scene_A",
        solve_after=1,
        raw_scene=_Scene(switches=[switch], lights=[light]),
    )

    strict_source_reset(environment, _frame(-1.0))

    assert switch.state is _BinaryState(light_state)


def test_strict_reset_fails_closed_on_switch_light_disagreement_without_partial_mutation() -> None:
    button_light = _Light(1)
    button = _Button(button_light, is_pressed=True)
    switch_light = _Light(1)
    switch = _Switch(switch_light, is_pressed=False)
    environment = _Environment(
        "calvin_scene_A",
        solve_after=1,
        raw_scene=_Scene(buttons=[button], switches=[switch], lights=[button_light, switch_light]),
    )

    with pytest.raises(CalvinDevStateError, match="switch physical state disagrees"):
        strict_source_reset(environment, _frame(-1.0))

    assert button.state is _BinaryState.OFF
    assert button.prev_is_pressed is False
    assert switch.state is _BinaryState.OFF


def test_strict_reset_fails_closed_when_toggle_light_is_not_scene_authenticated() -> None:
    attached_light = _Light(1)
    button = _Button(attached_light, is_pressed=False)
    environment = _Environment(
        "calvin_scene_A",
        solve_after=1,
        raw_scene=_Scene(buttons=[button], lights=[_Light(1)]),
    )

    with pytest.raises(CalvinDevStateError, match="button light is absent"):
        strict_source_reset(environment, _frame(-1.0))


@pytest.mark.parametrize("euler_index", [9, 10, 11, 15, 16, 17, 21, 22, 23])
def test_reset_validation_accepts_equivalent_euler_branch(euler_index: int) -> None:
    frame = _frame(1.0)
    frame.scene_obs[euler_index] = -np.pi + 1e-4
    environment = _Environment("calvin_scene_A", solve_after=1)
    observation = environment.reset(robot_obs=frame.robot_obs.copy(), scene_obs=frame.scene_obs.copy())
    observation["scene_obs"][euler_index] = np.pi - 1e-4
    object_index = (euler_index - 9) // 6
    movable_object = environment.scene.movable_objects[object_index]
    position, raw_quaternion = environment.p.poses[movable_object.uid]
    euler_slice = (slice(9, 12), slice(15, 18), slice(21, 24))[object_index]
    environment.p.set_pose(
        movable_object.uid,
        position,
        raw_quaternion,
        observation["scene_obs"][euler_slice],
    )

    _validate_reset_observation(observation, frame, environment)


def test_reset_validation_accepts_measured_near_gimbal_bullet_serialization() -> None:
    frame = _frame(1.0)
    source_euler = np.asarray(
        [3.0810634632984715, -1.5649422171108065, 2.8005713174419715],
        dtype=np.float64,
    )
    raw_quaternion = np.asarray(
        [-0.14221780846354004, -0.6936353489793308, -0.1397631130627423, 0.6921779899918524],
        dtype=np.float64,
    )
    observed_euler = np.asarray([0.0, -np.pi / 2, -0.4044596238568389], dtype=np.float64)
    frame.scene_obs[15:18] = source_euler
    environment = _Environment("calvin_scene_C", solve_after=1)
    observation = environment.reset(robot_obs=frame.robot_obs.copy(), scene_obs=frame.scene_obs.copy())
    observation["scene_obs"][15:18] = observed_euler
    movable_object = environment.scene.movable_objects[1]
    position, _original_quaternion = environment.p.poses[movable_object.uid]
    environment.p.set_pose(movable_object.uid, position, raw_quaternion, observed_euler)

    source_distance = _quaternion_geodesic(
        environment.p.getQuaternionFromEuler(source_euler),
        raw_quaternion,
        "measured source/raw object",
    )
    inverse_serialization_distance = _quaternion_geodesic(
        environment.p.getQuaternionFromEuler(observed_euler),
        raw_quaternion,
        "measured observed/raw object",
    )
    assert source_distance == pytest.approx(0.00458257110028981)
    assert inverse_serialization_distance == pytest.approx(0.0050219128866376165)
    assert source_distance < RESET_STATE_ATOL < inverse_serialization_distance

    _validate_reset_observation(observation, frame, environment)


def test_reset_validation_does_not_wrap_non_angular_scene_coordinates() -> None:
    frame = _frame(1.0)
    environment = _Environment("calvin_scene_A", solve_after=1)
    observation = environment.reset(robot_obs=frame.robot_obs.copy(), scene_obs=frame.scene_obs.copy())
    observation["scene_obs"][8] += 2 * np.pi

    with pytest.raises(CalvinDevStateError, match="fixture/object position drifted"):
        _validate_reset_observation(observation, frame, environment)


def test_reset_validation_rejects_real_quaternion_drift() -> None:
    frame = _frame(1.0)
    frame.scene_obs[15:18] = 0.0
    environment = _Environment("calvin_scene_A", solve_after=1)
    observation = environment.reset(robot_obs=frame.robot_obs.copy(), scene_obs=frame.scene_obs.copy())
    movable_object = environment.scene.movable_objects[1]
    position, _quaternion = environment.p.poses[movable_object.uid]
    drift_euler = np.asarray([2 * RESET_STATE_ATOL, 0.0, 0.0])
    drift_quaternion = environment.p.getQuaternionFromEuler(drift_euler)
    environment.p.set_pose(
        movable_object.uid,
        position,
        drift_quaternion,
        drift_euler,
    )
    observation["scene_obs"][15:18] = drift_euler

    with pytest.raises(CalvinDevStateError, match="object orientation drifted"):
        _validate_reset_observation(observation, frame, environment)


def test_reset_validation_binds_observed_euler_to_bullet_forward_serialization() -> None:
    frame = _frame(1.0)
    environment = _Environment("calvin_scene_A", solve_after=1)
    observation = environment.reset(robot_obs=frame.robot_obs.copy(), scene_obs=frame.scene_obs.copy())
    observation["scene_obs"][15] += 1e-6

    with pytest.raises(CalvinDevStateError, match="differs from Bullet serialization"):
        _validate_reset_observation(observation, frame, environment)


def test_reset_validation_fails_closed_on_movable_object_reordering() -> None:
    frame = _frame(1.0)
    environment = _Environment("calvin_scene_A", solve_after=1)
    observation = environment.reset(robot_obs=frame.robot_obs.copy(), scene_obs=frame.scene_obs.copy())
    environment.scene.movable_objects[0], environment.scene.movable_objects[1] = (
        environment.scene.movable_objects[1],
        environment.scene.movable_objects[0],
    )

    with pytest.raises(CalvinDevStateError, match="order differs from the pinned scene_obs layout"):
        _validate_reset_observation(observation, frame, environment)


def test_replay_uses_only_bundled_actions_checks_oracle_and_closes_env() -> None:
    frame = _frame(-1.0)
    actions = []
    members = []
    for index in range(2):
        action = np.zeros(7, dtype=np.float64)
        action[0] = 0.5
        action[6] = -1.0
        actions.append(action)
        members.append(
            {
                "global_index": index,
                "logical_bytes": 100 + index,
                "logical_sha256": f"{index + 1:064x}",
                "path": f"training/episode_{index:07d}.npz",
            }
        )
    candidate = CalvinDevCandidate(
        annotation_index=1,
        episode_index=0,
        global_start=0,
        global_end_exclusive=2,
        instruction="perform the fixture task",
        task="fixture_task",
        scene="calvin_scene_A",
    )
    environments: list[_Environment] = []

    def factory(scene: str) -> _Environment:
        environment = _Environment(scene, solve_after=2)
        environments.append(environment)
        return environment

    oracle = _Oracle(solve_after=2)
    replay = BundledCalvinReplay(
        candidate=candidate,
        frame=frame,
        actions=np.stack(actions),
        member_identities=tuple(members),
        record_sha256="a" * 64,
    )
    materialized, reason = replay_candidate(replay, oracle, factory)
    assert reason == ""
    assert materialized is not None and materialized.replay_actions == 2
    assert materialized.replay_bundle_record_sha256 == replay.record_sha256
    assert len(oracle.calls) == 3  # one unsolved-reset check plus one check per action
    assert all(call[2] == {"fixture_task"} for call in oracle.calls)
    assert environments[0].closed
    assert [action[0] for action in environments[0].actions] == [0.5, 0.5]


def test_rollout_replans_by_k_checks_oracle_each_action_and_discards_queue_on_success() -> None:
    environment = _Environment("calvin_scene_A", solve_after=2)
    oracle = _Oracle(solve_after=2)
    client = _Client()
    result = rollout_reset(
        environment,
        client,
        oracle,
        _record(),
        _frame(-1.0),
        reset_bank_sha256="a" * 64,
        reset_index=0,
        train_seed=17,
        evaluation_seed=91,
        execution_horizon=4,
        max_actions=10,
    )

    assert result["heldout_abc_subtask_success"] is True
    assert result["actions_executed"] == 2
    assert result["policy_calls"] == 1
    assert result["queued_actions_discarded_on_success"] == 2
    assert result["action_clip_fraction"] == pytest.approx(1 / 6)
    assert len(oracle.calls) == 3
    assert len(client.requests) == 1
    assert "scene_obs" not in client.requests[0]
    assert set(client.requests[0]) == {
        "annotation_index",
        "episode_index",
        "evaluation_seed",
        "execution_horizon",
        "global_start",
        "instruction",
        "replan_idx",
        "reset_bank_sha256",
        "reset_id_sha256",
        "reset_index",
        "rgb_gripper",
        "rgb_static",
        "scene",
        "state",
        "task",
        "train_seed",
    }
    assert client.requests[0]["state"][-1] == -1.0
    assert all(action[0] == 1.0 for action in environment.actions)


def test_evaluate_uses_fresh_environment_per_reset_and_fake_summary_never_passes() -> None:
    records = [_record(scene, index) for index, scene in enumerate(ABC_SCENES)]
    manifest = {"records": records, "root_sha256": "a" * 64}
    frames = [_frame(-1.0) for _record_value in records]
    robots = np.stack([frame.robot_obs for frame in frames])
    scenes = np.stack([frame.scene_obs for frame in frames])
    environments: list[_Environment] = []

    def factory(scene: str) -> _Environment:
        environment = _Environment(scene, solve_after=1)
        environments.append(environment)
        return environment

    results = evaluate_reset_indices(
        manifest,
        robots,
        scenes,
        [0, 1, 2],
        _Client(),
        _Oracle(solve_after=1),
        factory,
        train_seed=17,
        evaluation_seed=91,
        execution_horizon=1,
    )
    assert len(environments) == 3 and len({id(environment) for environment in environments}) == 3
    assert all(environment.closed for environment in environments)
    assert all(record["heldout_abc_subtask_success"] for record in results)

    fake = summarize_development(results, policy_mode="fake")
    real = summarize_development(results, policy_mode="real")
    assert fake["heldout_abc_subtask_success_rate"] == 1.0
    assert fake["plumbing_only"] is True and fake["gate_passed"] is False
    assert real["plumbing_only"] is False and real["gate_passed"] is True

    with pytest.raises(CalvinDevStateError, match="distinct"):
        evaluate_reset_indices(
            manifest,
            robots,
            scenes,
            [0, 0],
            _Client(),
            _Oracle(solve_after=1),
            factory,
            train_seed=17,
            evaluation_seed=91,
            execution_horizon=1,
        )


def test_fresh_environment_close_retires_pinned_destructor_ownership() -> None:
    environment = _PinnedLifecycleEnvironment()

    _close_fresh_environment(environment)
    environment.__del__()

    assert environment.disconnects == 1
    assert environment.connected is False
    assert environment.ownsPhysicsClient is False
    assert environment.cid == -1


def test_fresh_environment_close_preserves_error_after_retiring_ownership() -> None:
    environment = _PinnedLifecycleEnvironment()
    environment.connected = False

    with pytest.raises(RuntimeError, match="Not connected to physics server"):
        _close_fresh_environment(environment)

    assert environment.ownsPhysicsClient is False
    assert environment.cid == -1


def test_development_journal_finalizes_successful_run(tmp_path: Path) -> None:
    output_dir = tmp_path / "successful-development-run"
    with DevelopmentJournal(output_dir) as journal:
        journal.start_run({"schema": "fixture-run", "status": "running"})
        journal.append({"reset_index": 0})
        journal.write_json("summary.json", {"successes": 1})
        journal.complete()

    assert json.loads((output_dir / "run.json").read_text(encoding="utf-8"))["status"] == "complete"
    assert json.loads((output_dir / "summary.json").read_text(encoding="utf-8")) == {"successes": 1}
    assert json.loads((output_dir / "episodes.jsonl").read_text(encoding="utf-8")) == {"reset_index": 0}
    assert not list(output_dir.glob(".run.json.*.tmp"))


def test_development_journal_marks_failure_without_masking_rollout_error(tmp_path: Path) -> None:
    output_dir = tmp_path / "failed-development-run"
    with pytest.raises(RuntimeError, match="rollout failed"), DevelopmentJournal(output_dir) as journal:
        journal.start_run({"schema": "fixture-run", "status": "running"})
        journal.append({"reset_index": 0})
        raise RuntimeError("rollout failed")

    run = json.loads((output_dir / "run.json").read_text(encoding="utf-8"))
    assert run["status"] == "failed"
    assert run["error"] == {"message": "rollout failed", "type": "RuntimeError"}
    assert not (output_dir / "summary.json").exists()
    assert not list(output_dir.glob(".run.json.*.tmp"))


def test_calvin_action_copy_clips_continuous_channels_and_requires_binary_gripper() -> None:
    source = np.asarray([2.0, -2.0, 0.0, 0.5, -0.5, 1.5, -1.0], dtype=np.float32)
    result = calvin_env_action(source)
    assert result.flags.owndata and not np.shares_memory(result, source)
    np.testing.assert_array_equal(result, np.asarray([1.0, -1.0, 0.0, 0.5, -0.5, 1.0, -1.0], dtype=np.float32))
    with pytest.raises(CalvinDevStateError, match="gripper"):
        calvin_env_action(np.zeros(7, dtype=np.float32))


def test_live_policy_requires_the_exact_nested_calvin_dataset_identity() -> None:
    dataset = {
        "archive_bytes": 555_309_812_705,
        "archive_sha256": "c2036c67eb4c06966af1d1e1665bdb572c69e1404f5e77ffd46b384ff2b79f74",
        "central_directory_sha256": "b4f79bda7f6b966b51aa419badd0f7db7a8972a7b58d6d342af60aceff0ea31b",
        "dataset_manifest_file_sha256": "1" * 64,
        "dataset_manifest_schema": "duo-vla-calvin-dataset-manifest-v4",
        "dataset_manifest_sha256": "2" * 64,
        "member_index": {
            "bytes": 456,
            "path": "task_ABC_D.members-v2.sqlite3",
            "schema": "duo-vla-calvin-member-index-v2",
            "sha256": "3" * 64,
        },
        "member_inventory_sha256": "4" * 64,
        "metadata_files": [
            "ep_start_end_ids.npy",
            "lang_annotations/auto_lang_ann.npy",
            "scene_info.npy",
            ".hydra/merged_config.yaml",
        ],
        "metadata_sha256": "5" * 64,
        "name": "task_ABC_D",
        "reader_schema": "duo-vla-calvin-archive-reader-v1",
        "split": "training",
        "storage_identity_sha256": "6" * 64,
        "storage_mode": "archive-direct",
    }
    split = {"fixture": True}
    inputs = AuthenticatedCalvinDevInputs(
        identity={"metadata_sha256": dataset["metadata_sha256"], "normalization_content_sha256": "7" * 64},
        member_index={},
        split=split,
        stats={"dataset": dataset},
    )
    manifest = {
        "records": [_record()],
        "replay_bundle": {"root_sha256": "8" * 64},
        "root_sha256": "9" * 64,
    }
    health = {
        "allowed_scenes": list(ABC_SCENES),
        "calvin_identity": dataset,
        "execution_horizons": [1, 4],
        "mode": "real",
        "normalization_content_sha256": "7" * 64,
        "normalization_metadata_sha256": dataset["metadata_sha256"],
        "replay_bundle_sha256": "8" * 64,
        "reset_bank_sha256": "9" * 64,
        "reset_count": 1,
        "split_sha256": canonical_sha256(split),
        "train_seed": 17,
    }
    assert validate_live_policy(health, manifest, inputs, 4) == 17

    drifted = dict(health)
    drifted["calvin_identity"] = {**dataset, "metadata_sha256": "a" * 64}
    with pytest.raises(CalvinDevStateError, match="dataset identity differs"):
        validate_live_policy(drifted, manifest, inputs, 4)
