#!/usr/bin/env python3
"""Evaluate a policy on deterministic held-out CALVIN A/B/C reset states.

This is a development gate, not the official ABC-to-D evaluator.  Every reset
is independent, comes from a whole held-out training episode, and is executed
in a freshly constructed A/B/C environment.
"""

from __future__ import annotations

# ruff: noqa: UP006, UP035, UP045
import argparse
import json
import math
import os
import sys
import time
import types
from collections import deque
from contextlib import suppress
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_SOURCE_ROOT = _PROJECT_ROOT / "src"
_SCRIPT_ROOT = Path(__file__).resolve().parent
for _path in (_SOURCE_ROOT, _SCRIPT_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))
if sys.version_info < (3, 10) and "duo_vla" not in sys.modules:
    _legacy_package = types.ModuleType("duo_vla")
    _legacy_package.__path__ = [str(_SOURCE_ROOT / "duo_vla")]
    sys.modules["duo_vla"] = _legacy_package

from calvin_dev_bridge import (  # noqa: E402
    ACTION_DIM,
    ACTION_HORIZON,
    SUPPORTED_EXECUTION_HORIZONS,
    DevPolicyClient,
)
from generate_calvin_dev_states import (  # noqa: E402
    instantiate_abc_environment,
    instantiate_task_oracle,
    strict_source_reset,
)

from duo_vla.data.calvin_dev_states import (  # noqa: E402
    ABC_SCENES,
    AuthenticatedCalvinDevInputs,
    CalvinDevStateError,
    CalvinResetFrame,
    assert_bank_matches_inputs,
    authenticate_dev_inputs,
    canonical_json_bytes,
    canonical_sha256,
    load_bank,
)

DEVELOPMENT_RUN_SCHEMA = "duo-vla-calvin-heldout-abc-run-v2"
DEVELOPMENT_EPISODE_SCHEMA = "duo-vla-calvin-heldout-abc-episode-v2"
DEVELOPMENT_SUMMARY_SCHEMA = "duo-vla-calvin-heldout-abc-summary-v2"
DEFAULT_EVALUATION_SEED = 20260829
MAX_ACTIONS = 360


def require(condition: bool, message: str) -> None:
    if not condition:
        raise CalvinDevStateError(message)


def calvin_policy_state(robot_obs: Any) -> np.ndarray:
    source = np.asarray(robot_obs)
    require(source.shape == (15,) and bool(np.isfinite(source).all()), "CALVIN robot_obs must be finite shape (15,)")
    with np.errstate(over="ignore", invalid="ignore"):
        state = np.concatenate((source[:7], source[14:15])).astype(np.float32, copy=True)
    require(state.shape == (8,) and bool(np.isfinite(state).all()), "CALVIN policy state is invalid")
    require(state[7] in (-1.0, 1.0), "CALVIN policy gripper state must be exactly {-1,+1}")
    return np.ascontiguousarray(state)


def policy_observation(observation: Mapping[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    require(isinstance(observation, Mapping), "CALVIN observation must be an object")
    require("scene_obs" in observation, "CALVIN environment observation lacks reset-only scene state")
    rgb_obs = observation.get("rgb_obs")
    require(isinstance(rgb_obs, Mapping), "CALVIN observation lacks rgb_obs")
    static = np.asarray(rgb_obs.get("rgb_static"))
    gripper = np.asarray(rgb_obs.get("rgb_gripper"))
    require(static.shape == (200, 200, 3) and static.dtype == np.uint8, "CALVIN static RGB is invalid")
    require(gripper.shape == (84, 84, 3) and gripper.dtype == np.uint8, "CALVIN gripper RGB is invalid")
    # scene_obs intentionally stops here and cannot enter DevPolicyClient.predict.
    return static.copy(order="C"), gripper.copy(order="C"), calvin_policy_state(observation.get("robot_obs"))


def calvin_env_action(action: Any) -> np.ndarray:
    source = np.asarray(action)
    require(
        source.shape == (ACTION_DIM,) and bool(np.isfinite(source).all()), "CALVIN action must be finite shape (7,)"
    )
    result = np.asarray(source, dtype=np.float32).copy(order="C")
    np.clip(result[:6], -1.0, 1.0, out=result[:6])
    require(result[6] in (-1.0, 1.0), "CALVIN action gripper must be exactly {-1,+1}")
    return result


def _percentile(values: Sequence[float], percentage: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * percentage / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def validate_live_policy(
    health: Mapping[str, Any],
    manifest: Mapping[str, Any],
    inputs: AuthenticatedCalvinDevInputs,
    execution_horizon: int,
    allow_fake: bool = False,
) -> int:
    require(health.get("reset_bank_sha256") == manifest["root_sha256"], "live policy is bound to another reset bank")
    require(
        health.get("replay_bundle_sha256") == manifest["replay_bundle"]["root_sha256"],
        "live policy is bound to another replay bundle",
    )
    require(health.get("reset_count") == len(manifest["records"]), "live policy reset count differs")
    require(health.get("split_sha256") == canonical_sha256(inputs.split), "live policy split identity differs")
    require(
        health.get("calvin_identity") == inputs.stats["dataset"],
        "live policy CALVIN dataset identity differs",
    )
    require(health.get("allowed_scenes") == list(ABC_SCENES), "live policy accepts a non-A/B/C scene set")
    require(execution_horizon in SUPPORTED_EXECUTION_HORIZONS, "unsupported CALVIN execution horizon")
    require(execution_horizon in health.get("execution_horizons", []), "live policy does not support selected K")
    if not allow_fake:
        require(health.get("mode") == "real", "held-out policy gate refuses a fake policy")
        require(
            health.get("normalization_content_sha256") == inputs.identity["normalization_content_sha256"],
            "live policy normalization identity differs",
        )
        require(
            health.get("normalization_metadata_sha256") == inputs.identity["metadata_sha256"],
            "live policy normalization metadata differs",
        )
    train_seed = health.get("train_seed")
    require(type(train_seed) is int and 0 <= train_seed < 2**63, "live policy train_seed is invalid")
    return train_seed


def rollout_reset(
    environment: Any,
    client: Any,
    task_oracle: Any,
    record: Mapping[str, Any],
    frame: CalvinResetFrame,
    *,
    reset_bank_sha256: str,
    reset_index: int,
    train_seed: int,
    evaluation_seed: int,
    execution_horizon: int,
    max_actions: int = MAX_ACTIONS,
) -> Dict[str, Any]:
    """Run one independent held-out annotated subtask from a fresh A/B/C env."""

    require(record["scene"] in ABC_SCENES, "development rollout record is not from A/B/C")
    require(type(reset_index) is int and reset_index >= 0, "reset_index must be nonnegative")
    require(execution_horizon in SUPPORTED_EXECUTION_HORIZONS, "execution horizon must be one of {1,4}")
    require(type(max_actions) is int and 0 < max_actions <= MAX_ACTIONS, "development action budget is invalid")
    observation = strict_source_reset(environment, frame)
    start_info = environment.get_info()
    initially_solved = task_oracle.get_task_info_for_set(start_info, start_info, {record["task"]})
    require(not initially_solved, "held-out reset is successful before a policy action")

    queue = deque()
    replan_idx = 0
    actions_executed = 0
    discarded_actions = 0
    clipped_channels = 0
    latencies = []  # type: List[float]
    success = False
    started = time.perf_counter()
    current_observation = observation
    while actions_executed < max_actions and not success:
        if not queue:
            static, gripper, state = policy_observation(current_observation)
            request_started = time.perf_counter()
            actions, _metadata = client.predict(
                evaluation_seed=evaluation_seed,
                train_seed=train_seed,
                reset_bank_sha256=reset_bank_sha256,
                reset_id_sha256=record["reset_id_sha256"],
                reset_index=reset_index,
                scene=record["scene"],
                episode_index=record["episode_index"],
                annotation_index=record["annotation_index"],
                global_start=record["global_start"],
                task=record["task"],
                replan_idx=replan_idx,
                execution_horizon=execution_horizon,
                instruction=record["instruction"],
                rgb_static=static,
                rgb_gripper=gripper,
                state=state,
            )
            latencies.append(time.perf_counter() - request_started)
            require(
                isinstance(actions, np.ndarray)
                and actions.shape == (ACTION_HORIZON, ACTION_DIM)
                and actions.dtype == np.float32
                and bool(np.isfinite(actions).all()),
                "development policy returned an invalid action chunk",
            )
            queue.extend(np.asarray(row, dtype=np.float32).copy(order="C") for row in actions[:execution_horizon])
            replan_idx += 1

        queued = queue.popleft()
        clipped_channels += int(np.count_nonzero(np.abs(queued[:6]) > 1.0))
        env_action = calvin_env_action(queued)
        transition = environment.step(env_action)
        require(isinstance(transition, tuple) and len(transition) == 4, "CALVIN step must return a 4-tuple")
        current_observation, reward, done, current_info = transition
        actions_executed += 1
        try:
            finite_reward = math.isfinite(float(reward))
        except (TypeError, ValueError) as exc:
            raise CalvinDevStateError("development reward is not numeric") from exc
        require(finite_reward, "development reward is non-finite")
        require(not bool(done), "CALVIN environment terminated during development rollout")
        solved = task_oracle.get_task_info_for_set(start_info, current_info, {record["task"]})
        success = bool(solved)
        if success:
            discarded_actions = len(queue)
            queue.clear()

    return {
        "action_clip_fraction": clipped_channels / (actions_executed * 6) if actions_executed else 0.0,
        "actions_executed": actions_executed,
        "annotation_index": record["annotation_index"],
        "elapsed_seconds": time.perf_counter() - started,
        "episode_index": record["episode_index"],
        "evaluation_seed": evaluation_seed,
        "execution_horizon": execution_horizon,
        "global_start": record["global_start"],
        "heldout_abc_subtask_success": success,
        "instruction": record["instruction"],
        "max_actions": max_actions,
        "policy_calls": replan_idx,
        "policy_latency_p50_seconds": _percentile(latencies, 50.0),
        "policy_latency_p95_seconds": _percentile(latencies, 95.0),
        "queued_actions_discarded_on_success": discarded_actions,
        "reset_id_sha256": record["reset_id_sha256"],
        "reset_index": reset_index,
        "scene": record["scene"],
        "schema": DEVELOPMENT_EPISODE_SCHEMA,
        "steps_to_success": actions_executed if success else None,
        "task": record["task"],
        "train_seed": train_seed,
    }


def summarize_development(records: Sequence[Mapping[str, Any]], *, policy_mode: str = "real") -> Dict[str, Any]:
    require(bool(records), "cannot summarize zero held-out A/B/C reset rollouts")
    require(policy_mode in ("fake", "real"), "development policy mode is invalid")
    successes = sum(bool(record["heldout_abc_subtask_success"]) for record in records)
    by_scene = []
    for scene in ABC_SCENES:
        selected = [record for record in records if record["scene"] == scene]
        require(bool(selected), f"development rollout omitted scene {scene}")
        count = sum(bool(record["heldout_abc_subtask_success"]) for record in selected)
        by_scene.append(
            {
                "heldout_abc_subtask_success_rate": count / len(selected),
                "resets": len(selected),
                "scene": scene,
                "successes": count,
            }
        )
    return {
        "gate_passed": policy_mode == "real" and successes > 0,
        "heldout_abc_subtask_success_rate": successes / len(records),
        "per_scene": by_scene,
        "plumbing_only": policy_mode == "fake",
        "policy_mode": policy_mode,
        "resets": len(records),
        "schema": DEVELOPMENT_SUMMARY_SCHEMA,
        "successes": successes,
    }


def _close_fresh_environment(environment: Any) -> None:
    """Close one pinned CALVIN env without letting its destructor close a later env."""

    close = getattr(environment, "close", None)
    if not callable(close):
        return
    try:
        close()
    finally:
        # The pinned CALVIN PlayTableSimEnv.__del__ calls close() again.  PyBullet
        # reuses client id 0, so a delayed destructor can otherwise disconnect the
        # next freshly constructed environment.  Retire ownership after the
        # explicit close while preserving exceptions from that close operation.
        if hasattr(environment, "ownsPhysicsClient"):
            environment.ownsPhysicsClient = False
            if hasattr(environment, "cid"):
                environment.cid = -1


class DevelopmentJournal:
    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir.resolve()
        self.run_path = self.output_dir / "run.json"
        self.episodes_path = self.output_dir / "episodes.jsonl"
        self.summary_path = self.output_dir / "summary.json"
        self._handle = None
        self._run = None  # type: Optional[Dict[str, Any]]

    def __enter__(self) -> DevelopmentJournal:
        self.output_dir.mkdir(parents=True, exist_ok=False)
        self._handle = self.episodes_path.open("x", encoding="utf-8")
        return self

    def append(self, record: Mapping[str, Any]) -> None:
        require(self._handle is not None, "development journal is not open")
        self._handle.write(canonical_json_bytes(dict(record)).decode("utf-8") + "\n")
        self._handle.flush()
        os.fsync(self._handle.fileno())

    def start_run(self, payload: Mapping[str, Any]) -> None:
        require(self._run is None, "development run is already started")
        run = dict(payload)
        require(run.get("status") == "running", "development run must start in running state")
        self.write_json("run.json", run)
        self._run = run

    def write_json(self, name: str, payload: Mapping[str, Any]) -> None:
        path = self.output_dir / name
        with path.open("xb") as handle:
            handle.write(canonical_json_bytes(dict(payload), pretty=True))
            handle.flush()
            os.fsync(handle.fileno())
        self._fsync_output_directory()

    def _fsync_output_directory(self) -> None:
        directory_fd = os.open(str(self.output_dir), os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)

    def _close_episodes(self) -> None:
        if self._handle is not None:
            handle = self._handle
            try:
                handle.flush()
                os.fsync(handle.fileno())
            finally:
                try:
                    handle.close()
                finally:
                    self._handle = None

    def _replace_run(self, payload: Mapping[str, Any]) -> None:
        temporary = self.output_dir / f".run.json.{os.getpid()}.{time.time_ns()}.tmp"
        try:
            with temporary.open("xb") as handle:
                handle.write(canonical_json_bytes(dict(payload), pretty=True))
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(str(temporary), str(self.run_path))
            self._fsync_output_directory()
        finally:
            with suppress(FileNotFoundError):
                temporary.unlink()

    def complete(self) -> None:
        require(self._run is not None and self._run.get("status") == "running", "development run is not running")
        self._close_episodes()
        require(self.summary_path.is_file(), "development summary is missing")
        terminal = dict(self._run)
        terminal["status"] = "complete"
        self._replace_run(terminal)
        self._run = terminal

    def fail(self, error: BaseException) -> None:
        require(self._run is not None and self._run.get("status") == "running", "development run is not running")
        close_error = None  # type: Optional[BaseException]
        try:
            self._close_episodes()
        except BaseException as exc:
            close_error = exc
        message = str(error)
        if close_error is not None:
            message += f"; journal close failure: {type(close_error).__name__}: {close_error}"
        terminal = dict(self._run)
        terminal.update({"error": {"message": message, "type": type(error).__name__}, "status": "failed"})
        self._replace_run(terminal)
        self._run = terminal

    def __exit__(self, exc_type: object, error: object, _traceback: object) -> None:
        if self._run is not None and self._run.get("status") == "running":
            terminal_error = (
                error
                if isinstance(error, BaseException)
                else CalvinDevStateError("development journal exited before completion")
            )
            try:
                self.fail(terminal_error)
            except BaseException:
                if error is None:
                    raise
            if error is None:
                raise terminal_error
        else:
            self._close_episodes()


def evaluate_reset_indices(
    manifest: Mapping[str, Any],
    robot_obs: np.ndarray,
    scene_obs: np.ndarray,
    reset_indices: Sequence[int],
    client: Any,
    task_oracle: Any,
    environment_factory: Callable[[str], Any],
    *,
    train_seed: int,
    evaluation_seed: int,
    execution_horizon: int,
    callback: Optional[Callable[[Mapping[str, Any]], None]] = None,
) -> Tuple[Dict[str, Any], ...]:
    records = manifest["records"]
    requested_indices = list(reset_indices)
    require(bool(requested_indices), "development evaluation selected zero resets")
    require(len(requested_indices) == len(set(requested_indices)), "development reset indices must be distinct")
    results = []  # type: List[Dict[str, Any]]
    for reset_index in requested_indices:
        require(type(reset_index) is int and 0 <= reset_index < len(records), "reset index is outside the bank")
        record = records[reset_index]
        environment = environment_factory(record["scene"])
        try:
            result = rollout_reset(
                environment,
                client,
                task_oracle,
                record,
                CalvinResetFrame(
                    robot_obs=robot_obs[reset_index].copy(order="C"),
                    scene_obs=scene_obs[reset_index].copy(order="C"),
                    source_frame_sha256=record["source_frame_sha256"],
                ),
                reset_bank_sha256=manifest["root_sha256"],
                reset_index=reset_index,
                train_seed=train_seed,
                evaluation_seed=evaluation_seed,
                execution_horizon=execution_horizon,
            )
        finally:
            _close_fresh_environment(environment)
        results.append(result)
        if callback is not None:
            callback(result)
    return tuple(results)


def parse_args() -> argparse.Namespace:
    cache_root = Path(os.environ.get("DUO_VLA_CACHE_ROOT", "/root/.cache/duo-vla"))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-root", type=Path, default=cache_root / "data/calvin/task_ABC_D/training")
    parser.add_argument("--normalization", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, default=cache_root / "simulators/calvin")
    parser.add_argument("--revision-file", type=Path, default=_PROJECT_ROOT / "scripts/calvin/revisions.env")
    parser.add_argument("--reset-bank", type=Path, required=True)
    parser.add_argument("--socket", type=Path, required=True)
    parser.add_argument("--execution-horizon", type=int, choices=SUPPORTED_EXECUTION_HORIZONS, required=True)
    parser.add_argument("--evaluation-seed", type=int, default=DEFAULT_EVALUATION_SEED)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--all-resets", action="store_true")
    parser.add_argument(
        "--allow-fake-policy", action="store_true", help="IPC plumbing only; cannot pass the learned-policy gate"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    require(type(args.evaluation_seed) is int and 0 <= args.evaluation_seed < 2**63, "evaluation seed is invalid")
    inputs = authenticate_dev_inputs(
        args.training_root,
        args.normalization,
        args.source_root,
        args.revision_file,
    )
    manifest, robot_obs, scene_obs = load_bank(args.reset_bank)
    assert_bank_matches_inputs(manifest, inputs)
    reset_indices = (
        list(range(len(manifest["records"]))) if args.all_resets else list(manifest["selection"]["smoke_reset_indices"])
    )
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

    started = time.perf_counter()
    with DevPolicyClient(args.socket) as client:
        health = client.health()
        train_seed = validate_live_policy(
            health,
            manifest,
            inputs,
            args.execution_horizon,
            allow_fake=args.allow_fake_policy,
        )
        with DevelopmentJournal(args.output_dir) as journal:
            run = {
                "evaluation_seed": args.evaluation_seed,
                "execution_horizon": args.execution_horizon,
                "policy_health": health,
                "replay_bundle_sha256": manifest["replay_bundle"]["root_sha256"],
                "reset_bank_sha256": manifest["root_sha256"],
                "reset_indices": reset_indices,
                "schema": DEVELOPMENT_RUN_SCHEMA,
                "split_sha256": canonical_sha256(inputs.split),
                "status": "running",
            }
            journal.start_run(run)
            results = evaluate_reset_indices(
                manifest,
                robot_obs,
                scene_obs,
                reset_indices,
                client,
                task_oracle,
                environment_factory,
                train_seed=train_seed,
                evaluation_seed=args.evaluation_seed,
                execution_horizon=args.execution_horizon,
                callback=journal.append,
            )
            summary = summarize_development(results, policy_mode=health["mode"])
            summary["elapsed_seconds"] = time.perf_counter() - started
            summary["replay_bundle_sha256"] = manifest["replay_bundle"]["root_sha256"]
            summary["reset_bank_sha256"] = manifest["root_sha256"]
            journal.write_json("summary.json", summary)
            journal.complete()
    print(json.dumps(summary, allow_nan=False, sort_keys=True))


if __name__ == "__main__":
    main()
