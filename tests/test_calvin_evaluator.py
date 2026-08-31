"""Fake-simulator tests for the official CALVIN long-horizon state machine."""

from __future__ import annotations

import copy
import hashlib
import json
import sys
import types
from pathlib import Path
from typing import Any

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
CALVIN_SCRIPTS = ROOT / "scripts" / "calvin"
sys.path.insert(0, str(CALVIN_SCRIPTS))

import evaluate_calvin as EVALUATOR  # noqa: E402, I001


TASKS = tuple(f"task_{index}" for index in range(5))
ANNOTATIONS = {task: [f"first phrase for {task}", f"unused phrase for {task}"] for task in TASKS}
PREDICT_FIELDS = {
    "evaluation_seed",
    "execution_horizon",
    "instruction",
    "replan_idx",
    "rgb_gripper",
    "rgb_static",
    "sequence_idx",
    "sequence_sha256",
    "state",
    "subtask_idx",
    "subtask_name",
    "train_seed",
}


@pytest.mark.parametrize("mutated_name", EVALUATOR._EVALUATOR_SOURCE_NAMES)
def test_evaluator_import_snapshot_rejects_each_source_mutation(tmp_path: Path, mutated_name: str) -> None:
    for name in EVALUATOR._EVALUATOR_SOURCE_NAMES:
        (tmp_path / name).write_text(f"source:{name}\n", encoding="utf-8")
    snapshot = EVALUATOR._capture_evaluator_source_identities(tmp_path)
    (tmp_path / mutated_name).write_text("mutated source\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="changed after import-time snapshot"):
        EVALUATOR._require_evaluator_sources_unchanged(snapshot)


def test_imported_local_module_must_match_path_origin_and_raw_snapshot(tmp_path: Path) -> None:
    source = tmp_path / "preflight.py"
    source.write_text("source = 1\n", encoding="utf-8")
    identity = EVALUATOR._source_file_identity(source)
    expected = {"preflight.py": identity}
    module = types.SimpleNamespace(
        __file__=str(source),
        __spec__=types.SimpleNamespace(origin=str(source)),
    )
    EVALUATOR._require_imported_local_module(module, "preflight.py", expected)

    source.write_text("source = 2\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="differs from the pre-import snapshot"):
        EVALUATOR._require_imported_local_module(module, "preflight.py", expected)

    other = tmp_path / "other.py"
    other.write_text("source = 1\n", encoding="utf-8")
    module.__file__ = str(other)
    module.__spec__.origin = str(other)
    with pytest.raises(RuntimeError, match="unexpected __file__"):
        EVALUATOR._require_imported_local_module(module, "preflight.py", expected)


def test_authenticated_parser_rejects_symlink_path(tmp_path: Path) -> None:
    target = tmp_path / "target.yaml"
    target.write_text("value: 1\n", encoding="utf-8")
    link = tmp_path / "linked.yaml"
    link.symlink_to(target)
    identity = EVALUATOR._source_file_identity(target)

    with pytest.raises(RuntimeError, match="cannot open authenticated"):
        EVALUATOR._verify_attested_file(link, identity, name="test YAML")


def _observation() -> dict[str, Any]:
    robot_obs = np.linspace(-0.4, 0.4, 15, dtype=np.float64)
    robot_obs[14] = -1.0
    return {
        "rgb_obs": {
            "rgb_gripper": np.zeros(EVALUATOR.GRIPPER_IMAGE_SHAPE, dtype=np.uint8),
            "rgb_static": np.zeros(EVALUATOR.STATIC_IMAGE_SHAPE, dtype=np.uint8),
        },
        "robot_obs": robot_obs,
        "scene_obs": np.zeros(24, dtype=np.float64),
    }


class FakeEnvironment:
    """Mimic CALVIN's in-place action scaling without any simulator dependency."""

    def __init__(self) -> None:
        self.reset_calls: list[tuple[np.ndarray, np.ndarray]] = []
        self.actions: list[np.ndarray] = []
        self.step_count = 0
        self.closed = False

    def reset(self, *, robot_obs: np.ndarray, scene_obs: np.ndarray) -> dict[str, Any]:
        self.reset_calls.append((robot_obs.copy(), scene_obs.copy()))
        return _observation()

    def get_info(self) -> dict[str, int]:
        return {"step": self.step_count}

    def step(self, action: np.ndarray) -> tuple[dict[str, Any], float, bool, dict[str, int]]:
        self.actions.append(action.copy())
        # The pinned CALVIN robot scales the caller-owned input in place.
        action[:3] *= 0.02
        action[3:6] *= 0.05
        self.step_count += 1
        return _observation(), 0.0, False, {"step": self.step_count}

    def close(self) -> None:
        self.closed = True


class ScheduledOracle:
    def __init__(self, thresholds: dict[str, int | None]) -> None:
        self.thresholds = thresholds
        self.calls: list[tuple[int, int, str]] = []

    def get_task_info_for_set(
        self,
        start_info: dict[str, int],
        current_info: dict[str, int],
        tasks: set[str],
    ) -> set[str]:
        assert len(tasks) == 1
        task = next(iter(tasks))
        self.calls.append((start_info["step"], current_info["step"], task))
        threshold = self.thresholds[task]
        if threshold is not None and current_info["step"] - start_info["step"] >= threshold:
            return {task}
        return set()


class FakePolicyClient:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.chunk = np.zeros((EVALUATOR.ACTION_HORIZON, EVALUATOR.ACTION_DIM), dtype=np.float32)
        self.chunk[:, 0] = np.arange(1, EVALUATOR.ACTION_HORIZON + 1, dtype=np.float32) / 10.0
        self.chunk[:, 6] = np.asarray([-1.0, 1.0] * 4, dtype=np.float32)

    def predict(self, **kwargs: Any) -> tuple[np.ndarray, dict[str, Any]]:
        assert set(kwargs) == PREDICT_FIELDS
        self.calls.append(kwargs)
        response = {
            "evaluation_seed": kwargs["evaluation_seed"],
            "execution_horizon": kwargs["execution_horizon"],
            "policy_seconds": 0.001,
            "replan_idx": kwargs["replan_idx"],
            "sequence_idx": kwargs["sequence_idx"],
            "sequence_sha256": kwargs["sequence_sha256"],
            "subtask_idx": kwargs["subtask_idx"],
            "subtask_name": kwargs["subtask_name"],
            "train_seed": kwargs["train_seed"],
        }
        # Returning the same array every time makes in-place mutation leaks
        # visible to the tests.
        return self.chunk, response


def _state_converter_calls() -> tuple[list[dict[str, Any]], Any]:
    calls: list[dict[str, Any]] = []

    def convert(initial_state: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
        calls.append(initial_state)
        robot_obs = np.zeros(15, dtype=np.float64)
        robot_obs[14] = -1.0
        return robot_obs, np.zeros(24, dtype=np.float64)

    return calls, convert


def _evaluate_sequence(
    environment: FakeEnvironment,
    client: FakePolicyClient,
    oracle: ScheduledOracle,
    converter: Any,
    *,
    execution_horizon: int = 4,
) -> dict[str, Any]:
    return EVALUATOR.evaluate_sequence(
        environment,
        client,
        oracle,
        {"initial": "state"},
        TASKS,
        ANNOTATIONS,
        converter,
        train_seed=7,
        evaluation_seed=0,
        sequence_idx=0,
        execution_horizon=execution_horizon,
    )


def test_one_reset_five_sequential_subtasks_and_fresh_request_identity() -> None:
    environment = FakeEnvironment()
    client = FakePolicyClient()
    oracle = ScheduledOracle({task: 1 for task in TASKS})
    converter_calls, converter = _state_converter_calls()

    record = _evaluate_sequence(environment, client, oracle, converter)

    assert len(converter_calls) == 1
    assert len(environment.reset_calls) == 1
    assert environment.step_count == 5
    assert record["successful_subtasks"] == 5
    assert record["sequence_success"] is True
    assert [call["subtask_idx"] for call in client.calls] == [0, 1, 2, 3, 4]
    assert [call["subtask_name"] for call in client.calls] == list(TASKS)
    assert [call["replan_idx"] for call in client.calls] == [0, 0, 0, 0, 0]
    assert [call["instruction"] for call in client.calls] == [ANNOTATIONS[task][0] for task in TASKS]
    assert all("scene_obs" not in call for call in client.calls)
    assert all(call["state"].shape == (8,) for call in client.calls)
    assert [(start, end) for start, end, _task in oracle.calls] == [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]
    assert [subtask["discarded_queued_actions_on_success"] for subtask in record["subtasks"]] == [3] * 5


def test_k4_queue_replans_only_when_empty_and_discards_tail_on_early_success() -> None:
    environment = FakeEnvironment()
    observation = environment.reset(robot_obs=np.zeros(15), scene_obs=np.zeros(24))
    client = FakePolicyClient()
    original_chunk = client.chunk.copy()
    oracle = ScheduledOracle({TASKS[0]: 5})

    _observation_after, record = EVALUATOR.rollout_subtask(
        environment,
        client,
        oracle,
        observation,
        train_seed=7,
        evaluation_seed=0,
        sequence_idx=0,
        subtask_idx=0,
        subtask_name=TASKS[0],
        instruction=ANNOTATIONS[TASKS[0]][0],
        execution_horizon=4,
    )

    assert record["success"] is True
    assert record["environment_actions"] == 5
    assert record["policy_calls"] == 2
    assert record["discarded_queued_actions_on_success"] == 3
    assert [call["replan_idx"] for call in client.calls] == [0, 1]
    np.testing.assert_allclose([action[0] for action in environment.actions], [0.1, 0.2, 0.3, 0.4, 0.1])
    np.testing.assert_array_equal(client.chunk, original_chunk)
    assert len(oracle.calls) == len(environment.actions) == 5


def test_k1_replans_after_every_action() -> None:
    environment = FakeEnvironment()
    observation = environment.reset(robot_obs=np.zeros(15), scene_obs=np.zeros(24))
    client = FakePolicyClient()
    oracle = ScheduledOracle({TASKS[0]: 2})

    _observation_after, record = EVALUATOR.rollout_subtask(
        environment,
        client,
        oracle,
        observation,
        train_seed=7,
        evaluation_seed=0,
        sequence_idx=0,
        subtask_idx=0,
        subtask_name=TASKS[0],
        instruction=ANNOTATIONS[TASKS[0]][0],
        execution_horizon=1,
    )

    assert record["environment_actions"] == record["policy_calls"] == 2
    assert record["discarded_queued_actions_on_success"] == 0
    assert [call["replan_idx"] for call in client.calls] == [0, 1]


@pytest.mark.parametrize("gripper", [-1.0, 1.0])
def test_rel_action_is_owned_clipped_and_preserves_calvin_gripper_polarity(gripper: float) -> None:
    source = np.asarray([2.0, -2.0, 0.25, 9.0, -9.0, 0.5, gripper], dtype=np.float64)
    original = source.copy()

    action = EVALUATOR.calvin_env_action(source)

    assert action.dtype == np.float32 and action.flags.owndata and action.flags.c_contiguous
    assert not np.shares_memory(action, source)
    np.testing.assert_array_equal(action[:6], np.asarray([1.0, -1.0, 0.25, 1.0, -1.0, 0.5], dtype=np.float32))
    assert action[6] == gripper  # -1 closes, +1 opens in CALVIN.
    action[:] = 0
    np.testing.assert_array_equal(source, original)


def test_failure_consumes_exactly_360_actions_and_stops_the_sequence() -> None:
    environment = FakeEnvironment()
    client = FakePolicyClient()
    oracle = ScheduledOracle({task: None for task in TASKS})
    _converter_calls, converter = _state_converter_calls()

    record = _evaluate_sequence(environment, client, oracle, converter, execution_horizon=4)

    assert len(environment.reset_calls) == 1
    assert environment.step_count == 360
    assert len(oracle.calls) == 360
    assert record["successful_subtasks"] == 0
    assert record["sequence_success"] is False
    assert len(record["subtasks"]) == 1
    assert record["subtasks"][0]["environment_actions"] == 360
    assert record["subtasks"][0]["policy_calls"] == 90
    assert len(client.calls) == 90
    assert {call["subtask_idx"] for call in client.calls} == {0}
    assert [call["replan_idx"] for call in client.calls] == list(range(90))


def test_invalid_execution_horizon_and_seed_fail_before_policy_or_oracle() -> None:
    environment = FakeEnvironment()
    observation = environment.reset(robot_obs=np.zeros(15), scene_obs=np.zeros(24))
    client = FakePolicyClient()
    oracle = ScheduledOracle({TASKS[0]: 1})
    base = {
        "environment": environment,
        "client": client,
        "task_oracle": oracle,
        "observation": observation,
        "train_seed": 7,
        "evaluation_seed": 0,
        "sequence_idx": 0,
        "subtask_idx": 0,
        "subtask_name": TASKS[0],
        "instruction": ANNOTATIONS[TASKS[0]][0],
        "execution_horizon": 2,
    }
    with pytest.raises(RuntimeError, match=r"\{1, 4\}"):
        EVALUATOR.rollout_subtask(**base)
    assert not client.calls and not oracle.calls and environment.step_count == 0

    base["execution_horizon"] = 1
    base["evaluation_seed"] = 1
    with pytest.raises(RuntimeError, match="must equal 0"):
        EVALUATOR.rollout_subtask(**base)
    assert not client.calls and not oracle.calls and environment.step_count == 0


def test_sequence_contract_gate_rejects_before_any_policy_action() -> None:
    client = FakePolicyClient()
    environment = FakeEnvironment()
    oracle = ScheduledOracle({task: 1 for task in TASKS})
    _converter_calls, converter = _state_converter_calls()

    with pytest.raises(RuntimeError, match="exactly 1000"):
        EVALUATOR.evaluate_official_sequences(
            environment,
            client,
            oracle,
            [],
            ANNOTATIONS,
            converter,
            train_seed=7,
            evaluation_seed=0,
            execution_horizon=1,
        )
    assert not client.calls and not oracle.calls and not environment.reset_calls

    sample = [({"a": 1}, list(TASKS))]
    digest = EVALUATOR.canonical_sequence_sha256(sample)
    checked = EVALUATOR.verify_sequence_contract(sample, expected_count=1, expected_sha256=digest)
    assert checked == [({"a": 1}, TASKS)]


def test_official_metrics_are_avg_len_and_sr1_through_sr5() -> None:
    records = []
    for sequence_idx, length in enumerate(range(6)):
        records.append(
            {
                "sequence_idx": sequence_idx,
                "successful_subtasks": length,
                "subtasks": [
                    {"subtask_name": TASKS[index % 5], "success": index < length} for index in range(min(length + 1, 5))
                ],
            }
        )

    summary = EVALUATOR.summarize_sequences(records)

    assert summary["AvgLen"] == pytest.approx(2.5)
    assert [summary[f"SR{depth}"] for depth in range(1, 6)] == pytest.approx([5 / 6, 4 / 6, 3 / 6, 2 / 6, 1 / 6])
    assert summary["AvgLen"] == pytest.approx(sum(summary[f"SR{depth}"] for depth in range(1, 6)))


def _execution_geometry() -> dict[str, Any]:
    return {
        "expert_batch_isolation": "sample_isolated_grouped_mm_v1",
        "experts_implementation": "grouped_mm",
        "fixed_physical_prefix_width": 600,
        "physical_batch_size": 8,
        "prefix_geometry_content_sha256": "6" * 64,
    }


def _calvin_identity(*, metadata_sha256: str = "d" * 64) -> dict[str, Any]:
    return {
        "archive_bytes": EVALUATOR.ARCHIVE_BYTES,
        "archive_sha256": EVALUATOR.ARCHIVE_SHA256,
        "central_directory_sha256": EVALUATOR.CENTRAL_DIRECTORY_SHA256,
        "dataset_manifest_file_sha256": "1" * 64,
        "dataset_manifest_schema": EVALUATOR.DATASET_MANIFEST_SCHEMA,
        "dataset_manifest_sha256": "2" * 64,
        "member_index": {
            "bytes": 123,
            "path": EVALUATOR.MEMBER_INDEX_NAME,
            "schema": EVALUATOR.MEMBER_INDEX_SCHEMA,
            "sha256": "3" * 64,
        },
        "member_inventory_sha256": "4" * 64,
        "metadata_files": list(EVALUATOR.TRAINING_METADATA_FILES),
        "metadata_sha256": metadata_sha256,
        "name": "task_ABC_D",
        "reader_schema": EVALUATOR.ARCHIVE_READER_SCHEMA,
        "split": "training",
        "storage_identity_sha256": "5" * 64,
        "storage_mode": "archive-direct",
    }


def _health() -> dict[str, Any]:
    contract = EVALUATOR.selected_policy_contract("rectified_flow", 5)
    return {
        "action_dim": EVALUATOR.ACTION_DIM,
        "action_horizon": EVALUATOR.ACTION_HORIZON,
        "calvin_identity": _calvin_identity(),
        "checkpoint_manifest_sha256": "a" * 64,
        "execution_horizons": list(EVALUATOR.SUPPORTED_EXECUTION_HORIZONS),
        "execution_geometry": _execution_geometry(),
        "gripper_image_shape": list(EVALUATOR.GRIPPER_IMAGE_SHAPE),
        "mode": "real",
        "model_revision": EVALUATOR.MODEL_REVISION,
        "nfe": 5,
        "normalization_content_sha256": "c" * 64,
        "normalization_metadata_sha256": "d" * 64,
        "objective": "rectified_flow",
        "operation": "health",
        "policy_contract_sha256": EVALUATOR.canonical_sha256(contract),
        "protocol": EVALUATOR.PROTOCOL,
        "request_id": "health-1",
        "sampler": "euler_uniform",
        "schema": EVALUATOR.SCHEMA,
        "sequence_sha256": EVALUATOR.SEQUENCE_SHA256,
        "serving_runtime_sha256": "9" * 64,
        "state_dim": EVALUATOR.STATE_DIM,
        "static_image_shape": list(EVALUATOR.STATIC_IMAGE_SHAPE),
        "status": "ok",
        "train_seed": 0,
    }


def test_evaluator_health_inventory_matches_ipc_v4_exactly() -> None:
    assert EVALUATOR.SCHEMA == "duovla-calvin-policy-ipc-v4"
    assert EVALUATOR._HEALTH_FIELDS == EVALUATOR._calvin_bridge._HEALTH_RESPONSE_FIELDS
    assert EVALUATOR._CALVIN_IDENTITY_FIELDS == EVALUATOR._calvin_bridge._CALVIN_IDENTITY_FIELDS
    assert EVALUATOR._CALVIN_MEMBER_INDEX_FIELDS == EVALUATOR._calvin_bridge._CALVIN_MEMBER_INDEX_FIELDS


def _official_cells() -> list[dict[str, Any]]:
    cells = []
    for seed, objective, nfe, execution_horizon in sorted(EVALUATOR.official_factor_matrix()):
        contract = EVALUATOR.selected_policy_contract(objective, nfe)
        checkpoint = hashlib.sha256(f"checkpoint:{seed}:{objective}".encode()).hexdigest()
        runtime = hashlib.sha256(f"runtime:{seed}:{objective}".encode()).hexdigest()
        cells.append(
            {
                "cell_id": EVALUATOR.official_cell_id(seed, objective, nfe, execution_horizon),
                "checkpoint": {"sha256": checkpoint},
                "execution_geometry": _execution_geometry(),
                "execution_horizon": execution_horizon,
                "policy": {
                    "identity_sha256": EVALUATOR.canonical_sha256(contract),
                    "inference_seed_behavior": contract["inference_seed_behavior"],
                    "nfe": nfe,
                    "objective": objective,
                    "sampler": contract["sampler"],
                    "train_seed": seed,
                },
                "serving_runtime_sha256": runtime,
            }
        )
    return cells


def _preregistration(sequences: list[Any], digest: str, token: str) -> dict[str, Any]:
    return {
        "aggregation_python_version": EVALUATOR.PYTHON_VERSION,
        "aggregator_sha256": "a" * 64,
        "benchmark_protocol": EVALUATOR.PROTOCOL,
        "cells": _official_cells(),
        "direct_nfe": EVALUATOR.OFFICIAL_DIRECT_NFE,
        "evaluation_seed": EVALUATOR.EVALUATION_SEED,
        "execution_horizons": list(EVALUATOR.SUPPORTED_EXECUTION_HORIZONS),
        "final_checkpoint_update": EVALUATOR.FINAL_CHECKPOINT_UPDATE,
        "final_freeze_token_sha256": hashlib.sha256(token.encode()).hexdigest(),
        "flow_nfes": list(EVALUATOR.OFFICIAL_FLOW_NFES),
        "inference_seed_domain": EVALUATOR.INFERENCE_SEED_DOMAIN,
        "runtime_attestation_sha256": "c" * 64,
        "schema": EVALUATOR.PREREGISTRATION_SCHEMA,
        "sequence_count": len(sequences),
        "sequence_sha256": digest,
        "sequences": sequences,
        "subtasks_per_sequence": EVALUATOR.SUBTASKS_PER_SEQUENCE,
        "training_seeds": list(EVALUATOR.OFFICIAL_TRAIN_SEEDS),
    }


def test_policy_health_is_exact_and_pre_registered_train_seed_is_bound() -> None:
    health = _health()
    cell = next(
        value
        for value in _official_cells()
        if value["cell_id"] == EVALUATOR.official_cell_id(0, "rectified_flow", 5, 4)
    )
    health["checkpoint_manifest_sha256"] = cell["checkpoint"]["sha256"]
    health["serving_runtime_sha256"] = cell["serving_runtime_sha256"]
    assert (
        EVALUATOR.validate_policy_health(
            health,
            execution_horizon=4,
            expected_cell=cell,
            expected_calvin_identity=_calvin_identity(),
        )
        == health
    )
    with pytest.raises(RuntimeError, match="fields differ"):
        EVALUATOR.validate_policy_health(dict(health, fake=True), execution_horizon=4)
    with pytest.raises(RuntimeError, match="checkpoint identity"):
        changed_checkpoint = dict(health, checkpoint_manifest_sha256="f" * 64)
        EVALUATOR.validate_policy_health(changed_checkpoint, execution_horizon=4, expected_cell=cell)
    with pytest.raises(RuntimeError, match="policy identity"):
        changed_policy = dict(health, policy_contract_sha256="f" * 64)
        EVALUATOR.validate_policy_health(changed_policy, execution_horizon=4, expected_cell=cell)
    with pytest.raises(RuntimeError, match="serving runtime identity"):
        changed_runtime = dict(health, serving_runtime_sha256="8" * 64)
        EVALUATOR.validate_policy_health(changed_runtime, execution_horizon=4, expected_cell=cell)
    with pytest.raises(RuntimeError, match="execution geometry differs"):
        changed_cell = copy.deepcopy(cell)
        changed_cell["execution_geometry"]["fixed_physical_prefix_width"] = 601
        EVALUATOR.validate_policy_health(health, execution_horizon=4, expected_cell=changed_cell)
    with pytest.raises(RuntimeError, match="runtime/data attestation"):
        changed_calvin_identity = _calvin_identity()
        changed_calvin_identity["dataset_manifest_sha256"] = "f" * 64
        EVALUATOR.validate_policy_health(
            health,
            execution_horizon=4,
            expected_cell=cell,
            expected_calvin_identity=changed_calvin_identity,
        )
    with pytest.raises(RuntimeError, match="normalization/CALVIN metadata"):
        changed_metadata = copy.deepcopy(health)
        changed_metadata["calvin_identity"]["metadata_sha256"] = "e" * 64
        EVALUATOR.validate_policy_health(changed_metadata, execution_horizon=4)
    with pytest.raises(RuntimeError, match="refuses fake"):
        fake_health = dict(
            health,
            checkpoint_manifest_sha256=None,
            mode="fake",
            model_revision=None,
            nfe=0,
            normalization_content_sha256=None,
            normalization_metadata_sha256=None,
            objective="test_fake",
            policy_contract_sha256=None,
            sampler="seeded_test_normal",
            serving_runtime_sha256=None,
        )
        EVALUATOR.validate_policy_health(fake_health, execution_horizon=4)


def test_preregistration_binds_full_sequences_cell_k_and_explicit_token(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sequences = [({"initial": 0}, list(TASKS))]
    digest = EVALUATOR.canonical_sequence_sha256(sequences)
    token = "this cell is frozen before final evaluation"
    monkeypatch.setattr(EVALUATOR, "NUM_SEQUENCES", 1)
    monkeypatch.setattr(EVALUATOR, "SEQUENCE_SHA256", digest)
    manifest = _preregistration(sequences, digest, token)
    cell_id = EVALUATOR.official_cell_id(0, "rectified_flow", 5, 4)
    path = tmp_path / "preregistered.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    preregistration_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()

    _loaded, cell, manifest_sha = EVALUATOR.load_preregistration(
        path,
        sequences,
        cell_id=cell_id,
        execution_horizon=4,
        final_freeze_token=token,
        preregistration_sha256=preregistration_sha256,
        runtime_attestation_sha256="c" * 64,
    )

    assert cell["checkpoint"]["sha256"] == hashlib.sha256(b"checkpoint:0:rectified_flow").hexdigest()
    assert len(manifest_sha) == 64
    with pytest.raises(RuntimeError, match="freeze token"):
        EVALUATOR.load_preregistration(
            path,
            sequences,
            cell_id=cell_id,
            execution_horizon=4,
            final_freeze_token="post-hoc token",
            preregistration_sha256=preregistration_sha256,
            runtime_attestation_sha256="c" * 64,
        )
    with pytest.raises(RuntimeError, match="runtime/data attestation"):
        EVALUATOR.load_preregistration(
            path,
            sequences,
            cell_id=cell_id,
            execution_horizon=4,
            final_freeze_token=token,
            preregistration_sha256=preregistration_sha256,
            runtime_attestation_sha256="d" * 64,
        )
    with pytest.raises(RuntimeError, match="externally supplied"):
        EVALUATOR.load_preregistration(
            path,
            sequences,
            cell_id=cell_id,
            execution_horizon=4,
            final_freeze_token=token,
            preregistration_sha256="e" * 64,
            runtime_attestation_sha256="c" * 64,
        )

    invalid_path = tmp_path / "invalid-and-unfrozen.json"
    invalid_path.write_bytes(b"{not JSON")
    with pytest.raises(RuntimeError, match="externally supplied"):
        EVALUATOR.load_preregistration(
            invalid_path,
            sequences,
            cell_id=cell_id,
            execution_horizon=4,
            final_freeze_token=token,
            preregistration_sha256="0" * 64,
            runtime_attestation_sha256="c" * 64,
        )


def test_authenticated_annotation_and_oracle_parse_the_verified_bytes_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    conf = tmp_path / "conf"
    annotation_path = conf / "annotations" / "new_playtable_validation.yaml"
    oracle_path = conf / "callbacks" / "rollout" / "tasks" / "new_playtable_tasks.yaml"
    annotation_path.parent.mkdir(parents=True)
    oracle_path.parent.mkdir(parents=True)
    annotation_raw = b"task_0:\n  - authenticated phrase\n"
    oracle_raw = b"oracle:\n  authenticated: true\n"
    annotation_path.write_bytes(annotation_raw)
    oracle_path.write_bytes(oracle_raw)
    _raw, annotation_identity = EVALUATOR._read_stable_regular_bytes(
        annotation_path,
        name="annotation",
    )
    _raw, oracle_identity = EVALUATOR._read_stable_regular_bytes(oracle_path, name="oracle")
    monkeypatch.setattr(EVALUATOR, "_official_conf_dir", lambda: conf)
    monkeypatch.setattr(EVALUATOR, "VALIDATION_ANNOTATIONS_SHA256", annotation_identity["sha256"])
    monkeypatch.setattr(EVALUATOR, "TASK_ORACLE_SHA256", oracle_identity["sha256"])
    parsed_texts: list[str] = []

    class FakeOmegaConf:
        @staticmethod
        def create(text: str) -> dict[str, Any]:
            parsed_texts.append(text)
            if text == annotation_raw.decode():
                annotation_path.write_text("task_0: [mutated phrase]\n", encoding="utf-8")
                return {"task_0": ["authenticated phrase"]}
            oracle_path.write_text("oracle: {authenticated: false}\n", encoding="utf-8")
            return {"oracle": {"authenticated": True}}

    fake_hydra = types.SimpleNamespace(utils=types.SimpleNamespace(instantiate=lambda value: {"instantiated": value}))
    monkeypatch.setitem(sys.modules, "omegaconf", types.SimpleNamespace(OmegaConf=FakeOmegaConf))
    monkeypatch.setitem(sys.modules, "hydra", fake_hydra)

    annotations, observed_annotation = EVALUATOR.load_validation_annotations(annotation_identity)
    oracle, observed_oracle = EVALUATOR.load_task_oracle(oracle_identity)

    assert annotations == {"task_0": ["authenticated phrase"]}
    assert oracle == {"instantiated": {"oracle": {"authenticated": True}}}
    assert observed_annotation == annotation_identity
    assert observed_oracle == oracle_identity
    assert parsed_texts == [annotation_raw.decode(), oracle_raw.decode()]


def test_validation_environment_instantiates_authenticated_config_without_path_reopen(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset_root = tmp_path / "task_ABC_D"
    config_path = dataset_root / "validation" / ".hydra" / "merged_config.yaml"
    config_path.parent.mkdir(parents=True)
    config_raw = b"authenticated: merged-config\n"
    config_path.write_bytes(config_raw)
    _raw, config_identity = EVALUATOR._read_stable_regular_bytes(config_path, name="config")

    config = types.SimpleNamespace(
        cameras={
            "static": types.SimpleNamespace(width=200, height=200),
            "gripper": types.SimpleNamespace(width=84, height=84),
            "tactile": types.SimpleNamespace(width=120, height=160),
        },
        env=types.SimpleNamespace(
            _target_="calvin_env.envs.play_table_env.PlayTableSimEnv",
            control_freq=30,
            use_egl=True,
        ),
        scene=types.SimpleNamespace(name="calvin_scene_D"),
    )
    parsed: list[str] = []

    class FakeOmegaConf:
        @staticmethod
        def create(text: str) -> Any:
            parsed.append(text)
            config_path.write_text("mutated: after-authentication\n", encoding="utf-8")
            return config

        @staticmethod
        def select(value: Any, dotted: str) -> Any:
            current = value
            for name in dotted.split("."):
                current = current[name] if isinstance(current, dict) else getattr(current, name)
            return current

    instantiated: list[tuple[Any, dict[str, Any]]] = []
    environment = types.SimpleNamespace(control_freq=30)
    global_hydra = types.SimpleNamespace(is_initialized=lambda: True)
    fake_hydra = types.SimpleNamespace(
        core=types.SimpleNamespace(
            global_hydra=types.SimpleNamespace(GlobalHydra=types.SimpleNamespace(instance=lambda: global_hydra))
        ),
        initialize=lambda _path: (_ for _ in ()).throw(AssertionError("unexpected Hydra initialization")),
        utils=types.SimpleNamespace(
            instantiate=lambda value, **kwargs: instantiated.append((value, kwargs)) or environment
        ),
    )
    monkeypatch.setitem(sys.modules, "omegaconf", types.SimpleNamespace(OmegaConf=FakeOmegaConf))
    monkeypatch.setitem(sys.modules, "hydra", fake_hydra)

    observed_environment, observed_identity = EVALUATOR.construct_validation_environment(
        dataset_root,
        config_identity,
    )

    assert observed_environment is environment
    assert observed_identity["merged_config_sha256"] == config_identity["sha256"]
    assert parsed == [config_raw.decode()]
    assert set(config.cameras) == {"static", "gripper"}
    assert instantiated == [
        (
            config.env,
            {"show_gui": False, "use_vr": False, "use_scene_info": True},
        )
    ]


def test_official_score_attests_and_binds_preregistration_before_yaml_or_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    attestation = {
        "attestation_sha256": "c" * 64,
        "dataset": {
            "validation_critical_files": {
                "validation/.hydra/merged_config.yaml": {"bytes": 1, "path": "/config", "sha256": "d" * 64}
            }
        },
        "runtime": {
            "official_yaml": {
                "task_oracle": {"bytes": 1, "path": "/oracle", "sha256": "e" * 64},
                "validation_annotations": {"bytes": 1, "path": "/annotations", "sha256": "f" * 64},
            },
            "sources": EVALUATOR._IMPORT_EVALUATOR_SOURCE_IDENTITIES,
        },
    }

    def attest(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        events.append("attestation")
        return attestation

    def sequences(_seed: int) -> list[tuple[dict[str, int], tuple[str, ...]]]:
        events.append("sequences")
        return [({"initial": 0}, TASKS)]

    def preregister(*_args: Any, **kwargs: Any) -> tuple[dict[str, Any], dict[str, Any], str]:
        events.append("preregistration")
        assert kwargs["preregistration_sha256"] == "a" * 64
        assert kwargs["runtime_attestation_sha256"] == "c" * 64
        return {}, {}, "a" * 64

    class StopBeforeYaml(RuntimeError):
        pass

    def annotations(_identity: dict[str, Any]) -> Any:
        events.append("yaml")
        raise StopBeforeYaml

    def forbidden(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("policy/oracle/environment capability was reached before raw attestation")

    monkeypatch.setattr(EVALUATOR, "build_official_attestation", attest)
    monkeypatch.setattr(EVALUATOR, "regenerate_official_sequences", sequences)
    monkeypatch.setattr(EVALUATOR, "load_preregistration", preregister)
    monkeypatch.setattr(EVALUATOR, "load_validation_annotations", annotations)
    monkeypatch.setattr(EVALUATOR, "load_task_oracle", forbidden)
    monkeypatch.setattr(EVALUATOR, "construct_validation_environment", forbidden)
    monkeypatch.setattr(EVALUATOR, "PolicyClient", forbidden)
    args = types.SimpleNamespace(
        cell_id="cell",
        dataset_root=Path("/dataset"),
        evaluation_seed=0,
        execution_horizon=4,
        final_freeze_token="frozen",
        preregistration_manifest=Path("/preregistration.json"),
        preregistration_sha256="a" * 64,
        source_root=Path("/source"),
    )

    with pytest.raises(StopBeforeYaml):
        EVALUATOR.run_official_score_mode(args)

    assert events == ["attestation", "sequences", "preregistration", "yaml"]


def test_infrastructure_mode_cannot_step_predict_or_query_oracle(monkeypatch: pytest.MonkeyPatch) -> None:
    environment = FakeEnvironment()

    def forbidden_step(_action: np.ndarray) -> Any:
        raise AssertionError("infrastructure mode stepped the environment")

    def forbidden_info() -> Any:
        raise AssertionError("infrastructure mode queried outcome info")

    environment.step = forbidden_step  # type: ignore[method-assign]
    environment.get_info = forbidden_info  # type: ignore[method-assign]
    fake_sequences = [({"initial": 0}, TASKS)]
    monkeypatch.setattr(EVALUATOR, "regenerate_official_sequences", lambda _seed: fake_sequences)
    attestation = {
        "attestation_sha256": "d" * 64,
        "dataset": {
            "validation_critical_files": {
                "validation/.hydra/merged_config.yaml": {"bytes": 1, "path": "/config", "sha256": "c" * 64}
            }
        },
        "runtime": {
            "official_yaml": {"validation_annotations": {"bytes": 1, "path": "/annotations", "sha256": "a" * 64}},
            "sources": EVALUATOR._IMPORT_EVALUATOR_SOURCE_IDENTITIES,
        },
    }
    monkeypatch.setattr(EVALUATOR, "build_official_attestation", lambda *_args, **_kwargs: attestation)
    monkeypatch.setattr(
        EVALUATOR,
        "load_validation_annotations",
        lambda _identity: (ANNOTATIONS, {"sha256": "a" * 64}),
    )
    monkeypatch.setattr(
        EVALUATOR,
        "construct_validation_environment",
        lambda _root, _identity: (environment, {"scene": EVALUATOR.VALIDATION_SCENE}),
    )

    calvin_agent = types.ModuleType("calvin_agent")
    evaluation = types.ModuleType("calvin_agent.evaluation")
    utils = types.ModuleType("calvin_agent.evaluation.utils")

    def convert(_initial_state: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
        robot_obs = np.zeros(15)
        robot_obs[14] = -1
        return robot_obs, np.zeros(24)

    utils.get_env_state_for_initial_condition = convert  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "calvin_agent", calvin_agent)
    monkeypatch.setitem(sys.modules, "calvin_agent.evaluation", evaluation)
    monkeypatch.setitem(sys.modules, "calvin_agent.evaluation.utils", utils)

    report = EVALUATOR.run_infrastructure_mode(Path("unused"), evaluation_seed=0, source_root=Path("source"))

    assert report["policy_predict_calls"] == 0
    assert report["oracle_outcomes_inspected"] is False
    assert report["attestation_sha256"] == "d" * 64
    assert environment.step_count == 0
    assert len(environment.reset_calls) == 1
    assert environment.closed is True


def test_transactional_journal_writes_ordered_jsonl_then_summary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(EVALUATOR, "NUM_SEQUENCES", 2)
    output_dir = tmp_path / "official-run"
    journal = EVALUATOR.EvaluationJournal(output_dir, {"schema": EVALUATOR.RUN_SCHEMA})
    journal.start()
    journal.append_episode({"sequence_idx": 0, "successful_subtasks": 1})
    journal.append_episode({"sequence_idx": 1, "successful_subtasks": 2})
    summary = {"AvgLen": 1.5, "schema": EVALUATOR.SUMMARY_SCHEMA}
    journal.complete(summary)

    run = json.loads((output_dir / "run.json").read_text())
    episodes = [json.loads(line) for line in (output_dir / "episodes.jsonl").read_text().splitlines()]
    assert run["status"] == "complete"
    assert run["sequence_records"] == 2
    assert len(run["episodes_jsonl_sha256"]) == len(run["summary_json_sha256"]) == 64
    assert episodes == [
        {"sequence_idx": 0, "successful_subtasks": 1},
        {"sequence_idx": 1, "successful_subtasks": 2},
    ]
    assert json.loads((output_dir / "summary.json").read_text()) == summary


def test_journal_post_commit_guard_failure_demotes_complete_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(EVALUATOR, "NUM_SEQUENCES", 1)
    guard_calls = 0

    def guard() -> None:
        nonlocal guard_calls
        guard_calls += 1
        if guard_calls == 3:
            raise RuntimeError("source changed at post-commit boundary")

    output_dir = tmp_path / "guarded-run"
    journal = EVALUATOR.EvaluationJournal(
        output_dir,
        {"schema": EVALUATOR.RUN_SCHEMA},
        commit_guard=guard,
    )
    journal.start()
    journal.append_episode({"sequence_idx": 0, "successful_subtasks": 0})

    with pytest.raises(RuntimeError, match="post-commit"):
        journal.complete({"AvgLen": 0.0, "schema": EVALUATOR.SUMMARY_SCHEMA})

    run = json.loads((output_dir / "run.json").read_text())
    assert guard_calls == 3
    assert run["status"] == "failed"
    assert run["error"]["message"] == "source changed at post-commit boundary"


@pytest.mark.parametrize("mutation", ["inode", "content", "link-count"])
def test_journal_rejects_summary_target_mutation_during_commit_guard(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    monkeypatch.setattr(EVALUATOR, "NUM_SEQUENCES", 1)
    output_dir = tmp_path / f"guarded-summary-{mutation}"
    summary_path = output_dir / "summary.json"
    extra_link = output_dir / "summary-extra-link.json"
    guard_calls = 0

    def guard() -> None:
        nonlocal guard_calls
        guard_calls += 1
        if guard_calls == 2:
            if mutation == "inode":
                summary_path.unlink()
                summary_path.write_text("replacement\n", encoding="utf-8")
            elif mutation == "content":
                summary_path.write_text("tampered in place\n", encoding="utf-8")
            else:
                extra_link.hardlink_to(summary_path)

    journal = EVALUATOR.EvaluationJournal(
        output_dir,
        {"schema": EVALUATOR.RUN_SCHEMA},
        commit_guard=guard,
    )
    journal.start()
    journal.append_episode({"sequence_idx": 0, "successful_subtasks": 0})

    with pytest.raises(RuntimeError, match="published summary JSON"):
        journal.complete({"AvgLen": 0.0, "schema": EVALUATOR.SUMMARY_SCHEMA})

    run = json.loads((output_dir / "run.json").read_text())
    assert guard_calls == 2
    assert run["status"] == "failed"
    assert "summary JSON" in run["error"]["message"]


def test_journal_rejects_complete_run_inode_substitution_during_post_guard(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(EVALUATOR, "NUM_SEQUENCES", 1)
    output_dir = tmp_path / "guarded-run-inode"
    run_path = output_dir / "run.json"
    guard_calls = 0

    def guard() -> None:
        nonlocal guard_calls
        guard_calls += 1
        if guard_calls == 3:
            run_path.unlink()
            run_path.write_text("replacement\n", encoding="utf-8")

    journal = EVALUATOR.EvaluationJournal(
        output_dir,
        {"schema": EVALUATOR.RUN_SCHEMA},
        commit_guard=guard,
    )
    journal.start()
    journal.append_episode({"sequence_idx": 0, "successful_subtasks": 0})

    with pytest.raises(RuntimeError, match="complete run JSON"):
        journal.complete({"AvgLen": 0.0, "schema": EVALUATOR.SUMMARY_SCHEMA})

    run = json.loads(run_path.read_text())
    assert guard_calls == 3
    assert run["status"] == "failed"
    assert "complete run JSON" in run["error"]["message"]


def test_mode_arguments_make_official_score_opt_in() -> None:
    infrastructure = EVALUATOR.parse_args(["--mode", "infrastructure", "--dataset-root", "/dataset"])
    EVALUATOR.validate_mode_arguments(infrastructure)

    missing_freeze = EVALUATOR.parse_args(
        [
            "--mode",
            "official-score",
            "--dataset-root",
            "/dataset",
            "--execution-horizon",
            "4",
            "--output-dir",
            "/output",
            "--preregistration-manifest",
            "/frozen.json",
            "--preregistration-sha256",
            "a" * 64,
            "--cell-id",
            "cell",
        ]
    )
    with pytest.raises(RuntimeError, match="final-freeze-token"):
        EVALUATOR.validate_mode_arguments(missing_freeze)
