"""Fail-closed tests for the canonical CALVIN 24-cell evaluation matrix."""

from __future__ import annotations

import copy
import hashlib
import json
import sys
import types
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
CALVIN_SCRIPTS = ROOT / "scripts" / "calvin"
sys.path.insert(0, str(CALVIN_SCRIPTS))

import aggregate_calvin_official as AGGREGATOR  # noqa: E402
import create_calvin_preregistration as CREATOR  # noqa: E402
import evaluate_calvin as EVALUATOR  # noqa: E402

TASKS = [f"task_{index}" for index in range(5)]
INSTRUCTIONS = {task: f"perform task {index}" for index, task in enumerate(TASKS)}


@pytest.mark.parametrize("mutated_name", CREATOR._EVALUATOR_SOURCE_NAMES)
def test_preregistration_creator_import_snapshot_rejects_each_source_mutation(
    tmp_path: Path,
    mutated_name: str,
) -> None:
    for name in CREATOR._EVALUATOR_SOURCE_NAMES:
        (tmp_path / name).write_text(f"source:{name}\n", encoding="utf-8")
    snapshot = CREATOR._capture_evaluator_source_identities(tmp_path)
    (tmp_path / mutated_name).write_text("mutated source\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="changed after import-time snapshot"):
        CREATOR._require_evaluator_sources_unchanged(snapshot)


def test_preregistration_creator_rejects_its_own_source_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "create_calvin_preregistration.py"
    source.write_text("creator source\n", encoding="utf-8")
    monkeypatch.setattr(CREATOR, "_CREATOR_SOURCE_PATH", source)
    monkeypatch.setattr(CREATOR, "_IMPORT_CREATOR_SOURCE_IDENTITY", CREATOR._source_file_identity(source))
    CREATOR._require_creator_source_unchanged()

    source.write_text("mutated creator source\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="creator source changed"):
        CREATOR._require_creator_source_unchanged()


def _geometry() -> dict[str, Any]:
    return {
        "expert_batch_isolation": "sample_isolated_grouped_mm_v1",
        "experts_implementation": "grouped_mm",
        "fixed_physical_prefix_width": 600,
        "physical_batch_size": 8,
        "prefix_geometry_content_sha256": "6" * 64,
    }


def _calvin_identity(*, metadata_sha256: str = "5" * 64) -> dict[str, Any]:
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
        "storage_identity_sha256": "6" * 64,
        "storage_mode": "archive-direct",
    }


def _input_cells() -> list[dict[str, Any]]:
    cells = []
    for seed, objective, nfe, execution_horizon in EVALUATOR.official_factor_matrix():
        contract = EVALUATOR.selected_policy_contract(objective, nfe)
        cells.append(
            {
                "cell_id": EVALUATOR.official_cell_id(seed, objective, nfe, execution_horizon),
                "checkpoint": {"sha256": hashlib.sha256(f"checkpoint:{seed}:{objective}".encode()).hexdigest()},
                "execution_geometry": _geometry(),
                "execution_horizon": execution_horizon,
                "policy": {
                    "inference_seed_behavior": contract["inference_seed_behavior"],
                    "nfe": nfe,
                    "objective": objective,
                    "sampler": contract["sampler"],
                    "train_seed": seed,
                },
                "serving_runtime_sha256": hashlib.sha256(f"runtime:{seed}:{objective}".encode()).hexdigest(),
            }
        )
    return cells


@pytest.fixture
def manifest(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    sequences = [[{"initial": 0}, TASKS]]
    digest = EVALUATOR.canonical_sequence_sha256(sequences)
    monkeypatch.setattr(EVALUATOR, "NUM_SEQUENCES", 1)
    monkeypatch.setattr(EVALUATOR, "SEQUENCE_SHA256", digest)
    monkeypatch.setattr(CREATOR, "NUM_SEQUENCES", 1)
    monkeypatch.setattr(CREATOR, "SEQUENCE_SHA256", digest)
    return CREATOR.build_manifest(
        {"cells": _input_cells()},
        aggregator_sha256=AGGREGATOR._STARTUP_AGGREGATION_SOURCE_IDENTITIES["aggregate_calvin_official.py"]["sha256"],
        attestation_sha256="c" * 64,
        final_freeze_token="frozen before scoring",
        sequences=sequences,
    )


def test_creator_derives_exact_canonical_matrix_and_policy_identities(manifest: dict[str, Any]) -> None:
    assert len(manifest["cells"]) == 24
    assert len({cell["cell_id"] for cell in manifest["cells"]}) == 24
    assert len({cell["checkpoint"]["sha256"] for cell in manifest["cells"]}) == 6
    assert manifest["training_seeds"] == [0, 1, 2]
    assert manifest["flow_nfes"] == [1, 5, 10]
    assert manifest["aggregation_python_version"] == EVALUATOR.PYTHON_VERSION
    assert (
        manifest["aggregator_sha256"]
        == AGGREGATOR._STARTUP_AGGREGATION_SOURCE_IDENTITIES["aggregate_calvin_official.py"]["sha256"]
    )
    for cell in manifest["cells"]:
        policy = cell["policy"]
        expected = EVALUATOR.selected_policy_contract(policy["objective"], policy["nfe"])
        assert policy["identity_sha256"] == EVALUATOR.canonical_sha256(expected)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda value: value["cells"].pop(), "exact 24-cell"),
        (lambda value: value["cells"].append(copy.deepcopy(value["cells"][0])), "unique"),
        (lambda value: value["cells"][0].__setitem__("cell_id", "arbitrary"), "canonical"),
        (
            lambda value: value["cells"][0]["policy"].__setitem__("train_seed", 7),
            "one of",
        ),
        (
            lambda value: value["cells"][0]["policy"].__setitem__("identity_sha256", "f" * 64),
            "identity SHA-256",
        ),
        (
            lambda value: value["cells"][0]["checkpoint"].__setitem__("sha256", "f" * 64),
            "share one final checkpoint",
        ),
        (
            lambda value: value["cells"][0]["execution_geometry"].__setitem__("fixed_physical_prefix_width", 601),
            "share one execution geometry",
        ),
        (lambda value: value.__setitem__("aggregator_sha256", "not-a-digest"), "aggregator_sha256"),
        (lambda value: value.__setitem__("aggregation_python_version", "3.11.0"), "Python 3.8.20"),
    ],
)
def test_preregistration_rejects_missing_duplicate_and_off_matrix_cells(
    manifest: dict[str, Any], mutate: Any, message: str
) -> None:
    changed = copy.deepcopy(manifest)
    mutate(changed)
    with pytest.raises(RuntimeError, match=message):
        EVALUATOR.validate_preregistration_manifest(
            changed,
            changed["sequences"],
            runtime_attestation_sha256="c" * 64,
        )


def _inventory(manifest: dict[str, Any], root: Path) -> dict[str, Any]:
    runs = []
    for cell in manifest["cells"]:
        output_dir = root / cell["cell_id"]
        output_dir.mkdir()
        runs.append(
            {
                "cell_id": cell["cell_id"],
                "episodes_jsonl_sha256": "e" * 64,
                "output_dir": str(output_dir),
                "run_json_sha256": "a" * 64,
                "summary_json_sha256": "b" * 64,
            }
        )
    return {
        "preregistration_sha256": "d" * 64,
        "runs": runs,
        "schema": AGGREGATOR.RUN_INVENTORY_SCHEMA,
    }


def test_aggregator_requires_exactly_one_complete_run_per_cell(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    manifest: dict[str, Any],
) -> None:
    monkeypatch.setattr(AGGREGATOR, "NUM_SEQUENCES", 1)

    def validate(entry: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        cell = kwargs["registered_cell"]
        policy = cell["policy"]
        seed = policy["train_seed"]
        return {
            **{f"SR{depth}": (seed + depth) / 10 for depth in range(1, 6)},
            "AvgLen": sum((seed + depth) / 10 for depth in range(1, 6)),
            "cell_id": cell["cell_id"],
            "episodes_jsonl_sha256": entry["episodes_jsonl_sha256"],
            "execution_horizon": cell["execution_horizon"],
            "nfe": policy["nfe"],
            "objective": policy["objective"],
            "run_json_sha256": entry["run_json_sha256"],
            "summary_json_sha256": entry["summary_json_sha256"],
            "train_seed": seed,
        }

    monkeypatch.setattr(AGGREGATOR, "validate_run_artifact", validate)
    inventory = _inventory(manifest, tmp_path)
    result = AGGREGATOR.aggregate_matrix(
        manifest,
        "d" * 64,
        inventory,
        "f" * 64,
        inventory_dir=tmp_path,
    )
    assert result["cell_count"] == 24
    assert result["comparison_count"] == 8
    assert len(result["content_sha256"]) == 64
    assert result["aggregator_sha256"] == manifest["aggregator_sha256"]
    assert result["aggregation_source_identities"] == AGGREGATOR._STARTUP_AGGREGATION_SOURCE_IDENTITIES
    assert all(len(group["metrics"]["AvgLen"]["values_by_train_seed"]) == 3 for group in result["comparisons"])

    changed_source = copy.deepcopy(manifest)
    changed_source["aggregator_sha256"] = "a" * 64
    with pytest.raises(RuntimeError, match="pre-registered aggregator"):
        AGGREGATOR.aggregate_matrix(changed_source, "d" * 64, inventory, "f" * 64, inventory_dir=tmp_path)

    missing = copy.deepcopy(inventory)
    missing["runs"].pop()
    with pytest.raises(RuntimeError, match="exactly 24"):
        AGGREGATOR.aggregate_matrix(manifest, "d" * 64, missing, "f" * 64, inventory_dir=tmp_path)

    duplicate = copy.deepcopy(inventory)
    duplicate["runs"][-1]["cell_id"] = duplicate["runs"][0]["cell_id"]
    with pytest.raises(RuntimeError, match="duplicate"):
        AGGREGATOR.aggregate_matrix(manifest, "d" * 64, duplicate, "f" * 64, inventory_dir=tmp_path)

    off_matrix = copy.deepcopy(inventory)
    off_matrix["runs"][-1]["cell_id"] = "seed-99-flow-nfe-99-k-99"
    with pytest.raises(RuntimeError, match="missing or off-matrix"):
        AGGREGATOR.aggregate_matrix(manifest, "d" * 64, off_matrix, "f" * 64, inventory_dir=tmp_path)


def test_metric_aggregate_uses_n_minus_one_sample_standard_deviation() -> None:
    values = [
        {"train_seed": 0, "SR1": 0.0},
        {"train_seed": 1, "SR1": 1.0},
        {"train_seed": 2, "SR1": 2.0},
    ]
    aggregate = AGGREGATOR._metric_aggregate(values, "SR1")
    assert aggregate["mean"] == 1.0
    assert aggregate["sample_std"] == 1.0


@pytest.mark.parametrize("mutated_name", AGGREGATOR._AGGREGATION_SOURCE_NAMES)
def test_aggregation_source_snapshot_rejects_live_mutation(tmp_path: Path, mutated_name: str) -> None:
    for name in AGGREGATOR._AGGREGATION_SOURCE_NAMES:
        (tmp_path / name).write_text(f"source:{name}\n", encoding="utf-8")
    identities = AGGREGATOR.capture_aggregation_source_identities(tmp_path)
    AGGREGATOR.require_aggregation_sources_unchanged(identities)
    (tmp_path / mutated_name).write_text("mutated source\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="changed after process startup"):
        AGGREGATOR.require_aggregation_sources_unchanged(identities)


@pytest.mark.parametrize(
    ("helper", "source_name", "identity_reader"),
    [
        (CREATOR._require_imported_local_module, "evaluate_calvin.py", CREATOR._source_file_identity),
        (AGGREGATOR._require_imported_local_module, "evaluate_calvin.py", AGGREGATOR._source_file_identity),
    ],
)
def test_creator_and_aggregator_bind_imported_module_path_and_bytes(
    tmp_path: Path,
    helper: Any,
    source_name: str,
    identity_reader: Any,
) -> None:
    source = tmp_path / source_name
    source.write_text("source = 1\n", encoding="utf-8")
    identity = identity_reader(source)
    module = types.SimpleNamespace(
        __file__=str(source),
        __spec__=types.SimpleNamespace(origin=str(source)),
    )
    helper(module, source_name, {source_name: identity})

    source.write_text("source = 2\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="differs from"):
        helper(module, source_name, {source_name: identity})


def test_rehashed_attestation_cannot_substitute_evaluator_source() -> None:
    attestation = {
        "dataset": {},
        "runtime": {
            "sources": {
                name: copy.deepcopy(AGGREGATOR._STARTUP_AGGREGATION_SOURCE_IDENTITIES[name])
                for name in AGGREGATOR._EVALUATOR_SOURCE_NAMES
            }
        },
        "schema": AGGREGATOR.ATTESTATION_SCHEMA,
    }
    attestation["runtime"]["sources"]["evaluate_calvin.py"]["sha256"] = "f" * 64
    attestation["attestation_sha256"] = EVALUATOR.canonical_sha256(attestation)
    with pytest.raises(RuntimeError, match="differ from aggregation startup"):
        AGGREGATOR._validate_attestation(
            attestation,
            attestation["attestation_sha256"],
            aggregation_sources=AGGREGATOR._STARTUP_AGGREGATION_SOURCE_IDENTITIES,
        )


def test_exclusive_json_pair_publication_is_complete_and_collision_safe(tmp_path: Path) -> None:
    output = tmp_path / "preregistration.json"
    digest = CREATOR._write_exclusive_json(output, {"value": 1})
    companion = output.with_suffix(".json.sha256")
    assert companion.read_text(encoding="ascii") == f"{digest}  {output.name}\n"
    original = output.read_bytes()
    with pytest.raises(FileExistsError):
        CREATOR._write_exclusive_json(output, {"value": 2})
    assert output.read_bytes() == original

    companion_collision = tmp_path / "companion-collision.json"
    companion_collision_sha = companion_collision.with_suffix(".json.sha256")
    companion_collision_sha.write_text("pre-existing\n", encoding="ascii")
    with pytest.raises(FileExistsError):
        CREATOR._write_exclusive_json(companion_collision, {"value": 3})
    assert not companion_collision.exists()
    assert companion_collision_sha.read_text(encoding="ascii") == "pre-existing\n"

    payload_collision = tmp_path / "payload-collision.json"
    payload_collision.write_text("pre-existing\n", encoding="ascii")
    with pytest.raises(FileExistsError):
        CREATOR._write_exclusive_json(payload_collision, {"value": 4})
    assert payload_collision.read_text(encoding="ascii") == "pre-existing\n"
    assert not payload_collision.with_suffix(".json.sha256").exists()
    assert not list(tmp_path.glob(".*.tmp-*"))


@pytest.mark.parametrize("writer", [CREATOR._write_exclusive_json, AGGREGATOR._write_exclusive])
@pytest.mark.parametrize("failure_call", [1, 2, 3])
def test_exclusive_publication_guard_failure_removes_payload_and_sidecar(
    tmp_path: Path,
    writer: Any,
    failure_call: int,
) -> None:
    output = tmp_path / f"guarded-{writer.__module__}.json"
    calls = 0

    def guard() -> None:
        nonlocal calls
        calls += 1
        if calls == failure_call:
            raise RuntimeError("source changed after commit")

    with pytest.raises(RuntimeError, match="source changed"):
        writer(output, {"value": 1}, commit_guard=guard)

    assert calls == failure_call
    assert not output.exists()
    assert not output.with_suffix(".json.sha256").exists()
    assert not list(tmp_path.glob(".*.tmp-*"))


@pytest.mark.parametrize("writer", [CREATOR._write_exclusive_json, AGGREGATOR._write_exclusive])
@pytest.mark.parametrize("target_name", ["payload", "companion"])
def test_exclusive_publication_rejects_target_inode_substitution(
    tmp_path: Path,
    writer: Any,
    target_name: str,
) -> None:
    output = tmp_path / f"substituted-{writer.__module__}-{target_name}.json"
    companion = output.with_suffix(".json.sha256")
    replacement = output if target_name == "payload" else companion
    replacement_call = 3 if target_name == "payload" else 2
    calls = 0

    def guard() -> None:
        nonlocal calls
        calls += 1
        if calls == replacement_call:
            replacement.unlink()
            replacement.write_text("attacker replacement\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="target identity"):
        writer(output, {"value": 1}, commit_guard=guard)

    assert replacement.read_text(encoding="utf-8") == "attacker replacement\n"
    untouched = companion if target_name == "payload" else output
    assert not untouched.exists()
    assert not list(tmp_path.glob(".*.tmp-*"))


@pytest.mark.parametrize("writer", [CREATOR._write_exclusive_json, AGGREGATOR._write_exclusive])
@pytest.mark.parametrize("mutation", ["content", "link-count"])
def test_exclusive_publication_rejects_target_content_or_link_count_mutation(
    tmp_path: Path,
    writer: Any,
    mutation: str,
) -> None:
    output = tmp_path / f"mutated-{writer.__module__}-{mutation}.json"
    extra_link = tmp_path / f"extra-{writer.__module__}-{mutation}.json"
    calls = 0

    def guard() -> None:
        nonlocal calls
        calls += 1
        if calls == 3:
            if mutation == "content":
                output.write_text("tampered in place\n", encoding="utf-8")
            else:
                extra_link.hardlink_to(output)

    with pytest.raises(RuntimeError, match=r"identity|content|link count"):
        writer(output, {"value": 1}, commit_guard=guard)

    assert not output.exists()
    assert not output.with_suffix(".json.sha256").exists()
    assert extra_link.exists() is (mutation == "link-count")
    assert not list(tmp_path.glob(".*.tmp-*"))


def _failed_sequence_record(sequence_sha256: str) -> dict[str, Any]:
    calls = 90
    subtask = {
        "action_clip_fraction": 0.0,
        "action_clipped_channels": 0,
        "action_continuous_channels": 2160,
        "discarded_queued_actions_on_success": 0,
        "elapsed_seconds": 1.0,
        "environment_actions": 360,
        "execution_horizon": 4,
        "instruction": INSTRUCTIONS[TASKS[0]],
        "max_environment_actions": 360,
        "policy_calls": calls,
        "policy_identity_first_replan_idx": 0,
        "policy_latency_p50_seconds": 0.0,
        "policy_latency_p95_seconds": 0.0,
        "policy_latency_seconds": [0.0] * calls,
        "server_latency_p50_seconds": 0.0,
        "server_latency_p95_seconds": 0.0,
        "server_latency_seconds": [0.0] * calls,
        "steps_to_success": None,
        "subtask_idx": 0,
        "subtask_name": TASKS[0],
        "success": False,
    }
    return {
        "elapsed_seconds": 1.0,
        "evaluation_seed": 0,
        "execution_horizon": 4,
        "schema": EVALUATOR.EPISODE_SCHEMA,
        "sequence_idx": 0,
        "sequence_sha256": sequence_sha256,
        "sequence_success": False,
        "subtasks": [subtask],
        "successful_subtasks": 0,
    }


def test_episode_authentication_rejects_metric_and_task_tampering(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sequences = [[{"initial": 0}, TASKS]]
    digest = EVALUATOR.canonical_sequence_sha256(sequences)
    monkeypatch.setattr(AGGREGATOR, "NUM_SEQUENCES", 1)
    monkeypatch.setattr(AGGREGATOR, "SEQUENCE_SHA256", digest)
    record = _failed_sequence_record(digest)
    assert AGGREGATOR.validate_sequence_records(
        [record],
        sequences=sequences,
        execution_horizon=4,
        expected_instructions=INSTRUCTIONS,
    ) == [record]

    changed = copy.deepcopy(record)
    changed["subtasks"][0]["action_clip_fraction"] = 0.1
    with pytest.raises(RuntimeError, match="clip fraction"):
        AGGREGATOR.validate_sequence_records(
            [changed],
            sequences=sequences,
            execution_horizon=4,
            expected_instructions=INSTRUCTIONS,
        )

    changed = copy.deepcopy(record)
    changed["subtasks"][0]["subtask_name"] = TASKS[1]
    with pytest.raises(RuntimeError, match="differs from sequence"):
        AGGREGATOR.validate_sequence_records(
            [changed],
            sequences=sequences,
            execution_horizon=4,
            expected_instructions=INSTRUCTIONS,
        )

    changed = copy.deepcopy(record)
    changed["subtasks"][0]["instruction"] = "a different but non-empty instruction"
    with pytest.raises(RuntimeError, match="authenticated first validation phrase"):
        AGGREGATOR.validate_sequence_records(
            [changed],
            sequences=sequences,
            execution_horizon=4,
            expected_instructions=INSTRUCTIONS,
        )


def test_run_artifact_is_bound_and_summary_is_recomputed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    manifest: dict[str, Any],
) -> None:
    digest = manifest["sequence_sha256"]
    monkeypatch.setattr(AGGREGATOR, "NUM_SEQUENCES", 1)
    monkeypatch.setattr(AGGREGATOR, "SEQUENCE_SHA256", digest)
    monkeypatch.setattr(EVALUATOR, "NUM_SEQUENCES", 1)
    monkeypatch.setattr(EVALUATOR, "SEQUENCE_SHA256", digest)
    cell = next(
        value
        for value in manifest["cells"]
        if value["cell_id"] == EVALUATOR.official_cell_id(0, "rectified_flow", 5, 4)
    )
    record = _failed_sequence_record(digest)
    episodes_payload = (json.dumps(record, allow_nan=False, separators=(",", ":"), sort_keys=True) + "\n").encode()
    episodes_sha256 = hashlib.sha256(episodes_payload).hexdigest()
    summary = EVALUATOR.summarize_sequences([record])
    summary_payload = (json.dumps(summary, allow_nan=False, indent=2, sort_keys=True) + "\n").encode()
    summary_sha256 = hashlib.sha256(summary_payload).hexdigest()

    annotation = {"bytes": 1, "path": "/annotations", "sha256": "1" * 64}
    oracle = {"bytes": 1, "path": "/oracle", "sha256": "2" * 64}
    config = {
        "bytes": 1,
        "crc32": 1,
        "path": "/dataset/validation/.hydra/merged_config.yaml",
        "sha256": "3" * 64,
    }

    def load_annotations(expected_identity: dict[str, Any]) -> tuple[dict[str, list[str]], dict[str, Any]]:
        assert expected_identity == annotation
        return ({task: [INSTRUCTIONS[task], "unused phrase"] for task in TASKS}, dict(annotation))

    monkeypatch.setattr(AGGREGATOR, "load_validation_annotations", load_annotations)
    monkeypatch.setattr(AGGREGATOR, "validate_official_dataset_identity", lambda value: dict(value))
    calvin_identity = _calvin_identity()
    attestation = {
        "dataset": {
            "calvin_identity": calvin_identity,
            "validation_critical_files": {"validation/.hydra/merged_config.yaml": config},
        },
        "runtime": {
            "official_yaml": {"task_oracle": oracle, "validation_annotations": annotation},
            "sources": {
                name: copy.deepcopy(AGGREGATOR._STARTUP_AGGREGATION_SOURCE_IDENTITIES[name])
                for name in AGGREGATOR._EVALUATOR_SOURCE_NAMES
            },
        },
        "schema": AGGREGATOR.ATTESTATION_SCHEMA,
    }
    attestation["attestation_sha256"] = EVALUATOR.canonical_sha256(attestation)
    manifest["runtime_attestation_sha256"] = attestation["attestation_sha256"]
    policy = cell["policy"]
    health = {
        "action_dim": EVALUATOR.ACTION_DIM,
        "action_horizon": EVALUATOR.ACTION_HORIZON,
        "calvin_identity": calvin_identity,
        "checkpoint_manifest_sha256": cell["checkpoint"]["sha256"],
        "execution_horizons": list(EVALUATOR.SUPPORTED_EXECUTION_HORIZONS),
        "execution_geometry": _geometry(),
        "gripper_image_shape": list(EVALUATOR.GRIPPER_IMAGE_SHAPE),
        "mode": "real",
        "model_revision": EVALUATOR.MODEL_REVISION,
        "nfe": policy["nfe"],
        "normalization_content_sha256": "4" * 64,
        "normalization_metadata_sha256": "5" * 64,
        "objective": policy["objective"],
        "operation": "health",
        "policy_contract_sha256": policy["identity_sha256"],
        "protocol": EVALUATOR.PROTOCOL,
        "request_id": "health-1",
        "sampler": policy["sampler"],
        "schema": EVALUATOR.SCHEMA,
        "sequence_sha256": digest,
        "serving_runtime_sha256": cell["serving_runtime_sha256"],
        "state_dim": EVALUATOR.STATE_DIM,
        "static_image_shape": list(EVALUATOR.STATIC_IMAGE_SHAPE),
        "status": "ok",
        "train_seed": policy["train_seed"],
    }
    run = {
        "annotation": annotation,
        "attestation": attestation,
        "attestation_sha256": attestation["attestation_sha256"],
        "cell": cell,
        "created_utc": "2026-08-30T00:00:00+00:00",
        "environment": {
            "control_frequency_hz": 30,
            "merged_config_path": config["path"],
            "merged_config_sha256": config["sha256"],
            "scene": "calvin_scene_D",
            "validation_path": "/dataset/validation",
        },
        "episodes_jsonl_sha256": episodes_sha256,
        "evaluation_seed": 0,
        "execution_horizon": 4,
        "final_freeze_token_sha256": manifest["final_freeze_token_sha256"],
        "finished_utc": "2026-08-30T01:00:00+00:00",
        "mode": "official-score",
        "oracle": oracle,
        "policy_health": health,
        "policy_socket": "/run/calvin.sock",
        "preregistration_manifest": "/freeze/calvin.json",
        "preregistration_sha256": "d" * 64,
        "protocol": EVALUATOR.PROTOCOL,
        "schema": EVALUATOR.RUN_SCHEMA,
        "sequence_count": 1,
        "sequence_records": 1,
        "sequence_sha256": digest,
        "status": "complete",
        "summary_json_sha256": summary_sha256,
    }
    run_payload = (json.dumps(run, allow_nan=False, indent=2, sort_keys=True) + "\n").encode()
    run_sha256 = hashlib.sha256(run_payload).hexdigest()
    output_dir = tmp_path / "run"
    output_dir.mkdir()
    (output_dir / "episodes.jsonl").write_bytes(episodes_payload)
    (output_dir / "summary.json").write_bytes(summary_payload)
    (output_dir / "run.json").write_bytes(run_payload)
    entry = {
        "cell_id": cell["cell_id"],
        "episodes_jsonl_sha256": episodes_sha256,
        "output_dir": str(output_dir),
        "run_json_sha256": run_sha256,
        "summary_json_sha256": summary_sha256,
    }

    result = AGGREGATOR.validate_run_artifact(
        entry,
        inventory_dir=tmp_path,
        preregistration=manifest,
        preregistration_sha256="d" * 64,
        registered_cell=cell,
    )
    assert result["AvgLen"] == 0.0
    assert result["cell_id"] == cell["cell_id"]

    (output_dir / "summary.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="externally supplied SHA-256"):
        AGGREGATOR.validate_run_artifact(
            entry,
            inventory_dir=tmp_path,
            preregistration=manifest,
            preregistration_sha256="d" * 64,
            registered_cell=cell,
        )
