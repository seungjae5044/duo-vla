#!/usr/bin/env python3
# ruff: noqa: E402, I001, UP006, UP035, UP045
"""Authenticate, complete-check, and aggregate all 24 official CALVIN cells."""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import math
import os
import platform
import stat
import statistics
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Set, Tuple

_SCRIPT_DIR = Path(__file__).resolve().parent
_EVALUATOR_SOURCE_NAMES = ("evaluate_calvin.py", "calvin_bridge.py", "preflight.py")
_AGGREGATION_SOURCE_NAMES = (
    "aggregate_calvin_official.py",
    "evaluate_calvin.py",
    "calvin_bridge.py",
    "preflight.py",
)


def _source_file_identity(path: Path) -> Dict[str, Any]:
    """Hash one regular, non-symlink source through a stable open descriptor."""

    flags = os.O_RDONLY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(str(path), flags)
    except OSError as exc:
        raise RuntimeError(f"cannot open aggregation source: {path}") from exc
    digest = hashlib.sha256()
    size = 0
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise RuntimeError(f"aggregation source is not a regular file: {path}")
        with os.fdopen(descriptor, "rb") as source:
            descriptor = -1
            while True:
                block = source.read(1024 * 1024)
                if not block:
                    break
                size += len(block)
                digest.update(block)
            after = os.fstat(source.fileno())
        stable_fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
        if any(getattr(before, name) != getattr(after, name) for name in stable_fields) or size != after.st_size:
            raise RuntimeError(f"aggregation source changed while being hashed: {path}")
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    return {"bytes": size, "path": str(path.resolve()), "sha256": digest.hexdigest()}


def capture_aggregation_source_identities(script_dir: Path = _SCRIPT_DIR) -> Dict[str, Dict[str, Any]]:
    root = script_dir.resolve()
    return {name: _source_file_identity(root / name) for name in _AGGREGATION_SOURCE_NAMES}


# Capture every locally imported evaluator dependency before importing it.  The
# snapshot is later bound both to the pre-registration and each run attestation.
_STARTUP_AGGREGATION_SOURCE_IDENTITIES = capture_aggregation_source_identities()


def _require_imported_local_module(module: Any, source_name: str, expected: Mapping[str, Any]) -> None:
    identity = expected.get(source_name)
    if source_name not in _EVALUATOR_SOURCE_NAMES or not isinstance(identity, Mapping):
        raise RuntimeError(f"aggregation dependency identity {source_name} is invalid")
    if set(identity) != {"bytes", "path", "sha256"}:
        raise RuntimeError(f"aggregation dependency identity {source_name} fields are invalid")
    module_file = getattr(module, "__file__", None)
    module_spec = getattr(module, "__spec__", None)
    spec_origin = getattr(module_spec, "origin", None)
    if not isinstance(module_file, str) or Path(module_file).resolve() != Path(identity["path"]):
        raise RuntimeError(f"imported aggregation dependency {source_name} has an unexpected __file__")
    if not isinstance(spec_origin, str) or Path(spec_origin).resolve() != Path(identity["path"]):
        raise RuntimeError(f"imported aggregation dependency {source_name} has an unexpected spec origin")
    if _source_file_identity(Path(module_file)) != identity:
        raise RuntimeError(f"imported aggregation dependency {source_name} differs from the startup snapshot")


import evaluate_calvin as _evaluate_calvin

_require_imported_local_module(
    _evaluate_calvin,
    "evaluate_calvin.py",
    _STARTUP_AGGREGATION_SOURCE_IDENTITIES,
)

from evaluate_calvin import (
    EPISODE_SCHEMA,
    EVALUATION_SEED,
    MAX_ACTIONS_PER_SUBTASK,
    NUM_SEQUENCES,
    OFFICIAL_DIRECT_NFE,
    OFFICIAL_FLOW_NFES,
    OFFICIAL_TRAIN_SEEDS,
    PROTOCOL,
    RUN_SCHEMA,
    SEQUENCE_SHA256,
    SUBTASKS_PER_SEQUENCE,
    SUMMARY_SCHEMA,
    SUPPORTED_EXECUTION_HORIZONS,
    _percentile,
    canonical_json_bytes,
    canonical_sha256,
    first_language_phrase,
    load_validation_annotations,
    official_factor_matrix,
    publish_bytes_and_sha256_exclusive,
    summarize_sequences,
    validate_policy_health,
    validate_preregistration_manifest,
)

import preflight as _preflight

_require_imported_local_module(
    _preflight,
    "preflight.py",
    _STARTUP_AGGREGATION_SOURCE_IDENTITIES,
)

from preflight import ATTESTATION_SCHEMA, PYTHON_VERSION, validate_official_dataset_identity

RUN_INVENTORY_SCHEMA = "duovla-calvin-official-run-inventory-v2"
MATRIX_SUMMARY_SCHEMA = "duovla-calvin-official-matrix-summary-v3"
_INVENTORY_FIELDS = {"preregistration_sha256", "runs", "schema"}
_INVENTORY_RUN_FIELDS = {
    "cell_id",
    "episodes_jsonl_sha256",
    "output_dir",
    "run_json_sha256",
    "summary_json_sha256",
}
_COMPLETE_RUN_FIELDS = {
    "annotation",
    "attestation",
    "attestation_sha256",
    "cell",
    "created_utc",
    "environment",
    "episodes_jsonl_sha256",
    "evaluation_seed",
    "execution_horizon",
    "final_freeze_token_sha256",
    "finished_utc",
    "mode",
    "oracle",
    "policy_health",
    "policy_socket",
    "preregistration_manifest",
    "preregistration_sha256",
    "protocol",
    "schema",
    "sequence_count",
    "sequence_records",
    "sequence_sha256",
    "status",
    "summary_json_sha256",
}
_SEQUENCE_FIELDS = {
    "elapsed_seconds",
    "evaluation_seed",
    "execution_horizon",
    "schema",
    "sequence_idx",
    "sequence_sha256",
    "sequence_success",
    "subtasks",
    "successful_subtasks",
}
_SUBTASK_FIELDS = {
    "action_clip_fraction",
    "action_clipped_channels",
    "action_continuous_channels",
    "discarded_queued_actions_on_success",
    "elapsed_seconds",
    "environment_actions",
    "execution_horizon",
    "instruction",
    "max_environment_actions",
    "policy_calls",
    "policy_identity_first_replan_idx",
    "policy_latency_p50_seconds",
    "policy_latency_p95_seconds",
    "policy_latency_seconds",
    "server_latency_p50_seconds",
    "server_latency_p95_seconds",
    "server_latency_seconds",
    "steps_to_success",
    "subtask_idx",
    "subtask_name",
    "success",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _require_exact_keys(value: Mapping[str, Any], expected: Set[str], name: str) -> None:
    observed = set(value)
    require(
        observed == expected,
        f"{name} fields differ: missing={sorted(expected - observed)}, extra={sorted(observed - expected)}",
    )


def _require_sha256(value: Any, name: str) -> None:
    require(
        isinstance(value, str) and len(value) == 64 and all(character in "0123456789abcdef" for character in value),
        f"{name} must be 64 lowercase hexadecimal characters",
    )


def require_aggregation_sources_unchanged(expected: Mapping[str, Any]) -> None:
    """Fail if any source captured at process startup has changed since."""

    _require_exact_keys(expected, set(_AGGREGATION_SOURCE_NAMES), "aggregation source snapshot")
    observed = {}  # type: Dict[str, Dict[str, Any]]
    for name in _AGGREGATION_SOURCE_NAMES:
        identity = expected[name]
        require(isinstance(identity, Mapping), f"aggregation source identity {name} must be an object")
        _require_exact_keys(identity, {"bytes", "path", "sha256"}, f"aggregation source identity {name}")
        require(isinstance(identity["path"], str), f"aggregation source path {name} is invalid")
        observed[name] = _source_file_identity(Path(identity["path"]))
    require(
        canonical_json_bytes(observed) == canonical_json_bytes(expected),
        "aggregation sources changed after process startup",
    )


def _unique_object(pairs: List[Tuple[str, Any]]) -> Dict[str, Any]:
    value = {}  # type: Dict[str, Any]
    for name, item in pairs:
        if name in value:
            raise ValueError(f"duplicate JSON field {name!r}")
        value[name] = item
    return value


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant {value}")


def _read_json_object(path: Path, *, name: str, expected_sha256: str) -> Tuple[Dict[str, Any], str]:
    _require_sha256(expected_sha256, f"expected {name} SHA-256")
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise RuntimeError(f"cannot read {name}: {path}") from exc
    digest = hashlib.sha256(raw).hexdigest()
    require(hmac.compare_digest(digest, expected_sha256), f"{name} differs from its externally supplied SHA-256")
    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, ValueError) as exc:
        raise RuntimeError(f"{name} is not strict finite UTF-8 JSON") from exc
    require(isinstance(value, dict), f"{name} root must be an object")
    return value, digest


def _finite_number(value: Any, *, minimum: float = 0.0) -> bool:
    return type(value) in (int, float) and math.isfinite(value) and value >= minimum


def _validate_latency(values: Any, p50: Any, p95: Any, *, calls: int, name: str) -> None:
    require(isinstance(values, list) and len(values) == calls, f"{name} latency count differs from policy calls")
    require(all(_finite_number(value) for value in values), f"{name} latencies must be finite and nonnegative")
    for observed, percentile in ((p50, 50.0), (p95, 95.0)):
        expected = _percentile(values, percentile)
        require(_finite_number(observed), f"{name} latency percentile is invalid")
        require(
            math.isclose(float(observed), float(expected), rel_tol=0.0, abs_tol=1e-12),
            f"{name} latency percentile drifted",
        )


def validate_sequence_records(
    records: Sequence[Any],
    *,
    sequences: Sequence[Any],
    execution_horizon: int,
    expected_instructions: Mapping[str, str],
) -> List[Dict[str, Any]]:
    """Validate every persisted sequence and subtask invariant before aggregation."""

    require(
        type(execution_horizon) is int and execution_horizon in SUPPORTED_EXECUTION_HORIZONS,
        "sequence validation execution horizon must be one of {1, 4}",
    )
    require(len(records) == NUM_SEQUENCES, f"official run must contain exactly {NUM_SEQUENCES} sequence records")
    require(len(sequences) == NUM_SEQUENCES, "pre-registration sequence inventory is incomplete")
    require(isinstance(expected_instructions, Mapping), "expected instruction inventory must be a mapping")
    checked = []  # type: List[Dict[str, Any]]
    for sequence_idx, record in enumerate(records):
        planned = sequences[sequence_idx]
        require(isinstance(record, dict), f"sequence record {sequence_idx} must be an object")
        _require_exact_keys(record, _SEQUENCE_FIELDS, f"sequence record {sequence_idx}")
        require(record["schema"] == EPISODE_SCHEMA, "sequence record schema mismatch")
        require(
            type(record["sequence_idx"]) is int and record["sequence_idx"] == sequence_idx,
            "sequence order drifted",
        )
        require(record["sequence_sha256"] == SEQUENCE_SHA256, "sequence record digest mismatch")
        require(
            type(record["evaluation_seed"]) is int and record["evaluation_seed"] == EVALUATION_SEED,
            "sequence record evaluation seed mismatch",
        )
        require(
            type(record["execution_horizon"]) is int and record["execution_horizon"] == execution_horizon,
            "sequence record execution horizon mismatch",
        )
        require(_finite_number(record["elapsed_seconds"]), "sequence elapsed time is invalid")
        successful = record["successful_subtasks"]
        require(
            type(successful) is int and 0 <= successful <= SUBTASKS_PER_SEQUENCE,
            "successful_subtasks must be an integer in [0, 5]",
        )
        require(type(record["sequence_success"]) is bool, "sequence_success must be boolean")
        require(record["sequence_success"] is (successful == SUBTASKS_PER_SEQUENCE), "sequence success flag drifted")
        require(isinstance(planned, list) and len(planned) == 2, "registered sequence shape changed")
        planned_tasks = planned[1]
        require(
            isinstance(planned_tasks, list) and len(planned_tasks) == SUBTASKS_PER_SEQUENCE,
            "registered sequence task list changed",
        )
        subtasks = record["subtasks"]
        expected_attempts = successful if successful == SUBTASKS_PER_SEQUENCE else successful + 1
        require(
            isinstance(subtasks, list) and len(subtasks) == expected_attempts,
            "attempted subtask prefix is invalid",
        )
        for subtask_idx, subtask in enumerate(subtasks):
            require(isinstance(subtask, dict), "subtask record must be an object")
            _require_exact_keys(subtask, _SUBTASK_FIELDS, "subtask record")
            require(
                type(subtask["subtask_idx"]) is int and subtask["subtask_idx"] == subtask_idx,
                "subtask record order drifted",
            )
            require(subtask["subtask_name"] == planned_tasks[subtask_idx], "subtask name differs from sequence")
            expected_instruction = expected_instructions.get(subtask["subtask_name"])
            require(
                isinstance(expected_instruction, str) and bool(expected_instruction),
                "authenticated validation language has no fixed first phrase for subtask",
            )
            require(
                subtask["instruction"] == expected_instruction,
                "subtask instruction differs from the authenticated first validation phrase",
            )
            require(type(subtask["success"]) is bool, "subtask success must be boolean")
            require(subtask["success"] is (subtask_idx < successful), "subtask success prefix drifted")
            require(
                type(subtask["execution_horizon"]) is int and subtask["execution_horizon"] == execution_horizon,
                "subtask execution horizon mismatch",
            )
            require(
                type(subtask["max_environment_actions"]) is int
                and subtask["max_environment_actions"] == MAX_ACTIONS_PER_SUBTASK,
                "subtask action budget changed",
            )
            actions = subtask["environment_actions"]
            require(type(actions) is int and 1 <= actions <= MAX_ACTIONS_PER_SUBTASK, "subtask action count is invalid")
            calls = subtask["policy_calls"]
            require(type(calls) is int and calls == math.ceil(actions / execution_horizon), "policy call count drifted")
            require(
                type(subtask["policy_identity_first_replan_idx"]) is int
                and subtask["policy_identity_first_replan_idx"] == 0,
                "subtask did not begin at replan zero",
            )
            clipped = subtask["action_clipped_channels"]
            continuous = subtask["action_continuous_channels"]
            require(type(continuous) is int and continuous == actions * 6, "continuous action channel count drifted")
            require(type(clipped) is int and 0 <= clipped <= continuous, "clipped action channel count is invalid")
            require(
                _finite_number(subtask["action_clip_fraction"])
                and math.isclose(
                    float(subtask["action_clip_fraction"]),
                    clipped / continuous,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                ),
                "action clip fraction drifted",
            )
            discarded = subtask["discarded_queued_actions_on_success"]
            expected_discarded = calls * execution_horizon - actions if subtask["success"] else 0
            require(type(discarded) is int and discarded == expected_discarded, "discarded queued-action count drifted")
            if subtask["success"]:
                require(
                    type(subtask["steps_to_success"]) is int and subtask["steps_to_success"] == actions,
                    "successful subtask steps-to-success drifted",
                )
            else:
                require(actions == MAX_ACTIONS_PER_SUBTASK, "failed subtask did not exhaust the action budget")
                require(subtask["steps_to_success"] is None, "failed subtask has steps-to-success")
            require(_finite_number(subtask["elapsed_seconds"]), "subtask elapsed time is invalid")
            _validate_latency(
                subtask["policy_latency_seconds"],
                subtask["policy_latency_p50_seconds"],
                subtask["policy_latency_p95_seconds"],
                calls=calls,
                name="policy",
            )
            _validate_latency(
                subtask["server_latency_seconds"],
                subtask["server_latency_p50_seconds"],
                subtask["server_latency_p95_seconds"],
                calls=calls,
                name="server",
            )
        checked.append(dict(record))
    return checked


def _load_jsonl(path: Path, *, expected_sha256: str) -> List[Dict[str, Any]]:
    _require_sha256(expected_sha256, "expected episodes JSONL SHA-256")
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise RuntimeError(f"cannot read episodes JSONL: {path}") from exc
    require(hmac.compare_digest(hashlib.sha256(raw).hexdigest(), expected_sha256), "episodes JSONL SHA-256 mismatch")
    require(raw.endswith(b"\n"), "episodes JSONL must end with a newline")
    records = []  # type: List[Dict[str, Any]]
    for index, line in enumerate(raw.splitlines()):
        try:
            value = json.loads(
                line.decode("utf-8"),
                object_pairs_hook=_unique_object,
                parse_constant=_reject_json_constant,
            )
        except (UnicodeDecodeError, ValueError) as exc:
            raise RuntimeError(f"episodes JSONL line {index + 1} is not strict finite UTF-8 JSON") from exc
        require(isinstance(value, dict), f"episodes JSONL line {index + 1} must be an object")
        records.append(value)
    return records


def _validate_attestation(
    attestation: Any,
    expected_sha256: str,
    *,
    aggregation_sources: Mapping[str, Any],
) -> Dict[str, Any]:
    require(isinstance(attestation, dict), "run attestation must be an object")
    require(
        attestation.get("schema") == ATTESTATION_SCHEMA,
        "run attestation schema mismatch",
    )
    observed = attestation.get("attestation_sha256")
    _require_sha256(observed, "run attestation SHA-256")
    require(hmac.compare_digest(observed, expected_sha256), "run attestation differs from pre-registration")
    payload = {name: value for name, value in attestation.items() if name != "attestation_sha256"}
    require(hmac.compare_digest(canonical_sha256(payload), observed), "run attestation self-digest mismatch")
    try:
        attested_sources = attestation["runtime"]["sources"]
    except (KeyError, TypeError) as exc:
        raise RuntimeError("run attestation is missing evaluator source identities") from exc
    require(isinstance(attested_sources, Mapping), "run attestation evaluator sources must be an object")
    _require_exact_keys(attested_sources, set(_EVALUATOR_SOURCE_NAMES), "run attestation evaluator sources")
    live_sources = {name: aggregation_sources[name] for name in _EVALUATOR_SOURCE_NAMES}
    require(
        canonical_json_bytes(attested_sources) == canonical_json_bytes(live_sources),
        "run-attested evaluator sources differ from aggregation startup sources",
    )
    validate_official_dataset_identity(attestation.get("dataset"))
    return dict(attestation)


def validate_run_artifact(
    entry: Mapping[str, Any],
    *,
    inventory_dir: Path,
    preregistration: Mapping[str, Any],
    preregistration_sha256: str,
    registered_cell: Mapping[str, Any],
    aggregation_sources: Mapping[str, Any] = _STARTUP_AGGREGATION_SOURCE_IDENTITIES,
) -> Dict[str, Any]:
    """Authenticate one complete run and recompute its official summary."""

    _require_exact_keys(entry, _INVENTORY_RUN_FIELDS, "run inventory entry")
    for name in ("run_json_sha256", "episodes_jsonl_sha256", "summary_json_sha256"):
        _require_sha256(entry[name], f"inventory {name}")
    require(entry["cell_id"] == registered_cell["cell_id"], "run inventory cell ID mismatch")
    require(isinstance(entry["output_dir"], str) and bool(entry["output_dir"]), "run output_dir is invalid")
    output_dir = Path(entry["output_dir"])
    if not output_dir.is_absolute():
        output_dir = inventory_dir / output_dir
    output_dir = output_dir.resolve()
    require(output_dir.is_dir(), f"official run directory is missing: {output_dir}")
    run, run_sha256 = _read_json_object(
        output_dir / "run.json",
        name="official run.json",
        expected_sha256=entry["run_json_sha256"],
    )
    _require_exact_keys(run, _COMPLETE_RUN_FIELDS, "complete official run")
    require(run["schema"] == RUN_SCHEMA, "official run schema mismatch")
    require(run["status"] == "complete", "official run is not complete")
    require(run["mode"] == "official-score", "run is not an official score")
    require(run["protocol"] == PROTOCOL, "official run protocol mismatch")
    require(
        type(run["evaluation_seed"]) is int and run["evaluation_seed"] == EVALUATION_SEED,
        "official run evaluation seed mismatch",
    )
    require(
        type(run["sequence_count"]) is int and run["sequence_count"] == NUM_SEQUENCES,
        "official run sequence count mismatch",
    )
    require(
        type(run["sequence_records"]) is int and run["sequence_records"] == NUM_SEQUENCES,
        "official run record count mismatch",
    )
    require(run["sequence_sha256"] == SEQUENCE_SHA256, "official run sequence digest mismatch")
    require(run["preregistration_sha256"] == preregistration_sha256, "run pre-registration digest mismatch")
    require(
        run["final_freeze_token_sha256"] == preregistration["final_freeze_token_sha256"],
        "run freeze token digest mismatch",
    )
    require(
        canonical_json_bytes(run["cell"]) == canonical_json_bytes(registered_cell),
        "run cell differs from pre-registration",
    )
    execution_horizon = registered_cell["execution_horizon"]
    require(
        type(run["execution_horizon"]) is int and run["execution_horizon"] == execution_horizon,
        "run execution horizon differs from its cell",
    )
    require(isinstance(run["created_utc"], str) and bool(run["created_utc"]), "run creation timestamp is invalid")
    require(isinstance(run["finished_utc"], str) and bool(run["finished_utc"]), "run finish timestamp is invalid")
    require(isinstance(run["policy_socket"], str) and bool(run["policy_socket"]), "run policy socket is invalid")
    require(
        isinstance(run["preregistration_manifest"], str) and bool(run["preregistration_manifest"]),
        "run pre-registration path is invalid",
    )
    attestation = _validate_attestation(
        run["attestation"],
        preregistration["runtime_attestation_sha256"],
        aggregation_sources=aggregation_sources,
    )
    require(
        run["attestation_sha256"] == preregistration["runtime_attestation_sha256"],
        "run attestation binding mismatch",
    )
    try:
        official_yaml = attestation["runtime"]["official_yaml"]
        validation_config = attestation["dataset"]["validation_critical_files"]["validation/.hydra/merged_config.yaml"]
    except (KeyError, TypeError) as exc:
        raise RuntimeError("run attestation is missing official runtime/data identities") from exc
    annotations, annotation_identity = load_validation_annotations(official_yaml["validation_annotations"])
    require(
        canonical_json_bytes(run["annotation"]) == canonical_json_bytes(annotation_identity),
        "run validation annotation identity differs from authenticated YAML",
    )
    require(
        canonical_json_bytes(run["oracle"]) == canonical_json_bytes(official_yaml["task_oracle"]),
        "run task-oracle identity differs from attestation",
    )
    environment = run["environment"]
    require(isinstance(environment, dict), "run environment identity must be an object")
    _require_exact_keys(
        environment,
        {"control_frequency_hz", "merged_config_path", "merged_config_sha256", "scene", "validation_path"},
        "run environment identity",
    )
    require(environment["control_frequency_hz"] == 30, "run environment control frequency changed")
    require(environment["scene"] == "calvin_scene_D", "run environment is not validation scene D")
    require(
        environment["merged_config_path"] == validation_config["path"]
        and environment["merged_config_sha256"] == validation_config["sha256"],
        "run validation environment config differs from attestation",
    )
    require(
        isinstance(environment["validation_path"], str)
        and str(Path(environment["merged_config_path"]).parent.parent) == environment["validation_path"],
        "run validation environment path is inconsistent",
    )
    validate_policy_health(
        run["policy_health"],
        execution_horizon=execution_horizon,
        expected_cell=registered_cell,
        expected_calvin_identity=attestation["dataset"]["calvin_identity"],
    )
    require(run["episodes_jsonl_sha256"] == entry["episodes_jsonl_sha256"], "run/inventory episode digest mismatch")
    require(run["summary_json_sha256"] == entry["summary_json_sha256"], "run/inventory summary digest mismatch")
    records = _load_jsonl(output_dir / "episodes.jsonl", expected_sha256=entry["episodes_jsonl_sha256"])
    registered_tasks = {task_name for sequence in preregistration["sequences"] for task_name in sequence[1]}
    expected_instructions = {
        task_name: first_language_phrase(annotations, task_name) for task_name in sorted(registered_tasks)
    }
    checked_records = validate_sequence_records(
        records,
        sequences=preregistration["sequences"],
        execution_horizon=execution_horizon,
        expected_instructions=expected_instructions,
    )
    summary, summary_sha256 = _read_json_object(
        output_dir / "summary.json",
        name="official summary.json",
        expected_sha256=entry["summary_json_sha256"],
    )
    expected_summary = summarize_sequences(checked_records)
    require(summary.get("schema") == SUMMARY_SCHEMA, "official summary schema mismatch")
    require(
        canonical_json_bytes(summary) == canonical_json_bytes(expected_summary),
        "official summary does not match episodes",
    )
    policy = registered_cell["policy"]
    return {
        "AvgLen": summary["AvgLen"],
        "SR1": summary["SR1"],
        "SR2": summary["SR2"],
        "SR3": summary["SR3"],
        "SR4": summary["SR4"],
        "SR5": summary["SR5"],
        "cell_id": registered_cell["cell_id"],
        "checkpoint_sha256": registered_cell["checkpoint"]["sha256"],
        "episodes_jsonl_sha256": entry["episodes_jsonl_sha256"],
        "execution_horizon": execution_horizon,
        "inference_seed_behavior": policy["inference_seed_behavior"],
        "nfe": policy["nfe"],
        "objective": policy["objective"],
        "policy_identity_sha256": policy["identity_sha256"],
        "run_json_sha256": run_sha256,
        "sampler": policy["sampler"],
        "serving_runtime_sha256": registered_cell["serving_runtime_sha256"],
        "summary_json_sha256": summary_sha256,
        "train_seed": policy["train_seed"],
    }


def _metric_aggregate(cell_results: Sequence[Mapping[str, Any]], metric: str) -> Dict[str, Any]:
    ordered = sorted(cell_results, key=lambda value: value["train_seed"])
    values = [float(value[metric]) for value in ordered]
    require(
        [value["train_seed"] for value in ordered] == list(OFFICIAL_TRAIN_SEEDS),
        "aggregate seed set is incomplete",
    )
    return {
        "mean": statistics.fmean(values),
        "sample_std": statistics.stdev(values),
        "values_by_train_seed": [{"train_seed": value["train_seed"], "value": value[metric]} for value in ordered],
    }


def aggregate_matrix(
    preregistration: Mapping[str, Any],
    preregistration_sha256: str,
    inventory: Mapping[str, Any],
    inventory_sha256: str,
    *,
    inventory_dir: Path,
    aggregation_sources: Mapping[str, Any] = _STARTUP_AGGREGATION_SOURCE_IDENTITIES,
) -> Dict[str, Any]:
    """Reject incomplete/off-matrix inventories and aggregate three training seeds."""

    require_aggregation_sources_unchanged(aggregation_sources)
    _require_sha256(preregistration_sha256, "pre-registration SHA-256")
    _require_sha256(inventory_sha256, "run inventory SHA-256")
    require(isinstance(preregistration, dict), "pre-registration manifest must be an object")
    require("sequences" in preregistration, "pre-registration has no sequence inventory")
    require("runtime_attestation_sha256" in preregistration, "pre-registration has no attestation identity")
    registered_cells = validate_preregistration_manifest(
        preregistration,
        preregistration["sequences"],
        runtime_attestation_sha256=preregistration["runtime_attestation_sha256"],
    )
    aggregator_identity = aggregation_sources["aggregate_calvin_official.py"]
    require(isinstance(aggregator_identity, Mapping), "aggregation source identity must be an object")
    require(
        hmac.compare_digest(preregistration["aggregator_sha256"], aggregator_identity.get("sha256", "")),
        "aggregation startup source differs from the pre-registered aggregator",
    )
    _require_exact_keys(inventory, _INVENTORY_FIELDS, "official run inventory")
    require(inventory["schema"] == RUN_INVENTORY_SCHEMA, "official run inventory schema mismatch")
    require(inventory["preregistration_sha256"] == preregistration_sha256, "run inventory pre-registration mismatch")
    runs = inventory["runs"]
    require(isinstance(runs, list), "official run inventory runs must be a list")
    require(len(runs) == 24, "official run inventory must contain exactly 24 runs")
    require(all(isinstance(entry, dict) for entry in runs), "official run inventory entries must be objects")
    for entry in runs:
        _require_exact_keys(entry, _INVENTORY_RUN_FIELDS, "run inventory entry")
        require(isinstance(entry["cell_id"], str) and bool(entry["cell_id"]), "run inventory cell ID is invalid")
        require(isinstance(entry["output_dir"], str) and bool(entry["output_dir"]), "run output_dir is invalid")
        for name in ("run_json_sha256", "episodes_jsonl_sha256", "summary_json_sha256"):
            _require_sha256(entry[name], f"inventory {name}")
    inventory_ids = [entry.get("cell_id") for entry in runs]
    require(len(inventory_ids) == len(set(inventory_ids)), "official run inventory contains duplicate cell IDs")
    cells_by_id = {cell["cell_id"]: cell for cell in registered_cells}
    require(set(inventory_ids) == set(cells_by_id), "official run inventory has missing or off-matrix cells")
    resolved_dirs = []  # type: List[str]
    for entry in runs:
        path = Path(entry["output_dir"])
        if not path.is_absolute():
            path = inventory_dir / path
        resolved_dirs.append(str(path.resolve()))
    require(len(resolved_dirs) == len(set(resolved_dirs)), "official run inventory reuses an output directory")

    cell_results = [
        validate_run_artifact(
            entry,
            inventory_dir=inventory_dir,
            preregistration=preregistration,
            preregistration_sha256=preregistration_sha256,
            registered_cell=cells_by_id[entry["cell_id"]],
            aggregation_sources=aggregation_sources,
        )
        for entry in runs
    ]
    cell_results.sort(
        key=lambda value: (value["train_seed"], value["objective"], value["nfe"], value["execution_horizon"])
    )
    observed_factors = {
        (value["train_seed"], value["objective"], value["nfe"], value["execution_horizon"]) for value in cell_results
    }
    require(observed_factors == official_factor_matrix(), "validated run factors do not form the canonical matrix")

    comparisons = []  # type: List[Dict[str, Any]]
    for objective, nfes in (
        ("rectified_flow", OFFICIAL_FLOW_NFES),
        ("direct_regression", (OFFICIAL_DIRECT_NFE,)),
    ):
        for nfe in nfes:
            for execution_horizon in SUPPORTED_EXECUTION_HORIZONS:
                group = [
                    value
                    for value in cell_results
                    if value["objective"] == objective
                    and value["nfe"] == nfe
                    and value["execution_horizon"] == execution_horizon
                ]
                require(len(group) == len(OFFICIAL_TRAIN_SEEDS), "comparison does not contain all three seeds")
                label = "flow" if objective == "rectified_flow" else "direct"
                comparisons.append(
                    {
                        "comparison_id": f"{label}-nfe-{nfe}-k-{execution_horizon}",
                        "execution_horizon": execution_horizon,
                        "metrics": {
                            metric: _metric_aggregate(group, metric)
                            for metric in ("AvgLen", "SR1", "SR2", "SR3", "SR4", "SR5")
                        },
                        "nfe": nfe,
                        "objective": objective,
                    }
                )
    require_aggregation_sources_unchanged(aggregation_sources)
    result = {
        "aggregation_python_version": preregistration["aggregation_python_version"],
        "aggregation_source_identities": json.loads(canonical_json_bytes(aggregation_sources).decode("ascii")),
        "aggregator_sha256": preregistration["aggregator_sha256"],
        "benchmark_protocol": PROTOCOL,
        "cell_count": len(cell_results),
        "cells": cell_results,
        "comparison_count": len(comparisons),
        "comparisons": comparisons,
        "evaluation_seed": EVALUATION_SEED,
        "preregistration_sha256": preregistration_sha256,
        "run_inventory_sha256": inventory_sha256,
        "schema": MATRIX_SUMMARY_SCHEMA,
        "sequence_count_per_cell": NUM_SEQUENCES,
        "sequence_sha256": SEQUENCE_SHA256,
        "subtasks_per_sequence": SUBTASKS_PER_SEQUENCE,
        "training_seeds": list(OFFICIAL_TRAIN_SEEDS),
    }
    result["content_sha256"] = canonical_sha256(result)
    return result


def _write_exclusive(
    path: Path,
    value: Mapping[str, Any],
    *,
    commit_guard: Optional[Callable[[], None]] = None,
) -> str:
    payload = (json.dumps(value, allow_nan=False, ensure_ascii=True, indent=2, sort_keys=True) + "\n").encode("ascii")
    return publish_bytes_and_sha256_exclusive(path, payload, commit_guard=commit_guard)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preregistration-manifest", type=Path, required=True)
    parser.add_argument("--preregistration-sha256", required=True)
    parser.add_argument("--run-inventory", type=Path, required=True)
    parser.add_argument("--run-inventory-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    require(
        platform.python_version() == PYTHON_VERSION,
        f"CALVIN official aggregator requires Python {PYTHON_VERSION}, found {platform.python_version()}",
    )
    require_aggregation_sources_unchanged(_STARTUP_AGGREGATION_SOURCE_IDENTITIES)
    preregistration, preregistration_sha256 = _read_json_object(
        args.preregistration_manifest.resolve(),
        name="CALVIN pre-registration",
        expected_sha256=args.preregistration_sha256,
    )
    inventory_path = args.run_inventory.resolve()
    inventory, inventory_sha256 = _read_json_object(
        inventory_path,
        name="CALVIN official run inventory",
        expected_sha256=args.run_inventory_sha256,
    )
    result = aggregate_matrix(
        preregistration,
        preregistration_sha256,
        inventory,
        inventory_sha256,
        inventory_dir=inventory_path.parent,
        aggregation_sources=_STARTUP_AGGREGATION_SOURCE_IDENTITIES,
    )
    output = args.output.resolve()

    def commit_guard() -> None:
        require_aggregation_sources_unchanged(_STARTUP_AGGREGATION_SOURCE_IDENTITIES)

    commit_guard()
    digest = _write_exclusive(output, result, commit_guard=commit_guard)
    print(json.dumps({"matrix_summary": str(output), "sha256": digest}, sort_keys=True))


if __name__ == "__main__":
    main()
