#!/usr/bin/env python3
"""Authenticate and aggregate the exact 24-cell official LIBERO matrix."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import hmac
import importlib.util
import json
import math
import os
import platform
import secrets
import stat
import statistics
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

_SCRIPT_DIR = Path(__file__).resolve().parent
_EVALUATOR_SOURCE_NAMES = ("evaluate_libero.py", "libero_bridge.py")
_AGGREGATION_SOURCE_NAMES = (
    "aggregate_libero_official.py",
    *_EVALUATOR_SOURCE_NAMES,
    "preflight_libero_env.py",
)


def _activate_project_source_root() -> Path:
    source_root = _SCRIPT_DIR.parent / "src"
    if source_root.resolve(strict=True) != source_root or not stat.S_ISDIR(os.lstat(source_root).st_mode):
        raise RuntimeError("project src import root must be a canonical real directory")
    identity_fields = ("st_dev", "st_ino", "st_mode", "st_size", "st_mtime_ns", "st_ctime_ns", "st_nlink")

    def same_identity(left: os.stat_result, right: os.stat_result) -> bool:
        return all(getattr(left, name) == getattr(right, name) for name in identity_fields)

    def walk(directory: int, prefix: tuple[str, ...]) -> None:
        before = os.fstat(directory)
        names = sorted(os.listdir(directory))
        for name in names:
            context = "/".join((*prefix, name))
            observed = os.stat(name, dir_fd=directory, follow_symlinks=False)
            if stat.S_ISDIR(observed.st_mode):
                if name == "__pycache__":
                    continue
                child = os.open(
                    name,
                    os.O_RDONLY | os.O_NONBLOCK | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
                    dir_fd=directory,
                )
                try:
                    if not same_identity(observed, os.fstat(child)):
                        raise RuntimeError(f"project source directory changed while opening: {context}")
                    walk(child, (*prefix, name))
                finally:
                    os.close(child)
            elif not (stat.S_ISREG(observed.st_mode) and name.endswith(".py")):
                raise RuntimeError(f"project source import entry is unsafe: {context}")
        if names != sorted(os.listdir(directory)) or not same_identity(before, os.fstat(directory)):
            raise RuntimeError(f"project source directory changed during inventory: {'/'.join(prefix) or '.'}")

    root_descriptor = os.open(
        source_root,
        os.O_RDONLY | os.O_NONBLOCK | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
    )
    try:
        if sorted(os.listdir(root_descriptor)) != ["duo_vla"]:
            raise RuntimeError("project src import root must contain only the real duo_vla package directory")
        package_descriptor = os.open(
            "duo_vla",
            os.O_RDONLY | os.O_NONBLOCK | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
            dir_fd=root_descriptor,
        )
        try:
            walk(package_descriptor, ("duo_vla",))
        finally:
            os.close(package_descriptor)
    finally:
        os.close(root_descriptor)
    source_text = str(source_root)
    sys.path[:] = [source_text] + [
        entry for entry in sys.path if str(Path(entry or os.getcwd()).resolve()) != source_text
    ]
    return source_root


_PROJECT_SOURCE_ROOT = _activate_project_source_root()


def _source_file_identity(path: Path) -> dict[str, Any]:
    """Hash a stable regular source file without following a final symlink."""

    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
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
            while block := source.read(1024 * 1024):
                size += len(block)
                digest.update(block)
            after = os.fstat(source.fileno())
        stable = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
        if any(getattr(before, name) != getattr(after, name) for name in stable) or size != after.st_size:
            raise RuntimeError(f"aggregation source changed while being hashed: {path}")
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    return {"bytes": size, "path": str(path.resolve()), "sha256": digest.hexdigest()}


def capture_aggregation_source_identities(script_dir: Path = _SCRIPT_DIR) -> dict[str, dict[str, Any]]:
    root = script_dir.resolve()
    return {name: _source_file_identity(root / name) for name in _AGGREGATION_SOURCE_NAMES}


_STARTUP_AGGREGATION_SOURCE_IDENTITIES = capture_aggregation_source_identities()


def _load_local_module(name: str, path: Path) -> Any:
    expected = path.resolve(strict=True)
    existing = sys.modules.get(name)
    if existing is not None:
        module_file = getattr(existing, "__file__", None)
        origin = getattr(getattr(existing, "__spec__", None), "origin", None)
        if not isinstance(module_file, str) or not isinstance(origin, str):
            raise RuntimeError(f"existing local module {name} has no file origin")
        if Path(module_file).resolve() != expected or Path(origin).resolve() != expected:
            raise RuntimeError(f"existing local module {name} has an unexpected origin")
        return existing
    specification = importlib.util.spec_from_file_location(name, expected)
    if specification is None or specification.loader is None:
        raise RuntimeError(f"cannot construct import specification for {expected}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    try:
        specification.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


def _require_imported_local_module(module: Any, source_name: str, expected: Mapping[str, Any]) -> None:
    identity = expected.get(source_name)
    if source_name not in _EVALUATOR_SOURCE_NAMES or not isinstance(identity, Mapping):
        raise RuntimeError(f"aggregation dependency identity {source_name} is invalid")
    module_file = getattr(module, "__file__", None)
    module_spec = getattr(module, "__spec__", None)
    spec_origin = getattr(module_spec, "origin", None)
    if not isinstance(module_file, str) or Path(module_file).resolve() != Path(identity["path"]):
        raise RuntimeError(f"imported aggregation dependency {source_name} has an unexpected __file__")
    if not isinstance(spec_origin, str) or Path(spec_origin).resolve() != Path(identity["path"]):
        raise RuntimeError(f"imported aggregation dependency {source_name} has an unexpected spec origin")
    if _source_file_identity(Path(module_file)) != identity:
        raise RuntimeError(f"imported aggregation dependency {source_name} differs from the startup snapshot")


_evaluate_libero = _load_local_module("evaluate_libero", _SCRIPT_DIR / "evaluate_libero.py")
_libero_bridge = _load_local_module("libero_bridge", _SCRIPT_DIR / "libero_bridge.py")

_require_imported_local_module(
    _evaluate_libero,
    "evaluate_libero.py",
    _STARTUP_AGGREGATION_SOURCE_IDENTITIES,
)
_require_imported_local_module(
    _libero_bridge,
    "libero_bridge.py",
    _STARTUP_AGGREGATION_SOURCE_IDENTITIES,
)

from evaluate_libero import (  # noqa: E402
    AGGREGATION_PYTHON_VERSION,
    COMPLETION_SCHEMA,
    ENVIRONMENT_SEED,
    OFFICIAL_EXECUTION_HORIZONS,
    OFFICIAL_FLOW_NFES,
    OFFICIAL_POLICY_WARMUP_CALLS,
    OFFICIAL_PRIMARY_EPISODES,
    OFFICIAL_TRAIN_SEEDS,
    POLICY_BUDGETS,
    PROTOCOL,
    SETTLE_STEPS,
    SIMULATOR_ATTESTATION_SCHEMA,
    SUITES,
    _official_factor_matrix,
    bind_official_summary,
    canonical_json_bytes,
    canonical_sha256,
    load_contamination_contract,
    percentile,
    summarize_episodes,
    validate_claim_record,
    validate_official_output_roots,
    validate_policy_health,
    validate_preregistration_manifest,
    warmup_k_independent_response,
)

RUN_INVENTORY_SCHEMA = "duo-vla-libero-official-run-inventory-v2"
MATRIX_SUMMARY_SCHEMA = "duo-vla-libero-official-matrix-summary-v3"
_INVENTORY_FIELDS = {"preregistration_sha256", "runs", "schema"}
_INVENTORY_RUN_FIELDS = {
    "cell_id",
    "claim_json_sha256",
    "completion_json_sha256",
    "episodes_jsonl_sha256",
    "run_json_sha256",
    "summary_json_sha256",
}
_COMPLETE_RUN_FIELDS = {
    "cell",
    "checkpoint",
    "claim_json_sha256",
    "contamination",
    "created_utc",
    "episode_count",
    "episode_matrix_sha256",
    "episode_records",
    "episodes_jsonl_sha256",
    "evaluation_seed",
    "evaluator_environment",
    "execution_horizon",
    "final_freeze_token_sha256",
    "finished_utc",
    "init_state_ids",
    "mode",
    "output_claim",
    "policy_health",
    "policy_socket",
    "policy_warmup",
    "preregistration_manifest",
    "preregistration_sha256",
    "protocol",
    "reset_identity",
    "reset_source",
    "schema",
    "simulator_attestation_sha256",
    "simulator_preflight",
    "status",
    "summary_json_sha256",
    "suites",
    "task_ids",
    "validator_runtime_identity",
}
_EPISODE_FIELDS = {
    "action_clip_fraction",
    "action_clipped_channels",
    "action_continuous_channels",
    "elapsed_seconds",
    "environment_seed",
    "evaluation_seed",
    "execution_horizon",
    "init_state_id",
    "normalized_action_clip_fraction",
    "policy_budget",
    "policy_calls",
    "policy_latency_p50_seconds",
    "policy_latency_p95_seconds",
    "policy_latency_seconds",
    "policy_steps",
    "reset_id",
    "reset_source",
    "reset_state_sha256",
    "server_latency_p50_seconds",
    "server_latency_p95_seconds",
    "server_latency_seconds",
    "settle_steps",
    "simulator_done",
    "steps_to_success",
    "success",
    "suite",
    "task_id",
    "task_name",
}
_CHECKPOINT_IDENTITY_FIELDS = {
    "manifest_sha256",
    "path",
    "run_journal_latest",
    "run_root",
    "source_tree_sha256",
    "train_venv",
    "update",
}
_HEALTH_FIELDS = {
    "action_dim",
    "action_horizon",
    "checkpoint",
    "dataset_revision",
    "execution_geometry",
    "inference_seed_behavior",
    "latency_runtime_sha256",
    "mode",
    "model_revision",
    "nfe",
    "normalization_content_sha256",
    "objective",
    "operation",
    "prefix_cache_scope",
    "protocol",
    "request_id",
    "sampler",
    "schema",
    "serving_runtime_sha256",
    "state_dim",
    "status",
    "train_seed",
}
_HEALTH_CHECKPOINT_FIELDS = {
    "dataset_content_inventory_sha256",
    "dataset_files_verified",
    "dataset_total_bytes",
    "dataset_tree_sha256",
    "execution_geometry",
    "kind",
    "manifest_sha256",
    "model_content_inventory_sha256",
    "model_files_verified",
    "model_total_bytes",
    "model_tree_sha256",
    "path",
    "policy_contract",
    "policy_contract_sha256",
    "source_tree_sha256",
    "train_seed",
    "train_venv",
    "training_execution_environment",
    "training_execution_environment_sha256",
}
_WARMUP_FIELDS = {
    "actions_sha256",
    "evaluation_seed",
    "execution_horizon",
    "inference_seed",
    "inference_seed_behavior",
    "nfe",
    "objective",
    "policy_seconds",
    "replan_id",
    "reset_id",
    "reset_source",
    "reset_state_sha256",
    "sampler",
    "status",
    "warmup_index",
}
_COMPARISON_METRICS = (
    "episode_throughput_per_hour",
    "overall_40_task_macro_success",
    "overall_pooled_success_rate",
    "policy_call_throughput_per_second",
    "policy_latency_p50_seconds",
    "policy_latency_p95_seconds",
    "server_latency_p50_seconds",
    "server_latency_p95_seconds",
    *(f"{suite}_task_macro_success" for suite in SUITES),
    *(f"{suite}_pooled_success_rate" for suite in SUITES),
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _require_exact_keys(value: Mapping[str, Any], expected: set[str], name: str) -> None:
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
    _require_exact_keys(expected, set(_AGGREGATION_SOURCE_NAMES), "aggregation source snapshot")
    observed: dict[str, dict[str, Any]] = {}
    for name in _AGGREGATION_SOURCE_NAMES:
        identity = expected[name]
        require(isinstance(identity, Mapping), f"aggregation source identity {name} must be an object")
        _require_exact_keys(identity, {"bytes", "path", "sha256"}, f"aggregation source identity {name}")
        require(isinstance(identity["path"], str), f"aggregation source path {name} is invalid")
        observed[name] = _source_file_identity(Path(identity["path"]))
    require(
        canonical_json_bytes(observed) == canonical_json_bytes(expected), "aggregation sources changed after startup"
    )


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for name, item in pairs:
        if name in value:
            raise ValueError(f"duplicate JSON field {name!r}")
        value[name] = item
    return value


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant {value}")


def _read_json_object(path: Path, *, name: str, expected_sha256: str) -> tuple[dict[str, Any], str]:
    _require_sha256(expected_sha256, f"expected {name} SHA-256")
    raw = _read_single_link_regular_bytes(path, name=name)
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


def _read_single_link_regular_bytes(path: Path, *, name: str) -> bytes:
    require(hasattr(os, "O_NOFOLLOW"), "official artifact authentication requires O_NOFOLLOW")
    flags = os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
    try:
        descriptor = os.open(str(path), flags)
    except OSError as exc:
        raise RuntimeError(f"cannot open {name}: {path}") from exc
    chunks: list[bytes] = []
    try:
        before = os.fstat(descriptor)
        require(stat.S_ISREG(before.st_mode) and before.st_nlink == 1, f"{name} is not a single-link regular file")
        with os.fdopen(descriptor, "rb") as source:
            descriptor = -1
            while True:
                block = source.read(1024 * 1024)
                if not block:
                    break
                chunks.append(block)
            after = os.fstat(source.fileno())
        stable = ("st_dev", "st_ino", "st_nlink", "st_size", "st_mtime_ns", "st_ctime_ns")
        require(
            all(getattr(before, field) == getattr(after, field) for field in stable),
            f"{name} changed while being read",
        )
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    return b"".join(chunks)


def _load_jsonl(path: Path, *, expected_sha256: str) -> list[dict[str, Any]]:
    _require_sha256(expected_sha256, "expected episodes JSONL SHA-256")
    raw = _read_single_link_regular_bytes(path, name="episodes JSONL")
    require(hmac.compare_digest(hashlib.sha256(raw).hexdigest(), expected_sha256), "episodes JSONL SHA-256 mismatch")
    require(raw.endswith(b"\n"), "episodes JSONL must end with a newline")
    records: list[dict[str, Any]] = []
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


def _finite_number(value: Any, *, minimum: float = 0.0, maximum: float | None = None) -> bool:
    if type(value) not in (int, float) or not math.isfinite(float(value)) or float(value) < minimum:
        return False
    return maximum is None or float(value) <= maximum


def _same_number(left: Any, right: float) -> bool:
    return _finite_number(left) and math.isclose(float(left), right, rel_tol=0.0, abs_tol=1e-12)


def _validate_latency(values: Any, p50: Any, p95: Any, *, calls: int, name: str) -> None:
    require(isinstance(values, list) and len(values) == calls, f"{name} latency count differs from policy calls")
    require(all(_finite_number(value) for value in values), f"{name} latencies must be finite and nonnegative")
    for observed, q in ((p50, 50.0), (p95, 95.0)):
        expected = percentile(values, q)
        if expected is None:
            require(observed is None, f"empty {name} latency percentile must be null")
        else:
            require(_same_number(observed, expected), f"{name} latency percentile drifted")


def _validate_latency_pair(client_values: Sequence[Any], server_values: Sequence[Any]) -> None:
    require(len(client_values) == len(server_values), "client/server latency counts differ")
    require(
        all(float(client) >= float(server) for client, server in zip(client_values, server_values, strict=True)),
        "client round-trip latency must be at least the paired server latency",
    )


def validate_episode_records(
    records: Sequence[Any],
    *,
    planned_episodes: Sequence[Any],
    evaluation_seed: int,
    execution_horizon: int,
    expected_task_names: Mapping[tuple[str, int], str] | None = None,
) -> list[dict[str, Any]]:
    """Validate the canonical order, denominator, and every persisted episode invariant."""

    require(len(records) == OFFICIAL_PRIMARY_EPISODES, "official run must contain exactly 1,999 episodes")
    require(len(planned_episodes) == OFFICIAL_PRIMARY_EPISODES, "pre-registration episode inventory is incomplete")
    task_names: dict[tuple[str, int], str] = {}
    checked: list[dict[str, Any]] = []
    for index, (record, planned) in enumerate(zip(records, planned_episodes, strict=True)):
        require(isinstance(record, dict), f"episode record {index} must be an object")
        require(isinstance(planned, Mapping), f"planned episode {index} must be an object")
        _require_exact_keys(record, _EPISODE_FIELDS, f"episode record {index}")
        identity = {name: record[name] for name in ("reset_id", "suite", "task_id")}
        require(identity == planned, f"episode record {index} differs from the pre-registered order")
        suite = record["suite"]
        task_id = record["task_id"]
        reset_id = record["reset_id"]
        require(suite in SUITES, "episode suite is invalid")
        require(type(task_id) is int and 0 <= task_id < 10, "episode task ID is invalid")
        require(type(reset_id) is int and 0 <= reset_id < 50, "episode reset ID is invalid")
        require(record["init_state_id"] == reset_id, "official init-state identity drifted")
        require(record["reset_source"] == "official", "official episode used a non-official reset")
        require(record["reset_state_sha256"] is None, "official episode recorded a private reset-state hash")
        require(record["environment_seed"] == ENVIRONMENT_SEED, "official simulator seed drifted")
        require(record["evaluation_seed"] == evaluation_seed, "episode evaluation seed drifted")
        require(record["execution_horizon"] == execution_horizon, "episode execution horizon drifted")
        require(record["policy_budget"] == POLICY_BUDGETS[suite], "episode policy budget drifted")
        name = record["task_name"]
        require(isinstance(name, str) and bool(name), "episode task name is invalid")
        prior_name = task_names.setdefault((suite, task_id), name)
        require(prior_name == name, "episode task name changed within one suite/task identity")
        if expected_task_names is not None:
            require(expected_task_names.get((suite, task_id)) == name, "episode task name differs from attestation")

        require(type(record["success"]) is bool, "episode success flag must be boolean")
        require(type(record["simulator_done"]) is bool, "episode simulator_done flag must be boolean")
        settle_steps = record["settle_steps"]
        require(type(settle_steps) is int and 1 <= settle_steps <= SETTLE_STEPS, "episode settle-step count is invalid")
        steps = record["policy_steps"]
        budget = record["policy_budget"]
        require(type(steps) is int and 0 <= steps <= budget, "episode policy-step count is invalid")
        require(steps == 0 or settle_steps == SETTLE_STEPS, "policy action began before mandatory settling completed")
        if record["success"]:
            require(record["steps_to_success"] == steps, "successful episode steps-to-success drifted")
        else:
            require(steps == budget, "failed episode did not exhaust its policy budget")
            require(record["steps_to_success"] is None, "failed episode has steps-to-success")
        calls = record["policy_calls"]
        require(type(calls) is int and calls == math.ceil(steps / execution_horizon), "policy call count drifted")
        continuous = record["action_continuous_channels"]
        clipped = record["action_clipped_channels"]
        require(type(continuous) is int and continuous == steps * 6, "continuous action-channel count drifted")
        require(type(clipped) is int and 0 <= clipped <= continuous, "clipped action-channel count is invalid")
        expected_clip = clipped / continuous if continuous else 0.0
        require(_same_number(record["action_clip_fraction"], expected_clip), "action clip fraction drifted")
        require(
            _finite_number(record["normalized_action_clip_fraction"], maximum=1.0),
            "normalized action clip fraction is invalid",
        )
        require(_finite_number(record["elapsed_seconds"]), "episode elapsed time is invalid")
        _validate_latency(
            record["policy_latency_seconds"],
            record["policy_latency_p50_seconds"],
            record["policy_latency_p95_seconds"],
            calls=calls,
            name="policy",
        )
        _validate_latency(
            record["server_latency_seconds"],
            record["server_latency_p50_seconds"],
            record["server_latency_p95_seconds"],
            calls=calls,
            name="server",
        )
        _validate_latency_pair(record["policy_latency_seconds"], record["server_latency_seconds"])
        elapsed_seconds = float(record["elapsed_seconds"])
        summed_client_latency = sum(float(value) for value in record["policy_latency_seconds"])
        require(
            elapsed_seconds + max(1e-9, elapsed_seconds * 1e-12) >= summed_client_latency,
            "episode elapsed time is shorter than its client policy latency",
        )
        checked.append(dict(record))
    require(len(task_names) == 40, "official episodes do not cover all 40 suite/task identities")
    return checked


def _validate_warmups(value: Any, *, cell: Mapping[str, Any], evaluation_seed: int) -> dict[str, Any]:
    require(isinstance(value, Mapping), "run policy_warmup must be an object")
    _require_exact_keys(
        value,
        {
            "count",
            "included_in_episode_latency",
            "k_independent_response_sha256",
            "replan_id",
            "reports",
            "validated_before_scoring",
        },
        "run policy_warmup",
    )
    require(value["count"] == OFFICIAL_POLICY_WARMUP_CALLS, "official warm-up count drifted")
    require(value["included_in_episode_latency"] is False, "warm-up latency was included in episodes")
    require(value["replan_id"] == _evaluate_libero.OFFICIAL_POLICY_WARMUP_REPLAN_ID, "warm-up replan ID drifted")
    require(value["validated_before_scoring"] is True, "warm-up was not validated before scoring")
    _require_sha256(value["k_independent_response_sha256"], "warm-up K-independent response SHA-256")
    reports = value["reports"]
    require(
        isinstance(reports, list) and len(reports) == OFFICIAL_POLICY_WARMUP_CALLS, "warm-up reports are incomplete"
    )
    require(
        [report.get("execution_horizon") for report in reports] == list(OFFICIAL_EXECUTION_HORIZONS),
        "warm-up K probe inventory drifted",
    )
    for index, report in enumerate(reports):
        require(isinstance(report, Mapping), "warm-up report must be an object")
        _require_exact_keys(report, _WARMUP_FIELDS, "warm-up report")
        require(report["warmup_index"] == index, "warm-up report order drifted")
        require(report["status"] == "ok", "warm-up did not complete")
        require(report["evaluation_seed"] == evaluation_seed, "warm-up evaluation seed drifted")
        for name in ("inference_seed_behavior", "nfe", "objective", "sampler"):
            require(report[name] == cell[name], f"warm-up {name} differs from its cell")
        _require_sha256(report["actions_sha256"], "warm-up actions SHA-256")
        require(type(report["inference_seed"]) is int and 0 <= report["inference_seed"] < 2**63, "warm-up seed invalid")
        require(_finite_number(report["policy_seconds"]), "warm-up policy latency is invalid")
        require(
            report["replan_id"] == _evaluate_libero.OFFICIAL_POLICY_WARMUP_REPLAN_ID,
            "warm-up did not use the reserved replan identity",
        )
        require(report["reset_id"] == 0 and report["reset_source"] == "official", "warm-up identity drifted")
        require(report["reset_state_sha256"] is None, "warm-up official reset hash must be null")
    require(
        len({report["actions_sha256"] for report in reports}) == 1,
        "warm-up actions were not K-independent",
    )
    require(len({report["inference_seed"] for report in reports}) == 1, "warm-up inference seed depended on K")
    expected_response_sha256 = canonical_sha256(warmup_k_independent_response(reports[0]))
    require(
        value["k_independent_response_sha256"] == expected_response_sha256,
        "warm-up K-independent response digest mismatch",
    )
    return {
        "actions_sha256": reports[0]["actions_sha256"],
        "count": value["count"],
        "inference_seed": reports[0]["inference_seed"],
        "k_independent_response_sha256": expected_response_sha256,
        "replan_id": value["replan_id"],
    }


def _flatten_comparison_metrics(summary: Mapping[str, Any]) -> dict[str, float]:
    metrics = {
        "episode_throughput_per_hour": summary["episode_throughput_per_hour"],
        "overall_40_task_macro_success": summary["overall_40_task_macro_success"],
        "overall_pooled_success_rate": summary["overall_pooled_success_rate"],
        "policy_call_throughput_per_second": summary["policy_call_throughput_per_second"],
        "policy_latency_p50_seconds": summary["policy_latency_p50_seconds"],
        "policy_latency_p95_seconds": summary["policy_latency_p95_seconds"],
        "server_latency_p50_seconds": summary["server_latency_p50_seconds"],
        "server_latency_p95_seconds": summary["server_latency_p95_seconds"],
    }
    suites = summary["suites"]
    require(isinstance(suites, list), "summary suites must be a list")
    suite_map = {value["suite"]: value for value in suites if isinstance(value, Mapping) and "suite" in value}
    require(set(suite_map) == set(SUITES) and len(suites) == len(SUITES), "summary suite inventory is incomplete")
    for suite in SUITES:
        metrics[f"{suite}_task_macro_success"] = suite_map[suite]["suite_task_macro_success"]
        metrics[f"{suite}_pooled_success_rate"] = suite_map[suite]["pooled_success_rate"]
    require(set(metrics) == set(_COMPARISON_METRICS), "aggregate comparison metric inventory drifted")
    rate_metrics = {name for name in metrics if name.endswith("success") or name.endswith("success_rate")}
    require(
        all(_finite_number(metrics[name], maximum=1.0) for name in rate_metrics),
        "summary success metric is invalid",
    )
    require(
        all(_finite_number(value) for name, value in metrics.items() if name not in rate_metrics),
        "summary latency/throughput metric is invalid",
    )
    require(
        float(metrics["policy_latency_p50_seconds"]) >= float(metrics["server_latency_p50_seconds"])
        and float(metrics["policy_latency_p95_seconds"]) >= float(metrics["server_latency_p95_seconds"]),
        "summary client latency is below server latency",
    )
    return {name: float(value) for name, value in metrics.items()}


def _validate_sha256_sidecar(path: Path, digest: str, *, name: str) -> None:
    companion = path.with_suffix(path.suffix + ".sha256")
    raw = _read_single_link_regular_bytes(companion, name=f"{name} SHA-256 companion")
    expected = f"{digest}  {path.name}\n".encode("ascii")
    require(raw == expected, f"{name} SHA-256 companion mismatch")


def _require_exact_output_inventories(
    preregistration: Mapping[str, Any],
    registered_cells: Sequence[Mapping[str, Any]],
) -> None:
    """Reject missing, extra, linked, or alternate artifacts under frozen roots."""

    roots = validate_official_output_roots(preregistration["official_output_roots"], require_live=True)
    cell_ids = {cell["cell_id"] for cell in registered_cells}
    runs_root = Path(roots["runs"]["path"])
    claims_root = Path(roots["claims"]["path"])
    try:
        run_entries = list(os.scandir(str(runs_root)))
        claim_entries = list(os.scandir(str(claims_root)))
    except OSError as exc:
        raise RuntimeError("cannot enumerate official output roots") from exc
    require({entry.name for entry in run_entries} == cell_ids, "official run root inventory is not exact")
    for entry in run_entries:
        require(entry.is_dir(follow_symlinks=False), f"official run root entry is not a real directory: {entry.name}")
        expected = Path(
            next(cell["output_claim"]["output_dir"] for cell in registered_cells if cell["cell_id"] == entry.name)
        )
        require(Path(entry.path) == expected, f"official run directory is not pre-registered: {entry.name}")
    expected_claims = {name for cell_id in cell_ids for name in (f"{cell_id}.json", f"{cell_id}.json.sha256")}
    require({entry.name for entry in claim_entries} == expected_claims, "official claim root inventory is not exact")
    for entry in claim_entries:
        require(entry.is_file(follow_symlinks=False), f"official claim root entry is not regular: {entry.name}")
        require(
            os.stat(entry.path, follow_symlinks=False).st_nlink == 1,
            f"official claim artifact is linked: {entry.name}",
        )
    validate_official_output_roots(roots, require_live=True)


def _require_exact_run_directory(output_dir: Path) -> None:
    require(output_dir.is_absolute() and output_dir.resolve(strict=True) == output_dir, "official run path is not real")
    require(stat.S_ISDIR(os.lstat(str(output_dir)).st_mode), "official run path is not a real directory")
    try:
        entries = list(os.scandir(str(output_dir)))
    except OSError as exc:
        raise RuntimeError(f"cannot enumerate official run directory: {output_dir}") from exc
    expected = {"completion.json", "completion.json.sha256", "episodes.jsonl", "run.json", "summary.json"}
    require({entry.name for entry in entries} == expected, "official run directory inventory is not exact")
    for entry in entries:
        require(entry.is_file(follow_symlinks=False), f"official run artifact is not regular: {entry.name}")
        require(
            os.stat(entry.path, follow_symlinks=False).st_nlink == 1,
            f"official run artifact is linked: {entry.name}",
        )


def validate_run_artifact(
    entry: Mapping[str, Any],
    *,
    preregistration: Mapping[str, Any],
    preregistration_sha256: str,
    registered_cell: Mapping[str, Any],
    aggregation_sources: Mapping[str, Any] = _STARTUP_AGGREGATION_SOURCE_IDENTITIES,
) -> dict[str, Any]:
    """Authenticate one raw run/episodes/summary triple and recompute its summary."""

    _require_exact_keys(entry, _INVENTORY_RUN_FIELDS, "run inventory entry")
    for name in (
        "claim_json_sha256",
        "completion_json_sha256",
        "run_json_sha256",
        "episodes_jsonl_sha256",
        "summary_json_sha256",
    ):
        _require_sha256(entry[name], f"inventory {name}")
    require(entry["cell_id"] == registered_cell["cell_id"], "run inventory cell ID mismatch")
    output_claim = registered_cell["output_claim"]
    output_dir = Path(output_claim["output_dir"])
    require(output_dir.is_dir(), f"official run directory is missing: {output_dir}")
    _require_exact_run_directory(output_dir)
    claim_path = Path(output_claim["claim_path"])
    claim, claim_sha256 = _read_json_object(
        claim_path,
        name="official claim JSON",
        expected_sha256=entry["claim_json_sha256"],
    )
    _validate_sha256_sidecar(claim_path, claim_sha256, name="official claim JSON")
    validate_claim_record(
        claim,
        output_claim=output_claim,
        cell_id=registered_cell["cell_id"],
        preregistration_sha256=preregistration_sha256,
        final_freeze_token_sha256=preregistration["final_freeze_token_sha256"],
    )
    run, run_sha256 = _read_json_object(
        output_dir / "run.json",
        name="official run.json",
        expected_sha256=entry["run_json_sha256"],
    )
    _require_exact_keys(run, _COMPLETE_RUN_FIELDS, "complete official run")
    require(run["schema"] == _evaluate_libero.OFFICIAL_RUN_SCHEMA, "official run schema mismatch")
    require(run["status"] == "complete", "official run is not complete")
    require(run["mode"] == "official-score", "run is not an official score")
    require(run["protocol"] == PROTOCOL, "official run protocol mismatch")
    require(run["preregistration_sha256"] == preregistration_sha256, "run pre-registration digest mismatch")
    require(run["claim_json_sha256"] == claim_sha256, "run/claim digest binding mismatch")
    require(
        canonical_json_bytes(run["output_claim"]) == canonical_json_bytes(output_claim),
        "run output claim differs from pre-registration",
    )
    require(
        run["final_freeze_token_sha256"] == preregistration["final_freeze_token_sha256"],
        "run freeze-token digest mismatch",
    )
    require(
        canonical_json_bytes(run["cell"]) == canonical_json_bytes(registered_cell),
        "run cell differs from pre-registration",
    )
    require(run["evaluation_seed"] == preregistration["evaluation_seed"], "run evaluation seed mismatch")
    execution_horizon = registered_cell["execution_horizon"]
    require(run["execution_horizon"] == execution_horizon, "run execution horizon differs from its cell")
    require(run["suites"] == list(SUITES), "run suite inventory drifted")
    require(run["task_ids"] == list(range(10)), "run task inventory drifted")
    require(run["init_state_ids"] == list(range(50)), "run reset inventory drifted")
    require(run["reset_source"] == "official", "official run used a non-official reset source")
    require(run["episode_count"] == OFFICIAL_PRIMARY_EPISODES, "run episode denominator drifted")
    require(run["episode_records"] == OFFICIAL_PRIMARY_EPISODES, "run episode record count drifted")
    require(run["episode_matrix_sha256"] == preregistration["episode_matrix_sha256"], "run episode matrix drifted")
    require(run["contamination"] == preregistration["contamination"], "run contamination contract drifted")
    require(isinstance(run["created_utc"], str) and bool(run["created_utc"]), "run timestamp is invalid")
    require(isinstance(run["finished_utc"], str) and bool(run["finished_utc"]), "run finish timestamp is invalid")
    require(isinstance(run["policy_socket"], str) and bool(run["policy_socket"]), "run policy socket is invalid")
    require(
        isinstance(run["preregistration_manifest"], str) and bool(run["preregistration_manifest"]),
        "run pre-registration path is invalid",
    )
    reset_identity = run["reset_identity"]
    require(isinstance(reset_identity, Mapping), "run reset identity is invalid")
    require(
        reset_identity
        == {
            "bank": None,
            "id_field": "published_init_state_id",
            "source": "official",
            "state_sha256": None,
        },
        "run reset identity drifted",
    )
    evaluator_environment = run["evaluator_environment"]
    require(isinstance(evaluator_environment, Mapping), "run evaluator environment is invalid")
    expected_environment_fields = {
        *_evaluate_libero._REQUIRED_EVALUATOR_ENVIRONMENT,
        "DUO_VLA_CACHE_ROOT",
        "HF_HOME",
        "LANG",
        "LC_ALL",
        "LIBERO_CONFIG_PATH",
        "PATH",
    }
    _require_exact_keys(evaluator_environment, expected_environment_fields, "run evaluator environment")
    for name, expected in _evaluate_libero._REQUIRED_EVALUATOR_ENVIRONMENT.items():
        require(evaluator_environment[name] == expected, f"run evaluator environment {name} drifted")
    cache_root = evaluator_environment["DUO_VLA_CACHE_ROOT"]
    require(isinstance(cache_root, str) and Path(cache_root).is_absolute(), "run evaluator cache root is invalid")
    require(evaluator_environment["HF_HOME"] == "/root/.cache/huggingface", "run evaluator HF_HOME drifted")
    require(evaluator_environment["LANG"] == evaluator_environment["LC_ALL"] == "C.UTF-8", "run locale drifted")
    require(
        evaluator_environment["LIBERO_CONFIG_PATH"] == str((Path(cache_root) / "simulators/libero/config").resolve()),
        "run LIBERO config path drifted",
    )
    require(
        evaluator_environment["PATH"] == f"{(Path(cache_root) / 'venvs/libero-eval/bin').resolve()}:/usr/bin:/bin",
        "run evaluator PATH drifted",
    )
    qualified_validator_runtime = _evaluate_libero.validate_validator_runtime_identity(
        preregistration["expert_replay_qualification"]["validator_runtime_identity"]
    )
    run_validator_runtime = _evaluate_libero.validate_validator_runtime_identity(run["validator_runtime_identity"])
    require(
        canonical_json_bytes(run_validator_runtime) == canonical_json_bytes(qualified_validator_runtime),
        "run validator runtime differs from expert replay qualification",
    )

    simulator_preflight = run["simulator_preflight"]
    require(isinstance(simulator_preflight, Mapping), "run simulator preflight is invalid")
    require(
        simulator_preflight.get("schema") == SIMULATOR_ATTESTATION_SCHEMA
        and simulator_preflight.get("status") == "ok"
        and simulator_preflight.get("environment_constructed") is True,
        "run simulator preflight is not a successful full attestation",
    )
    simulator_sha256 = canonical_sha256(simulator_preflight)
    require(run["simulator_attestation_sha256"] == simulator_sha256, "run simulator attestation self-binding mismatch")
    require(
        simulator_sha256 == preregistration["simulator_attestation_sha256"],
        "run simulator attestation differs from pre-registration",
    )
    _evaluate_libero.validate_validator_runtime_against_attestation(
        qualified_validator_runtime,
        simulator_preflight,
    )
    project_sources = simulator_preflight.get("project_sources")
    require(isinstance(project_sources, Mapping), "run simulator attestation has no project source identities")
    require(
        project_sources.get("evaluator") == aggregation_sources["evaluate_libero.py"]["sha256"],
        "run-attested evaluator source differs from aggregation source",
    )
    require(
        project_sources.get("bridge") == aggregation_sources["libero_bridge.py"]["sha256"],
        "run-attested bridge source differs from aggregation source",
    )
    require(
        project_sources.get("preflight") == aggregation_sources["preflight_libero_env.py"]["sha256"],
        "run-attested preflight source differs from aggregation source",
    )
    task_inventory = simulator_preflight.get("task_inventory")
    require(isinstance(task_inventory, list) and len(task_inventory) == 40, "simulator task inventory is incomplete")
    expected_task_names: dict[tuple[str, int], str] = {}
    for task in task_inventory:
        require(isinstance(task, Mapping), "simulator task inventory entry is invalid")
        identity = (task.get("suite"), task.get("task_id"))
        task_name = task.get("task_name")
        require(
            identity in {(suite, task_id) for suite in SUITES for task_id in range(10)},
            "simulator task identity is invalid",
        )
        require(isinstance(task_name, str) and bool(task_name), "simulator task name is invalid")
        require(identity not in expected_task_names, "simulator task inventory contains a duplicate identity")
        expected_task_names[identity] = task_name
    require(len(expected_task_names) == 40, "simulator task inventory does not cover all 40 tasks")

    checkpoint = run["checkpoint"]
    require(isinstance(checkpoint, Mapping), "run checkpoint identity must be an object")
    _require_exact_keys(checkpoint, _CHECKPOINT_IDENTITY_FIELDS, "run checkpoint identity")
    require(checkpoint["run_journal_latest"] is True, "run checkpoint was not the journal tip")
    require(checkpoint["update"] == registered_cell["checkpoint"]["update"], "run checkpoint update drifted")
    for name in ("manifest_sha256", "source_tree_sha256"):
        require(checkpoint[name] == registered_cell["checkpoint"][name], f"run checkpoint {name} drifted")
    require(
        all(isinstance(checkpoint[name], str) and bool(checkpoint[name]) for name in ("path", "run_root")),
        "run checkpoint path identity is invalid",
    )
    checkpoint_path = Path(checkpoint["path"])
    require(
        checkpoint_path.is_absolute()
        and checkpoint_path.parent.name == "checkpoints"
        and checkpoint_path.parent.parent == Path(checkpoint["run_root"]),
        "run checkpoint path is not inside its canonical run root",
    )
    qualified_train_venv = _evaluate_libero.validate_train_venv_identity(
        preregistration["expert_replay_qualification"]["train_venv_identity"],
        name="pre-registered expert replay train-venv identity",
    )
    run_train_venv = _evaluate_libero.validate_train_venv_identity(
        checkpoint["train_venv"],
        name="run checkpoint train-venv identity",
    )
    require(
        canonical_json_bytes(run_train_venv) == canonical_json_bytes(qualified_train_venv),
        "run checkpoint train-venv differs from expert replay qualification",
    )

    health = run["policy_health"]
    require(isinstance(health, dict), "run policy health must be an object")
    _require_exact_keys(health, _HEALTH_FIELDS, "run policy health")
    validate_policy_health(health, allow_fake_policy=False)
    require(
        health["schema"] == "duo-vla-libero-policy-ipc-v5"
        and health["status"] == "ok"
        and health["operation"] == "health",
        "run policy health envelope drifted",
    )
    require(isinstance(health["request_id"], str) and bool(health["request_id"]), "run policy request ID is invalid")
    require(health["protocol"] == PROTOCOL, "run policy protocol drifted")
    require(
        (health["action_dim"], health["action_horizon"], health["state_dim"]) == (7, 8, 8),
        "run policy tensor geometry drifted",
    )
    require(health["mode"] == "real", "official run recorded fake policy health")
    require(health["train_seed"] == registered_cell["train_seed"], "run policy train seed drifted")
    require(
        health["serving_runtime_sha256"] == registered_cell["serving_runtime_sha256"], "run serving runtime drifted"
    )
    require(
        health["latency_runtime_sha256"] == registered_cell["latency_runtime_sha256"],
        "run latency runtime drifted",
    )
    require(health["execution_geometry"] == registered_cell["execution_geometry"], "run execution geometry drifted")
    for name in ("inference_seed_behavior", "nfe", "objective", "sampler"):
        require(health[name] == registered_cell[name], f"run policy {name} differs from its cell")
    health_checkpoint = health["checkpoint"]
    require(isinstance(health_checkpoint, Mapping), "run health checkpoint must be an object")
    _require_exact_keys(health_checkpoint, _HEALTH_CHECKPOINT_FIELDS, "run health checkpoint")
    require(health_checkpoint["kind"] == "resumable-libero-training", "health checkpoint kind drifted")
    require(health_checkpoint["train_seed"] == registered_cell["train_seed"], "health checkpoint seed drifted")
    require(
        health_checkpoint["manifest_sha256"] == registered_cell["checkpoint"]["manifest_sha256"],
        "health checkpoint manifest drifted",
    )
    require(
        health_checkpoint["source_tree_sha256"] == registered_cell["checkpoint"]["source_tree_sha256"],
        "health checkpoint source drifted",
    )
    require(
        health_checkpoint["policy_contract_sha256"] == registered_cell["policy_contract_sha256"],
        "health checkpoint contract drifted",
    )
    require(
        health_checkpoint["execution_geometry"] == registered_cell["execution_geometry"],
        "health checkpoint geometry drifted",
    )
    require(health_checkpoint["path"] == checkpoint["path"], "run and health checkpoint paths differ")
    qualification = preregistration["expert_replay_qualification"]
    require(
        health_checkpoint["dataset_tree_sha256"] == registered_cell["checkpoint"]["dataset_tree_sha256"]
        and health_checkpoint["dataset_content_inventory_sha256"]
        == registered_cell["checkpoint"]["dataset_content_inventory_sha256"],
        "health checkpoint dataset identity drifted",
    )
    require(
        health_checkpoint["dataset_files_verified"] == qualification["dataset_snapshot_files_verified"]
        and health_checkpoint["dataset_total_bytes"] == qualification["dataset_snapshot_total_bytes"],
        "health checkpoint dataset snapshot counts drifted",
    )
    for name in ("model_content_inventory_sha256", "model_tree_sha256"):
        _require_sha256(health_checkpoint[name], f"health checkpoint {name}")
    for name in ("model_files_verified", "model_total_bytes"):
        require(
            type(health_checkpoint[name]) is int and health_checkpoint[name] > 0,
            f"health checkpoint {name} is invalid",
        )
    health_train_venv = _evaluate_libero.validate_train_venv_identity(
        health_checkpoint["train_venv"],
        name="health checkpoint train-venv identity",
    )
    require(
        canonical_json_bytes(health_train_venv)
        == canonical_json_bytes(run_train_venv)
        == canonical_json_bytes(qualified_train_venv),
        "health checkpoint train-venv differs from the run and expert replay qualification",
    )
    require(
        canonical_sha256(health_checkpoint["policy_contract"]) == registered_cell["policy_contract_sha256"],
        "health checkpoint policy contract self-binding mismatch",
    )
    _require_sha256(
        health_checkpoint["training_execution_environment_sha256"],
        "health training execution environment SHA-256",
    )
    require(
        canonical_sha256(health_checkpoint["training_execution_environment"])
        == health_checkpoint["training_execution_environment_sha256"],
        "health training execution environment self-binding mismatch",
    )
    training_environment = health_checkpoint["training_execution_environment"]
    require(isinstance(training_environment, Mapping), "health training execution environment is invalid")
    authenticated_runtime = training_environment.get("authenticated_runtime")
    require(isinstance(authenticated_runtime, Mapping), "health training environment has no authenticated runtime")
    authenticated_train_venv = _evaluate_libero.validate_train_venv_identity(
        authenticated_runtime.get("train_venv"),
        name="health authenticated-runtime train-venv identity",
    )
    require(
        canonical_json_bytes(authenticated_train_venv) == canonical_json_bytes(qualified_train_venv),
        "health authenticated train-venv differs from expert replay qualification",
    )
    warmup_output = _validate_warmups(
        run["policy_warmup"],
        cell=registered_cell,
        evaluation_seed=run["evaluation_seed"],
    )

    require(run["episodes_jsonl_sha256"] == entry["episodes_jsonl_sha256"], "run episode digest drifted")
    require(run["summary_json_sha256"] == entry["summary_json_sha256"], "run summary digest drifted")

    records = _load_jsonl(output_dir / "episodes.jsonl", expected_sha256=entry["episodes_jsonl_sha256"])
    checked_records = validate_episode_records(
        records,
        planned_episodes=preregistration["episodes"],
        evaluation_seed=preregistration["evaluation_seed"],
        execution_horizon=execution_horizon,
        expected_task_names=expected_task_names,
    )
    summary, summary_sha256 = _read_json_object(
        output_dir / "summary.json",
        name="official summary.json",
        expected_sha256=entry["summary_json_sha256"],
    )
    expected_summary = bind_official_summary(
        summarize_episodes(checked_records, execution_horizon=execution_horizon),
        contamination=preregistration["contamination"],
        episode_matrix_sha256=preregistration["episode_matrix_sha256"],
    )
    require(summary.get("schema") == _evaluate_libero.OFFICIAL_SUMMARY_SCHEMA, "official summary schema mismatch")
    require(
        canonical_json_bytes(summary) == canonical_json_bytes(expected_summary),
        "official summary does not match episodes",
    )
    completion_path = output_dir / "completion.json"
    completion, completion_sha256 = _read_json_object(
        completion_path,
        name="official completion JSON",
        expected_sha256=entry["completion_json_sha256"],
    )
    _validate_sha256_sidecar(completion_path, completion_sha256, name="official completion JSON")
    _require_exact_keys(completion, _evaluate_libero._COMPLETION_FIELDS, "official completion record")
    require(completion["schema"] == COMPLETION_SCHEMA, "official completion schema mismatch")
    expected_completion = {
        "cell_id": registered_cell["cell_id"],
        "claim_json_sha256": claim_sha256,
        "episode_records": OFFICIAL_PRIMARY_EPISODES,
        "episodes_jsonl_sha256": entry["episodes_jsonl_sha256"],
        "preregistration_sha256": preregistration_sha256,
        "run_json_sha256": run_sha256,
        "schema": COMPLETION_SCHEMA,
        "summary_json_sha256": summary_sha256,
    }
    require(
        canonical_json_bytes(completion) == canonical_json_bytes(expected_completion),
        "official completion record binding mismatch",
    )
    return {
        "cell_id": registered_cell["cell_id"],
        "checkpoint_manifest_sha256": registered_cell["checkpoint"]["manifest_sha256"],
        "checkpoint_source_tree_sha256": registered_cell["checkpoint"]["source_tree_sha256"],
        "claim_json_sha256": claim_sha256,
        "completion_json_sha256": completion_sha256,
        "episodes_jsonl_sha256": entry["episodes_jsonl_sha256"],
        "execution_horizon": execution_horizon,
        "inference_seed_behavior": registered_cell["inference_seed_behavior"],
        "latency_runtime_sha256": registered_cell["latency_runtime_sha256"],
        "metrics": _flatten_comparison_metrics(summary),
        "nfe": registered_cell["nfe"],
        "objective": registered_cell["objective"],
        "policy_contract_sha256": registered_cell["policy_contract_sha256"],
        "run_json_sha256": run_sha256,
        "sampler": registered_cell["sampler"],
        "serving_policy_sha256": registered_cell["serving_policy_sha256"],
        "serving_runtime_sha256": registered_cell["serving_runtime_sha256"],
        "summary": summary,
        "summary_json_sha256": summary_sha256,
        "train_seed": registered_cell["train_seed"],
        "warmup_output": warmup_output,
    }


def _metric_aggregate(cell_results: Sequence[Mapping[str, Any]], metric: str) -> dict[str, Any]:
    ordered = sorted(cell_results, key=lambda value: value["train_seed"])
    require(
        [value["train_seed"] for value in ordered] == list(OFFICIAL_TRAIN_SEEDS), "aggregate seed set is incomplete"
    )
    values = [float(value["metrics"][metric]) for value in ordered]
    return {
        "mean": statistics.fmean(values),
        "sample_std": statistics.stdev(values),
        "values_by_train_seed": [
            {"train_seed": value["train_seed"], "value": value["metrics"][metric]} for value in ordered
        ],
    }


def aggregate_matrix(
    preregistration: Mapping[str, Any],
    preregistration_sha256: str,
    inventory: Mapping[str, Any],
    inventory_sha256: str,
    *,
    aggregation_sources: Mapping[str, Any] = _STARTUP_AGGREGATION_SOURCE_IDENTITIES,
) -> dict[str, Any]:
    """Reject incomplete/off-matrix inputs and compare three seeds without pooling K."""

    require_aggregation_sources_unchanged(aggregation_sources)
    _require_sha256(preregistration_sha256, "pre-registration SHA-256")
    _require_sha256(inventory_sha256, "run inventory SHA-256")
    contamination = load_contamination_contract(_SCRIPT_DIR.parent)
    registered_cells = validate_preregistration_manifest(
        preregistration,
        contamination=contamination,
        simulator_attestation_sha256=preregistration.get("simulator_attestation_sha256", ""),
    )
    aggregator_identity = aggregation_sources["aggregate_libero_official.py"]
    require(
        hmac.compare_digest(preregistration["aggregator_sha256"], aggregator_identity["sha256"]),
        "aggregation startup source differs from the pre-registered aggregator",
    )
    require(
        preregistration["aggregation_python_version"] == AGGREGATION_PYTHON_VERSION,
        "pre-registered aggregation Python changed",
    )
    _require_exact_keys(inventory, _INVENTORY_FIELDS, "official run inventory")
    require(inventory["schema"] == RUN_INVENTORY_SCHEMA, "official run inventory schema mismatch")
    require(inventory["preregistration_sha256"] == preregistration_sha256, "run inventory pre-registration mismatch")
    runs = inventory["runs"]
    require(isinstance(runs, list), "official run inventory runs must be a list")
    require(len(runs) == 24, "official run inventory must contain exactly 24 runs")
    for entry in runs:
        require(isinstance(entry, dict), "official run inventory entries must be objects")
        _require_exact_keys(entry, _INVENTORY_RUN_FIELDS, "run inventory entry")
        require(isinstance(entry["cell_id"], str) and bool(entry["cell_id"]), "run inventory cell ID is invalid")
        for name in (
            "claim_json_sha256",
            "completion_json_sha256",
            "run_json_sha256",
            "episodes_jsonl_sha256",
            "summary_json_sha256",
        ):
            _require_sha256(entry[name], f"inventory {name}")
    inventory_ids = [entry["cell_id"] for entry in runs]
    require(len(inventory_ids) == len(set(inventory_ids)), "official run inventory contains duplicate cell IDs")
    cells_by_id = {cell["cell_id"]: cell for cell in registered_cells}
    require(set(inventory_ids) == set(cells_by_id), "official run inventory has missing or off-matrix cells")
    _require_exact_output_inventories(preregistration, registered_cells)

    cell_results = [
        validate_run_artifact(
            entry,
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
    factors = {
        (value["train_seed"], value["objective"], value["nfe"], value["execution_horizon"]) for value in cell_results
    }
    require(factors == _official_factor_matrix(), "validated run factors do not form the canonical matrix")
    for train_seed in OFFICIAL_TRAIN_SEEDS:
        for objective, nfes in (("rectified_flow", OFFICIAL_FLOW_NFES), ("direct_regression", (1,))):
            for nfe in nfes:
                cross_k = [
                    value
                    for value in cell_results
                    if value["train_seed"] == train_seed and value["objective"] == objective and value["nfe"] == nfe
                ]
                require(len(cross_k) == len(OFFICIAL_EXECUTION_HORIZONS), "cross-K warm-up group is incomplete")
                require(
                    len({canonical_json_bytes(value["warmup_output"]) for value in cross_k}) == 1,
                    "warm-up output or inference seed differs across K",
                )
    comparisons: list[dict[str, Any]] = []
    for objective, nfes in (("rectified_flow", OFFICIAL_FLOW_NFES), ("direct_regression", (1,))):
        for nfe in nfes:
            for execution_horizon in OFFICIAL_EXECUTION_HORIZONS:
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
                        "latency_runtime_sha256": group[0]["latency_runtime_sha256"],
                        "metrics": {metric: _metric_aggregate(group, metric) for metric in _COMPARISON_METRICS},
                        "nfe": nfe,
                        "objective": objective,
                    }
                )
    require(len(comparisons) == 8, "official matrix must emit exactly eight K-specific comparisons")
    latency_runtime_sha256 = {cell["latency_runtime_sha256"] for cell in cell_results}
    require(len(latency_runtime_sha256) == 1, "matrix mixes incomparable latency runtime identities")
    require_aggregation_sources_unchanged(aggregation_sources)
    _require_exact_output_inventories(preregistration, registered_cells)
    source_tree_sha256 = {cell["checkpoint"]["source_tree_sha256"] for cell in registered_cells}
    require(len(source_tree_sha256) == 1, "official checkpoint source identity is not unique")
    result = {
        "aggregation_python_version": AGGREGATION_PYTHON_VERSION,
        "aggregation_source_identities": json.loads(canonical_json_bytes(aggregation_sources).decode("ascii")),
        "aggregator_sha256": preregistration["aggregator_sha256"],
        "benchmark_protocol": PROTOCOL,
        "cell_count": len(cell_results),
        "cells": cell_results,
        "comparison_count": len(comparisons),
        "comparisons": comparisons,
        "episode_count_per_cell": OFFICIAL_PRIMARY_EPISODES,
        "evaluation_seed": preregistration["evaluation_seed"],
        "expert_replay_qualification": preregistration["expert_replay_qualification"],
        "latency_runtime_sha256": next(iter(latency_runtime_sha256)),
        "latency_scope": "episode_policy_calls_only",
        "policy_warmup_calls": preregistration["policy_warmup_calls"],
        "policy_warmup_included_in_latency": False,
        "preregistration_sha256": preregistration_sha256,
        "run_inventory_sha256": inventory_sha256,
        "schema": MATRIX_SUMMARY_SCHEMA,
        "total_episode_count": OFFICIAL_PRIMARY_EPISODES * len(cell_results),
        "training_seeds": list(OFFICIAL_TRAIN_SEEDS),
        "training_source_tree_sha256": next(iter(source_tree_sha256)),
    }
    result["content_sha256"] = canonical_sha256(result)
    return result


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(str(path), os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _unlink_if_same(path: Path, identity: tuple[int, int] | None) -> None:
    if identity is None:
        return
    try:
        observed = os.stat(path, follow_symlinks=False)
        if stat.S_ISREG(observed.st_mode) and (observed.st_dev, observed.st_ino) == identity:
            path.unlink()
    except OSError:
        return


def _verify_published_file(
    path: Path,
    identity: tuple[int, int],
    *,
    expected_bytes: int,
    expected_sha256: str,
    links: int,
) -> None:
    """Verify one publication name through a stable, no-follow descriptor."""

    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    require(hasattr(os, "O_NOFOLLOW"), "exclusive publication requires O_NOFOLLOW")
    flags |= os.O_NOFOLLOW
    descriptor = os.open(str(path), flags)
    digest = hashlib.sha256()
    size = 0
    try:
        before = os.fstat(descriptor)
        require(
            stat.S_ISREG(before.st_mode) and (before.st_dev, before.st_ino) == identity and before.st_nlink == links,
            f"published target identity changed: {path}",
        )
        with os.fdopen(descriptor, "rb") as source:
            descriptor = -1
            while block := source.read(1024 * 1024):
                size += len(block)
                digest.update(block)
            after = os.fstat(source.fileno())
        stable = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns", "st_nlink")
        require(
            all(getattr(before, name) == getattr(after, name) for name in stable)
            and size == expected_bytes == after.st_size
            and hmac.compare_digest(digest.hexdigest(), expected_sha256),
            f"published target content changed: {path}",
        )
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def publish_bytes_and_sha256_exclusive(
    path: Path,
    payload: bytes,
    *,
    commit_guard: Callable[[], None] | None = None,
) -> str:
    """Durably publish an immutable payload/sidecar pair; payload is the commit marker."""

    require(isinstance(payload, bytes), "exclusive publication payload must be bytes")
    parent = path.parent
    parent.mkdir(parents=True, exist_ok=True)
    companion = path.with_suffix(path.suffix + ".sha256")
    require(companion != path, "exclusive publication companion path collides with payload path")
    digest = hashlib.sha256(payload).hexdigest()
    companion_payload = f"{digest}  {path.name}\n".encode("ascii")
    companion_digest = hashlib.sha256(companion_payload).hexdigest()
    nonce = f"{os.getpid()}-{time.time_ns()}-{secrets.token_hex(8)}"
    temporary_payload = parent / f".{path.name}.tmp-{nonce}"
    temporary_companion = parent / f".{companion.name}.tmp-{nonce}"
    payload_identity: tuple[int, int] | None = None
    companion_identity: tuple[int, int] | None = None
    final_payload = False
    final_companion = False

    def write_temporary(candidate: Path, content: bytes) -> tuple[int, int]:
        require(hasattr(os, "O_NOFOLLOW"), "exclusive publication requires O_NOFOLLOW")
        descriptor = os.open(str(candidate), os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        with os.fdopen(descriptor, "wb") as sink:
            sink.write(content)
            sink.flush()
            os.fsync(sink.fileno())
        observed = os.stat(candidate, follow_symlinks=False)
        identity = (observed.st_dev, observed.st_ino)
        _verify_published_file(
            candidate,
            identity,
            expected_bytes=len(content),
            expected_sha256=hashlib.sha256(content).hexdigest(),
            links=1,
        )
        return identity

    try:
        payload_identity = write_temporary(temporary_payload, payload)
        companion_identity = write_temporary(temporary_companion, companion_payload)
        _verify_published_file(
            temporary_payload,
            payload_identity,
            expected_bytes=len(payload),
            expected_sha256=digest,
            links=1,
        )
        _verify_published_file(
            temporary_companion,
            companion_identity,
            expected_bytes=len(companion_payload),
            expected_sha256=companion_digest,
            links=1,
        )
        if commit_guard is not None:
            commit_guard()
        os.link(temporary_companion, companion, follow_symlinks=False)
        final_companion = True
        _fsync_directory(parent)
        for candidate in (temporary_companion, companion):
            _verify_published_file(
                candidate,
                companion_identity,
                expected_bytes=len(companion_payload),
                expected_sha256=companion_digest,
                links=2,
            )
        _verify_published_file(
            temporary_payload,
            payload_identity,
            expected_bytes=len(payload),
            expected_sha256=digest,
            links=1,
        )
        if commit_guard is not None:
            commit_guard()
        os.link(temporary_payload, path, follow_symlinks=False)
        final_payload = True
        _fsync_directory(parent)
        for candidate in (temporary_payload, path):
            _verify_published_file(
                candidate,
                payload_identity,
                expected_bytes=len(payload),
                expected_sha256=digest,
                links=2,
            )
        for candidate in (temporary_companion, companion):
            _verify_published_file(
                candidate,
                companion_identity,
                expected_bytes=len(companion_payload),
                expected_sha256=companion_digest,
                links=2,
            )
        if commit_guard is not None:
            commit_guard()
        _unlink_if_same(temporary_payload, payload_identity)
        _unlink_if_same(temporary_companion, companion_identity)
        _fsync_directory(parent)
        _verify_published_file(
            path,
            payload_identity,
            expected_bytes=len(payload),
            expected_sha256=digest,
            links=1,
        )
        _verify_published_file(
            companion,
            companion_identity,
            expected_bytes=len(companion_payload),
            expected_sha256=companion_digest,
            links=1,
        )
        return digest
    except BaseException:
        if final_payload:
            _unlink_if_same(path, payload_identity)
        if final_companion:
            _unlink_if_same(companion, companion_identity)
        _unlink_if_same(temporary_payload, payload_identity)
        _unlink_if_same(temporary_companion, companion_identity)
        with contextlib.suppress(OSError):
            _fsync_directory(parent)
        raise
    finally:
        _unlink_if_same(temporary_payload, payload_identity)
        _unlink_if_same(temporary_companion, companion_identity)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preregistration-manifest", type=Path, required=True)
    parser.add_argument("--preregistration-sha256", required=True)
    parser.add_argument("--run-inventory", type=Path, required=True)
    parser.add_argument("--run-inventory-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    require(
        platform.python_version() == AGGREGATION_PYTHON_VERSION,
        f"LIBERO official aggregator requires Python {AGGREGATION_PYTHON_VERSION}, found {platform.python_version()}",
    )
    project_root = _SCRIPT_DIR.parent
    _evaluate_libero.validate_evaluator_process_environment(project_root)
    require_aggregation_sources_unchanged(_STARTUP_AGGREGATION_SOURCE_IDENTITIES)
    preregistration, preregistration_sha256 = _read_json_object(
        args.preregistration_manifest.resolve(),
        name="LIBERO pre-registration",
        expected_sha256=args.preregistration_sha256,
    )
    inventory_path = args.run_inventory.resolve()
    inventory, inventory_sha256 = _read_json_object(
        inventory_path,
        name="LIBERO official run inventory",
        expected_sha256=args.run_inventory_sha256,
    )
    result = aggregate_matrix(
        preregistration,
        preregistration_sha256,
        inventory,
        inventory_sha256,
        aggregation_sources=_STARTUP_AGGREGATION_SOURCE_IDENTITIES,
    )
    output = args.output.resolve()

    def commit_guard() -> None:
        require_aggregation_sources_unchanged(_STARTUP_AGGREGATION_SOURCE_IDENTITIES)

    payload = (json.dumps(result, allow_nan=False, ensure_ascii=True, indent=2, sort_keys=True) + "\n").encode("ascii")
    digest = publish_bytes_and_sha256_exclusive(output, payload, commit_guard=commit_guard)
    print(json.dumps({"matrix_summary": str(output), "sha256": digest}, sort_keys=True))


if __name__ == "__main__":
    main()
