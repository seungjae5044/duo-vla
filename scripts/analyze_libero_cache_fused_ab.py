#!/usr/bin/env python3
"""Analyze the frozen 100-update LIBERO cache/fused grouped-MM A/B run.

This program is deliberately outside the training path.  It treats both run
directories as immutable inputs, authenticates their final checkpoints, and
publishes a new analysis report without modifying either run.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import random
import secrets
import stat
import statistics
from collections.abc import Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPORT_SCHEMA = "duovla-libero-cache-fused-ab-analysis-v1"
CRITERIA_SCHEMA = "duovla-libero-cache-fused-ab-source-freeze-v1"
EXPECTED_UPDATES = 100
EXPECTED_BASELINE_BACKEND = "sample_isolated_grouped_mm_v1"
EXPECTED_CANDIDATE_BACKEND = "sample_isolated_grouped_mm_v2"
EXPECTED_BASELINE_CACHE_FILES = 128
EXPECTED_CANDIDATE_CACHE_FILES = 377
EXPECTED_BASELINE_CONFIG = "configs/libero_single_gpu_ab_v1_b64.toml"
EXPECTED_CANDIDATE_CONFIG = "configs/libero_single_gpu_fused_v2_b64.toml"
EXPECTED_BASELINE_PROFILE = "duovla-single-gpu-tp1-sequential-v1-train-b64-serve-b8-v1"
EXPECTED_CANDIDATE_PROFILE = "duovla-single-gpu-tp1-fused-v2-train-b64-serve-b8-v1"
DEFAULT_BOOTSTRAP_SEED = 20260904
DEFAULT_BOOTSTRAP_REPLICATES = 20_000
RELATIVE_DENOMINATOR_FLOOR = 1e-12


class AnalysisError(RuntimeError):
    """Raised when an input cannot support a trustworthy A/B decision."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AnalysisError(message)


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("ascii")


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


@dataclass(frozen=True, slots=True)
class FileSnapshot:
    path: Path
    raw: bytes
    sha256: str
    size: int
    device: int
    inode: int
    mtime_ns: int

    def identity(self) -> dict[str, Any]:
        return {
            "bytes": self.size,
            "path": str(self.path),
            "sha256": self.sha256,
        }


def _read_stable_file(path: Path, *, name: str) -> FileSnapshot:
    path = path.absolute()
    try:
        before_path = path.lstat()
    except OSError as exc:
        raise AnalysisError(f"cannot stat {name}: {path}: {exc}") from exc
    require(not stat.S_ISLNK(before_path.st_mode), f"{name} must not be a symlink: {path}")
    require(stat.S_ISREG(before_path.st_mode), f"{name} is not a regular file: {path}")
    require(before_path.st_nlink == 1, f"{name} must have exactly one hard link: {path}")

    flags = os.O_RDONLY | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise AnalysisError(f"cannot open {name}: {path}: {exc}") from exc
    try:
        before = os.fstat(descriptor)
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    raw = b"".join(chunks)
    identity_before = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_nlink)
    identity_after = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_nlink)
    require(identity_before == identity_after, f"{name} changed while it was being read: {path}")
    require(before.st_nlink == 1, f"{name} hard-link count changed while reading: {path}")
    require(len(raw) == before.st_size, f"{name} byte count changed while reading: {path}")
    return FileSnapshot(
        path=path.resolve(strict=True),
        raw=raw,
        sha256=_sha256(raw),
        size=len(raw),
        device=before.st_dev,
        inode=before.st_ino,
        mtime_ns=before.st_mtime_ns,
    )


def _assert_snapshot_unchanged(snapshot: FileSnapshot, *, name: str) -> None:
    observed = _read_stable_file(snapshot.path, name=name)
    require(
        (
            observed.device,
            observed.inode,
            observed.size,
            observed.mtime_ns,
            observed.sha256,
        )
        == (
            snapshot.device,
            snapshot.inode,
            snapshot.size,
            snapshot.mtime_ns,
            snapshot.sha256,
        ),
        f"{name} changed during analysis: {snapshot.path}",
    )


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise AnalysisError(f"JSON object contains duplicate key {key!r}")
        result[key] = value
    return result


def _reject_nonfinite_constant(value: str) -> None:
    raise AnalysisError(f"JSON contains forbidden non-finite number {value}")


def _decode_json(raw: bytes, *, name: str) -> Any:
    try:
        text = raw.decode("utf-8", errors="strict")
        return json.loads(
            text,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonfinite_constant,
        )
    except AnalysisError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AnalysisError(f"cannot decode {name} as strict JSON: {exc}") from exc


def _read_json(path: Path, *, name: str) -> tuple[dict[str, Any], FileSnapshot]:
    snapshot = _read_stable_file(path, name=name)
    value = _decode_json(snapshot.raw, name=name)
    require(isinstance(value, dict), f"{name} must contain one JSON object")
    return value, snapshot


def _mapping(value: Any, *, name: str) -> dict[str, Any]:
    require(isinstance(value, dict), f"{name} must be an object")
    return value


def _integer(value: Any, *, name: str, minimum: int | None = None) -> int:
    require(isinstance(value, int) and not isinstance(value, bool), f"{name} must be an integer")
    if minimum is not None:
        require(value >= minimum, f"{name} must be at least {minimum}")
    return value


def _number(value: Any, *, name: str, minimum: float | None = None, strict_minimum: bool = False) -> float:
    require(isinstance(value, (int, float)) and not isinstance(value, bool), f"{name} must be numeric")
    result = float(value)
    require(math.isfinite(result), f"{name} must be finite")
    if minimum is not None:
        if strict_minimum:
            require(result > minimum, f"{name} must be greater than {minimum}")
        else:
            require(result >= minimum, f"{name} must be at least {minimum}")
    return result


def _string(value: Any, *, name: str) -> str:
    require(isinstance(value, str) and value, f"{name} must be a non-empty string")
    return value


def _at(value: Mapping[str, Any], *keys: str, name: str) -> Any:
    current: Any = value
    for key in keys:
        require(isinstance(current, dict) and key in current, f"{name} is missing {'.'.join(keys)}")
        current = current[key]
    return current


def _resolve_directory(path: Path, *, name: str) -> Path:
    try:
        before = path.lstat()
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise AnalysisError(f"cannot resolve {name}: {path}: {exc}") from exc
    require(not stat.S_ISLNK(before.st_mode), f"{name} must not be a symlink: {path}")
    require(resolved.is_dir(), f"{name} is not a directory: {resolved}")
    return resolved


def _load_metrics(path: Path, *, arm: str) -> tuple[list[dict[str, Any]], FileSnapshot]:
    snapshot = _read_stable_file(path, name=f"{arm} metrics")
    require(snapshot.raw.endswith(b"\n"), f"{arm} metrics must end with a newline")
    lines = snapshot.raw.splitlines()
    require(len(lines) == EXPECTED_UPDATES, f"{arm} metrics must contain exactly {EXPECTED_UPDATES} records")
    records: list[dict[str, Any]] = []
    for line_number, raw_line in enumerate(lines, start=1):
        require(bool(raw_line), f"{arm} metrics line {line_number} is empty")
        value = _decode_json(raw_line, name=f"{arm} metrics line {line_number}")
        require(isinstance(value, dict), f"{arm} metrics line {line_number} must be an object")
        records.append(value)
    return records, snapshot


def percentile(values: Sequence[float], q: float) -> float:
    require(bool(values), "cannot compute a percentile of an empty sequence")
    require(0.0 <= q <= 100.0, "percentile must be in [0, 100]")
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * q / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _summary(values: Sequence[float]) -> dict[str, float]:
    require(bool(values), "cannot summarize an empty sequence")
    return {
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "p95": percentile(values, 95.0),
    }


def _aggregate_speedup(baseline: Mapping[str, float], candidate: Mapping[str, float]) -> dict[str, float]:
    result: dict[str, float] = {}
    for statistic_name in ("mean", "median", "p95"):
        baseline_value = baseline[statistic_name]
        candidate_value = candidate[statistic_name]
        require(baseline_value > 0.0 and candidate_value > 0.0, "update timings must be positive")
        result[statistic_name] = 1.0 - candidate_value / baseline_value
    return result


def block_bootstrap_speedup_ci(
    baseline: Sequence[float],
    candidate: Sequence[float],
    *,
    block_size: int,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    require(len(baseline) == len(candidate), "bootstrap arms must have equal lengths")
    require(block_size > 0 and len(baseline) % block_size == 0, "bootstrap window must divide into full blocks")
    require(replicates >= 1, "bootstrap replicates must be positive")
    block_count = len(baseline) // block_size
    baseline_sums = [sum(baseline[start : start + block_size]) for start in range(0, len(baseline), block_size)]
    candidate_sums = [sum(candidate[start : start + block_size]) for start in range(0, len(candidate), block_size)]
    generator = random.Random(seed)
    draws: list[float] = []
    for _ in range(replicates):
        baseline_sum = 0.0
        candidate_sum = 0.0
        for _ in range(block_count):
            block = generator.randrange(block_count)
            baseline_sum += baseline_sums[block]
            candidate_sum += candidate_sums[block]
        require(baseline_sum > 0.0, "bootstrap sampled a non-positive baseline duration")
        draws.append(1.0 - candidate_sum / baseline_sum)
    return {
        "block_count": block_count,
        "block_size_updates": block_size,
        "confidence_level": 0.95,
        "lower": percentile(draws, 2.5),
        "replicates": replicates,
        "seed": seed,
        "statistic": "1 - candidate_mean_update_seconds / baseline_mean_update_seconds",
        "upper": percentile(draws, 97.5),
    }


def _normalize_arm_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Erase only the differences predeclared by the frozen A/B design."""

    normalized = copy.deepcopy(dict(config))
    _mapping(normalized["model"], name="config.model")["expert_batch_isolation"] = "<arm>"
    normalized["execution_profile"] = "<arm>"
    geometry = _mapping(normalized["execution_geometry"], name="config.execution_geometry")
    geometry["execution_profile"] = "<arm>"
    geometry["expert_batch_isolation"] = "<arm>"
    geometry.pop("shared_weight_kernel_sha256", None)
    _mapping(normalized["run"], name="config.run")["max_cached_files"] = "<arm>"
    environment = _mapping(normalized["execution_environment"], name="config.execution_environment")
    environment["gpu_uuids"] = "<arm>"
    authenticated = _mapping(environment["authenticated_runtime"], name="authenticated_runtime")
    _mapping(authenticated["environment"], name="authenticated_runtime.environment")["CUDA_VISIBLE_DEVICES"] = (
        "<arm>"
    )
    authenticated["static_environment_sha256"] = "<arm>"
    return normalized


def _validate_criteria(criteria: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    require(criteria.get("schema") == CRITERIA_SCHEMA, "frozen criteria schema mismatch")
    require(criteria.get("status") == "frozen", "criteria document is not frozen")
    arms = _mapping(criteria.get("arms"), name="criteria.arms")
    baseline_arm = _mapping(arms.get("baseline"), name="criteria.arms.baseline")
    candidate_arm = _mapping(arms.get("candidate"), name="criteria.arms.candidate")
    require(
        baseline_arm.get("backend") == EXPECTED_BASELINE_BACKEND,
        "criteria baseline backend is not sample-isolated v1",
    )
    require(
        candidate_arm.get("backend") == EXPECTED_CANDIDATE_BACKEND,
        "criteria candidate backend is not sample-isolated v2",
    )
    require(
        baseline_arm.get("max_cached_files") == EXPECTED_BASELINE_CACHE_FILES,
        "criteria baseline cache capacity is not 128",
    )
    require(
        candidate_arm.get("max_cached_files") == EXPECTED_CANDIDATE_CACHE_FILES,
        "criteria candidate cache capacity is not 377",
    )
    require(baseline_arm.get("config") == EXPECTED_BASELINE_CONFIG, "criteria baseline config path mismatch")
    require(candidate_arm.get("config") == EXPECTED_CANDIDATE_CONFIG, "criteria candidate config path mismatch")
    execution = _mapping(criteria.get("execution"), name="criteria.execution")
    require(execution.get("stop_after_updates") == EXPECTED_UPDATES, "criteria must declare a 100-update A/B run")
    global_batch = _integer(execution.get("global_batch_size"), name="criteria global batch", minimum=1)
    canonical_batch = _integer(
        execution.get("canonical_stream_batch_size"), name="criteria canonical stream batch", minimum=1
    )
    stream_plans = _integer(
        execution.get("canonical_stream_plans_per_update"), name="criteria stream plans", minimum=1
    )
    require(canonical_batch * stream_plans == global_batch, "criteria canonical stream does not form the global batch")
    _integer(execution.get("physical_batch_size"), name="criteria physical batch", minimum=1)
    _integer(execution.get("serving_batch_size"), name="criteria serving batch", minimum=1)
    _integer(execution.get("total_updates"), name="criteria total updates", minimum=EXPECTED_UPDATES)
    _integer(execution.get("seed"), name="criteria seed", minimum=0)

    acceptance = _mapping(criteria.get("acceptance"), name="criteria.acceptance")
    required_acceptance = {
        "candidate_peak_memory_gib_max",
        "finite_loss_and_gradient_every_update",
        "gradient_norm_ratio_interval",
        "gradient_norm_ratio_min_fraction",
        "mean_speedup_updates_51_100_min",
        "median_relative_loss_difference_max",
        "p95_candidate_update_seconds_below_baseline",
        "tail_mean_relative_loss_difference_max",
        "tail_updates",
    }
    require(required_acceptance <= acceptance.keys(), "criteria acceptance contract is incomplete")
    _number(acceptance["candidate_peak_memory_gib_max"], name="candidate peak limit", minimum=0.0)
    require(
        acceptance["finite_loss_and_gradient_every_update"] is True,
        "criteria must require finite loss and gradient on every update",
    )
    interval = acceptance["gradient_norm_ratio_interval"]
    require(isinstance(interval, list) and len(interval) == 2, "gradient ratio interval must have two bounds")
    lower = _number(interval[0], name="gradient ratio lower", minimum=0.0)
    upper = _number(interval[1], name="gradient ratio upper", minimum=0.0)
    require(lower <= upper, "gradient ratio interval is reversed")
    for key in (
        "gradient_norm_ratio_min_fraction",
        "mean_speedup_updates_51_100_min",
        "median_relative_loss_difference_max",
        "tail_mean_relative_loss_difference_max",
    ):
        _number(acceptance[key], name=f"criteria {key}")
    require(
        acceptance["p95_candidate_update_seconds_below_baseline"] is True,
        "criteria must require candidate p95 below baseline p95",
    )
    tail_updates = acceptance["tail_updates"]
    require(
        isinstance(tail_updates, list)
        and len(tail_updates) == 2
        and all(isinstance(value, int) and not isinstance(value, bool) for value in tail_updates),
        "tail update window must contain two integers",
    )
    require(1 <= tail_updates[0] <= tail_updates[1] <= EXPECTED_UPDATES, "tail update window is invalid")
    _string(criteria.get("source_tree_sha256"), name="criteria source-tree SHA-256")
    _string(criteria.get("shared_weight_kernel_sha256"), name="criteria shared-weight kernel SHA-256")
    return baseline_arm, candidate_arm


def _validate_resolved_config(
    value: Mapping[str, Any],
    *,
    arm: str,
    arm_contract: Mapping[str, Any],
    criteria: Mapping[str, Any],
) -> tuple[dict[str, Any], str]:
    require(set(value) == {"config", "config_sha256"}, f"{arm} resolved config envelope has unexpected keys")
    config = _mapping(value.get("config"), name=f"{arm} config")
    config_sha256 = _string(value.get("config_sha256"), name=f"{arm} config SHA-256")
    calculated = hashlib.sha256(
        json.dumps(config, allow_nan=False, ensure_ascii=True, separators=(",", ":"), sort_keys=True).encode("ascii")
    ).hexdigest()
    require(calculated == config_sha256, f"{arm} resolved config SHA-256 mismatch")

    expected_backend = EXPECTED_BASELINE_BACKEND if arm == "baseline" else EXPECTED_CANDIDATE_BACKEND
    expected_cache = EXPECTED_BASELINE_CACHE_FILES if arm == "baseline" else EXPECTED_CANDIDATE_CACHE_FILES
    expected_profile = EXPECTED_BASELINE_PROFILE if arm == "baseline" else EXPECTED_CANDIDATE_PROFILE
    execution = _mapping(criteria["execution"], name="criteria.execution")
    require(config.get("source_tree_sha256") == criteria["source_tree_sha256"], f"{arm} source tree is not frozen")
    require(_at(config, "run", "seed", name=f"{arm} config") == execution["seed"], f"{arm} seed mismatch")
    require(_at(config, "run", "max_cached_files", name=f"{arm} config") == expected_cache, f"{arm} cache mismatch")
    require(
        _at(config, "model", "expert_batch_isolation", name=f"{arm} config") == expected_backend,
        f"{arm} model backend mismatch",
    )
    require(
        _at(config, "execution_geometry", "expert_batch_isolation", name=f"{arm} config") == expected_backend,
        f"{arm} execution backend mismatch",
    )
    for key in ("global_batch_size", "physical_batch_size", "serving_batch_size", "total_updates"):
        require(
            _at(config, "optimization", key, name=f"{arm} config") == execution[key],
            f"{arm} optimization.{key} differs from frozen execution",
        )
    require(
        _at(config, "optimization", "microbatch_size", name=f"{arm} config") == execution["physical_batch_size"],
        f"{arm} microbatch is not the physical batch",
    )
    require(
        _at(config, "optimization", "gradient_accumulation_steps", name=f"{arm} config") == 1,
        f"{arm} must use one physical B=64 step per update",
    )
    require(
        _at(config, "execution_geometry", "physical_batch_size", name=f"{arm} config")
        == execution["physical_batch_size"],
        f"{arm} execution physical batch mismatch",
    )
    require(
        _at(config, "execution_geometry", "serving_batch_size", name=f"{arm} config")
        == execution["serving_batch_size"],
        f"{arm} execution serving batch mismatch",
    )
    require(
        _at(config, "policy", "objective", name=f"{arm} config") == "rectified_flow",
        f"{arm} objective is not rectified flow",
    )
    require(
        config.get("execution_profile") == expected_profile,
        f"{arm} execution profile differs from the frozen v1/v2 B=64 design",
    )
    require(
        _at(config, "execution_geometry", "execution_profile", name=f"{arm} config") == expected_profile,
        f"{arm} execution-geometry profile differs from the frozen v1/v2 B=64 design",
    )
    gpu_uuids = _at(config, "execution_environment", "gpu_uuids", name=f"{arm} config")
    require(
        isinstance(gpu_uuids, list) and gpu_uuids == [arm_contract.get("gpu_uuid")],
        f"{arm} GPU UUID differs from the frozen arm",
    )
    visible_device = _at(
        config,
        "execution_environment",
        "authenticated_runtime",
        "environment",
        "CUDA_VISIBLE_DEVICES",
        name=f"{arm} config",
    )
    require(visible_device == str(arm_contract.get("gpu_index")), f"{arm} physical GPU index mismatch")
    if arm == "candidate":
        expected_kernel = criteria["shared_weight_kernel_sha256"]
        require(
            _at(config, "execution_geometry", "shared_weight_kernel_sha256", name="candidate config")
            == expected_kernel,
            "candidate shared-weight kernel identity mismatch",
        )
    else:
        require(
            "shared_weight_kernel_sha256" not in _mapping(config["execution_geometry"], name="baseline geometry"),
            "baseline unexpectedly declares a shared-weight kernel",
        )
    return config, config_sha256


def _validate_metric_pair(
    baseline: Sequence[Mapping[str, Any]],
    candidate: Sequence[Mapping[str, Any]],
    *,
    global_batch_size: int,
    objective: str,
) -> None:
    require(len(baseline) == len(candidate) == EXPECTED_UPDATES, "metrics are not exactly 100 aligned updates")
    for expected_update, (base, cand) in enumerate(zip(baseline, candidate, strict=True), start=1):
        for arm, record in (("baseline", base), ("candidate", cand)):
            update = _integer(record.get("update"), name=f"{arm} update {expected_update}.update", minimum=1)
            require(update == expected_update, f"{arm} metrics are not contiguous at update {expected_update}")
            examples = _integer(
                record.get("examples_seen"), name=f"{arm} update {expected_update}.examples_seen", minimum=1
            )
            require(
                examples == expected_update * global_batch_size,
                f"{arm} examples_seen mismatch at update {expected_update}",
            )
            require(record.get("objective") == objective, f"{arm} objective mismatch at update {expected_update}")
            _number(record.get("train_loss"), name=f"{arm} loss at update {expected_update}", minimum=0.0)
            _number(record.get("gradient_norm"), name=f"{arm} gradient norm at update {expected_update}", minimum=0.0)
            _number(
                record.get("update_seconds"),
                name=f"{arm} update seconds at update {expected_update}",
                minimum=0.0,
                strict_minimum=True,
            )
            _number(
                record.get("interface_learning_rate"),
                name=f"{arm} interface LR at update {expected_update}",
                minimum=0.0,
            )
            _number(
                record.get("lora_learning_rate"),
                name=f"{arm} LoRA LR at update {expected_update}",
                minimum=0.0,
            )
        require(
            base["examples_seen"] == cand["examples_seen"],
            f"examples_seen is not aligned at update {expected_update}",
        )
        for key in ("interface_learning_rate", "lora_learning_rate"):
            require(base[key] == cand[key], f"{key} is not aligned at update {expected_update}")


def _require_relative_path(root: Path, value: Any, *, name: str) -> tuple[Path, Path]:
    relative = Path(_string(value, name=name))
    require(not relative.is_absolute() and ".." not in relative.parts, f"{name} must be a contained relative path")
    candidate = root.joinpath(relative)
    current = root
    for part in relative.parts:
        current = current / part
        try:
            mode = current.lstat().st_mode
        except OSError as exc:
            raise AnalysisError(f"cannot stat {name} component {current}: {exc}") from exc
        require(not stat.S_ISLNK(mode), f"{name} traverses a symlink: {current}")
    resolved = candidate.resolve(strict=True)
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise AnalysisError(f"{name} escapes its run/checkpoint directory") from exc
    return relative, resolved


def _authenticate_final_manifest(
    run_dir: Path,
    *,
    arm: str,
    config: Mapping[str, Any],
    config_snapshot: FileSnapshot,
    config_sha256: str,
    metrics: Sequence[Mapping[str, Any]],
    criteria: Mapping[str, Any],
) -> tuple[dict[str, Any], list[FileSnapshot], dict[str, Any]]:
    journal, journal_snapshot = _read_json(run_dir / "run_journal.json", name=f"{arm} run journal")
    require(journal.get("schema") == "duo-vla-run-journal-v1", f"{arm} run journal schema mismatch")
    require(journal.get("config_sha256") == config_sha256, f"{arm} journal config identity mismatch")
    run_uuid = _string(journal.get("run_uuid"), name=f"{arm} run UUID")
    latest = _mapping(journal.get("latest_checkpoint"), name=f"{arm} latest checkpoint")
    require(latest.get("update") == EXPECTED_UPDATES, f"{arm} final journal is not at update 100")
    journal_metric = _mapping(latest.get("last_metrics"), name=f"{arm} journal last metrics")
    for key in (
        "examples_seen",
        "gradient_norm",
        "interface_learning_rate",
        "lora_learning_rate",
        "objective",
        "train_loss",
        "update",
        "update_seconds",
    ):
        require(journal_metric.get(key) == metrics[-1].get(key), f"{arm} journal metric {key} mismatch")
    relative_checkpoint, checkpoint_dir = _require_relative_path(
        run_dir, latest.get("relative_path"), name=f"{arm} latest checkpoint path"
    )
    require(checkpoint_dir.is_dir(), f"{arm} latest checkpoint is not a directory")
    manifest, manifest_snapshot = _read_json(checkpoint_dir / "manifest.json", name=f"{arm} final manifest")
    require(latest.get("manifest_sha256") == manifest_snapshot.sha256, f"{arm} journal manifest hash mismatch")
    require(manifest.get("schema") == "duo-vla-checkpoint-v1", f"{arm} checkpoint schema mismatch")
    require(manifest.get("config_sha256") == config_sha256, f"{arm} manifest config identity mismatch")
    require(manifest.get("run_uuid") == run_uuid, f"{arm} manifest run UUID mismatch")
    require(manifest.get("source_tree_sha256") == criteria["source_tree_sha256"], f"{arm} manifest source mismatch")
    require(manifest.get("run_seed") == criteria["execution"]["seed"], f"{arm} manifest seed mismatch")
    state = _mapping(manifest.get("trainer_state"), name=f"{arm} trainer state")
    require(state.get("next_update") == EXPECTED_UPDATES, f"{arm} trainer state is not at update 100")
    require(
        state.get("examples_seen") == EXPECTED_UPDATES * criteria["execution"]["global_batch_size"],
        f"{arm} manifest example count mismatch",
    )
    expected_backend = EXPECTED_BASELINE_BACKEND if arm == "baseline" else EXPECTED_CANDIDATE_BACKEND
    require(manifest.get("expert_batch_isolation") == expected_backend, f"{arm} manifest backend mismatch")
    require(
        manifest.get("physical_batch_size") == criteria["execution"]["physical_batch_size"],
        f"{arm} manifest physical batch mismatch",
    )
    require(
        manifest.get("serving_batch_size") == criteria["execution"]["serving_batch_size"],
        f"{arm} manifest serving batch mismatch",
    )
    require(manifest.get("execution_profile") == config["execution_profile"], f"{arm} manifest profile mismatch")
    if arm == "candidate":
        require(
            manifest.get("shared_weight_kernel_sha256") == criteria["shared_weight_kernel_sha256"],
            "candidate manifest kernel identity mismatch",
        )
    else:
        require("shared_weight_kernel_sha256" not in manifest, "baseline manifest declares a fused kernel")
    final_metric = _mapping(manifest.get("last_metrics"), name=f"{arm} manifest last metrics")
    for key in (
        "examples_seen",
        "gradient_norm",
        "interface_learning_rate",
        "lora_learning_rate",
        "objective",
        "train_loss",
        "update",
        "update_seconds",
    ):
        require(final_metric.get(key) == metrics[-1].get(key), f"{arm} manifest metric {key} mismatch")

    artifacts = _mapping(manifest.get("artifacts"), name=f"{arm} manifest artifacts")
    require(bool(artifacts), f"{arm} final manifest has no artifacts")
    artifact_snapshots: list[FileSnapshot] = []
    artifact_report: dict[str, Any] = {}
    for artifact_name in sorted(artifacts):
        descriptor = _mapping(artifacts[artifact_name], name=f"{arm} artifact {artifact_name}")
        _, artifact_path = _require_relative_path(
            checkpoint_dir,
            descriptor.get("path"),
            name=f"{arm} artifact {artifact_name} path",
        )
        snapshot = _read_stable_file(artifact_path, name=f"{arm} artifact {artifact_name}")
        require(descriptor.get("bytes") == snapshot.size, f"{arm} artifact {artifact_name} byte count mismatch")
        require(descriptor.get("sha256") == snapshot.sha256, f"{arm} artifact {artifact_name} SHA-256 mismatch")
        artifact_snapshots.append(snapshot)
        artifact_report[artifact_name] = {
            "bytes": snapshot.size,
            "path": str(artifact_path),
            "sha256": snapshot.sha256,
        }
    resolved_artifact = artifact_report.get("resolved_config")
    require(resolved_artifact is not None, f"{arm} checkpoint does not bind the resolved config")
    require(
        resolved_artifact["sha256"] == config_snapshot.sha256
        and resolved_artifact["bytes"] == config_snapshot.size,
        f"{arm} checkpoint resolved config differs from the run root",
    )
    report = {
        "artifact_count": len(artifact_report),
        "artifact_total_bytes": sum(value["bytes"] for value in artifact_report.values()),
        "artifacts": artifact_report,
        "bytes": manifest_snapshot.size,
        "checkpoint_relative_path": relative_checkpoint.as_posix(),
        "execution_environment_sha256": manifest.get("execution_environment_sha256"),
        "manifest_path": str(manifest_snapshot.path),
        "manifest_sha256": manifest_snapshot.sha256,
        "run_uuid": run_uuid,
        "update": EXPECTED_UPDATES,
    }
    return report, [journal_snapshot, manifest_snapshot, *artifact_snapshots], manifest


def _common_manifest_contract(baseline: Mapping[str, Any], candidate: Mapping[str, Any]) -> None:
    comparable_keys = (
        "dataset_content_inventory_sha256",
        "dataset_files_verified",
        "dataset_id",
        "dataset_revision",
        "dataset_total_bytes",
        "dataset_tree_sha256",
        "experts_implementation",
        "fixed_physical_prefix_width",
        "model_content_inventory_sha256",
        "model_files_verified",
        "model_id",
        "model_revision",
        "model_total_bytes",
        "model_tree_sha256",
        "normalization_sha256",
        "optimizer_parameter_schema_sha256",
        "policy_contract",
        "policy_contract_sha256",
        "prefix_geometry_content_sha256",
        "task",
        "tensor_parallel_size",
        "train_episode_count",
        "validation_episode_count",
    )
    for key in comparable_keys:
        require(baseline.get(key) == candidate.get(key), f"final manifests disagree on common field {key}")


def _window_report(
    baseline_records: Sequence[Mapping[str, Any]],
    candidate_records: Sequence[Mapping[str, Any]],
    *,
    start: int,
    end: int,
    bootstrap_replicates: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    base_seconds = [float(record["update_seconds"]) for record in baseline_records[start - 1 : end]]
    candidate_seconds = [float(record["update_seconds"]) for record in candidate_records[start - 1 : end]]
    baseline_summary = _summary(base_seconds)
    candidate_summary = _summary(candidate_seconds)
    paired_speedups = [
        1.0 - candidate / baseline
        for baseline, candidate in zip(base_seconds, candidate_seconds, strict=True)
    ]
    effective_seed = bootstrap_seed ^ (start << 16) ^ end
    return {
        "baseline_update_seconds": baseline_summary,
        "block_bootstrap_speedup_fraction_95_ci": block_bootstrap_speedup_ci(
            base_seconds,
            candidate_seconds,
            block_size=5,
            replicates=bootstrap_replicates,
            seed=effective_seed,
        ),
        "candidate_update_seconds": candidate_summary,
        "count": len(base_seconds),
        "paired_update_speedup_fraction": _summary(paired_speedups),
        "speedup_fraction": _aggregate_speedup(baseline_summary, candidate_summary),
        "updates": [start, end],
    }


def _peak_memory(
    run_dir: Path,
    *,
    arm: str,
    override: float | None,
) -> tuple[dict[str, Any], list[FileSnapshot]]:
    if override is not None:
        return {
            "available": True,
            "peak_memory_gib": _number(override, name=f"{arm} peak-memory override", minimum=0.0),
            "source": "command_line",
        }, []
    discovered: list[tuple[float, FileSnapshot]] = []
    for filename in ("completion.json", "run_completion.json", "summary.json"):
        path = run_dir / filename
        if not path.exists():
            continue
        value, snapshot = _read_json(path, name=f"{arm} {filename}")
        if "peak_memory_gib" not in value:
            continue
        peak = _number(value["peak_memory_gib"], name=f"{arm} {filename} peak memory", minimum=0.0)
        if "final_update" in value:
            require(value["final_update"] == EXPECTED_UPDATES, f"{arm} {filename} is not the 100-update completion")
        if "output_dir" in value:
            require(Path(value["output_dir"]).resolve(strict=True) == run_dir, f"{arm} {filename} run path mismatch")
        discovered.append((peak, snapshot))
    if not discovered:
        return {
            "available": False,
            "peak_memory_gib": None,
            "source": None,
        }, []
    require(len({peak for peak, _ in discovered}) == 1, f"{arm} completion files disagree on peak memory")
    peak, snapshot = discovered[0]
    return {
        "available": True,
        "completion_file": snapshot.identity(),
        "peak_memory_gib": peak,
        "source": "run_completion_file",
    }, [entry[1] for entry in discovered]


def analyze(
    baseline_run_dir: Path,
    candidate_run_dir: Path,
    criteria_json: Path,
    *,
    baseline_peak_memory_gib: float | None = None,
    candidate_peak_memory_gib: float | None = None,
    bootstrap_replicates: int = DEFAULT_BOOTSTRAP_REPLICATES,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> dict[str, Any]:
    baseline_dir = _resolve_directory(baseline_run_dir, name="baseline run directory")
    candidate_dir = _resolve_directory(candidate_run_dir, name="candidate run directory")
    require(baseline_dir != candidate_dir, "baseline and candidate run directories must be distinct")
    criteria, criteria_snapshot = _read_json(criteria_json, name="frozen criteria")
    baseline_arm, candidate_arm = _validate_criteria(criteria)
    require(
        Path(_string(baseline_arm.get("output_dir"), name="criteria baseline output")).resolve(strict=True)
        == baseline_dir,
        "baseline run directory differs from the frozen arm",
    )
    require(
        Path(_string(candidate_arm.get("output_dir"), name="criteria candidate output")).resolve(strict=True)
        == candidate_dir,
        "candidate run directory differs from the frozen arm",
    )

    baseline_resolved, baseline_config_snapshot = _read_json(
        baseline_dir / "resolved_config.json", name="baseline resolved config"
    )
    candidate_resolved, candidate_config_snapshot = _read_json(
        candidate_dir / "resolved_config.json", name="candidate resolved config"
    )
    baseline_config, baseline_config_sha256 = _validate_resolved_config(
        baseline_resolved,
        arm="baseline",
        arm_contract=baseline_arm,
        criteria=criteria,
    )
    candidate_config, candidate_config_sha256 = _validate_resolved_config(
        candidate_resolved,
        arm="candidate",
        arm_contract=candidate_arm,
        criteria=criteria,
    )
    require(
        _normalize_arm_config(baseline_config) == _normalize_arm_config(candidate_config),
        "resolved configs differ outside the predeclared backend/cache/profile/GPU fields",
    )

    baseline_metrics, baseline_metrics_snapshot = _load_metrics(
        baseline_dir / "metrics.jsonl", arm="baseline"
    )
    candidate_metrics, candidate_metrics_snapshot = _load_metrics(
        candidate_dir / "metrics.jsonl", arm="candidate"
    )
    objective = _string(
        _at(baseline_config, "policy", "objective", name="baseline config"),
        name="A/B objective",
    )
    _validate_metric_pair(
        baseline_metrics,
        candidate_metrics,
        global_batch_size=criteria["execution"]["global_batch_size"],
        objective=objective,
    )

    baseline_manifest, baseline_manifest_snapshots, baseline_manifest_value = _authenticate_final_manifest(
        baseline_dir,
        arm="baseline",
        config=baseline_config,
        config_snapshot=baseline_config_snapshot,
        config_sha256=baseline_config_sha256,
        metrics=baseline_metrics,
        criteria=criteria,
    )
    candidate_manifest, candidate_manifest_snapshots, candidate_manifest_value = _authenticate_final_manifest(
        candidate_dir,
        arm="candidate",
        config=candidate_config,
        config_snapshot=candidate_config_snapshot,
        config_sha256=candidate_config_sha256,
        metrics=candidate_metrics,
        criteria=criteria,
    )
    _common_manifest_contract(baseline_manifest_value, candidate_manifest_value)

    all_window = _window_report(
        baseline_metrics,
        candidate_metrics,
        start=1,
        end=EXPECTED_UPDATES,
        bootstrap_replicates=bootstrap_replicates,
        bootstrap_seed=bootstrap_seed,
    )
    warm_window = _window_report(
        baseline_metrics,
        candidate_metrics,
        start=51,
        end=100,
        bootstrap_replicates=bootstrap_replicates,
        bootstrap_seed=bootstrap_seed,
    )
    relative_loss_differences = [
        abs(float(candidate["train_loss"]) - float(baseline["train_loss"]))
        / max(abs(float(baseline["train_loss"])), RELATIVE_DENOMINATOR_FLOOR)
        for baseline, candidate in zip(baseline_metrics, candidate_metrics, strict=True)
    ]
    gradient_ratios = [
        float(candidate["gradient_norm"])
        / max(abs(float(baseline["gradient_norm"])), RELATIVE_DENOMINATOR_FLOOR)
        for baseline, candidate in zip(baseline_metrics, candidate_metrics, strict=True)
    ]
    acceptance = criteria["acceptance"]
    ratio_lower, ratio_upper = (float(value) for value in acceptance["gradient_norm_ratio_interval"])
    gradient_passes = [ratio_lower <= ratio <= ratio_upper for ratio in gradient_ratios]
    tail_start, tail_end = acceptance["tail_updates"]
    loss_report = {
        "all_updates": _summary(relative_loss_differences),
        "definition": "abs(candidate_loss - baseline_loss) / max(abs(baseline_loss), 1e-12)",
        "tail_mean": statistics.fmean(relative_loss_differences[tail_start - 1 : tail_end]),
        "tail_updates": [tail_start, tail_end],
    }
    gradient_report = {
        "all_updates": _summary(gradient_ratios),
        "definition": "candidate_gradient_norm / max(abs(baseline_gradient_norm), 1e-12)",
        "pass_count": sum(gradient_passes),
        "pass_fraction": statistics.fmean(gradient_passes),
        "required_interval_inclusive": [ratio_lower, ratio_upper],
    }
    baseline_peak, baseline_peak_snapshots = _peak_memory(
        baseline_dir,
        arm="baseline",
        override=baseline_peak_memory_gib,
    )
    candidate_peak, candidate_peak_snapshots = _peak_memory(
        candidate_dir,
        arm="candidate",
        override=candidate_peak_memory_gib,
    )

    checks = {
        "candidate_peak_memory_gib_max": candidate_peak["available"]
        and candidate_peak["peak_memory_gib"] <= float(acceptance["candidate_peak_memory_gib_max"]),
        "finite_loss_and_gradient_every_update": True,
        "gradient_norm_ratio_min_fraction": gradient_report["pass_fraction"]
        >= float(acceptance["gradient_norm_ratio_min_fraction"]),
        "mean_speedup_updates_51_100_min": warm_window["speedup_fraction"]["mean"]
        >= float(acceptance["mean_speedup_updates_51_100_min"]),
        "median_relative_loss_difference_max": loss_report["all_updates"]["median"]
        <= float(acceptance["median_relative_loss_difference_max"]),
        "p95_candidate_update_seconds_below_baseline": warm_window["candidate_update_seconds"]["p95"]
        < warm_window["baseline_update_seconds"]["p95"],
        "tail_mean_relative_loss_difference_max": loss_report["tail_mean"]
        <= float(acceptance["tail_mean_relative_loss_difference_max"]),
    }
    check_evidence = {
        "candidate_peak_memory_gib_max": {
            "limit": float(acceptance["candidate_peak_memory_gib_max"]),
            "observed": candidate_peak["peak_memory_gib"],
        },
        "finite_loss_and_gradient_every_update": {
            "finite_updates": EXPECTED_UPDATES,
            "required_updates": EXPECTED_UPDATES,
        },
        "gradient_norm_ratio_min_fraction": {
            "minimum": float(acceptance["gradient_norm_ratio_min_fraction"]),
            "observed": gradient_report["pass_fraction"],
        },
        "mean_speedup_updates_51_100_min": {
            "minimum": float(acceptance["mean_speedup_updates_51_100_min"]),
            "observed": warm_window["speedup_fraction"]["mean"],
        },
        "median_relative_loss_difference_max": {
            "limit": float(acceptance["median_relative_loss_difference_max"]),
            "observed": loss_report["all_updates"]["median"],
        },
        "p95_candidate_update_seconds_below_baseline": {
            "baseline_p95_seconds": warm_window["baseline_update_seconds"]["p95"],
            "candidate_p95_seconds": warm_window["candidate_update_seconds"]["p95"],
            "updates": [51, 100],
        },
        "tail_mean_relative_loss_difference_max": {
            "limit": float(acceptance["tail_mean_relative_loss_difference_max"]),
            "observed": loss_report["tail_mean"],
            "updates": [tail_start, tail_end],
        },
    }

    snapshots = [
        criteria_snapshot,
        baseline_config_snapshot,
        candidate_config_snapshot,
        baseline_metrics_snapshot,
        candidate_metrics_snapshot,
        *baseline_manifest_snapshots,
        *candidate_manifest_snapshots,
        *baseline_peak_snapshots,
        *candidate_peak_snapshots,
    ]
    for index, snapshot in enumerate(snapshots):
        _assert_snapshot_unchanged(snapshot, name=f"analysis input {index}")

    report: dict[str, Any] = {
        "acceptance": {
            "all_passed": all(checks.values()),
            "checks": checks,
            "criteria": acceptance,
            "evidence": check_evidence,
        },
        "bootstrap": {
            "base_seed": bootstrap_seed,
            "method": "paired non-overlapping 5-update block resampling with replacement",
            "replicates": bootstrap_replicates,
        },
        "criteria": criteria_snapshot.identity(),
        "gradient_norm_ratio": gradient_report,
        "geometry": {
            "canonical_stream_batch_size": criteria["execution"]["canonical_stream_batch_size"],
            "canonical_stream_plans_per_update": criteria["execution"]["canonical_stream_plans_per_update"],
            "global_batch_size": criteria["execution"]["global_batch_size"],
            "physical_batch_size": criteria["execution"]["physical_batch_size"],
            "serving_batch_size": criteria["execution"]["serving_batch_size"],
            "source_tree_sha256": criteria["source_tree_sha256"],
            "total_updates": criteria["execution"]["total_updates"],
        },
        "inputs": {
            "baseline": {
                "backend": EXPECTED_BASELINE_BACKEND,
                "config": baseline_config_snapshot.identity(),
                "config_sha256": baseline_config_sha256,
                "max_cached_files": EXPECTED_BASELINE_CACHE_FILES,
                "metrics": baseline_metrics_snapshot.identity(),
                "peak_memory": baseline_peak,
                "run_dir": str(baseline_dir),
            },
            "candidate": {
                "backend": EXPECTED_CANDIDATE_BACKEND,
                "config": candidate_config_snapshot.identity(),
                "config_sha256": candidate_config_sha256,
                "max_cached_files": EXPECTED_CANDIDATE_CACHE_FILES,
                "metrics": candidate_metrics_snapshot.identity(),
                "peak_memory": candidate_peak,
                "run_dir": str(candidate_dir),
                "shared_weight_kernel_sha256": criteria["shared_weight_kernel_sha256"],
            },
        },
        "loss_relative_difference": loss_report,
        "manifests": {
            "baseline": baseline_manifest,
            "candidate": candidate_manifest,
            "common_contract_equal": True,
        },
        "schema": REPORT_SCHEMA,
        "status": "pass" if all(checks.values()) else "rejected",
        "timing": {
            "updates_1_100": all_window,
            "updates_51_100": warm_window,
        },
        "validation": {
            "aligned_updates": EXPECTED_UPDATES,
            "examples_learning_rates_and_objective_aligned": True,
            "finite_metrics": True,
            "inputs_unchanged_during_analysis": True,
            "manifests_and_artifacts_authenticated": True,
            "resolved_configs_equal_outside_predeclared_arm_differences": True,
        },
    }
    unsigned = dict(report)
    report["report_sha256"] = _sha256(canonical_json_bytes(unsigned))
    return report


def write_canonical_json_atomic_exclusive(path: Path, value: Mapping[str, Any]) -> Path:
    """Atomically publish one report, refusing to replace an existing path."""

    parent = path.parent.resolve(strict=True)
    require(parent.is_dir(), f"report parent is not a directory: {parent}")
    filename = path.name
    require(filename not in {"", ".", ".."}, "report filename is invalid")
    directory_fd = os.open(parent, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
    temporary = f".{filename}.tmp-{secrets.token_hex(12)}"
    raw = canonical_json_bytes(value)
    temporary_created = False
    try:
        descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC,
            0o600,
            dir_fd=directory_fd,
        )
        temporary_created = True
        try:
            view = memoryview(raw)
            while view:
                written = os.write(descriptor, view)
                require(written > 0, "zero-byte write while publishing A/B report")
                view = view[written:]
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        try:
            os.link(
                temporary,
                filename,
                src_dir_fd=directory_fd,
                dst_dir_fd=directory_fd,
                follow_symlinks=False,
            )
        except FileExistsError as exc:
            raise AnalysisError(f"refusing to replace existing report: {parent / filename}") from exc
        os.unlink(temporary, dir_fd=directory_fd)
        temporary_created = False
        os.fsync(directory_fd)
    finally:
        if temporary_created:
            with suppress(FileNotFoundError):
                os.unlink(temporary, dir_fd=directory_fd)
        os.close(directory_fd)
    return parent / filename


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline_run_dir", type=Path)
    parser.add_argument("candidate_run_dir", type=Path)
    parser.add_argument("criteria_json", type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--baseline-peak-memory-gib", type=float)
    parser.add_argument("--candidate-peak-memory-gib", type=float)
    parser.add_argument("--bootstrap-replicates", type=int, default=DEFAULT_BOOTSTRAP_REPLICATES)
    parser.add_argument("--bootstrap-seed", type=int, default=DEFAULT_BOOTSTRAP_SEED)
    args = parser.parse_args(argv)
    if args.bootstrap_replicates <= 0:
        parser.error("--bootstrap-replicates must be positive")
    output = args.output_json.absolute()
    if output.is_relative_to(args.baseline_run_dir.absolute()) or output.is_relative_to(
        args.candidate_run_dir.absolute()
    ):
        parser.error("--output-json must be outside both read-only run directories")
    return args


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    report = analyze(
        args.baseline_run_dir,
        args.candidate_run_dir,
        args.criteria_json,
        baseline_peak_memory_gib=args.baseline_peak_memory_gib,
        candidate_peak_memory_gib=args.candidate_peak_memory_gib,
        bootstrap_replicates=args.bootstrap_replicates,
        bootstrap_seed=args.bootstrap_seed,
    )
    published = write_canonical_json_atomic_exclusive(args.output_json, report)
    print(canonical_json_bytes(report).decode("ascii"), end="")
    print(f"report_path={published}")


if __name__ == "__main__":
    main()
