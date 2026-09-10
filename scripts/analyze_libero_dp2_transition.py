#!/usr/bin/env python3
"""Authenticate and score the TP1-update-1000 to DP2 transition.

The one-step semantic comparison and warm throughput comparison are separate:
update 1001 compares the old B64 continuation with the DP2 fork; reference
updates 901..1000 and DP2 updates 1002..1100 are the sealed timing windows,
with a distinct update-1100 performance checkpoint anchoring memory evidence.
"""

# ruff: noqa: E402 -- authenticate the isolated checkout before project imports.

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import stat
import statistics
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SOURCE_ROOT = _PROJECT_ROOT / "src"
if _SOURCE_ROOT.resolve(strict=True) != _SOURCE_ROOT or not stat.S_ISDIR(os.lstat(_SOURCE_ROOT).st_mode):
    raise RuntimeError("project source root must be a canonical real directory")
sys.path.insert(0, str(_SOURCE_ROOT))

from duo_vla.checkpointing import load_checkpoint_manifest
from duo_vla.dp2_fork import dp2_semantic_recipe_sha256, validate_dp2_checkpoint_lineage
from duo_vla.run_config import load_verified_resolved_config

SCHEMA = "duovla-libero-dp2-transition-analysis-v2"
PARENT_UPDATE = 1000
COMPARISON_UPDATE = 1001
PERFORMANCE_UPDATE = 1100
REFERENCE_METRICS_START = 1
REFERENCE_METRICS_END = COMPARISON_UPDATE
CANDIDATE_METRICS_START = COMPARISON_UPDATE
CANDIDATE_METRICS_END = PERFORMANCE_UPDATE
BASELINE_TIMING_START = 901
BASELINE_TIMING_END = PARENT_UPDATE
CANDIDATE_TIMING_START = 1002
CANDIDATE_TIMING_END = PERFORMANCE_UPDATE
GLOBAL_BATCH_SIZE = 64
DP2_PROFILE = "duovla-dp2-tp1-fused-v2-train-b32-serve-b8-v1"
TP1_PROFILE = "duovla-single-gpu-tp1-fused-v2-train-b64-serve-b8-v1"
DP2_GPU_UUIDS = [
    "GPU-30424b03-3051-615a-832e-186511378a61",
    "GPU-84fa4004-92fb-8f86-cc65-01d62a27950e",
]
LOSS_RELATIVE_DIFFERENCE_MAX = 0.01
GRADIENT_NORM_RELATIVE_DIFFERENCE_MAX = 0.05
PARAMETER_DELTA_COSINE_MIN = 0.99
PARAMETER_DELTA_RELATIVE_L2_MAX = 0.15
MEDIAN_SPEEDUP_MIN = 0.02
PEAK_MEMORY_GIB_MAX = 75.0
_TENSOR_BLOCK_ELEMENTS = 1_000_000


class AnalysisError(RuntimeError):
    """An input cannot support the predeclared transition decision."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AnalysisError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _strict_json_line(raw: str, *, source: str) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise AnalysisError(f"non-finite JSON value in {source}: {value}")

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        output: dict[str, Any] = {}
        for key, value in pairs:
            require(key not in output, f"duplicate JSON key in {source}: {key}")
            output[key] = value
        return output

    try:
        value = json.loads(
            raw,
            object_pairs_hook=reject_duplicates,
            parse_constant=reject_constant,
        )
    except (json.JSONDecodeError, UnicodeError) as exc:
        raise AnalysisError(f"cannot parse strict JSON from {source}: {exc}") from exc
    require(isinstance(value, dict), f"{source} must contain one JSON object")
    return value


def _stable_regular_bytes(path: Path, *, name: str) -> tuple[Path, bytes]:
    canonical = path.resolve(strict=True)
    before = os.stat(canonical, follow_symlinks=False)
    require(stat.S_ISREG(before.st_mode), f"{name} must be a regular non-symlink file")
    require(before.st_nlink == 1, f"{name} must have exactly one link")
    descriptor = os.open(canonical, os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW | os.O_CLOEXEC)
    try:
        opened = os.fstat(descriptor)
        identity = (
            opened.st_dev,
            opened.st_ino,
            opened.st_mode,
            opened.st_size,
            opened.st_mtime_ns,
            opened.st_ctime_ns,
        )
        require(
            identity
            == (
                before.st_dev,
                before.st_ino,
                before.st_mode,
                before.st_size,
                before.st_mtime_ns,
                before.st_ctime_ns,
            ),
            f"{name} changed while opening",
        )
        blocks: list[bytes] = []
        while block := os.read(descriptor, 1024 * 1024):
            blocks.append(block)
        raw = b"".join(blocks)
        for observed in (os.fstat(descriptor), os.stat(canonical, follow_symlinks=False)):
            require(
                identity
                == (
                    observed.st_dev,
                    observed.st_ino,
                    observed.st_mode,
                    observed.st_size,
                    observed.st_mtime_ns,
                    observed.st_ctime_ns,
                ),
                f"{name} changed while reading",
            )
        require(len(raw) == opened.st_size, f"{name} byte count changed while reading")
        return canonical, raw
    finally:
        os.close(descriptor)


def load_metrics(
    path: Path,
    *,
    name: str,
    expected_start: int,
    expected_end: int,
) -> tuple[dict[int, dict[str, Any]], dict[str, Any]]:
    canonical, raw_bytes = _stable_regular_bytes(path, name=f"{name} metrics")
    try:
        raw = raw_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise AnalysisError(f"{name} metrics are not UTF-8") from exc
    require(raw.endswith("\n"), f"{name} metrics must end with a newline")
    records: dict[int, dict[str, Any]] = {}
    observed_updates: list[int] = []
    for line_number, line in enumerate(raw.splitlines(), start=1):
        value = _strict_json_line(line, source=f"{name} metrics line {line_number}")
        update = value.get("update")
        require(type(update) is int and update > 0, f"{name} metric update is invalid")
        require(update not in records, f"{name} metrics contain duplicate update {update}")
        for field in ("train_loss", "gradient_norm", "update_seconds"):
            observed = value.get(field)
            require(
                isinstance(observed, (int, float))
                and not isinstance(observed, bool)
                and math.isfinite(float(observed)),
                f"{name} update {update} has invalid {field}",
            )
        require(value.get("examples_seen") == update * GLOBAL_BATCH_SIZE, f"{name} example count differs")
        records[update] = value
        observed_updates.append(update)
    expected_updates = list(range(expected_start, expected_end + 1))
    require(observed_updates == expected_updates, f"{name} metrics update range or ordering differs")
    return records, {
        "bytes": len(raw_bytes),
        "path": str(canonical),
        "sha256": hashlib.sha256(raw_bytes).hexdigest(),
    }


def _artifact_path(checkpoint: Path, manifest: Mapping[str, Any], name: str) -> Path:
    artifacts = manifest.get("artifacts")
    require(isinstance(artifacts, Mapping), "checkpoint artifact table is missing")
    record = artifacts.get(name)
    require(isinstance(record, Mapping), f"checkpoint artifact {name} is missing")
    relative = record.get("path")
    require(isinstance(relative, str) and relative, f"checkpoint artifact {name} path is invalid")
    path = (checkpoint / relative).resolve(strict=True)
    require(path.is_relative_to(checkpoint), f"checkpoint artifact {name} escapes its checkpoint")
    require(path.stat().st_size == record.get("bytes"), f"checkpoint artifact {name} byte count differs")
    require(sha256_file(path) == record.get("sha256"), f"checkpoint artifact {name} hash differs")
    return path


def authenticate_checkpoint(path: Path, *, expected_update: int, name: str) -> tuple[Path, dict[str, Any], str]:
    checkpoint = path.resolve(strict=True)
    require(checkpoint.is_dir(), f"{name} checkpoint is not a directory")
    manifest = load_checkpoint_manifest(checkpoint, verify_hashes=True)
    manifest_sha256 = sha256_file(checkpoint / "manifest.json")
    state = manifest.get("trainer_state")
    require(isinstance(state, Mapping), f"{name} trainer state is missing")
    require(state.get("next_update") == expected_update, f"{name} checkpoint update differs")
    require(
        state.get("examples_seen") == expected_update * GLOBAL_BATCH_SIZE,
        f"{name} checkpoint example count differs",
    )
    return checkpoint, manifest, manifest_sha256


def _exact_nested(left: Any, right: Any, *, path: str) -> None:
    if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
        require(isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor), f"{path} tensor type differs")
        require(left.dtype == right.dtype and tuple(left.shape) == tuple(right.shape), f"{path} tensor schema differs")
        require(torch.equal(left, right), f"{path} tensor values differ")
        return
    require(type(left) is type(right), f"{path} value type differs")
    if isinstance(left, dict):
        require(set(left) == set(right), f"{path} mapping fields differ")
        for key in sorted(left):
            _exact_nested(left[key], right[key], path=f"{path}.{key}")
        return
    if isinstance(left, (list, tuple)):
        require(len(left) == len(right), f"{path} sequence length differs")
        for index, (left_value, right_value) in enumerate(zip(left, right, strict=True)):
            _exact_nested(left_value, right_value, path=f"{path}[{index}]")
        return
    require(left == right, f"{path} value differs")


def validate_candidate_rank_replicas(checkpoint: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    hashes = manifest.get("training_rank_state_sha256")
    require(isinstance(hashes, list) and len(hashes) == 2, "DP2 checkpoint must contain two rank states")
    states = []
    artifact_hashes = []
    for rank in range(2):
        path = _artifact_path(checkpoint, manifest, f"training_rank_{rank:03d}")
        require(sha256_file(path) == hashes[rank], f"DP2 rank {rank} state hash list differs")
        states.append(torch.load(path, map_location="cpu", weights_only=True))
        artifact_hashes.append(hashes[rank])
    for rank, state in enumerate(states):
        require(state.get("rank") == rank and state.get("world_size") == 2, f"DP2 rank {rank} topology differs")
    for field in (
        "optimizer",
        "optimizer_parameter_inventory",
        "run_contract",
        "scheduler",
        "trainer_state",
    ):
        _exact_nested(states[0][field], states[1][field], path=f"rank_state.{field}")
    left_rng = states[0].get("rng")
    right_rng = states[1].get("rng")
    require(isinstance(left_rng, Mapping) and isinstance(right_rng, Mapping), "DP2 rank RNG state is missing")
    require(left_rng.get("cuda_device_index") == 0, "DP2 rank-zero CUDA RNG device differs")
    require(right_rng.get("cuda_device_index") == 1, "DP2 rank-one CUDA RNG device differs")
    for field in ("torch_cpu", "torch_cuda"):
        left_tensor = left_rng.get(field)
        right_tensor = right_rng.get(field)
        require(
            isinstance(left_tensor, torch.Tensor) and isinstance(right_tensor, torch.Tensor),
            f"DP2 rank RNG {field} is invalid",
        )
        require(not torch.equal(left_tensor, right_tensor), f"DP2 rank RNG {field} is not domain-separated")
    return {
        "artifact_sha256": artifact_hashes,
        "optimizer_scheduler_trainer_bitwise_replicated": True,
        "rng_states_distinct": True,
    }


def _tensor_delta_statistics(parent: Path, reference: Path, candidate: Path) -> dict[str, Any]:
    totals = {
        "reference_delta_squared": 0.0,
        "candidate_delta_squared": 0.0,
        "difference_squared": 0.0,
        "delta_dot": 0.0,
    }
    tensor_count = 0
    element_count = 0
    max_absolute_final_difference = 0.0
    with (
        safe_open(parent, framework="pt", device="cpu") as parent_file,
        safe_open(reference, framework="pt", device="cpu") as reference_file,
        safe_open(candidate, framework="pt", device="cpu") as candidate_file,
    ):
        keys = tuple(parent_file.keys())
        require(keys == tuple(reference_file.keys()) == tuple(candidate_file.keys()), "trainable tensor names differ")
        for key in keys:
            parent_tensor = parent_file.get_tensor(key)
            reference_tensor = reference_file.get_tensor(key)
            candidate_tensor = candidate_file.get_tensor(key)
            require(
                parent_tensor.dtype == reference_tensor.dtype == candidate_tensor.dtype
                and tuple(parent_tensor.shape) == tuple(reference_tensor.shape) == tuple(candidate_tensor.shape),
                f"trainable tensor schema differs for {key}",
            )
            require(
                bool(torch.isfinite(parent_tensor).all())
                and bool(torch.isfinite(reference_tensor).all())
                and bool(torch.isfinite(candidate_tensor).all()),
                f"trainable tensor is non-finite for {key}",
            )
            parent_flat = parent_tensor.reshape(-1)
            reference_flat = reference_tensor.reshape(-1)
            candidate_flat = candidate_tensor.reshape(-1)
            for start in range(0, parent_flat.numel(), _TENSOR_BLOCK_ELEMENTS):
                stop = min(start + _TENSOR_BLOCK_ELEMENTS, parent_flat.numel())
                parent_block = parent_flat[start:stop].double()
                reference_block = reference_flat[start:stop].double()
                candidate_block = candidate_flat[start:stop].double()
                reference_delta = reference_block - parent_block
                candidate_delta = candidate_block - parent_block
                difference = candidate_delta - reference_delta
                totals["reference_delta_squared"] += float(torch.dot(reference_delta, reference_delta))
                totals["candidate_delta_squared"] += float(torch.dot(candidate_delta, candidate_delta))
                totals["difference_squared"] += float(torch.dot(difference, difference))
                totals["delta_dot"] += float(torch.dot(reference_delta, candidate_delta))
                if difference.numel():
                    max_absolute_final_difference = max(
                        max_absolute_final_difference,
                        float(difference.abs().max()),
                    )
            tensor_count += 1
            element_count += parent_flat.numel()
    reference_norm = math.sqrt(totals["reference_delta_squared"])
    candidate_norm = math.sqrt(totals["candidate_delta_squared"])
    difference_norm = math.sqrt(totals["difference_squared"])
    require(reference_norm > 0.0 and candidate_norm > 0.0, "one-step trainable delta is zero")
    cosine = totals["delta_dot"] / (reference_norm * candidate_norm)
    return {
        "candidate_delta_l2": candidate_norm,
        "delta_cosine_similarity": cosine,
        "delta_difference_l2": difference_norm,
        "delta_relative_l2_difference": difference_norm / reference_norm,
        "elements": element_count,
        "max_absolute_final_difference": max_absolute_final_difference,
        "reference_delta_l2": reference_norm,
        "tensors": tensor_count,
    }


def trainable_delta_statistics(
    parent_checkpoint: Path,
    parent_manifest: Mapping[str, Any],
    reference_checkpoint: Path,
    reference_manifest: Mapping[str, Any],
    candidate_checkpoint: Path,
    candidate_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    result = {}
    for name in ("interface", "lora_weights"):
        result[name] = _tensor_delta_statistics(
            _artifact_path(parent_checkpoint, parent_manifest, name),
            _artifact_path(reference_checkpoint, reference_manifest, name),
            _artifact_path(candidate_checkpoint, candidate_manifest, name),
        )
    reference_squared = sum(value["reference_delta_l2"] ** 2 for value in result.values())
    candidate_squared = sum(value["candidate_delta_l2"] ** 2 for value in result.values())
    difference_squared = sum(value["delta_difference_l2"] ** 2 for value in result.values())
    dot = sum(
        value["delta_cosine_similarity"] * value["reference_delta_l2"] * value["candidate_delta_l2"]
        for value in result.values()
    )
    reference_norm = math.sqrt(reference_squared)
    candidate_norm = math.sqrt(candidate_squared)
    difference_norm = math.sqrt(difference_squared)
    result["combined"] = {
        "candidate_delta_l2": candidate_norm,
        "delta_cosine_similarity": dot / (reference_norm * candidate_norm),
        "delta_difference_l2": difference_norm,
        "delta_relative_l2_difference": difference_norm / reference_norm,
        "elements": sum(value["elements"] for value in result.values()),
        "max_absolute_final_difference": max(value["max_absolute_final_difference"] for value in result.values()),
        "reference_delta_l2": reference_norm,
        "tensors": sum(value["tensors"] for value in result.values()),
    }
    return result


def _relative_difference(left: float, right: float) -> float:
    denominator = max(abs(left), abs(right), 1e-12)
    return abs(left - right) / denominator


def _timing_window(records: Mapping[int, Mapping[str, Any]], start: int, end: int, *, name: str) -> list[float]:
    require(0 < start <= end, f"{name} timing window is invalid")
    expected = list(range(start, end + 1))
    require(all(update in records for update in expected), f"{name} timing window is incomplete")
    values = [float(records[update]["update_seconds"]) for update in expected]
    require(all(value > 0.0 for value in values), f"{name} timing window contains a non-positive value")
    return values


def _crosscheck_manifest_metric(
    manifest: Mapping[str, Any],
    metric: Mapping[str, Any],
    *,
    name: str,
) -> None:
    observed = manifest.get("last_metrics")
    require(isinstance(observed, Mapping), f"{name} manifest last_metrics is missing")
    _exact_nested(dict(observed), dict(metric), path=f"{name}.last_metrics")


def _crosscheck_run_identity(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    *,
    name: str,
) -> None:
    for field in ("run_uuid", "config_sha256", "source_tree_sha256"):
        value = left.get(field)
        require(isinstance(value, str) and value, f"{name} {field} is missing")
        require(right.get(field) == value, f"{name} {field} differs")


def _validate_dp2_lineage(manifest: Mapping[str, Any], *, name: str) -> tuple[dict[str, str], dict[str, Any]]:
    try:
        return validate_dp2_checkpoint_lineage(manifest)
    except (TypeError, ValueError) as exc:
        raise AnalysisError(f"{name} DP2 lineage is invalid: {exc}") from exc


def _validate_dp2_environment(
    manifest: Mapping[str, Any],
    parent: Mapping[str, Any],
    *,
    name: str,
) -> None:
    environment = manifest.get("execution_environment")
    require(isinstance(environment, Mapping), f"{name} execution environment is missing")
    gpu_uuids = environment.get("gpu_uuids")
    require(
        isinstance(gpu_uuids, list) and all(isinstance(value, str) for value in gpu_uuids),
        f"{name} DP2 GPU inventory is invalid",
    )
    normalized_gpu_uuids = [value if value.startswith("GPU-") else f"GPU-{value}" for value in gpu_uuids]
    require(normalized_gpu_uuids == DP2_GPU_UUIDS, f"{name} DP2 GPU inventory differs")
    require(environment.get("cuda_runtime") == "12.9", f"{name} CUDA runtime differs")
    parent_environment = parent.get("execution_environment")
    require(isinstance(parent_environment, Mapping), "parent execution environment is missing")
    parent_runtime = parent_environment.get("authenticated_runtime")
    runtime = environment.get("authenticated_runtime")
    require(
        isinstance(parent_runtime, Mapping) and isinstance(runtime, Mapping), "training runtime identity is missing"
    )
    require(runtime.get("train_venv") == parent_runtime.get("train_venv"), f"{name} train venv differs from parent")


def _candidate_peak_memory(manifest: Mapping[str, Any]) -> dict[str, Any]:
    values = manifest.get("cuda_allocator_peak_memory_bytes_by_rank")
    maximum = manifest.get("cuda_allocator_peak_memory_bytes_max")
    require(
        isinstance(values, list) and len(values) == 2 and all(type(value) is int and value > 0 for value in values),
        "performance checkpoint allocator peak inventory must contain two positive integers",
    )
    require(type(maximum) is int and maximum == max(values), "performance checkpoint allocator peak maximum differs")
    return {
        "bytes_by_rank": list(values),
        "bytes_max": maximum,
        "gib_by_rank": [value / 2**30 for value in values],
        "gib_max": maximum / 2**30,
    }


def _checkpoint_semantic_recipe_sha256(
    checkpoint: Path,
    manifest: Mapping[str, Any],
    *,
    name: str,
) -> tuple[str, str]:
    """Authenticate one resolved config and return raw-artifact and semantic digests."""

    config_sha256 = manifest.get("config_sha256")
    require(isinstance(config_sha256, str), f"{name} config SHA-256 is missing")
    artifacts = manifest.get("artifacts")
    require(isinstance(artifacts, Mapping), f"{name} artifact table is missing")
    record = artifacts.get("resolved_config")
    require(isinstance(record, Mapping), f"{name} resolved-config artifact is missing")
    raw_sha256 = record.get("sha256")
    require(isinstance(raw_sha256, str), f"{name} resolved-config artifact SHA-256 is missing")
    path = _artifact_path(checkpoint, manifest, "resolved_config")
    try:
        config, observed_config_sha256 = load_verified_resolved_config(path, expected_sha256=config_sha256)
        semantic_sha256 = dp2_semantic_recipe_sha256(config)
    except (TypeError, ValueError) as exc:
        raise AnalysisError(f"{name} resolved semantic recipe is invalid: {exc}") from exc
    require(observed_config_sha256 == config_sha256, f"{name} resolved config digest differs")
    return raw_sha256, semantic_sha256


def _canonical_json(value: Any) -> bytes:
    return (json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n").encode("utf-8")


def run(args: argparse.Namespace) -> tuple[dict[str, Any], str]:
    parent_path, parent, parent_sha = authenticate_checkpoint(
        args.parent_checkpoint,
        expected_update=PARENT_UPDATE,
        name="parent",
    )
    reference_path, reference, reference_sha = authenticate_checkpoint(
        args.reference_checkpoint,
        expected_update=COMPARISON_UPDATE,
        name="reference",
    )
    candidate_path, candidate, candidate_sha = authenticate_checkpoint(
        args.candidate_checkpoint,
        expected_update=COMPARISON_UPDATE,
        name="candidate",
    )
    performance_input = args.candidate_performance_checkpoint
    if performance_input is None:
        performance_input = candidate_path.with_name(f"update-{PERFORMANCE_UPDATE:06d}")
    performance_path, performance, performance_sha = authenticate_checkpoint(
        performance_input,
        expected_update=PERFORMANCE_UPDATE,
        name="performance",
    )
    require(parent.get("execution_profile") == TP1_PROFILE, "parent profile differs")
    require(reference.get("execution_profile") == TP1_PROFILE, "reference profile differs")
    require(candidate.get("execution_profile") == DP2_PROFILE, "candidate profile differs")
    require(performance.get("execution_profile") == DP2_PROFILE, "performance profile differs")
    _crosscheck_run_identity(parent, reference, name="parent/reference")
    _crosscheck_run_identity(candidate, performance, name="candidate/performance")
    require(reference.get("parent_manifest_sha256") == parent_sha, "reference does not continue the parent")
    require(candidate.get("parent_manifest_sha256") is None, "first DP2 checkpoint must be a new-run root")
    require(
        performance.get("parent_manifest_sha256") == candidate_sha,
        "performance checkpoint does not directly continue the semantic candidate",
    )
    candidate_contract, lineage = _validate_dp2_lineage(candidate, name="candidate")
    performance_contract, performance_lineage = _validate_dp2_lineage(performance, name="performance")
    _exact_nested(candidate_contract, performance_contract, path="candidate_performance.fork_contract")
    _exact_nested(lineage, performance_lineage, path="candidate_performance.fork_lineage")
    _candidate_peak_memory(candidate)
    require(lineage.get("parent_manifest_sha256") == parent_sha, "candidate fork parent differs")
    require(lineage.get("parent_update") == PARENT_UPDATE, "candidate fork update differs")
    require(lineage.get("parent_run_uuid") == parent.get("run_uuid"), "candidate fork run UUID differs")
    require(
        lineage.get("parent_source_tree_sha256") == parent.get("source_tree_sha256"),
        "candidate fork source differs",
    )
    require(lineage.get("parent_config_sha256") == parent.get("config_sha256"), "candidate fork config differs")
    require(
        lineage.get("parent_optimizer_parameter_schema_sha256") == parent.get("optimizer_parameter_schema_sha256"),
        "candidate fork optimizer schema differs",
    )
    parent_resolved_sha256, parent_semantic_sha256 = _checkpoint_semantic_recipe_sha256(
        parent_path,
        parent,
        name="parent",
    )
    _, reference_semantic_sha256 = _checkpoint_semantic_recipe_sha256(
        reference_path,
        reference,
        name="reference",
    )
    _, candidate_semantic_sha256 = _checkpoint_semantic_recipe_sha256(
        candidate_path,
        candidate,
        name="candidate",
    )
    _, performance_semantic_sha256 = _checkpoint_semantic_recipe_sha256(
        performance_path,
        performance,
        name="performance",
    )
    require(
        lineage.get("parent_resolved_config_sha256") == parent_resolved_sha256,
        "candidate fork parent resolved-config artifact differs",
    )
    require(
        parent_semantic_sha256
        == reference_semantic_sha256
        == candidate_semantic_sha256
        == performance_semantic_sha256
        == lineage.get("semantic_recipe_sha256"),
        "parent/reference/candidate/performance semantic recipes differ",
    )
    parent_directory = os.stat(parent_path, follow_symlinks=False)
    require(lineage.get("parent_checkpoint") == str(parent_path), "candidate fork parent path differs")
    require(lineage.get("parent_checkpoint_device") == parent_directory.st_dev, "candidate fork device differs")
    require(lineage.get("parent_checkpoint_inode") == parent_directory.st_ino, "candidate fork inode differs")
    _validate_dp2_environment(candidate, parent, name="candidate")
    _validate_dp2_environment(performance, parent, name="performance")
    memory = _candidate_peak_memory(performance)

    reference_metrics, reference_metrics_identity = load_metrics(
        args.reference_metrics,
        name="reference",
        expected_start=REFERENCE_METRICS_START,
        expected_end=REFERENCE_METRICS_END,
    )
    candidate_metrics, candidate_metrics_identity = load_metrics(
        args.candidate_metrics,
        name="candidate",
        expected_start=CANDIDATE_METRICS_START,
        expected_end=CANDIDATE_METRICS_END,
    )
    _crosscheck_manifest_metric(parent, reference_metrics[PARENT_UPDATE], name="parent")
    _crosscheck_manifest_metric(reference, reference_metrics[COMPARISON_UPDATE], name="reference")
    _crosscheck_manifest_metric(candidate, candidate_metrics[COMPARISON_UPDATE], name="candidate")
    _crosscheck_manifest_metric(performance, candidate_metrics[PERFORMANCE_UPDATE], name="performance")
    reference_step = reference_metrics[COMPARISON_UPDATE]
    candidate_step = candidate_metrics[COMPARISON_UPDATE]
    for name in ("examples_seen", "interface_learning_rate", "lora_learning_rate", "objective", "update"):
        require(reference_step.get(name) == candidate_step.get(name), f"comparison metric {name} differs")
    loss_difference = _relative_difference(reference_step["train_loss"], candidate_step["train_loss"])
    gradient_difference = _relative_difference(reference_step["gradient_norm"], candidate_step["gradient_norm"])
    baseline_timing = _timing_window(
        reference_metrics,
        BASELINE_TIMING_START,
        BASELINE_TIMING_END,
        name="baseline",
    )
    candidate_timing = _timing_window(
        candidate_metrics,
        CANDIDATE_TIMING_START,
        CANDIDATE_TIMING_END,
        name="candidate",
    )
    baseline_median = statistics.median(baseline_timing)
    candidate_median = statistics.median(candidate_timing)
    median_speedup = 1.0 - candidate_median / baseline_median
    deltas = trainable_delta_statistics(
        parent_path,
        parent,
        reference_path,
        reference,
        candidate_path,
        candidate,
    )
    replicas = validate_candidate_rank_replicas(candidate_path, candidate)
    combined = deltas["combined"]
    checks = {
        "candidate_peak_memory_within_limit": memory["gib_max"] <= PEAK_MEMORY_GIB_MAX,
        "gradient_norm_relative_difference_within_limit": (
            gradient_difference <= GRADIENT_NORM_RELATIVE_DIFFERENCE_MAX
        ),
        "loss_relative_difference_within_limit": loss_difference <= LOSS_RELATIVE_DIFFERENCE_MAX,
        "median_speedup_meets_minimum": median_speedup >= MEDIAN_SPEEDUP_MIN,
        "parameter_delta_cosine_meets_minimum": (combined["delta_cosine_similarity"] >= PARAMETER_DELTA_COSINE_MIN),
        "parameter_delta_relative_l2_within_limit": (
            combined["delta_relative_l2_difference"] <= PARAMETER_DELTA_RELATIVE_L2_MAX
        ),
        "rank_optimizer_scheduler_trainer_bitwise_replicated": (
            replicas["optimizer_scheduler_trainer_bitwise_replicated"]
        ),
        "rank_rng_states_domain_separated": replicas["rng_states_distinct"],
    }
    report = {
        "schema": SCHEMA,
        "status": "passed" if all(checks.values()) else "rejected",
        "checks": checks,
        "criteria": {
            "candidate_peak_memory_gib_max": PEAK_MEMORY_GIB_MAX,
            "gradient_norm_relative_difference_max": GRADIENT_NORM_RELATIVE_DIFFERENCE_MAX,
            "loss_relative_difference_max": LOSS_RELATIVE_DIFFERENCE_MAX,
            "median_speedup_min": MEDIAN_SPEEDUP_MIN,
            "parameter_delta_cosine_min": PARAMETER_DELTA_COSINE_MIN,
            "parameter_delta_relative_l2_max": PARAMETER_DELTA_RELATIVE_L2_MAX,
        },
        "inputs": {
            "candidate_checkpoint": str(candidate_path),
            "candidate_manifest_sha256": candidate_sha,
            "candidate_metrics": candidate_metrics_identity,
            "candidate_performance_checkpoint": str(performance_path),
            "candidate_performance_manifest_sha256": performance_sha,
            "parent_checkpoint": str(parent_path),
            "parent_manifest_sha256": parent_sha,
            "reference_checkpoint": str(reference_path),
            "reference_manifest_sha256": reference_sha,
            "reference_metrics": reference_metrics_identity,
            "script_sha256": sha256_file(Path(__file__).resolve()),
        },
        "measurements": {
            "candidate_peak_memory_bytes_by_rank": memory["bytes_by_rank"],
            "candidate_peak_memory_bytes_max": memory["bytes_max"],
            "candidate_peak_memory_gib": memory["gib_max"],
            "candidate_peak_memory_gib_by_rank": memory["gib_by_rank"],
            "gradient_norm_relative_difference": gradient_difference,
            "loss_relative_difference": loss_difference,
            "parameter_deltas": deltas,
            "rank_replicas": replicas,
            "timing": {
                "baseline_median_update_seconds": baseline_median,
                "baseline_window": [BASELINE_TIMING_START, BASELINE_TIMING_END],
                "candidate_median_update_seconds": candidate_median,
                "candidate_window": [CANDIDATE_TIMING_START, CANDIDATE_TIMING_END],
                "median_speedup": median_speedup,
            },
        },
        "scope": {
            "adopts_dp2_for_long_run_only_if_passed": True,
            "global_batch_semantics_preserved": True,
            "official_benchmark_claim": False,
            "qualification_only": True,
        },
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = _canonical_json(report)
    with output.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    digest = hashlib.sha256(payload).hexdigest()
    sidecar = output.with_name(f"{output.name}.sha256")
    with sidecar.open("xb") as handle:
        handle.write(f"{digest}  {output.name}\n".encode("ascii"))
        handle.flush()
        os.fsync(handle.fileno())
    return report, digest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-checkpoint", type=Path, required=True)
    parser.add_argument("--reference-checkpoint", type=Path, required=True)
    parser.add_argument("--candidate-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--candidate-performance-checkpoint",
        type=Path,
        help="DP2 update-1100 checkpoint; defaults to update-001100 beside --candidate-checkpoint.",
    )
    parser.add_argument("--reference-metrics", type=Path, required=True)
    parser.add_argument("--candidate-metrics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    report, digest = run(parse_args())
    print(json.dumps({"report_sha256": digest, "status": report["status"]}, sort_keys=True))
    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
