"""Authenticated TP1-to-DP2 training fork contracts.

The fork is a new run, not an in-place resume.  A frozen manifest authenticates
the TP1 update-1000 journal tip and the child DP2 recipe.  At bootstrap both DP
ranks restore identical model / optimizer / scheduler / trainer state, then
receive deterministic domain-separated mutable RNG states.  LIBERO's canonical
data and flow streams remain keyed only by ``(run_seed, update, microstep)``.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import random
import re
import stat
import uuid
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from duo_vla.checkpointing import load_checkpoint_manifest
from duo_vla.run_config import canonical_config_sha256, load_resolved_toml, load_verified_resolved_config
from duo_vla.run_journal import validate_resume_checkpoint
from duo_vla.training import TrainerState
from duo_vla.training_checkpoint import (
    optimizer_parameter_inventory,
    optimizer_parameter_inventory_sha256,
    validate_optimizer_state_dict,
)

DP2_FORK_SCHEMA = "duo-vla-libero-dp2-fork-v2"
DP2_SOURCE_IDENTITY_SCHEMA = "duo-vla-dp2-source-identity-v1"
DP2_RNG_DERIVATION = "blake2b-domain-separated-parent-state-child-run-rank-v1"
DP2_RNG_DOMAIN = "duo-vla-libero-dp2-rank-rng-v1"
DP2_SEMANTIC_RECIPE_SCHEMA = "duo-vla-libero-dp2-semantic-recipe-v1"
DP2_EXECUTION_PROFILE = "duovla-dp2-tp1-fused-v2-train-b32-serve-b8-v1"
TP1_PARENT_EXECUTION_PROFILE = "duovla-single-gpu-tp1-fused-v2-train-b64-serve-b8-v1"
FUSED_V2_BACKEND = "sample_isolated_grouped_mm_v2"
PARENT_UPDATE = 1000
GLOBAL_BATCH_SIZE = 64
RANK_PHYSICAL_BATCH_SIZE = 32
SERVING_BATCH_SIZE = 8
DP_WORLD_SIZE = 2
MODEL_TENSOR_PARALLEL_SIZE = 1
EXPECTED_PHYSICAL_GPU_INDICES = (0, 1)
EXPECTED_TRAINING_GPU_UUIDS = (
    "GPU-30424b03-3051-615a-832e-186511378a61",
    "GPU-84fa4004-92fb-8f86-cc65-01d62a27950e",
)

_SHA256 = re.compile(r"[0-9a-f]{64}")
_GPU_UUID = re.compile(r"(?:GPU-)?[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}")
_PARENT_ARTIFACTS = (
    "interface",
    "lora_config",
    "lora_weights",
    "resolved_config",
    "training_rank_000",
)
DP2_FORK_LINEAGE_FIELDS = frozenset(
    {
        "fork_manifest_sha256",
        "parent_checkpoint",
        "parent_checkpoint_device",
        "parent_checkpoint_inode",
        "parent_config_sha256",
        "parent_manifest_sha256",
        "parent_optimizer_parameter_schema_sha256",
        "parent_resolved_config_sha256",
        "parent_run_uuid",
        "parent_source_tree_sha256",
        "parent_update",
        "semantic_recipe_sha256",
    }
)
DP2_FORK_CHECKPOINT_CONTRACT_FIELDS = frozenset(
    {
        "fork_manifest_sha256",
        "fork_parent_config_sha256",
        "fork_parent_checkpoint_device",
        "fork_parent_checkpoint_inode",
        "fork_parent_checkpoint_path",
        "fork_parent_manifest_sha256",
        "fork_parent_optimizer_parameter_schema_sha256",
        "fork_parent_resolved_config_sha256",
        "fork_parent_run_uuid",
        "fork_parent_source_tree_sha256",
        "fork_parent_training_rank_state_sha256",
        "fork_parent_update",
        "fork_parent_venv_root_sha256",
        "fork_rng_derivation",
        "fork_schema",
        "fork_semantic_recipe_sha256",
        "fork_static_resolved_toml_sha256",
        "serving_tensor_parallel_size",
        "serving_world_size",
        "training_gpu_uuids",
        "training_tensor_parallel_size",
        "training_world_size",
    }
)
_SOURCE_EXPLICIT_PATHS = (
    "envs/train-single-gpu/pyproject.toml",
    "envs/train-single-gpu/uv.lock",
    "pyproject.toml",
    "scripts/bootstrap_train_single_gpu_env.sh",
    "scripts/create_libero_dp2_fork.py",
    "scripts/create_libero_topology_fork.py",
    "scripts/run_libero_train.sh",
    "scripts/run_libero_train_dp2.sh",
    "scripts/run_libero_train_single_gpu.sh",
    "scripts/train_libero.py",
    "uv.lock",
)


@dataclass(frozen=True, slots=True)
class DP2ForkRestoreResult:
    """Authenticated lineage returned after mutating one child rank's state."""

    trainer_state: TrainerState
    fork_manifest_sha256: str
    parent_checkpoint: str
    parent_checkpoint_device: int
    parent_checkpoint_inode: int
    parent_manifest_sha256: str
    parent_run_uuid: str
    parent_run_seed: int
    parent_update: int
    parent_source_tree_sha256: str
    parent_config_sha256: str
    parent_optimizer_parameter_schema_sha256: str
    parent_resolved_config_sha256: str
    child_run_uuid: str
    child_run_seed: int
    child_source_tree_sha256: str
    child_static_resolved_toml_sha256: str
    semantic_recipe_sha256: str

    def lineage_dict(self) -> dict[str, Any]:
        """Return JSON-safe fields for the first child manifest and journal."""

        return {
            "fork_manifest_sha256": self.fork_manifest_sha256,
            "parent_checkpoint": self.parent_checkpoint,
            "parent_checkpoint_device": self.parent_checkpoint_device,
            "parent_checkpoint_inode": self.parent_checkpoint_inode,
            "parent_config_sha256": self.parent_config_sha256,
            "parent_manifest_sha256": self.parent_manifest_sha256,
            "parent_optimizer_parameter_schema_sha256": self.parent_optimizer_parameter_schema_sha256,
            "parent_resolved_config_sha256": self.parent_resolved_config_sha256,
            "parent_run_uuid": self.parent_run_uuid,
            "parent_source_tree_sha256": self.parent_source_tree_sha256,
            "parent_update": self.parent_update,
            "semantic_recipe_sha256": self.semantic_recipe_sha256,
        }


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _exact_keys(value: object, expected: set[str], context: str) -> Mapping[str, Any]:
    _require(isinstance(value, Mapping), f"{context} must be an object")
    assert isinstance(value, Mapping)
    _require(set(value) == expected, f"{context} field inventory differs")
    return value


def _sha256(value: object, context: str) -> str:
    _require(isinstance(value, str) and _SHA256.fullmatch(value) is not None, f"{context} must be a SHA-256")
    assert isinstance(value, str)
    return value


def _uuid(value: object, context: str) -> str:
    _require(isinstance(value, str), f"{context} must be a UUID")
    assert isinstance(value, str)
    try:
        parsed = uuid.UUID(value)
    except (ValueError, AttributeError) as exc:
        raise ValueError(f"{context} must be a UUID") from exc
    _require(str(parsed) == value, f"{context} must use canonical UUID text")
    return value


def _canonical_json_bytes(value: object, *, pretty: bool = False) -> bytes:
    separators = None if pretty else (",", ":")
    suffix = "\n" if pretty else ""
    return (
        json.dumps(value, allow_nan=False, indent=2 if pretty else None, separators=separators, sort_keys=True) + suffix
    ).encode()


_SEMANTIC_RECIPE_TABLES = (
    "action",
    "benchmark",
    "lora",
    "model",
    "policy",
    "reproducibility",
    "sampling",
)
_NORMALIZED_OPTIMIZATION_FIELDS = frozenset(
    {
        "gradient_accumulation_steps",
        "microbatch_size",
        "physical_batch_size",
    }
)


def dp2_semantic_recipe(config: Mapping[str, Any]) -> dict[str, Any]:
    """Return training semantics after removing only qualified batch topology differences."""

    _require(isinstance(config, Mapping), "semantic recipe config must be an object")
    protocol = config.get("protocol")
    _require(isinstance(protocol, str) and protocol, "semantic recipe protocol is missing")
    recipe: dict[str, Any] = {
        "schema": DP2_SEMANTIC_RECIPE_SCHEMA,
        "normalized_optimization_fields": sorted(_NORMALIZED_OPTIMIZATION_FIELDS),
        "protocol": protocol,
    }
    for name in _SEMANTIC_RECIPE_TABLES:
        value = config.get(name)
        _require(isinstance(value, Mapping), f"semantic recipe {name} table is missing")
        recipe[name] = dict(value)

    optimization = config.get("optimization")
    _require(isinstance(optimization, Mapping), "semantic recipe optimization table is missing")
    missing_optimization = _NORMALIZED_OPTIMIZATION_FIELDS - set(optimization)
    _require(not missing_optimization, f"semantic recipe batch fields are missing: {sorted(missing_optimization)}")
    recipe["optimization"] = {
        name: value for name, value in optimization.items() if name not in _NORMALIZED_OPTIMIZATION_FIELDS
    }

    training = config.get("training")
    _require(isinstance(training, Mapping), "semantic recipe training table is missing")
    normalized_training = dict(training)
    run = config.get("run")
    run_cache = run.get("max_cached_files") if isinstance(run, Mapping) else None
    run_seed = run.get("seed") if isinstance(run, Mapping) else None
    run_task = run.get("task") if isinstance(run, Mapping) else None
    training_cache = normalized_training.get("max_cached_files")
    effective_cache = training_cache if training_cache is not None else run_cache
    _require(type(effective_cache) is int and effective_cache > 0, "semantic recipe cache capacity is missing")
    if training_cache is not None and run_cache is not None:
        _require(training_cache == run_cache, "training and run cache capacities differ")
    normalized_training["max_cached_files"] = effective_cache
    recipe["training"] = normalized_training
    reproducibility_seed = recipe["reproducibility"].get("seed")
    effective_seed = reproducibility_seed if run_seed is None else run_seed
    _require(effective_seed == reproducibility_seed, "run and reproducibility seeds differ")
    _require(run_task is None, "DP2 semantic recipe requires the full LIBERO task mixture")
    recipe["run"] = {"max_cached_files": effective_cache, "seed": effective_seed, "task": None}

    # Round-trip through strict JSON so Mapping subclasses and tuples cannot
    # create equality or hashing behavior that differs across creator/trainer.
    try:
        normalized = json.loads(_canonical_json_bytes(recipe).decode("utf-8"))
    except (TypeError, ValueError) as exc:
        raise ValueError("semantic recipe is not strict JSON") from exc
    assert isinstance(normalized, dict)
    return normalized


def dp2_semantic_recipe_sha256(config: Mapping[str, Any]) -> str:
    """Hash the normalized model/objective/optimizer/schedule recipe."""

    return hashlib.sha256(_canonical_json_bytes(dp2_semantic_recipe(config))).hexdigest()


def validate_dp2_semantic_recipe_continuity(
    parent_config: Mapping[str, Any],
    child_config: Mapping[str, Any],
) -> str:
    """Require exact TP1-parent/DP2-child semantics outside batch topology."""

    parent_recipe = dp2_semantic_recipe(parent_config)
    child_recipe = dp2_semantic_recipe(child_config)
    _require(parent_recipe == child_recipe, "DP2 child semantic recipe differs from the authenticated TP1 parent")
    return hashlib.sha256(_canonical_json_bytes(parent_recipe)).hexdigest()


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_nonfinite_json(value: str) -> None:
    raise ValueError(f"non-finite JSON value: {value}")


def _stable_regular_bytes(path: Path, *, single_link: bool = True) -> tuple[bytes, os.stat_result]:
    before = os.stat(path, follow_symlinks=False)
    _require(stat.S_ISREG(before.st_mode), f"input must be a regular non-symlink file: {path}")
    if single_link:
        _require(before.st_nlink == 1, f"input must have exactly one link: {path}")
    descriptor = os.open(path, os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW | os.O_CLOEXEC)
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
        _require(
            identity
            == (before.st_dev, before.st_ino, before.st_mode, before.st_size, before.st_mtime_ns, before.st_ctime_ns),
            f"input changed while opening: {path}",
        )
        blocks: list[bytes] = []
        while block := os.read(descriptor, 1024 * 1024):
            blocks.append(block)
        raw = b"".join(blocks)
        after = os.fstat(descriptor)
        rebound = os.stat(path, follow_symlinks=False)
        for observed in (after, rebound):
            _require(
                identity
                == (
                    observed.st_dev,
                    observed.st_ino,
                    observed.st_mode,
                    observed.st_size,
                    observed.st_mtime_ns,
                    observed.st_ctime_ns,
                ),
                f"input changed while reading: {path}",
            )
        _require(len(raw) == opened.st_size, f"input size changed while reading: {path}")
        return raw, opened
    finally:
        os.close(descriptor)


def _strict_json_bytes(raw: bytes, *, source: Path) -> dict[str, Any]:
    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonfinite_json,
        )
    except (UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"cannot parse strict JSON: {source}") from exc
    _require(isinstance(value, dict), f"JSON document must be an object: {source}")
    return value


def _source_paths(project_root: Path) -> tuple[Path, ...]:
    root = project_root.resolve(strict=True)
    paths = [path for path in (root / "src/duo_vla").rglob("*.py") if "__pycache__" not in path.parts]
    paths.extend(path for path in (root / "configs").rglob("*") if path.is_file())
    paths.extend(root / relative for relative in _SOURCE_EXPLICIT_PATHS)
    unique = tuple(sorted(set(paths)))
    _require(unique, "DP2 source inventory is empty")
    for path in unique:
        _require(path.is_file() and not path.is_symlink(), f"DP2 source input is not a real file: {path}")
        _require(path.resolve(strict=True).is_relative_to(root), f"DP2 source input escapes the workspace: {path}")
    return unique


def dp2_source_identity(project_root: str | Path) -> dict[str, Any]:
    """Hash the exact DP2 trainer, launcher, fork, config, venv, and source inputs."""

    root = Path(project_root).resolve(strict=True)
    records = []
    digest = hashlib.sha256()
    for path in _source_paths(root):
        raw, _ = _stable_regular_bytes(path, single_link=False)
        relative = path.relative_to(root).as_posix()
        records.append(
            {
                "bytes": len(raw),
                "path": relative,
                "sha256": hashlib.sha256(raw).hexdigest(),
            }
        )
        # This is intentionally byte-for-byte the trainer's source hash
        # algorithm; the richer inventory above is an independently checked
        # explanation of the same input set.
        digest.update(relative.encode())
        digest.update(raw)
    tree_sha256 = digest.hexdigest()
    return {
        "schema": DP2_SOURCE_IDENTITY_SCHEMA,
        "files": records,
        "files_verified": len(records),
        "inventory_sha256": hashlib.sha256(_canonical_json_bytes(records)).hexdigest(),
        "root": str(root),
        "source_tree_sha256": tree_sha256,
        "total_bytes": sum(record["bytes"] for record in records),
    }


def _artifact_record(value: object, context: str) -> dict[str, Any]:
    record = _exact_keys(value, {"bytes", "path", "sha256"}, context)
    path = record["path"]
    size = record["bytes"]
    _require(
        isinstance(path, str) and path and not Path(path).is_absolute() and ".." not in Path(path).parts,
        f"{context} path is invalid",
    )
    _require(type(size) is int and size > 0, f"{context} byte size is invalid")
    return {"bytes": size, "path": path, "sha256": _sha256(record["sha256"], f"{context} SHA-256")}


def _parent_rank_state(
    checkpoint: Path,
    manifest: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    artifact = _artifact_record(manifest["artifacts"]["training_rank_000"], "parent rank-state artifact")
    path = checkpoint / artifact["path"]
    _require(path.resolve(strict=True).is_relative_to(checkpoint), "parent rank-state artifact escapes checkpoint")
    raw, _ = _stable_regular_bytes(path)
    _require(len(raw) == artifact["bytes"], "parent rank-state artifact size mismatch")
    _require(hashlib.sha256(raw).hexdigest() == artifact["sha256"], "parent rank-state artifact hash mismatch")
    try:
        payload = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
    except Exception as exc:
        raise ValueError("cannot deserialize authenticated parent rank state") from exc
    required = {
        "optimizer",
        "optimizer_parameter_inventory",
        "rank",
        "rng",
        "run_contract",
        "scheduler",
        "schema",
        "trainer_state",
        "world_size",
    }
    _exact_keys(payload, required, "parent rank state")
    _require(payload["schema"] == "duo-vla-training-rank-state-v2", "parent rank-state schema differs")
    _require(payload["rank"] == 0 and payload["world_size"] == 1, "parent rank-state topology is not TP1")
    trainer_state = TrainerState.from_dict(payload["trainer_state"])
    _require(trainer_state.next_update == PARENT_UPDATE, "parent rank state is not update 1000")
    _require(trainer_state.examples_seen == PARENT_UPDATE * GLOBAL_BATCH_SIZE, "parent example count differs")
    _require(payload["trainer_state"] == manifest["trainer_state"], "parent trainer state disagrees with manifest")
    inventory_sha256 = optimizer_parameter_inventory_sha256(payload["optimizer_parameter_inventory"])
    _require(
        inventory_sha256 == manifest["optimizer_parameter_schema_sha256"],
        "parent optimizer parameter schema disagrees with manifest",
    )
    validate_optimizer_state_dict(
        payload["optimizer"],
        payload["optimizer_parameter_inventory"],
        expected_update=PARENT_UPDATE,
    )
    scheduler = payload["scheduler"]
    _require(isinstance(scheduler, Mapping), "parent scheduler state must be an object")
    _require(scheduler.get("last_epoch") == PARENT_UPDATE, "parent scheduler is not at update 1000")
    run_contract = payload["run_contract"]
    _require(isinstance(run_contract, Mapping), "parent rank-state run contract must be an object")
    parent_contract = {
        "config_sha256": manifest["config_sha256"],
        "execution_profile": TP1_PARENT_EXECUTION_PROFILE,
        "expert_batch_isolation": FUSED_V2_BACKEND,
        "optimizer_parameter_schema_sha256": manifest["optimizer_parameter_schema_sha256"],
        "physical_batch_size": str(RANK_PHYSICAL_BATCH_SIZE * DP_WORLD_SIZE),
        "run_uuid": manifest["run_uuid"],
        "serving_batch_size": str(SERVING_BATCH_SIZE),
        "source_tree_sha256": manifest["source_tree_sha256"],
        "tensor_parallel_size": "1",
    }
    for name, expected in parent_contract.items():
        _require(run_contract.get(name) == expected, f"parent rank-state run contract differs for {name}")
    rng = payload["rng"]
    _require(isinstance(rng, Mapping), "parent RNG state must be an object")
    _require(
        set(rng) == {"python", "numpy", "torch_cpu", "torch_cuda", "cuda_device_index"},
        "parent RNG generator inventory differs",
    )
    _require(rng["cuda_device_index"] == 0, "parent TP1 CUDA RNG must use logical device zero")
    return payload, artifact


def _parent_resolved_config(
    checkpoint: Path,
    manifest: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load the parent recipe through its manifest-authenticated artifact."""

    artifact = _artifact_record(manifest["artifacts"]["resolved_config"], "parent resolved-config artifact")
    path = checkpoint / artifact["path"]
    _require(path.resolve(strict=True).is_relative_to(checkpoint), "parent resolved config escapes checkpoint")
    raw_before, _ = _stable_regular_bytes(path)
    _require(len(raw_before) == artifact["bytes"], "parent resolved-config artifact size mismatch")
    _require(
        hashlib.sha256(raw_before).hexdigest() == artifact["sha256"],
        "parent resolved-config artifact hash mismatch",
    )
    config, config_sha256 = load_verified_resolved_config(
        path,
        expected_sha256=_sha256(manifest.get("config_sha256"), "parent config SHA-256"),
    )
    raw_after, _ = _stable_regular_bytes(path)
    _require(raw_after == raw_before, "parent resolved config changed during authentication")
    _require(config_sha256 == manifest["config_sha256"], "parent resolved config digest differs")
    _validate_tp1_parent_config(config)
    return config, artifact


def _validate_tp1_parent_config(config: Mapping[str, Any]) -> None:
    """Require the resolved parent side of the one qualified B64-to-DP2 fork."""

    _require(config.get("execution_profile") == TP1_PARENT_EXECUTION_PROFILE, "parent resolved profile differs")
    model = config.get("model")
    _require(isinstance(model, Mapping), "parent resolved model table is missing")
    _require(model.get("tensor_parallel_size") == 1, "parent resolved model is not TP1")
    _require(model.get("expert_batch_isolation") == FUSED_V2_BACKEND, "parent resolved backend is not fused v2")
    optimization = config.get("optimization")
    _require(isinstance(optimization, Mapping), "parent resolved optimization table is missing")
    expected_optimization = {
        "global_batch_size": GLOBAL_BATCH_SIZE,
        "gradient_accumulation_steps": 1,
        "microbatch_size": GLOBAL_BATCH_SIZE,
        "physical_batch_size": GLOBAL_BATCH_SIZE,
        "serving_batch_size": SERVING_BATCH_SIZE,
    }
    _require(
        all(optimization.get(name) == expected for name, expected in expected_optimization.items()),
        "parent resolved batch geometry differs",
    )
    run = config.get("run")
    _require(
        isinstance(run, Mapping)
        and run.get("seed") == 0
        and run.get("task") is None
        and run.get("max_cached_files") == 377,
        "parent resolved run contract differs",
    )


def _canonical_parent_checkpoint(
    parent_checkpoint: str | Path,
    *,
    expected_manifest_sha256: str,
    expected_source_tree_sha256: str,
    expected_run_uuid: str,
) -> tuple[Path, dict[str, Any], dict[str, Any], os.stat_result, dict[str, Any], dict[str, Any]]:
    supplied = Path(parent_checkpoint)
    _require(supplied.is_absolute(), "parent checkpoint path must be absolute")
    checkpoint = supplied.resolve(strict=True)
    _require(checkpoint == supplied, "parent checkpoint path must be canonical and contain no symlink")
    directory = os.stat(checkpoint, follow_symlinks=False)
    _require(stat.S_ISDIR(directory.st_mode), "parent checkpoint must be a real directory")
    _require(
        checkpoint.name == "update-001000" and checkpoint.parent.name == "checkpoints", "parent must be update-001000"
    )
    manifest_raw, _ = _stable_regular_bytes(checkpoint / "manifest.json")
    _require(hashlib.sha256(manifest_raw).hexdigest() == expected_manifest_sha256, "parent manifest SHA-256 mismatch")
    strict_manifest = _strict_json_bytes(manifest_raw, source=checkpoint / "manifest.json")
    manifest = load_checkpoint_manifest(checkpoint)
    _require(strict_manifest == manifest, "parent manifest strict parse disagrees")
    _require(manifest.get("kind") == "resumable-libero-training", "parent is not a resumable LIBERO checkpoint")
    _require(manifest.get("run_uuid") == expected_run_uuid, "parent run UUID mismatch")
    _require(manifest.get("source_tree_sha256") == expected_source_tree_sha256, "parent source SHA-256 mismatch")
    _require(manifest.get("execution_profile") == TP1_PARENT_EXECUTION_PROFILE, "parent execution profile differs")
    _require(manifest.get("tensor_parallel_size") == 1, "parent tensor-parallel size is not one")
    _require(manifest.get("physical_batch_size") == 64, "parent physical batch size is not 64")
    _require(manifest.get("serving_batch_size") == SERVING_BATCH_SIZE, "parent serving batch size is not eight")
    _require(manifest.get("expert_batch_isolation") == FUSED_V2_BACKEND, "parent backend is not fused v2")
    _require(manifest.get("run_seed") == 0, "parent run seed is not the frozen seed zero")
    _require(
        manifest.get("trainer_state") == TrainerState(PARENT_UPDATE, 64_000).to_dict(), "parent trainer state differs"
    )
    geometry = manifest.get("execution_geometry")
    _require(isinstance(geometry, Mapping), "parent execution geometry is missing")
    for name, expected in {
        "execution_profile": TP1_PARENT_EXECUTION_PROFILE,
        "tensor_parallel_size": 1,
        "physical_batch_size": 64,
        "serving_batch_size": SERVING_BATCH_SIZE,
        "expert_batch_isolation": FUSED_V2_BACKEND,
    }.items():
        _require(geometry.get(name) == expected, f"parent execution geometry differs for {name}")
    environment = manifest.get("execution_environment")
    _require(isinstance(environment, Mapping), "parent execution environment is missing")
    _require(environment.get("world_size") == 1, "parent execution environment is not TP1")
    _require(environment.get("cuda_runtime") == "12.9", "parent did not use the cu129 runtime family")
    _require(environment.get("torch") == "2.13.0+cu129", "parent did not use the pinned TP1 Torch")
    authenticated_runtime = environment.get("authenticated_runtime")
    _require(isinstance(authenticated_runtime, Mapping), "parent authenticated runtime is missing")
    runtime_environment = authenticated_runtime.get("environment")
    _require(isinstance(runtime_environment, Mapping), "parent process environment is missing")
    _require(
        str(runtime_environment.get("DUO_VLA_TRAIN_VENV", "")).endswith("/venvs/train-single-gpu"),
        "parent did not use the train-single-gpu venv",
    )
    parent_venv = authenticated_runtime.get("train_venv")
    _require(isinstance(parent_venv, Mapping), "parent train-venv identity is missing")
    for field in ("content_inventory_sha256", "root_sha256", "tree_metadata_sha256"):
        _sha256(parent_venv.get(field), f"parent train-venv {field}")
    record = validate_resume_checkpoint(checkpoint.parent.parent, checkpoint)
    _require(record.update == PARENT_UPDATE, "parent run journal tip is not update 1000")
    _require(record.manifest_sha256 == expected_manifest_sha256, "parent run journal manifest SHA-256 differs")
    payload, rank_artifact = _parent_rank_state(checkpoint, manifest)
    parent_config, resolved_config_artifact = _parent_resolved_config(checkpoint, manifest)
    recorded_rank_hashes = manifest.get("training_rank_state_sha256")
    _require(recorded_rank_hashes == [rank_artifact["sha256"]], "parent rank-state hash list differs")
    return checkpoint, manifest, payload, directory, parent_config, resolved_config_artifact


def _validate_dp2_config(config: Mapping[str, Any]) -> None:
    distributed = _exact_keys(
        config.get("distributed"),
        {
            "canonical_plan_partition",
            "data_parallel_size",
            "gradient_reduction",
            "rank_physical_batch_size",
            "strategy",
            "tensor_parallel_size",
            "world_size",
        },
        "DP2 distributed config",
    )
    expected_distributed = {
        "canonical_plan_partition": "contiguous-b8-chunks-by-rank",
        "data_parallel_size": 2,
        "gradient_reduction": "sum_globally_normalized_sse_gradients",
        "rank_physical_batch_size": RANK_PHYSICAL_BATCH_SIZE,
        "strategy": "data_parallel",
        "tensor_parallel_size": MODEL_TENSOR_PARALLEL_SIZE,
        "world_size": DP_WORLD_SIZE,
    }
    _require(dict(distributed) == expected_distributed, "DP2 distributed config differs from the qualified contract")
    optimization = config.get("optimization")
    _require(isinstance(optimization, Mapping), "DP2 optimization config is missing")
    for name, expected in {
        "physical_batch_size": RANK_PHYSICAL_BATCH_SIZE,
        "microbatch_size": RANK_PHYSICAL_BATCH_SIZE,
        "gradient_accumulation_steps": 1,
        "global_batch_size": GLOBAL_BATCH_SIZE,
        "serving_batch_size": SERVING_BATCH_SIZE,
    }.items():
        _require(optimization.get(name) == expected, f"DP2 optimization config differs for {name}")
    model = config.get("model")
    _require(isinstance(model, Mapping), "DP2 model config is missing")
    _require(model.get("tensor_parallel_size") == MODEL_TENSOR_PARALLEL_SIZE, "DP2 model TP size must be one")
    _require(model.get("expert_batch_isolation") == FUSED_V2_BACKEND, "DP2 model backend must be fused v2")
    serving = _exact_keys(
        config.get("serving"),
        {
            "checkpoint_artifact_layout",
            "hardware_identity_scope",
            "physical_batch_size",
            "strategy",
            "tensor_parallel_size",
            "training_gpu_identity_reuse_required",
            "world_size",
        },
        "DP2 serving config",
    )
    expected_serving = {
        "checkpoint_artifact_layout": "consolidated-tp1-trainables",
        "hardware_identity_scope": "serving-runtime-attestation",
        "physical_batch_size": SERVING_BATCH_SIZE,
        "strategy": "single_gpu",
        "tensor_parallel_size": MODEL_TENSOR_PARALLEL_SIZE,
        "training_gpu_identity_reuse_required": False,
        "world_size": 1,
    }
    _require(dict(serving) == expected_serving, "DP2 serving config differs from the qualified contract")
    _require(config.get("execution_profile") == DP2_EXECUTION_PROFILE, "DP2 execution profile differs")
    reproducibility = config.get("reproducibility")
    _require(isinstance(reproducibility, Mapping), "DP2 reproducibility config is missing")
    _require(reproducibility.get("seed") == 0, "DP2 fork must preserve parent seed zero")
    training = config.get("training")
    _require(isinstance(training, Mapping), "DP2 training config is missing")
    _require(training.get("max_cached_files") == 377, "DP2 training cache capacity must be 377")


def derive_dp2_rank_rng_seeds(
    *,
    parent_training_state_sha256: str,
    parent_manifest_sha256: str,
    parent_run_uuid: str,
    child_run_uuid: str,
    rank: int,
) -> dict[str, int]:
    """Derive deterministic, distinct mutable RNG seeds for one DP rank."""

    parent_state = _sha256(parent_training_state_sha256, "parent training-state SHA-256")
    parent_manifest = _sha256(parent_manifest_sha256, "parent manifest SHA-256")
    parent_uuid = _uuid(parent_run_uuid, "parent run UUID")
    child_uuid = _uuid(child_run_uuid, "child run UUID")
    _require(type(rank) is int and 0 <= rank < DP_WORLD_SIZE, "DP rank must be zero or one")
    result: dict[str, int] = {}
    for generator, bits in (("python", 63), ("numpy", 32), ("torch_cpu", 63), ("torch_cuda", 63)):
        identity = [
            DP2_RNG_DOMAIN,
            parent_state,
            parent_manifest,
            parent_uuid,
            child_uuid,
            rank,
            generator,
        ]
        digest = hashlib.blake2b(_canonical_json_bytes(identity), digest_size=8).digest()
        result[generator] = int.from_bytes(digest, "little") & ((1 << bits) - 1)
    return result


def create_dp2_fork_manifest(
    *,
    project_root: str | Path,
    parent_checkpoint: str | Path,
    expected_parent_manifest_sha256: str,
    expected_parent_source_tree_sha256: str,
    expected_parent_run_uuid: str,
    config_path: str | Path,
    child_run_uuid: str,
    child_output_dir: str | Path,
) -> dict[str, Any]:
    """Authenticate a TP1 journal tip and construct a new-run DP2 fork manifest."""

    expected_manifest = _sha256(expected_parent_manifest_sha256, "expected parent manifest SHA-256")
    expected_source = _sha256(expected_parent_source_tree_sha256, "expected parent source SHA-256")
    expected_uuid = _uuid(expected_parent_run_uuid, "expected parent run UUID")
    new_uuid = _uuid(child_run_uuid, "child run UUID")
    _require(new_uuid != expected_uuid, "child run UUID must differ from parent run UUID")
    supplied_child_output = Path(child_output_dir)
    _require(supplied_child_output.is_absolute(), "child output directory must be absolute")
    child_output = supplied_child_output.resolve()
    _require(child_output == supplied_child_output, "child output directory must be a canonical path")
    _require(not child_output.exists(), "child output directory must not exist before the fork launch")
    checkpoint, parent, parent_rank, directory, parent_config, resolved_config_artifact = _canonical_parent_checkpoint(
        parent_checkpoint,
        expected_manifest_sha256=expected_manifest,
        expected_source_tree_sha256=expected_source,
        expected_run_uuid=expected_uuid,
    )
    config_file = Path(config_path).resolve(strict=True)
    root = Path(project_root).resolve(strict=True)
    _require(config_file.is_relative_to(root), "DP2 config must remain inside the child workspace")
    config_raw, _ = _stable_regular_bytes(config_file, single_link=False)
    config = load_resolved_toml(config_file)
    _validate_dp2_config(config)
    semantic_recipe_sha256 = validate_dp2_semantic_recipe_continuity(parent_config, config)
    source_identity = dp2_source_identity(root)
    artifacts = {
        name: _artifact_record(parent["artifacts"][name], f"parent {name} artifact") for name in _PARENT_ARTIFACTS
    }
    _require(
        artifacts["resolved_config"] == resolved_config_artifact,
        "parent resolved-config artifact changed during fork creation",
    )
    creator_path = root / "scripts/create_libero_dp2_fork.py"
    creator_raw, _ = _stable_regular_bytes(creator_path, single_link=False)
    parent_environment = parent["execution_environment"]
    parent_runtime = parent_environment["authenticated_runtime"]
    train_venv_identity = json.loads(json.dumps(parent_runtime["train_venv"], allow_nan=False))
    rank_state_sha256 = artifacts["training_rank_000"]["sha256"]
    rng_seeds = [
        {
            "rank": rank,
            "seeds": derive_dp2_rank_rng_seeds(
                parent_training_state_sha256=rank_state_sha256,
                parent_manifest_sha256=expected_manifest,
                parent_run_uuid=expected_uuid,
                child_run_uuid=new_uuid,
                rank=rank,
            ),
        }
        for rank in range(DP_WORLD_SIZE)
    ]
    return {
        "schema": DP2_FORK_SCHEMA,
        "creator": {
            "path": creator_path.relative_to(root).as_posix(),
            "sha256": hashlib.sha256(creator_raw).hexdigest(),
        },
        "parent": {
            "artifacts": artifacts,
            "checkpoint_identity": {
                "device": directory.st_dev,
                "inode": directory.st_ino,
                "path": str(checkpoint),
            },
            "config_sha256": parent["config_sha256"],
            "execution_profile": TP1_PARENT_EXECUTION_PROFILE,
            "manifest_sha256": expected_manifest,
            "optimizer_parameter_schema_sha256": parent["optimizer_parameter_schema_sha256"],
            "semantic_recipe_sha256": semantic_recipe_sha256,
            "run_uuid": expected_uuid,
            "run_seed": parent["run_seed"],
            "source_tree_sha256": expected_source,
            "train_venv_identity": train_venv_identity,
            "trainer_state": dict(parent["trainer_state"]),
            "training_gpu_uuids": list(parent["execution_environment"]["gpu_uuids"]),
            "update": PARENT_UPDATE,
        },
        "child": {
            "config": {
                "file_sha256": hashlib.sha256(config_raw).hexdigest(),
                "path": config_file.relative_to(root).as_posix(),
                "semantic_recipe_sha256": semantic_recipe_sha256,
                "static_resolved_toml_sha256": canonical_config_sha256(config),
            },
            "execution_profile": DP2_EXECUTION_PROFILE,
            "output_dir": str(child_output),
            "run_uuid": new_uuid,
            "run_seed": 0,
            "serving_topology": dict(config["serving"]),
            "source_identity": source_identity,
            "train_venv_identity": train_venv_identity,
            "training_hardware": {
                "physical_gpu_indices": list(EXPECTED_PHYSICAL_GPU_INDICES),
                "physical_gpu_uuids": list(EXPECTED_TRAINING_GPU_UUIDS),
            },
            "training_runtime": {
                "cuda_runtime": "12.9",
                "environment_family": "train-single-gpu",
                "max_cached_files": 377,
                "torch": "2.13.0+cu129",
                "venv_continuity": "same-content-identity-as-authenticated-tp1-parent",
            },
            "training_topology": dict(config["distributed"]),
        },
        "continuity": {
            "canonical_stream": {
                "canonical_chunk_size": 8,
                "canonical_microsteps_per_update": 8,
                "global_batch_size": GLOBAL_BATCH_SIZE,
                "identity": "run-seed-update-microstep-stateless-v1",
                "next_update": PARENT_UPDATE,
                "partition": "contiguous-b8-chunks-by-rank",
            },
            "model": {
                "interface_sha256": artifacts["interface"]["sha256"],
                "lora_config_sha256": artifacts["lora_config"]["sha256"],
                "lora_weights_sha256": artifacts["lora_weights"]["sha256"],
                "restore": "same-authenticated-tp1-trainables-on-both-dp-ranks",
            },
            "optimizer_scheduler_trainer": {
                "optimizer_parameter_schema_sha256": parent["optimizer_parameter_schema_sha256"],
                "restore": "validate-parent-contract-then-replicate-identical-state-to-both-ranks",
                "scheduler_last_epoch": parent_rank["scheduler"]["last_epoch"],
                "source_rank_state_sha256": rank_state_sha256,
                "trainer_state": dict(parent["trainer_state"]),
            },
            "rng": {
                "derivation": DP2_RNG_DERIVATION,
                "domain": DP2_RNG_DOMAIN,
                "post_fork_stochastic_model_ops": "disabled-or-explicitly-pinned",
                "rank_seeds": rng_seeds,
            },
            "semantic_recipe": {
                "comparison": "exact-after-qualified-batch-topology-normalization",
                "normalized_optimization_fields": sorted(_NORMALIZED_OPTIMIZATION_FIELDS),
                "schema": DP2_SEMANTIC_RECIPE_SCHEMA,
                "sha256": semantic_recipe_sha256,
            },
        },
    }


def validate_dp2_fork_manifest(value: object) -> dict[str, Any]:
    """Validate exact fork fields and all topology / continuity cross-links."""

    manifest = _exact_keys(value, {"child", "continuity", "creator", "parent", "schema"}, "DP2 fork manifest")
    _require(manifest["schema"] == DP2_FORK_SCHEMA, "DP2 fork schema differs")
    creator = _exact_keys(manifest["creator"], {"path", "sha256"}, "DP2 fork creator")
    _require(creator["path"] == "scripts/create_libero_dp2_fork.py", "DP2 fork creator path differs")
    _sha256(creator["sha256"], "DP2 fork creator SHA-256")
    parent = _exact_keys(
        manifest["parent"],
        {
            "artifacts",
            "checkpoint_identity",
            "config_sha256",
            "execution_profile",
            "manifest_sha256",
            "optimizer_parameter_schema_sha256",
            "semantic_recipe_sha256",
            "run_uuid",
            "run_seed",
            "source_tree_sha256",
            "train_venv_identity",
            "trainer_state",
            "training_gpu_uuids",
            "update",
        },
        "DP2 fork parent",
    )
    _require(parent["execution_profile"] == TP1_PARENT_EXECUTION_PROFILE, "fork parent profile differs")
    _require(parent["update"] == PARENT_UPDATE, "fork parent update differs")
    _sha256(parent["manifest_sha256"], "fork parent manifest SHA-256")
    _sha256(parent["source_tree_sha256"], "fork parent source SHA-256")
    _sha256(parent["config_sha256"], "fork parent config SHA-256")
    _sha256(parent["optimizer_parameter_schema_sha256"], "fork parent optimizer schema SHA-256")
    _sha256(parent["semantic_recipe_sha256"], "fork parent semantic recipe SHA-256")
    _uuid(parent["run_uuid"], "fork parent run UUID")
    _require(parent["run_seed"] == 0, "fork parent run seed differs")
    _require(
        parent["trainer_state"] == TrainerState(PARENT_UPDATE, 64_000).to_dict(), "fork parent trainer state differs"
    )
    checkpoint_identity = _exact_keys(
        parent["checkpoint_identity"], {"device", "inode", "path"}, "fork parent checkpoint identity"
    )
    _require(
        type(checkpoint_identity["device"]) is int and checkpoint_identity["device"] >= 0, "parent device is invalid"
    )
    _require(type(checkpoint_identity["inode"]) is int and checkpoint_identity["inode"] > 0, "parent inode is invalid")
    _require(
        isinstance(checkpoint_identity["path"], str) and Path(checkpoint_identity["path"]).is_absolute(),
        "parent path is invalid",
    )
    artifacts = _exact_keys(parent["artifacts"], set(_PARENT_ARTIFACTS), "fork parent artifacts")
    normalized_artifacts = {
        name: _artifact_record(artifacts[name], f"fork parent {name}") for name in _PARENT_ARTIFACTS
    }
    gpu_uuids = parent["training_gpu_uuids"]
    _require(
        isinstance(gpu_uuids, list)
        and len(gpu_uuids) == 1
        and isinstance(gpu_uuids[0], str)
        and _GPU_UUID.fullmatch(gpu_uuids[0]) is not None,
        "fork parent training GPU UUID inventory differs",
    )
    parent_train_venv = parent["train_venv_identity"]
    _require(isinstance(parent_train_venv, Mapping), "fork parent train-venv identity is missing")
    for field in ("content_inventory_sha256", "root_sha256", "tree_metadata_sha256"):
        _sha256(parent_train_venv.get(field), f"fork parent train-venv {field}")
    child = _exact_keys(
        manifest["child"],
        {
            "config",
            "execution_profile",
            "output_dir",
            "run_uuid",
            "run_seed",
            "serving_topology",
            "source_identity",
            "train_venv_identity",
            "training_hardware",
            "training_runtime",
            "training_topology",
        },
        "DP2 fork child",
    )
    _require(child["execution_profile"] == DP2_EXECUTION_PROFILE, "fork child profile differs")
    child_uuid = _uuid(child["run_uuid"], "fork child run UUID")
    _require(child["run_seed"] == parent["run_seed"] == 0, "fork child does not preserve parent run seed")
    _require(child_uuid != parent["run_uuid"], "fork child UUID equals parent UUID")
    _require(
        isinstance(child["output_dir"], str) and Path(child["output_dir"]).is_absolute(), "child output path is invalid"
    )
    config = _exact_keys(
        child["config"],
        {"file_sha256", "path", "semantic_recipe_sha256", "static_resolved_toml_sha256"},
        "fork child config",
    )
    _sha256(config["file_sha256"], "fork child config file SHA-256")
    _sha256(config["semantic_recipe_sha256"], "fork child semantic recipe SHA-256")
    _sha256(config["static_resolved_toml_sha256"], "fork child static resolved TOML SHA-256")
    _require(
        config["semantic_recipe_sha256"] == parent["semantic_recipe_sha256"],
        "fork parent and child semantic recipe digests differ",
    )
    _require(config["path"] == "configs/libero_dp2_fused_v2_b32.toml", "fork child config path differs")
    source = _exact_keys(
        child["source_identity"],
        {
            "files",
            "files_verified",
            "inventory_sha256",
            "root",
            "schema",
            "source_tree_sha256",
            "total_bytes",
        },
        "fork child source identity",
    )
    _require(source["schema"] == DP2_SOURCE_IDENTITY_SCHEMA, "fork child source schema differs")
    _sha256(source["source_tree_sha256"], "fork child source tree SHA-256")
    _sha256(source["inventory_sha256"], "fork child source inventory SHA-256")
    _require(
        isinstance(source["files"], list) and len(source["files"]) == source["files_verified"] > 0,
        "source file count differs",
    )
    _require(type(source["total_bytes"]) is int and source["total_bytes"] > 0, "source total bytes is invalid")
    _require(isinstance(source["root"], str) and Path(source["root"]).is_absolute(), "source root is invalid")
    source_paths = []
    source_bytes = 0
    normalized_source_records: dict[str, dict[str, Any]] = {}
    for index, record in enumerate(source["files"]):
        normalized = _artifact_record(record, f"source file {index}")
        source_paths.append(normalized["path"])
        source_bytes += normalized["bytes"]
        normalized_source_records[normalized["path"]] = normalized
    _require(source_paths == sorted(set(source_paths)), "source file paths are duplicated or unordered")
    _require(source_bytes == source["total_bytes"], "source file byte total differs")
    _require(
        hashlib.sha256(_canonical_json_bytes(source["files"])).hexdigest() == source["inventory_sha256"],
        "source inventory SHA-256 differs",
    )
    _require(creator["path"] in normalized_source_records, "fork creator is absent from child source inventory")
    _require(config["path"] in normalized_source_records, "fork config is absent from child source inventory")
    _require(
        normalized_source_records[creator["path"]]["sha256"] == creator["sha256"],
        "fork creator SHA-256 differs from the child source inventory",
    )
    _require(
        normalized_source_records[config["path"]]["sha256"] == config["file_sha256"],
        "fork config SHA-256 differs from the child source inventory",
    )
    train_venv_identity = child["train_venv_identity"]
    _require(isinstance(train_venv_identity, Mapping), "fork child train-venv identity is missing")
    for field in ("content_inventory_sha256", "root_sha256", "tree_metadata_sha256"):
        _sha256(train_venv_identity.get(field), f"fork child train-venv {field}")
    _require(train_venv_identity == parent_train_venv, "child train-venv identity differs from the TP1 parent")
    hardware = _exact_keys(
        child["training_hardware"],
        {"physical_gpu_indices", "physical_gpu_uuids"},
        "fork child training hardware",
    )
    _require(
        hardware
        == {
            "physical_gpu_indices": list(EXPECTED_PHYSICAL_GPU_INDICES),
            "physical_gpu_uuids": list(EXPECTED_TRAINING_GPU_UUIDS),
        },
        "fork child training hardware differs",
    )
    runtime = _exact_keys(
        child["training_runtime"],
        {"cuda_runtime", "environment_family", "max_cached_files", "torch", "venv_continuity"},
        "fork child training runtime",
    )
    _require(
        runtime
        == {
            "cuda_runtime": "12.9",
            "environment_family": "train-single-gpu",
            "max_cached_files": 377,
            "torch": "2.13.0+cu129",
            "venv_continuity": "same-content-identity-as-authenticated-tp1-parent",
        },
        "fork child training runtime differs",
    )
    topology_config = {
        "execution_profile": child["execution_profile"],
        "distributed": child["training_topology"],
        "optimization": {
            "physical_batch_size": RANK_PHYSICAL_BATCH_SIZE,
            "microbatch_size": RANK_PHYSICAL_BATCH_SIZE,
            "gradient_accumulation_steps": 1,
            "global_batch_size": GLOBAL_BATCH_SIZE,
            "serving_batch_size": SERVING_BATCH_SIZE,
        },
        "model": {"tensor_parallel_size": MODEL_TENSOR_PARALLEL_SIZE, "expert_batch_isolation": FUSED_V2_BACKEND},
        "reproducibility": {"seed": 0},
        "serving": child["serving_topology"],
        "training": {"max_cached_files": child["training_runtime"]["max_cached_files"]},
    }
    _validate_dp2_config(topology_config)
    continuity = _exact_keys(
        manifest["continuity"],
        {"canonical_stream", "model", "optimizer_scheduler_trainer", "rng", "semantic_recipe"},
        "DP2 fork continuity",
    )
    stream = _exact_keys(
        continuity["canonical_stream"],
        {
            "canonical_chunk_size",
            "canonical_microsteps_per_update",
            "global_batch_size",
            "identity",
            "next_update",
            "partition",
        },
        "DP2 fork canonical stream",
    )
    _require(
        dict(stream)
        == {
            "canonical_chunk_size": 8,
            "canonical_microsteps_per_update": 8,
            "global_batch_size": 64,
            "identity": "run-seed-update-microstep-stateless-v1",
            "next_update": PARENT_UPDATE,
            "partition": "contiguous-b8-chunks-by-rank",
        },
        "DP2 canonical stream contract differs",
    )
    model = _exact_keys(
        continuity["model"],
        {"interface_sha256", "lora_config_sha256", "lora_weights_sha256", "restore"},
        "DP2 fork model continuity",
    )
    for field, artifact_name in (
        ("interface_sha256", "interface"),
        ("lora_config_sha256", "lora_config"),
        ("lora_weights_sha256", "lora_weights"),
    ):
        _require(model[field] == normalized_artifacts[artifact_name]["sha256"], f"model continuity differs for {field}")
    _require(model["restore"] == "same-authenticated-tp1-trainables-on-both-dp-ranks", "model restore contract differs")
    optimizer = _exact_keys(
        continuity["optimizer_scheduler_trainer"],
        {
            "optimizer_parameter_schema_sha256",
            "restore",
            "scheduler_last_epoch",
            "source_rank_state_sha256",
            "trainer_state",
        },
        "DP2 fork optimizer continuity",
    )
    _require(
        optimizer["optimizer_parameter_schema_sha256"] == parent["optimizer_parameter_schema_sha256"],
        "optimizer schema cross-link differs",
    )
    _require(
        optimizer["source_rank_state_sha256"] == normalized_artifacts["training_rank_000"]["sha256"],
        "rank-state cross-link differs",
    )
    _require(optimizer["scheduler_last_epoch"] == PARENT_UPDATE, "scheduler continuity update differs")
    _require(optimizer["trainer_state"] == parent["trainer_state"], "trainer-state continuity cross-link differs")
    _require(
        optimizer["restore"] == "validate-parent-contract-then-replicate-identical-state-to-both-ranks",
        "optimizer restore contract differs",
    )
    semantic_recipe = _exact_keys(
        continuity["semantic_recipe"],
        {"comparison", "normalized_optimization_fields", "schema", "sha256"},
        "DP2 semantic recipe continuity",
    )
    _require(
        semantic_recipe
        == {
            "comparison": "exact-after-qualified-batch-topology-normalization",
            "normalized_optimization_fields": sorted(_NORMALIZED_OPTIMIZATION_FIELDS),
            "schema": DP2_SEMANTIC_RECIPE_SCHEMA,
            "sha256": parent["semantic_recipe_sha256"],
        },
        "DP2 semantic recipe continuity differs",
    )
    rng = _exact_keys(
        continuity["rng"],
        {"derivation", "domain", "post_fork_stochastic_model_ops", "rank_seeds"},
        "DP2 fork RNG continuity",
    )
    _require(rng["derivation"] == DP2_RNG_DERIVATION and rng["domain"] == DP2_RNG_DOMAIN, "DP2 RNG derivation differs")
    _require(
        rng["post_fork_stochastic_model_ops"] == "disabled-or-explicitly-pinned",
        "post-fork stochastic-op contract differs",
    )
    _require(
        isinstance(rng["rank_seeds"], list) and len(rng["rank_seeds"]) == DP_WORLD_SIZE,
        "DP2 RNG rank inventory differs",
    )
    for rank, record in enumerate(rng["rank_seeds"]):
        entry = _exact_keys(record, {"rank", "seeds"}, f"DP2 RNG rank {rank}")
        _require(entry["rank"] == rank, "DP2 RNG rank order differs")
        expected = derive_dp2_rank_rng_seeds(
            parent_training_state_sha256=normalized_artifacts["training_rank_000"]["sha256"],
            parent_manifest_sha256=parent["manifest_sha256"],
            parent_run_uuid=parent["run_uuid"],
            child_run_uuid=child_uuid,
            rank=rank,
        )
        _require(entry["seeds"] == expected, f"DP2 RNG seeds differ for rank {rank}")
    _require(rng["rank_seeds"][0]["seeds"] != rng["rank_seeds"][1]["seeds"], "DP2 rank RNG seeds are not distinct")
    return json.loads(json.dumps(manifest, allow_nan=False))


def write_dp2_fork_manifest(path: str | Path, manifest: Mapping[str, Any]) -> str:
    """Publish a sidecar first, then the JSON commit marker, both exclusively."""

    validated = validate_dp2_fork_manifest(manifest)
    output = Path(path)
    _require(output.is_absolute(), "fork manifest output path must be absolute")
    _require(output.name.endswith(".json"), "fork manifest output must be a JSON path")
    output.parent.mkdir(parents=True, exist_ok=True)
    _require(
        output.parent.resolve(strict=True) == output.parent, "fork manifest parent must be canonical and non-symlink"
    )
    payload = _canonical_json_bytes(validated, pretty=True)
    digest = hashlib.sha256(payload).hexdigest()
    sidecar = output.with_name(f"{output.name}.sha256")
    with sidecar.open("xb") as handle:
        handle.write(f"{digest}  {output.name}\n".encode())
        handle.flush()
        os.fsync(handle.fileno())
    try:
        descriptor = os.open(output.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        with output.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        sidecar.unlink(missing_ok=True)
        raise
    descriptor = os.open(output.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return digest


def load_dp2_fork_manifest(path: str | Path, *, expected_sha256: str) -> dict[str, Any]:
    """Load a frozen fork manifest only after checking an external raw digest."""

    expected = _sha256(expected_sha256, "expected fork manifest SHA-256")
    source = Path(path).resolve(strict=True)
    raw, _ = _stable_regular_bytes(source)
    _require(hashlib.sha256(raw).hexdigest() == expected, "fork manifest raw SHA-256 mismatch")
    return validate_dp2_fork_manifest(_strict_json_bytes(raw, source=source))


def load_published_dp2_fork_manifest(path: str | Path) -> tuple[dict[str, Any], str]:
    """Authenticate a committed fork manifest through its adjacent strict sidecar."""

    supplied = Path(path)
    _require(supplied.is_absolute(), "published fork manifest path must be absolute")
    source = supplied.resolve(strict=True)
    _require(source == supplied, "published fork manifest path must be canonical and contain no symlink")
    sidecar = source.with_name(f"{source.name}.sha256")
    sidecar_raw, _ = _stable_regular_bytes(sidecar)
    try:
        line = sidecar_raw.decode("ascii")
    except UnicodeError as exc:
        raise ValueError("fork manifest SHA-256 sidecar is not ASCII") from exc
    match = re.fullmatch(r"([0-9a-f]{64})  ([^/\n]+)\n", line)
    _require(match is not None, "fork manifest SHA-256 sidecar format differs")
    assert match is not None
    _require(match.group(2) == source.name, "fork manifest SHA-256 sidecar names another file")
    digest = match.group(1)
    return load_dp2_fork_manifest(source, expected_sha256=digest), digest


def _reauthenticate_live_fork_manifest(
    manifest: Mapping[str, Any],
    *,
    fork_manifest_path: str | Path,
    fork_manifest_sha256: str,
) -> dict[str, Any]:
    """Re-read the immutable publication and bind it to the in-memory document."""

    expected = _sha256(fork_manifest_sha256, "live fork manifest SHA-256")
    published, observed = load_published_dp2_fork_manifest(fork_manifest_path)
    _require(observed == expected, "live fork manifest sidecar digest changed")
    validated = validate_dp2_fork_manifest(manifest)
    _require(published == validated, "live fork manifest content differs from the authenticated document")
    return validated


def authenticate_dp2_fork_parent(
    manifest: Mapping[str, Any],
    *,
    fork_manifest_path: str | Path,
    fork_manifest_sha256: str,
) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    """Reauthenticate the frozen TP1 journal tip and all checkpoint artifacts."""

    validated = _reauthenticate_live_fork_manifest(
        manifest,
        fork_manifest_path=fork_manifest_path,
        fork_manifest_sha256=fork_manifest_sha256,
    )
    parent = validated["parent"]
    checkpoint, checkpoint_manifest, rank_state, directory, parent_config, resolved_config_artifact = (
        _canonical_parent_checkpoint(
        parent["checkpoint_identity"]["path"],
        expected_manifest_sha256=parent["manifest_sha256"],
        expected_source_tree_sha256=parent["source_tree_sha256"],
        expected_run_uuid=parent["run_uuid"],
        )
    )
    identity = parent["checkpoint_identity"]
    _require(
        directory.st_dev == identity["device"] and directory.st_ino == identity["inode"],
        "parent checkpoint path identity changed after fork creation",
    )
    _require(
        resolved_config_artifact == parent["artifacts"]["resolved_config"],
        "parent resolved-config artifact differs from fork manifest",
    )
    _require(
        dp2_semantic_recipe_sha256(parent_config) == parent["semantic_recipe_sha256"],
        "live parent semantic recipe digest differs from fork manifest",
    )
    return checkpoint, checkpoint_manifest, rank_state


def authenticate_dp2_child_environment(
    manifest: Mapping[str, Any],
    *,
    fork_manifest_path: str | Path,
    fork_manifest_sha256: str,
    project_root: str | Path,
    config_path: str | Path,
    child_output_dir: str | Path,
    child_run_uuid: str,
    live_train_venv_identity: Mapping[str, Any],
    live_training_gpu_uuids: Iterable[str],
) -> dict[str, Any]:
    """Cross-check the fork document against the live child launch inputs."""

    validated = _reauthenticate_live_fork_manifest(
        manifest,
        fork_manifest_path=fork_manifest_path,
        fork_manifest_sha256=fork_manifest_sha256,
    )
    child = validated["child"]
    root = Path(project_root).resolve(strict=True)
    _require(str(root) == child["source_identity"]["root"], "child workspace path differs from fork manifest")
    live_source = dp2_source_identity(root)
    _require(live_source == child["source_identity"], "live child source identity differs from fork manifest")
    config_file = Path(config_path).resolve(strict=True)
    _require(config_file.is_relative_to(root), "live child config escapes the workspace")
    _require(
        config_file.relative_to(root).as_posix() == child["config"]["path"],
        "live child config path differs from fork manifest",
    )
    config_raw, _ = _stable_regular_bytes(config_file, single_link=False)
    _require(
        hashlib.sha256(config_raw).hexdigest() == child["config"]["file_sha256"],
        "live child config file SHA-256 differs from fork manifest",
    )
    static_config = load_resolved_toml(config_file)
    _validate_dp2_config(static_config)
    _require(
        canonical_config_sha256(static_config) == child["config"]["static_resolved_toml_sha256"],
        "live child static resolved TOML SHA-256 differs from fork manifest",
    )
    _require(
        dp2_semantic_recipe_sha256(static_config) == child["config"]["semantic_recipe_sha256"],
        "live child semantic recipe digest differs from fork manifest",
    )
    output = Path(child_output_dir)
    _require(output.is_absolute(), "live child output directory must be absolute")
    _require(str(output.resolve()) == child["output_dir"], "live child output directory differs from fork manifest")
    _require(_uuid(child_run_uuid, "live child run UUID") == child["run_uuid"], "live child run UUID differs")
    _require(
        dict(live_train_venv_identity) == child["train_venv_identity"],
        "live child train-venv identity differs from the authenticated TP1 parent",
    )
    normalized_gpu_uuids = [
        value if value.startswith("GPU-") else f"GPU-{value}"
        for value in live_training_gpu_uuids
        if isinstance(value, str)
    ]
    _require(
        normalized_gpu_uuids == child["training_hardware"]["physical_gpu_uuids"],
        "live DP2 GPU UUID inventory differs from fork manifest",
    )
    return validated


def dp2_fork_run_contract(manifest: Mapping[str, Any], *, fork_manifest_sha256: str) -> dict[str, str]:
    """Return lineage fields every child rank state/checkpoint must repeat."""

    validated = validate_dp2_fork_manifest(manifest)
    digest = _sha256(fork_manifest_sha256, "fork manifest SHA-256")
    parent = validated["parent"]
    child = validated["child"]
    identity = parent["checkpoint_identity"]
    return {
        "execution_profile": DP2_EXECUTION_PROFILE,
        "data_parallel_size": str(DP_WORLD_SIZE),
        "fork_manifest_sha256": digest,
        "fork_parent_config_sha256": parent["config_sha256"],
        "fork_parent_checkpoint_device": str(identity["device"]),
        "fork_parent_checkpoint_inode": str(identity["inode"]),
        "fork_parent_checkpoint_path": identity["path"],
        "fork_parent_manifest_sha256": parent["manifest_sha256"],
        "fork_parent_optimizer_parameter_schema_sha256": parent["optimizer_parameter_schema_sha256"],
        "fork_parent_resolved_config_sha256": parent["artifacts"]["resolved_config"]["sha256"],
        "fork_parent_run_uuid": parent["run_uuid"],
        "fork_parent_source_tree_sha256": parent["source_tree_sha256"],
        "fork_parent_training_rank_state_sha256": parent["artifacts"]["training_rank_000"]["sha256"],
        "fork_parent_update": str(PARENT_UPDATE),
        "fork_parent_venv_root_sha256": parent["train_venv_identity"]["root_sha256"],
        "fork_rng_derivation": DP2_RNG_DERIVATION,
        "fork_schema": DP2_FORK_SCHEMA,
        "fork_semantic_recipe_sha256": parent["semantic_recipe_sha256"],
        "fork_static_resolved_toml_sha256": child["config"]["static_resolved_toml_sha256"],
        "max_cached_files": "377",
        "serving_tensor_parallel_size": "1",
        "serving_world_size": "1",
        "training_gpu_uuids": ",".join(EXPECTED_TRAINING_GPU_UUIDS),
        "training_tensor_parallel_size": "1",
        "training_world_size": str(DP_WORLD_SIZE),
        "run_uuid": child["run_uuid"],
        "source_tree_sha256": child["source_identity"]["source_tree_sha256"],
    }


def has_dp2_checkpoint_lineage(manifest: Mapping[str, Any]) -> bool:
    """Return whether a checkpoint carries any DP2-only ancestry field."""

    return "fork_lineage" in manifest or any(
        name in manifest for name in DP2_FORK_CHECKPOINT_CONTRACT_FIELDS if name.startswith("fork_")
    )


def validate_dp2_checkpoint_lineage(
    manifest: Mapping[str, Any],
) -> tuple[dict[str, str], dict[str, Any]]:
    """Validate the exact DP2 checkpoint ancestry and run-contract cross-links."""

    _require(isinstance(manifest, Mapping), "DP2 checkpoint manifest must be an object")
    expected_topology = {
        "canonical_plan_partition": "contiguous-b8-chunks-by-rank",
        "data_parallel_size": DP_WORLD_SIZE,
        "expert_batch_isolation": FUSED_V2_BACKEND,
        "execution_profile": DP2_EXECUTION_PROFILE,
        "gradient_reduction": "sum_globally_normalized_sse_gradients",
        "max_cached_files": 377,
        "physical_batch_size": RANK_PHYSICAL_BATCH_SIZE,
        "rank_physical_batch_size": RANK_PHYSICAL_BATCH_SIZE,
        "run_seed": 0,
        "serving_batch_size": SERVING_BATCH_SIZE,
        "strategy": "data_parallel",
        "tensor_parallel_size": MODEL_TENSOR_PARALLEL_SIZE,
        "world_size": DP_WORLD_SIZE,
    }
    mismatches = {
        name: {"expected": expected, "observed": manifest.get(name)}
        for name, expected in expected_topology.items()
        if manifest.get(name) != expected
    }
    _require(not mismatches, f"DP2 checkpoint topology differs: {mismatches}")
    _uuid(manifest.get("run_uuid"), "DP2 checkpoint run UUID")
    allocator_peaks = manifest.get("cuda_allocator_peak_memory_bytes_by_rank")
    _require(
        isinstance(allocator_peaks, list)
        and len(allocator_peaks) == DP_WORLD_SIZE
        and all(type(value) is int and value >= 0 for value in allocator_peaks),
        "DP2 checkpoint allocator peak inventory differs",
    )
    _require(
        manifest.get("cuda_allocator_peak_memory_bytes_max") == max(allocator_peaks),
        "DP2 checkpoint allocator peak maximum differs",
    )

    lineage = _exact_keys(manifest.get("fork_lineage"), set(DP2_FORK_LINEAGE_FIELDS), "DP2 checkpoint lineage")
    integer_lineage_fields = {"parent_checkpoint_device", "parent_checkpoint_inode", "parent_update"}
    for name in integer_lineage_fields:
        _require(type(lineage[name]) is int, f"DP2 checkpoint lineage {name} must be an integer")
    _require(lineage["parent_checkpoint_device"] >= 0, "DP2 checkpoint parent device is invalid")
    _require(lineage["parent_checkpoint_inode"] > 0, "DP2 checkpoint parent inode is invalid")
    _require(lineage["parent_update"] == PARENT_UPDATE, "DP2 checkpoint parent update differs")
    for name in DP2_FORK_LINEAGE_FIELDS - integer_lineage_fields:
        _require(isinstance(lineage[name], str), f"DP2 checkpoint lineage {name} must be a string")

    observed_fork_fields = {name for name in manifest if isinstance(name, str) and name.startswith("fork_")}
    expected_fork_fields = {
        name for name in DP2_FORK_CHECKPOINT_CONTRACT_FIELDS if name.startswith("fork_")
    } | {"fork_lineage"}
    _require(
        observed_fork_fields == expected_fork_fields,
        "DP2 checkpoint fork field inventory differs",
    )
    contract = {name: manifest.get(name) for name in DP2_FORK_CHECKPOINT_CONTRACT_FIELDS}
    _require(
        all(isinstance(value, str) for value in contract.values()),
        "DP2 checkpoint fork run-contract fields must all be strings",
    )
    string_contract = {name: str(value) for name, value in contract.items()}
    sha256_fields = {
        "fork_manifest_sha256",
        "fork_parent_config_sha256",
        "fork_parent_manifest_sha256",
        "fork_parent_optimizer_parameter_schema_sha256",
        "fork_parent_resolved_config_sha256",
        "fork_parent_source_tree_sha256",
        "fork_parent_training_rank_state_sha256",
        "fork_parent_venv_root_sha256",
        "fork_semantic_recipe_sha256",
        "fork_static_resolved_toml_sha256",
    }
    for name in sha256_fields:
        _sha256(string_contract[name], f"DP2 checkpoint {name}")
    for name in (
        "fork_manifest_sha256",
        "parent_config_sha256",
        "parent_manifest_sha256",
        "parent_optimizer_parameter_schema_sha256",
        "parent_resolved_config_sha256",
        "parent_source_tree_sha256",
        "semantic_recipe_sha256",
    ):
        _sha256(lineage[name], f"DP2 checkpoint lineage {name}")
    _uuid(lineage["parent_run_uuid"], "DP2 checkpoint parent run UUID")
    _require(lineage["parent_run_uuid"] != manifest["run_uuid"], "DP2 parent and child run UUIDs are equal")

    cross_links = {
        "fork_manifest_sha256": "fork_manifest_sha256",
        "parent_checkpoint": "fork_parent_checkpoint_path",
        "parent_config_sha256": "fork_parent_config_sha256",
        "parent_manifest_sha256": "fork_parent_manifest_sha256",
        "parent_optimizer_parameter_schema_sha256": "fork_parent_optimizer_parameter_schema_sha256",
        "parent_resolved_config_sha256": "fork_parent_resolved_config_sha256",
        "parent_run_uuid": "fork_parent_run_uuid",
        "parent_source_tree_sha256": "fork_parent_source_tree_sha256",
        "semantic_recipe_sha256": "fork_semantic_recipe_sha256",
    }
    cross_link_mismatches = {
        lineage_name: {
            "lineage": lineage[lineage_name],
            "run_contract": string_contract[contract_name],
        }
        for lineage_name, contract_name in cross_links.items()
        if lineage[lineage_name] != string_contract[contract_name]
    }
    _require(not cross_link_mismatches, f"DP2 checkpoint fork lineage cross-links differ: {cross_link_mismatches}")
    _require(
        str(lineage["parent_checkpoint_device"]) == string_contract["fork_parent_checkpoint_device"]
        and str(lineage["parent_checkpoint_inode"]) == string_contract["fork_parent_checkpoint_inode"]
        and str(lineage["parent_update"]) == string_contract["fork_parent_update"],
        "DP2 checkpoint fork identity integer cross-links differ",
    )
    fixed_contract = {
        "fork_parent_update": "1000",
        "fork_rng_derivation": DP2_RNG_DERIVATION,
        "fork_schema": DP2_FORK_SCHEMA,
        "serving_tensor_parallel_size": "1",
        "serving_world_size": "1",
        "training_gpu_uuids": ",".join(EXPECTED_TRAINING_GPU_UUIDS),
        "training_tensor_parallel_size": "1",
        "training_world_size": "2",
    }
    _require(
        all(string_contract[name] == expected for name, expected in fixed_contract.items()),
        "DP2 checkpoint fixed fork contract differs",
    )
    _require(
        string_contract["fork_parent_optimizer_parameter_schema_sha256"]
        == manifest.get("optimizer_parameter_schema_sha256"),
        "DP2 parent/current optimizer parameter schemas differ",
    )
    parent_path = Path(string_contract["fork_parent_checkpoint_path"])
    _require(parent_path.is_absolute(), "DP2 checkpoint fork parent path must be absolute")
    return dict(string_contract), json.loads(json.dumps(dict(lineage), allow_nan=False))


def _activate_dp2_rank_rng(manifest: Mapping[str, Any], *, rank: int, device: torch.device | str | None) -> None:
    seeds = manifest["continuity"]["rng"]["rank_seeds"][rank]["seeds"]
    random.seed(seeds["python"])
    np.random.seed(seeds["numpy"])
    cpu_generator = torch.Generator(device="cpu")
    cpu_generator.manual_seed(seeds["torch_cpu"])
    torch.set_rng_state(cpu_generator.get_state())
    if device is not None:
        resolved = torch.device(device)
        _require(resolved.type == "cuda" and resolved.index == rank, "DP2 CUDA RNG device must equal local rank")
        cuda_generator = torch.Generator(device=resolved)
        cuda_generator.manual_seed(seeds["torch_cuda"])
        torch.cuda.set_rng_state(cuda_generator.get_state(), resolved)


def restore_tp1_fork_training_state(
    manifest: Mapping[str, Any],
    *,
    fork_manifest_path: str | Path,
    fork_manifest_sha256: str,
    rank: int,
    world_size: int,
    optimizer: torch.optim.Optimizer,
    named_parameters: Iterable[tuple[str, torch.nn.Parameter]],
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    child_run_contract: Mapping[str, str],
    device: torch.device | str | None = None,
) -> DP2ForkRestoreResult:
    """Restore authenticated TP1 state into one DP rank at the explicit fork boundary."""

    validated = validate_dp2_fork_manifest(manifest)
    _require(world_size == DP_WORLD_SIZE and 0 <= rank < world_size, "fork restore requires DP world size two")
    required_child_contract = dp2_fork_run_contract(validated, fork_manifest_sha256=fork_manifest_sha256)
    for name, expected in required_child_contract.items():
        _require(child_run_contract.get(name) == expected, f"child run contract differs for {name}")
    parent_checkpoint, parent_manifest, payload = authenticate_dp2_fork_parent(
        validated,
        fork_manifest_path=fork_manifest_path,
        fork_manifest_sha256=fork_manifest_sha256,
    )
    parent_contract = payload["run_contract"]
    _require(parent_contract.get("run_uuid") == parent_manifest["run_uuid"], "parent contract was not authenticated")
    current_inventory = optimizer_parameter_inventory(optimizer, named_parameters)
    _require(
        current_inventory == payload["optimizer_parameter_inventory"],
        "child optimizer parameter inventory differs from authenticated TP1 parent",
    )
    _require(
        optimizer_parameter_inventory_sha256(current_inventory)
        == validated["parent"]["optimizer_parameter_schema_sha256"],
        "child optimizer parameter schema differs from authenticated TP1 parent",
    )
    validate_optimizer_state_dict(payload["optimizer"], current_inventory, expected_update=PARENT_UPDATE)
    trainer_state = TrainerState.from_dict(payload["trainer_state"])
    _require(
        payload["scheduler"].get("last_epoch") == trainer_state.next_update, "parent scheduler/trainer state differ"
    )
    # Mutation begins only after parent, child contract, topology, optimizer, scheduler,
    # model-artifact lineage, and rank RNG derivation have all been authenticated.
    optimizer.load_state_dict(payload["optimizer"])
    scheduler.load_state_dict(payload["scheduler"])
    _activate_dp2_rank_rng(validated, rank=rank, device=device)
    parent = validated["parent"]
    child = validated["child"]
    return DP2ForkRestoreResult(
        trainer_state=trainer_state,
        fork_manifest_sha256=_sha256(fork_manifest_sha256, "fork manifest SHA-256"),
        parent_checkpoint=str(parent_checkpoint),
        parent_checkpoint_device=parent["checkpoint_identity"]["device"],
        parent_checkpoint_inode=parent["checkpoint_identity"]["inode"],
        parent_manifest_sha256=parent["manifest_sha256"],
        parent_run_uuid=parent["run_uuid"],
        parent_run_seed=parent["run_seed"],
        parent_update=parent["update"],
        parent_source_tree_sha256=parent["source_tree_sha256"],
        parent_config_sha256=parent["config_sha256"],
        parent_optimizer_parameter_schema_sha256=parent["optimizer_parameter_schema_sha256"],
        parent_resolved_config_sha256=parent["artifacts"]["resolved_config"]["sha256"],
        child_run_uuid=child["run_uuid"],
        child_run_seed=child["run_seed"],
        child_source_tree_sha256=child["source_identity"]["source_tree_sha256"],
        child_static_resolved_toml_sha256=child["config"]["static_resolved_toml_sha256"],
        semantic_recipe_sha256=parent["semantic_recipe_sha256"],
    )
