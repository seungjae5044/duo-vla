"""Authenticated LIBERO TP1/DP2 topology transitions at 2,500-update boundaries.

This module complements :mod:`duo_vla.dp2_fork`.  The original DP2-v2 fork is
kept frozen for the qualified update-1000 transition.  A topology fork starts a
new run from an immutable journal tip while preserving model, optimizer,
scheduler, trainer, and canonical data-stream progress.  It supports both
TP1->DP2 and DP2->TP1, and can therefore be used for temporary GPU availability
changes without pretending that differently shaped rank checkpoints are an
ordinary resume.
"""

from __future__ import annotations

import hashlib
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
from duo_vla.dp2_fork import (
    DP2_EXECUTION_PROFILE,
    EXPECTED_TRAINING_GPU_UUIDS,
    GLOBAL_BATCH_SIZE,
    RANK_PHYSICAL_BATCH_SIZE,
    TP1_PARENT_EXECUTION_PROFILE,
    _canonical_json_bytes,
    _stable_regular_bytes,
    _strict_json_bytes,
    dp2_semantic_recipe_sha256,
    dp2_source_identity,
    validate_dp2_semantic_recipe_continuity,
)
from duo_vla.run_config import canonical_config_sha256, load_resolved_toml, load_verified_resolved_config
from duo_vla.run_journal import validate_resume_checkpoint
from duo_vla.training import TrainerState
from duo_vla.training_checkpoint import (
    TRAINING_RANK_STATE_SCHEMA,
    optimizer_parameter_inventory,
    optimizer_parameter_inventory_sha256,
    validate_optimizer_state_dict,
)

TOPOLOGY_FORK_SCHEMA = "duo-vla-libero-topology-fork-v1"
TOPOLOGY_FORK_RNG_DERIVATION = "blake2b-parent-rank-states-child-run-rank-v1"
TOPOLOGY_FORK_RNG_DOMAIN = "duo-vla-libero-topology-fork-rng-v1"
TOPOLOGY_FORK_BOUNDARY = 2500
TP1_WORLD_SIZE = 1
DP2_WORLD_SIZE = 2
TP1_PHYSICAL_BATCH_SIZE = 64
MAX_CACHED_FILES = 377

_SHA256 = re.compile(r"[0-9a-f]{64}")
_ARTIFACT_NAMES = ("interface", "lora_config", "lora_weights", "resolved_config")
_LINEAGE_KEYS = frozenset(
    {
        "fork_manifest_sha256",
        "parent_checkpoint",
        "parent_checkpoint_device",
        "parent_checkpoint_inode",
        "parent_config_sha256",
        "parent_manifest_sha256",
        "parent_optimizer_parameter_schema_sha256",
        "parent_rank_state_sha256",
        "parent_run_uuid",
        "parent_source_tree_sha256",
        "parent_update",
        "semantic_recipe_sha256",
    }
)
TOPOLOGY_FORK_CHECKPOINT_FIELDS = frozenset(
    {
        "topology_fork_manifest_sha256",
        "topology_fork_parent_checkpoint_device",
        "topology_fork_parent_checkpoint_inode",
        "topology_fork_parent_checkpoint_path",
        "topology_fork_parent_config_sha256",
        "topology_fork_parent_manifest_sha256",
        "topology_fork_parent_optimizer_parameter_schema_sha256",
        "topology_fork_parent_rank_state_sha256",
        "topology_fork_parent_run_uuid",
        "topology_fork_parent_source_tree_sha256",
        "topology_fork_parent_update",
        "topology_fork_rng_derivation",
        "topology_fork_schema",
        "topology_fork_semantic_recipe_sha256",
        "topology_fork_static_resolved_toml_sha256",
    }
)


@dataclass(frozen=True, slots=True)
class TopologyForkRestoreResult:
    trainer_state: TrainerState
    lineage: dict[str, Any]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


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


def _artifact(value: object, context: str) -> dict[str, Any]:
    _require(isinstance(value, Mapping) and set(value) == {"bytes", "path", "sha256"}, f"{context} differs")
    assert isinstance(value, Mapping)
    relative = value["path"]
    size = value["bytes"]
    _require(
        isinstance(relative, str)
        and relative
        and not Path(relative).is_absolute()
        and ".." not in Path(relative).parts,
        f"{context} path is invalid",
    )
    _require(type(size) is int and size > 0, f"{context} byte size is invalid")
    return {"bytes": size, "path": relative, "sha256": _sha256(value["sha256"], f"{context} SHA-256")}


def _topology_from_config(config: Mapping[str, Any]) -> dict[str, Any]:
    optimization = config.get("optimization")
    _require(isinstance(optimization, Mapping), "topology config has no optimization table")
    profile = config.get("execution_profile")
    distributed = config.get("distributed")
    if isinstance(distributed, Mapping) and distributed.get("strategy") == "data_parallel":
        expected = {
            "data_parallel_size": DP2_WORLD_SIZE,
            "execution_profile": DP2_EXECUTION_PROFILE,
            "gradient_accumulation_steps": 1,
            "physical_batch_size": RANK_PHYSICAL_BATCH_SIZE,
            "world_size": DP2_WORLD_SIZE,
        }
        observed = {
            "data_parallel_size": distributed.get("data_parallel_size"),
            "execution_profile": profile,
            "gradient_accumulation_steps": optimization.get("gradient_accumulation_steps"),
            "physical_batch_size": optimization.get("physical_batch_size"),
            "world_size": distributed.get("world_size"),
        }
        _require(observed == expected, f"unsupported DP2 child topology: {observed}")
        return {"kind": "dp2", **expected}
    expected = {
        "data_parallel_size": 1,
        "execution_profile": TP1_PARENT_EXECUTION_PROFILE,
        "gradient_accumulation_steps": 1,
        "physical_batch_size": TP1_PHYSICAL_BATCH_SIZE,
        "world_size": TP1_WORLD_SIZE,
    }
    observed = {
        "data_parallel_size": 1,
        "execution_profile": profile,
        "gradient_accumulation_steps": optimization.get("gradient_accumulation_steps"),
        "physical_batch_size": optimization.get("physical_batch_size"),
        "world_size": 1,
    }
    _require(observed == expected, f"unsupported TP1 child topology: {observed}")
    return {"kind": "tp1", **expected}


def _topology_from_checkpoint(manifest: Mapping[str, Any]) -> dict[str, Any]:
    geometry = manifest.get("execution_geometry")
    _require(isinstance(geometry, Mapping), "parent checkpoint has no execution geometry")
    profile = manifest.get("execution_profile")
    if profile == DP2_EXECUTION_PROFILE:
        _require(
            manifest.get("world_size") == 2
            and manifest.get("data_parallel_size") == 2
            and manifest.get("physical_batch_size") == 32
            and geometry.get("strategy") == "data_parallel",
            "parent DP2 topology differs",
        )
        return {
            "kind": "dp2",
            "data_parallel_size": 2,
            "execution_profile": profile,
            "physical_batch_size": 32,
            "world_size": 2,
        }
    _require(profile == TP1_PARENT_EXECUTION_PROFILE, "parent execution profile is not qualified TP1 or DP2")
    _require(
        geometry.get("tensor_parallel_size") == 1 and geometry.get("physical_batch_size") == 64,
        "parent TP1 topology differs",
    )
    return {
        "kind": "tp1",
        "data_parallel_size": 1,
        "execution_profile": profile,
        "physical_batch_size": 64,
        "world_size": 1,
    }


def _recipe_config(config: Mapping[str, Any], *, seed: int = 0) -> dict[str, Any]:
    """Apply the sealed runtime-only fields used by the trainer before hashing."""

    normalized = json.loads(json.dumps(dict(config), allow_nan=False))
    normalized["run"] = {"max_cached_files": MAX_CACHED_FILES, "seed": seed, "task": None}
    return normalized


def _exact_nested(left: object, right: object) -> bool:
    if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
        return isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor) and torch.equal(left, right)
    if type(left) is not type(right):
        return False
    if isinstance(left, Mapping):
        return set(left) == set(right) and all(_exact_nested(left[key], right[key]) for key in left)
    if isinstance(left, (list, tuple)):
        return len(left) == len(right) and all(_exact_nested(a, b) for a, b in zip(left, right, strict=True))
    return bool(left == right)


def _rank_payloads(
    checkpoint: Path, manifest: Mapping[str, Any], *, update: int, world_size: int
) -> tuple[list[dict[str, Any]], list[str]]:
    recorded = manifest.get("training_rank_state_sha256")
    _require(
        isinstance(recorded, list)
        and len(recorded) == world_size
        and all(isinstance(item, str) and _SHA256.fullmatch(item) is not None for item in recorded),
        "parent rank-state hash inventory differs",
    )
    payloads: list[dict[str, Any]] = []
    for rank in range(world_size):
        name = f"training_rank_{rank:03d}"
        artifact = _artifact(manifest.get("artifacts", {}).get(name), f"parent {name}")
        _require(artifact["sha256"] == recorded[rank], f"parent {name} hash cross-link differs")
        path = checkpoint / artifact["path"]
        _require(path.resolve(strict=True).is_relative_to(checkpoint), f"parent {name} escapes checkpoint")
        raw, _ = _stable_regular_bytes(path)
        _require(
            len(raw) == artifact["bytes"] and hashlib.sha256(raw).hexdigest() == artifact["sha256"],
            f"parent {name} changed",
        )
        value = torch.load(__import__("io").BytesIO(raw), map_location="cpu", weights_only=True)
        _require(
            isinstance(value, dict) and value.get("schema") == TRAINING_RANK_STATE_SCHEMA,
            f"parent {name} schema differs",
        )
        _require(value.get("rank") == rank and value.get("world_size") == world_size, f"parent {name} topology differs")
        trainer = TrainerState.from_dict(value.get("trainer_state"))
        _require(
            trainer.next_update == update and trainer.examples_seen == update * GLOBAL_BATCH_SIZE,
            f"parent {name} progress differs",
        )
        inventory = value.get("optimizer_parameter_inventory")
        validate_optimizer_state_dict(value.get("optimizer"), inventory, expected_update=update)
        _require(value.get("scheduler", {}).get("last_epoch") == update, f"parent {name} scheduler differs")
        payloads.append(value)
    canonical = payloads[0]
    for rank, value in enumerate(payloads[1:], start=1):
        for field in ("trainer_state", "optimizer", "optimizer_parameter_inventory", "scheduler", "run_contract"):
            _require(_exact_nested(value[field], canonical[field]), f"parent rank {rank} {field} is not replicated")
    return payloads, list(recorded)


def _canonical_parent(
    checkpoint_value: str | Path,
    *,
    expected_manifest_sha256: str,
    expected_source_tree_sha256: str,
    expected_run_uuid: str,
    expected_update: int,
) -> tuple[Path, dict[str, Any], dict[str, Any], list[str], dict[str, Any], os.stat_result]:
    supplied = Path(checkpoint_value)
    _require(supplied.is_absolute(), "parent checkpoint must be absolute")
    checkpoint = supplied.resolve(strict=True)
    _require(checkpoint == supplied and not checkpoint.is_symlink(), "parent checkpoint path must be canonical")
    directory = os.stat(checkpoint, follow_symlinks=False)
    _require(stat.S_ISDIR(directory.st_mode), "parent checkpoint must be a real directory")
    raw, _ = _stable_regular_bytes(checkpoint / "manifest.json")
    _require(hashlib.sha256(raw).hexdigest() == expected_manifest_sha256, "parent manifest SHA-256 differs")
    manifest = load_checkpoint_manifest(checkpoint)
    _require(manifest.get("source_tree_sha256") == expected_source_tree_sha256, "parent source SHA-256 differs")
    _require(manifest.get("run_uuid") == expected_run_uuid, "parent run UUID differs")
    _require(
        type(expected_update) is int and expected_update > 0 and expected_update % TOPOLOGY_FORK_BOUNDARY == 0,
        "topology fork update must be a positive 2500 multiple",
    )
    _require(
        manifest.get("trainer_state") == TrainerState(expected_update, expected_update * GLOBAL_BATCH_SIZE).to_dict(),
        "parent trainer progress differs",
    )
    run_root = checkpoint.parent.parent
    record = validate_resume_checkpoint(run_root, checkpoint)
    _require(
        record.update == expected_update and record.manifest_sha256 == expected_manifest_sha256,
        "parent is not the authenticated journal tip",
    )
    topology = _topology_from_checkpoint(manifest)
    payloads, rank_hashes = _rank_payloads(
        checkpoint, manifest, update=expected_update, world_size=topology["world_size"]
    )
    artifact = _artifact(manifest["artifacts"]["resolved_config"], "parent resolved config")
    parent_config, config_digest = load_verified_resolved_config(checkpoint / artifact["path"])
    _require(config_digest == manifest.get("config_sha256"), "parent resolved config digest differs")
    _require(_topology_from_config(parent_config)["kind"] == topology["kind"], "parent config topology differs")
    return checkpoint, manifest, payloads[0], rank_hashes, parent_config, directory


def derive_topology_rank_rng_seeds(
    *,
    parent_rank_state_sha256: Iterable[str],
    parent_manifest_sha256: str,
    parent_run_uuid: str,
    child_run_uuid: str,
    rank: int,
) -> dict[str, int]:
    hashes = tuple(_sha256(value, "parent rank-state SHA-256") for value in parent_rank_state_sha256)
    _require(hashes and rank >= 0, "topology RNG inputs are invalid")
    material = json.dumps(
        {
            "child_run_uuid": _uuid(child_run_uuid, "child run UUID"),
            "domain": TOPOLOGY_FORK_RNG_DOMAIN,
            "parent_manifest_sha256": _sha256(parent_manifest_sha256, "parent manifest SHA-256"),
            "parent_rank_state_sha256": hashes,
            "parent_run_uuid": _uuid(parent_run_uuid, "parent run UUID"),
            "rank": rank,
        },
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    names = ("python", "numpy", "torch_cpu", "torch_cuda")
    return {
        name: int.from_bytes(hashlib.blake2b(material + b"\0" + name.encode(), digest_size=8).digest(), "big")
        % (1 << (32 if name == "numpy" else 63))
        for name in names
    }


def create_topology_fork_manifest(
    *,
    project_root: str | Path,
    parent_checkpoint: str | Path,
    expected_parent_manifest_sha256: str,
    expected_parent_source_tree_sha256: str,
    expected_parent_run_uuid: str,
    expected_parent_update: int,
    child_config_path: str | Path,
    child_run_uuid: str,
    child_output_dir: str | Path,
    child_physical_gpu_indices: Iterable[int],
) -> dict[str, Any]:
    expected_manifest = _sha256(expected_parent_manifest_sha256, "expected parent manifest SHA-256")
    expected_source = _sha256(expected_parent_source_tree_sha256, "expected parent source SHA-256")
    expected_uuid = _uuid(expected_parent_run_uuid, "expected parent run UUID")
    child_uuid = _uuid(child_run_uuid, "child run UUID")
    _require(child_uuid != expected_uuid, "child and parent run UUIDs must differ")
    child_output = Path(child_output_dir)
    _require(
        child_output.is_absolute() and child_output.resolve() == child_output and not child_output.exists(),
        "child output must be a new canonical absolute path",
    )
    checkpoint, parent, parent_rank, rank_hashes, parent_config, directory = _canonical_parent(
        parent_checkpoint,
        expected_manifest_sha256=expected_manifest,
        expected_source_tree_sha256=expected_source,
        expected_run_uuid=expected_uuid,
        expected_update=expected_parent_update,
    )
    root = Path(project_root).resolve(strict=True)
    config_path = Path(child_config_path).resolve(strict=True)
    _require(config_path.is_relative_to(root), "child config escapes project root")
    config_raw, _ = _stable_regular_bytes(config_path, single_link=False)
    child_config = load_resolved_toml(config_path)
    child_topology = _topology_from_config(child_config)
    child_recipe_config = _recipe_config(child_config, seed=int(parent["run_seed"]))
    semantic_sha = validate_dp2_semantic_recipe_continuity(parent_config, child_recipe_config)
    gpu_indices = tuple(child_physical_gpu_indices)
    expected_indices = (0, 1) if child_topology["kind"] == "dp2" else (0,)
    _require(
        len(gpu_indices) == child_topology["world_size"] and len(set(gpu_indices)) == len(gpu_indices),
        "child GPU index inventory differs",
    )
    _require(all(index in {0, 1} for index in gpu_indices), "only qualified physical GPUs 0 and 1 are supported")
    if child_topology["kind"] == "dp2":
        _require(gpu_indices == expected_indices, "DP2 child must use physical GPUs 0,1 in rank order")
    gpu_uuids = [EXPECTED_TRAINING_GPU_UUIDS[index] for index in gpu_indices]
    artifacts = {name: _artifact(parent["artifacts"][name], f"parent {name}") for name in _ARTIFACT_NAMES}
    train_venv = parent.get("execution_environment", {}).get("authenticated_runtime", {}).get("train_venv")
    _require(
        isinstance(train_venv, Mapping) and isinstance(train_venv.get("root_sha256"), str),
        "parent train venv identity is missing",
    )
    rng = [
        {
            "rank": rank,
            "seeds": derive_topology_rank_rng_seeds(
                parent_rank_state_sha256=rank_hashes,
                parent_manifest_sha256=expected_manifest,
                parent_run_uuid=expected_uuid,
                child_run_uuid=child_uuid,
                rank=rank,
            ),
        }
        for rank in range(child_topology["world_size"])
    ]
    return validate_topology_fork_manifest(
        {
            "schema": TOPOLOGY_FORK_SCHEMA,
            "parent": {
                "artifacts": artifacts,
                "checkpoint_identity": {"device": directory.st_dev, "inode": directory.st_ino, "path": str(checkpoint)},
                "config_sha256": parent["config_sha256"],
                "execution_profile": parent["execution_profile"],
                "manifest_sha256": expected_manifest,
                "optimizer_parameter_schema_sha256": parent["optimizer_parameter_schema_sha256"],
                "rank_state_sha256": rank_hashes,
                "replicated_optimizer_sha256": parent["replicated_optimizer_sha256"],
                "replicated_parameter_sha256": parent["replicated_parameter_sha256"],
                "run_seed": parent["run_seed"],
                "run_uuid": expected_uuid,
                "semantic_recipe_sha256": semantic_sha,
                "source_tree_sha256": expected_source,
                "topology": _topology_from_checkpoint(parent),
                "trainer_state": dict(parent["trainer_state"]),
                "update": expected_parent_update,
            },
            "child": {
                "config": {
                    "file_sha256": hashlib.sha256(config_raw).hexdigest(),
                    "path": config_path.relative_to(root).as_posix(),
                    "semantic_recipe_sha256": semantic_sha,
                    "static_resolved_toml_sha256": canonical_config_sha256(child_config),
                },
                "max_cached_files": MAX_CACHED_FILES,
                "output_dir": str(child_output),
                "physical_gpu_indices": list(gpu_indices),
                "physical_gpu_uuids": gpu_uuids,
                "run_seed": parent["run_seed"],
                "run_uuid": child_uuid,
                "source_identity": dp2_source_identity(root),
                "topology": child_topology,
                "train_venv_identity": json.loads(json.dumps(dict(train_venv), allow_nan=False)),
            },
            "continuity": {
                "canonical_stream": "run-seed-update-microstep-stateless-v1",
                "global_batch_size": GLOBAL_BATCH_SIZE,
                "next_update": expected_parent_update,
                "optimizer_parameter_inventory_sha256": optimizer_parameter_inventory_sha256(
                    parent_rank["optimizer_parameter_inventory"]
                ),
                "rng_derivation": TOPOLOGY_FORK_RNG_DERIVATION,
                "rank_rng": rng,
                "scheduler_last_epoch": parent_rank["scheduler"]["last_epoch"],
                "semantic_recipe_sha256": semantic_sha,
            },
        }
    )


def validate_topology_fork_manifest(value: object) -> dict[str, Any]:
    required = {"schema", "parent", "child", "continuity"}
    _require(isinstance(value, Mapping) and set(value) == required, "topology fork field inventory differs")
    assert isinstance(value, Mapping)
    _require(value["schema"] == TOPOLOGY_FORK_SCHEMA, "unsupported topology fork schema")
    parent = value["parent"]
    child = value["child"]
    continuity = value["continuity"]
    _require(
        isinstance(parent, Mapping) and isinstance(child, Mapping) and isinstance(continuity, Mapping),
        "topology fork sections must be objects",
    )
    _require(
        set(parent)
        == {
            "artifacts",
            "checkpoint_identity",
            "config_sha256",
            "execution_profile",
            "manifest_sha256",
            "optimizer_parameter_schema_sha256",
            "rank_state_sha256",
            "replicated_optimizer_sha256",
            "replicated_parameter_sha256",
            "run_seed",
            "run_uuid",
            "semantic_recipe_sha256",
            "source_tree_sha256",
            "topology",
            "trainer_state",
            "update",
        },
        "topology fork parent field inventory differs",
    )
    _require(
        set(child)
        == {
            "config",
            "max_cached_files",
            "output_dir",
            "physical_gpu_indices",
            "physical_gpu_uuids",
            "run_seed",
            "run_uuid",
            "source_identity",
            "topology",
            "train_venv_identity",
        },
        "topology fork child field inventory differs",
    )
    _require(
        set(continuity)
        == {
            "canonical_stream",
            "global_batch_size",
            "next_update",
            "optimizer_parameter_inventory_sha256",
            "rank_rng",
            "rng_derivation",
            "scheduler_last_epoch",
            "semantic_recipe_sha256",
        },
        "topology fork continuity field inventory differs",
    )
    for name in (
        "manifest_sha256",
        "config_sha256",
        "source_tree_sha256",
        "optimizer_parameter_schema_sha256",
        "replicated_optimizer_sha256",
        "replicated_parameter_sha256",
        "semantic_recipe_sha256",
    ):
        _sha256(parent.get(name), f"parent {name}")
    _uuid(parent.get("run_uuid"), "parent run UUID")
    _uuid(child.get("run_uuid"), "child run UUID")
    _require(parent["run_uuid"] != child["run_uuid"], "parent and child run UUIDs are equal")
    update = parent.get("update")
    _require(
        type(update) is int and update > 0 and update % TOPOLOGY_FORK_BOUNDARY == 0,
        "parent update is not a 2500 boundary",
    )
    _require(
        parent.get("trainer_state") == TrainerState(update, update * GLOBAL_BATCH_SIZE).to_dict(),
        "parent trainer state differs",
    )
    rank_hashes = parent.get("rank_state_sha256")
    parent_topology = parent.get("topology")
    child_topology = child.get("topology")
    _require(isinstance(parent_topology, Mapping) and isinstance(child_topology, Mapping), "fork topology is missing")
    _require(
        parent_topology.get("kind") in {"tp1", "dp2"} and child_topology.get("kind") in {"tp1", "dp2"},
        "fork topology kind differs",
    )
    _require(parent_topology.get("kind") != child_topology.get("kind"), "topology fork must change TP1/DP2 topology")
    _require(
        parent.get("execution_profile") == parent_topology.get("execution_profile"),
        "parent execution profile cross-link differs",
    )
    _require(
        isinstance(rank_hashes, list) and len(rank_hashes) == parent_topology.get("world_size"),
        "parent rank-state count differs",
    )
    for digest in rank_hashes:
        _sha256(digest, "parent rank-state SHA-256")
    config = child.get("config")
    _require(
        isinstance(config, Mapping)
        and set(config) == {"file_sha256", "path", "semantic_recipe_sha256", "static_resolved_toml_sha256"},
        "child config is missing or has extra fields",
    )
    for name in ("file_sha256", "semantic_recipe_sha256", "static_resolved_toml_sha256"):
        _sha256(config.get(name), f"child config {name}")
    _require(
        config.get("semantic_recipe_sha256")
        == parent.get("semantic_recipe_sha256")
        == continuity.get("semantic_recipe_sha256"),
        "semantic recipe cross-link differs",
    )
    _require(
        continuity.get("next_update") == update and continuity.get("scheduler_last_epoch") == update,
        "fork progress cross-link differs",
    )
    _require(
        continuity.get("canonical_stream") == "run-seed-update-microstep-stateless-v1"
        and continuity.get("global_batch_size") == GLOBAL_BATCH_SIZE
        and continuity.get("optimizer_parameter_inventory_sha256") == parent.get("optimizer_parameter_schema_sha256"),
        "fork stream or optimizer continuity differs",
    )
    _require(continuity.get("rng_derivation") == TOPOLOGY_FORK_RNG_DERIVATION, "fork RNG derivation differs")
    rank_rng = continuity.get("rank_rng")
    _require(
        isinstance(rank_rng, list) and len(rank_rng) == child_topology.get("world_size"),
        "child rank RNG inventory differs",
    )
    for rank, record in enumerate(rank_rng):
        _require(isinstance(record, Mapping) and record.get("rank") == rank, "child rank RNG ordering differs")
        seeds = record.get("seeds")
        _require(
            isinstance(seeds, Mapping) and set(seeds) == {"python", "numpy", "torch_cpu", "torch_cuda"},
            "child rank RNG seed fields differ",
        )
        _require(all(type(seed) is int and seed >= 0 for seed in seeds.values()), "child rank RNG seed differs")
        _require(
            dict(seeds)
            == derive_topology_rank_rng_seeds(
                parent_rank_state_sha256=rank_hashes,
                parent_manifest_sha256=parent["manifest_sha256"],
                parent_run_uuid=parent["run_uuid"],
                child_run_uuid=child["run_uuid"],
                rank=rank,
            ),
            "child rank RNG derivation cross-link differs",
        )
    _require(
        child.get("max_cached_files") == MAX_CACHED_FILES and child.get("run_seed") == parent.get("run_seed") == 0,
        "child seed/cache contract differs",
    )
    _require(len(child.get("physical_gpu_indices", [])) == child_topology.get("world_size"), "child GPU count differs")
    _require(
        len(child.get("physical_gpu_uuids", [])) == child_topology.get("world_size"), "child GPU UUID count differs"
    )
    gpu_indices = child["physical_gpu_indices"]
    _require(
        all(type(index) is int and index in {0, 1} for index in gpu_indices)
        and len(set(gpu_indices)) == len(gpu_indices),
        "child GPU index values differ",
    )
    _require(
        child["physical_gpu_uuids"] == [EXPECTED_TRAINING_GPU_UUIDS[index] for index in gpu_indices],
        "child GPU index/UUID cross-links differ",
    )
    _require(
        (child_topology["kind"] == "tp1" and len(gpu_indices) == 1)
        or (child_topology["kind"] == "dp2" and gpu_indices == [0, 1]),
        "child physical GPU topology differs",
    )
    identity = parent.get("checkpoint_identity")
    _require(
        isinstance(identity, Mapping) and set(identity) == {"device", "inode", "path"},
        "parent checkpoint identity differs",
    )
    _require(
        type(identity["device"]) is int
        and identity["device"] >= 0
        and type(identity["inode"]) is int
        and identity["inode"] > 0
        and Path(identity["path"]).is_absolute(),
        "parent checkpoint identity is invalid",
    )
    _require(
        isinstance(child.get("output_dir"), str) and Path(child["output_dir"]).is_absolute(),
        "child output path is invalid",
    )
    source_identity = child.get("source_identity")
    _require(
        isinstance(source_identity, Mapping)
        and _SHA256.fullmatch(str(source_identity.get("source_tree_sha256"))) is not None,
        "child source identity differs",
    )
    train_venv = child.get("train_venv_identity")
    _require(
        isinstance(train_venv, Mapping) and _SHA256.fullmatch(str(train_venv.get("root_sha256"))) is not None,
        "child train venv identity differs",
    )
    return json.loads(json.dumps(dict(value), allow_nan=False))


def write_topology_fork_manifest(path: str | Path, manifest: Mapping[str, Any]) -> str:
    validated = validate_topology_fork_manifest(manifest)
    output = Path(path)
    _require(
        output.is_absolute() and output.name.endswith(".json"), "topology fork output must be an absolute JSON path"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    _require(output.parent.resolve(strict=True) == output.parent, "topology fork parent path must be canonical")
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


def load_published_topology_fork_manifest(path: str | Path) -> tuple[dict[str, Any], str]:
    supplied = Path(path)
    _require(supplied.is_absolute(), "topology fork manifest path must be absolute")
    source = supplied.resolve(strict=True)
    _require(source == supplied, "topology fork manifest path must be canonical")
    raw, _ = _stable_regular_bytes(source)
    sidecar_raw, _ = _stable_regular_bytes(source.with_name(f"{source.name}.sha256"))
    match = re.fullmatch(rb"([0-9a-f]{64})  ([^/\n]+)\n", sidecar_raw)
    _require(match is not None and match.group(2).decode() == source.name, "topology fork sidecar differs")
    assert match is not None
    digest = match.group(1).decode()
    _require(hashlib.sha256(raw).hexdigest() == digest, "topology fork manifest SHA-256 differs")
    return validate_topology_fork_manifest(_strict_json_bytes(raw, source=source)), digest


def _reauthenticate_manifest(manifest: Mapping[str, Any], path: str | Path, digest: str) -> dict[str, Any]:
    expected = _sha256(digest, "topology fork manifest SHA-256")
    published, observed = load_published_topology_fork_manifest(path)
    validated = validate_topology_fork_manifest(manifest)
    _require(observed == expected and published == validated, "published topology fork changed")
    return validated


def authenticate_topology_fork_parent(
    manifest: Mapping[str, Any], *, manifest_path: str | Path, manifest_sha256: str
) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    value = _reauthenticate_manifest(manifest, manifest_path, manifest_sha256)
    parent = value["parent"]
    checkpoint, checkpoint_manifest, payload, rank_hashes, parent_config, directory = _canonical_parent(
        parent["checkpoint_identity"]["path"],
        expected_manifest_sha256=parent["manifest_sha256"],
        expected_source_tree_sha256=parent["source_tree_sha256"],
        expected_run_uuid=parent["run_uuid"],
        expected_update=parent["update"],
    )
    _require(
        directory.st_dev == parent["checkpoint_identity"]["device"]
        and directory.st_ino == parent["checkpoint_identity"]["inode"],
        "parent checkpoint inode changed",
    )
    _require(rank_hashes == parent["rank_state_sha256"], "parent rank-state inventory changed")
    _require(
        dp2_semantic_recipe_sha256(parent_config) == parent["semantic_recipe_sha256"], "parent semantic recipe changed"
    )
    return checkpoint, checkpoint_manifest, payload


def authenticate_topology_child_environment(
    manifest: Mapping[str, Any],
    *,
    manifest_path: str | Path,
    manifest_sha256: str,
    project_root: str | Path,
    config_path: str | Path,
    child_output_dir: str | Path,
    child_run_uuid: str,
    live_train_venv_identity: Mapping[str, Any],
    live_training_gpu_uuids: Iterable[str],
) -> dict[str, Any]:
    value = _reauthenticate_manifest(manifest, manifest_path, manifest_sha256)
    child = value["child"]
    root = Path(project_root).resolve(strict=True)
    _require(dp2_source_identity(root) == child["source_identity"], "live child source identity differs")
    config_file = Path(config_path).resolve(strict=True)
    raw, _ = _stable_regular_bytes(config_file, single_link=False)
    _require(
        config_file.relative_to(root).as_posix() == child["config"]["path"]
        and hashlib.sha256(raw).hexdigest() == child["config"]["file_sha256"],
        "live child config file differs",
    )
    config = load_resolved_toml(config_file)
    _require(_topology_from_config(config) == child["topology"], "live child topology differs")
    recipe_config = _recipe_config(config, seed=int(child["run_seed"]))
    _require(
        canonical_config_sha256(config) == child["config"]["static_resolved_toml_sha256"]
        and dp2_semantic_recipe_sha256(recipe_config) == child["config"]["semantic_recipe_sha256"],
        "live child config semantics differ",
    )
    _require(str(Path(child_output_dir).resolve()) == child["output_dir"], "live child output path differs")
    _require(_uuid(child_run_uuid, "live child UUID") == child["run_uuid"], "live child UUID differs")
    _require(dict(live_train_venv_identity) == child["train_venv_identity"], "live child train venv differs")
    observed = [value if str(value).startswith("GPU-") else f"GPU-{value}" for value in live_training_gpu_uuids]
    _require(observed == child["physical_gpu_uuids"], "live child GPU UUID inventory differs")
    return value


def topology_fork_run_contract(manifest: Mapping[str, Any], *, manifest_sha256: str) -> dict[str, str]:
    value = validate_topology_fork_manifest(manifest)
    parent = value["parent"]
    child = value["child"]
    identity = parent["checkpoint_identity"]
    return {
        "topology_fork_manifest_sha256": _sha256(manifest_sha256, "topology fork manifest SHA-256"),
        "topology_fork_parent_checkpoint_device": str(identity["device"]),
        "topology_fork_parent_checkpoint_inode": str(identity["inode"]),
        "topology_fork_parent_checkpoint_path": identity["path"],
        "topology_fork_parent_config_sha256": parent["config_sha256"],
        "topology_fork_parent_manifest_sha256": parent["manifest_sha256"],
        "topology_fork_parent_optimizer_parameter_schema_sha256": parent["optimizer_parameter_schema_sha256"],
        "topology_fork_parent_rank_state_sha256": ",".join(parent["rank_state_sha256"]),
        "topology_fork_parent_run_uuid": parent["run_uuid"],
        "topology_fork_parent_source_tree_sha256": parent["source_tree_sha256"],
        "topology_fork_parent_update": str(parent["update"]),
        "topology_fork_rng_derivation": TOPOLOGY_FORK_RNG_DERIVATION,
        "topology_fork_schema": TOPOLOGY_FORK_SCHEMA,
        "topology_fork_semantic_recipe_sha256": parent["semantic_recipe_sha256"],
        "topology_fork_static_resolved_toml_sha256": child["config"]["static_resolved_toml_sha256"],
    }


def topology_fork_lineage(manifest: Mapping[str, Any], *, manifest_sha256: str) -> dict[str, Any]:
    value = validate_topology_fork_manifest(manifest)
    parent = value["parent"]
    identity = parent["checkpoint_identity"]
    return {
        "fork_manifest_sha256": _sha256(manifest_sha256, "topology fork manifest SHA-256"),
        "parent_checkpoint": identity["path"],
        "parent_checkpoint_device": identity["device"],
        "parent_checkpoint_inode": identity["inode"],
        "parent_config_sha256": parent["config_sha256"],
        "parent_manifest_sha256": parent["manifest_sha256"],
        "parent_optimizer_parameter_schema_sha256": parent["optimizer_parameter_schema_sha256"],
        "parent_rank_state_sha256": list(parent["rank_state_sha256"]),
        "parent_run_uuid": parent["run_uuid"],
        "parent_source_tree_sha256": parent["source_tree_sha256"],
        "parent_update": parent["update"],
        "semantic_recipe_sha256": parent["semantic_recipe_sha256"],
    }


def has_topology_fork_lineage(manifest: Mapping[str, Any]) -> bool:
    return "topology_fork_lineage" in manifest or any(name in manifest for name in TOPOLOGY_FORK_CHECKPOINT_FIELDS)


def validate_topology_checkpoint_lineage(manifest: Mapping[str, Any]) -> tuple[dict[str, str], dict[str, Any]]:
    _require(isinstance(manifest, Mapping), "topology-fork checkpoint manifest must be an object")
    observed = {name for name in manifest if str(name).startswith("topology_fork_") and name != "topology_fork_lineage"}
    _require(observed == set(TOPOLOGY_FORK_CHECKPOINT_FIELDS), "topology-fork checkpoint field inventory differs")
    fields = {name: manifest.get(name) for name in TOPOLOGY_FORK_CHECKPOINT_FIELDS}
    _require(
        all(isinstance(value, str) for value in fields.values()), "topology-fork checkpoint fields must be strings"
    )
    string_fields = {name: str(value) for name, value in fields.items()}
    for name in (
        "topology_fork_manifest_sha256",
        "topology_fork_parent_config_sha256",
        "topology_fork_parent_manifest_sha256",
        "topology_fork_parent_optimizer_parameter_schema_sha256",
        "topology_fork_parent_source_tree_sha256",
        "topology_fork_semantic_recipe_sha256",
        "topology_fork_static_resolved_toml_sha256",
    ):
        _sha256(string_fields[name], f"topology-fork checkpoint {name}")
    _uuid(string_fields["topology_fork_parent_run_uuid"], "topology-fork parent run UUID")
    _require(
        Path(string_fields["topology_fork_parent_checkpoint_path"]).is_absolute(),
        "topology-fork parent checkpoint path differs",
    )
    lineage = manifest.get("topology_fork_lineage")
    _require(isinstance(lineage, Mapping) and set(lineage) == _LINEAGE_KEYS, "topology-fork checkpoint lineage differs")
    assert isinstance(lineage, Mapping)
    _require(
        type(lineage["parent_update"]) is int
        and lineage["parent_update"] > 0
        and lineage["parent_update"] % TOPOLOGY_FORK_BOUNDARY == 0,
        "topology-fork parent update differs",
    )
    _require(
        isinstance(lineage["parent_rank_state_sha256"], list) and lineage["parent_rank_state_sha256"],
        "topology-fork parent rank hashes differ",
    )
    for digest in lineage["parent_rank_state_sha256"]:
        _sha256(digest, "topology-fork parent rank-state SHA-256")
    expected = {
        "fork_manifest_sha256": string_fields["topology_fork_manifest_sha256"],
        "parent_checkpoint": string_fields["topology_fork_parent_checkpoint_path"],
        "parent_checkpoint_device": int(string_fields["topology_fork_parent_checkpoint_device"]),
        "parent_checkpoint_inode": int(string_fields["topology_fork_parent_checkpoint_inode"]),
        "parent_config_sha256": string_fields["topology_fork_parent_config_sha256"],
        "parent_manifest_sha256": string_fields["topology_fork_parent_manifest_sha256"],
        "parent_optimizer_parameter_schema_sha256": string_fields[
            "topology_fork_parent_optimizer_parameter_schema_sha256"
        ],
        "parent_rank_state_sha256": string_fields["topology_fork_parent_rank_state_sha256"].split(","),
        "parent_run_uuid": string_fields["topology_fork_parent_run_uuid"],
        "parent_source_tree_sha256": string_fields["topology_fork_parent_source_tree_sha256"],
        "parent_update": int(string_fields["topology_fork_parent_update"]),
        "semantic_recipe_sha256": string_fields["topology_fork_semantic_recipe_sha256"],
    }
    _require(dict(lineage) == expected, "topology-fork lineage cross-links differ")
    _require(
        string_fields["topology_fork_schema"] == TOPOLOGY_FORK_SCHEMA
        and string_fields["topology_fork_rng_derivation"] == TOPOLOGY_FORK_RNG_DERIVATION,
        "topology-fork fixed contract differs",
    )
    _require(
        string_fields["topology_fork_parent_optimizer_parameter_schema_sha256"]
        == manifest.get("optimizer_parameter_schema_sha256"),
        "topology-fork optimizer schema changed",
    )
    return string_fields, json.loads(json.dumps(expected, allow_nan=False))


def _activate_rng(manifest: Mapping[str, Any], *, rank: int, device: torch.device | str | None) -> None:
    seeds = manifest["continuity"]["rank_rng"][rank]["seeds"]
    random.seed(seeds["python"])
    np.random.seed(seeds["numpy"])
    cpu = torch.Generator(device="cpu")
    cpu.manual_seed(seeds["torch_cpu"])
    torch.set_rng_state(cpu.get_state())
    if device is not None:
        resolved = torch.device(device)
        _require(resolved.type == "cuda" and resolved.index == rank, "child CUDA device must equal local rank")
        cuda = torch.Generator(device=resolved)
        cuda.manual_seed(seeds["torch_cuda"])
        torch.cuda.set_rng_state(cuda.get_state(), resolved)


def restore_topology_fork_training_state(
    manifest: Mapping[str, Any],
    *,
    manifest_path: str | Path,
    manifest_sha256: str,
    rank: int,
    world_size: int,
    optimizer: torch.optim.Optimizer,
    named_parameters: Iterable[tuple[str, torch.nn.Parameter]],
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    child_run_contract: Mapping[str, str],
    device: torch.device | str | None = None,
) -> TopologyForkRestoreResult:
    value = validate_topology_fork_manifest(manifest)
    _require(
        world_size == value["child"]["topology"]["world_size"] and 0 <= rank < world_size,
        "child topology differs at restore",
    )
    required = topology_fork_run_contract(value, manifest_sha256=manifest_sha256)
    for name, expected in required.items():
        _require(child_run_contract.get(name) == expected, f"child topology fork contract differs for {name}")
    _, parent_manifest, payload = authenticate_topology_fork_parent(
        value, manifest_path=manifest_path, manifest_sha256=manifest_sha256
    )
    current_inventory = optimizer_parameter_inventory(optimizer, named_parameters)
    _require(
        current_inventory == payload["optimizer_parameter_inventory"], "child optimizer inventory differs from parent"
    )
    _require(
        optimizer_parameter_inventory_sha256(current_inventory) == parent_manifest["optimizer_parameter_schema_sha256"],
        "child optimizer schema differs from parent",
    )
    update = value["parent"]["update"]
    validate_optimizer_state_dict(payload["optimizer"], current_inventory, expected_update=update)
    trainer_state = TrainerState.from_dict(payload["trainer_state"])
    optimizer.load_state_dict(payload["optimizer"])
    scheduler.load_state_dict(payload["scheduler"])
    _activate_rng(value, rank=rank, device=device)
    return TopologyForkRestoreResult(
        trainer_state=trainer_state,
        lineage=topology_fork_lineage(value, manifest_sha256=manifest_sha256),
    )
