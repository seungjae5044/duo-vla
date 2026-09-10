#!/usr/bin/env python3
"""Resumable TP1, TP2, or DP2/TP1 Duo-VLA trainer for pinned CALVIN ABC-to-D data."""

# ruff: noqa: E402 -- authenticate project sources before importing project code.

from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import random
import re
import site
import stat
import sys
import time
from collections.abc import Sequence
from datetime import timedelta
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from PIL import Image
from transformers import AutoProcessor


def _inventory_project_source_root() -> Path:
    source_root = Path(__file__).resolve().parents[1] / "src"
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
    resolved_entries = [str(Path(entry or os.getcwd()).resolve()) for entry in sys.path]
    if source_text not in resolved_entries:
        raise RuntimeError("sealed editable install does not expose the canonical project src import root")
    sys.path[:] = [source_text] + [
        entry for entry in sys.path if str(Path(entry or os.getcwd()).resolve()) != source_text
    ]
    return source_root


_PROJECT_SOURCE_ROOT = _inventory_project_source_root()


def _validate_project_module_origins(required_modules: set[str]) -> dict[str, str]:
    package_root = (_PROJECT_SOURCE_ROOT / "duo_vla").resolve(strict=True)
    loaded = {name: module for name, module in sys.modules.items() if name == "duo_vla" or name.startswith("duo_vla.")}
    missing = sorted(required_modules - set(loaded))
    if missing:
        raise RuntimeError(f"required checkout modules are not loaded: {missing}")
    origins: dict[str, str] = {}
    for name, module in sorted(loaded.items()):
        origin = getattr(getattr(module, "__spec__", None), "origin", None)
        module_file = getattr(module, "__file__", None)
        if not isinstance(origin, str) or not isinstance(module_file, str):
            raise RuntimeError(f"checkout module has no file origin: {name}")
        resolved_origin = Path(origin).resolve(strict=True)
        resolved_file = Path(module_file).resolve(strict=True)
        if resolved_origin != resolved_file or not resolved_file.is_relative_to(package_root):
            raise RuntimeError(f"checkout module origin escapes authenticated source root: {name}")
        origins[name] = str(resolved_file)
    return origins


from duo_vla.action_interface import ActionInputProjector, VelocityHead
from duo_vla.backbones.diffusion_gemma import (
    DiffusionGemmaActionDecoder,
    apply_decoder_attention_lora,
    decoder_lora_parameter_partition,
    encode_diffusion_gemma_prefix,
)
from duo_vla.backbones.loading import (
    DEFAULT_DIFFUSION_GEMMA_SPEC,
    load_diffusion_gemma_bf16_tp,
    validate_decoder_attention_lora_adapter_config,
    validate_decoder_attention_lora_weights,
)
from duo_vla.backbones.sample_isolated_experts import (
    GROUPED_MM_EXPERTS_IMPLEMENTATION,
)
from duo_vla.calvin_execution import (
    EXTENDED_GEOMETRY_FIELDS,
    OPTIMIZED_SOURCE_PATHS,
    SUPPORTED_BATCHES,
    canonical_training_pair,
    execution_from_config,
    expert_backend_functions,
    group_plans,
    reduce_scalar,
    sum_gradients_,
)
from duo_vla.calvin_execution import (
    execution_geometry as calvin_execution_geometry,
)
from duo_vla.calvin_source_identity import calvin_source_tree_sha256
from duo_vla.checkpointing import (
    load_checkpoint_manifest,
    load_interface_state_dict,
    load_lora_checkpoint,
    save_trainable_checkpoint,
)
from duo_vla.config import ActionInterfaceConfig
from duo_vla.data.calvin import (
    CalvinAnchor,
    CalvinAnnotation,
    CalvinNpzDataset,
    CalvinTaskUniformAnchorSampler,
    make_calvin_episode_split,
)
from duo_vla.data.calvin_archive import (
    CALVIN_ARCHIVE_READER_SCHEMA,
    CALVIN_INDEX_NAME,
    CALVIN_MEMBER_INDEX_SCHEMA,
    OFFICIAL_CENTRAL_DIRECTORY_SHA256,
)
from duo_vla.data.calvin_batching import CalvinBatch, coalesce_calvin_batches, collate_calvin_samples
from duo_vla.data.calvin_stats import (
    CALVIN_ABC_D_ARCHIVE_BYTES,
    CALVIN_ABC_D_ARCHIVE_SHA256,
    CALVIN_CRITICAL_TRAIN_METADATA,
    CALVIN_DATASET_MANIFEST_SCHEMA,
    CALVIN_STATS_SCHEMA,
    CALVIN_STORAGE_MODE_ARCHIVE_DIRECT,
    AuthenticatedCalvinDatasetGeneration,
    authenticate_calvin_dataset_generation,
    load_calvin_state_normalizer,
    verify_calvin_dataset_generation,
)
from duo_vla.hf_snapshot import verify_huggingface_snapshot
from duo_vla.modeling import DuoVLADenoiser
from duo_vla.optimization import (
    OptimizationConfig,
    assert_replicated_parameter_values,
    assert_replicated_tensor,
    clip_tensor_parallel_grad_norm_,
    create_optimizer_and_scheduler,
    finite_gradient_audit,
    learning_rate_scale,
)
from duo_vla.policy_contract import PolicyContract, policy_contract_from_config, validate_manifest_policy_contract
from duo_vla.prefix_geometry import (
    CameraGeometry,
    SnapshotTreeIdentity,
    apply_fixed_prefix_chat_template,
    instruction_inventory_sha256,
    load_prefix_geometry_contract,
)
from duo_vla.run_config import (
    canonical_config_sha256,
    load_resolved_toml,
    load_verified_resolved_config,
    save_resolved_config,
)
from duo_vla.run_journal import (
    apply_checkpoint_retention,
    create_run_journal,
    load_run_journal,
    make_checkpoint_retention_contract,
    quarantine_uncommitted_training_artifacts,
    reconcile_metrics_jsonl,
    record_latest_checkpoint,
    validate_resume_checkpoint,
)
from duo_vla.runtime_integrity import (
    canonical_visible_cuda_world_size,
    content_address_train_venv,
    static_environment_identity,
    validate_torchrun_rank_environment,
)
from duo_vla.training import TrainerState, make_microbatch_plan, make_update_plan, masked_element_count, masked_sse
from duo_vla.training_checkpoint import (
    capture_rng_state,
    file_sha256,
    load_training_rank_state,
    optimizer_parameter_schema_sha256,
    restore_rng_state,
    save_training_rank_state,
    validate_training_progress,
)

_validate_project_module_origins({"duo_vla", "duo_vla.runtime_integrity"})

CALVIN_PROTOCOL = "duovla-calvin-abc-to-d-v1"
CALVIN_EXPECTED_SCENES = ("calvin_scene_A", "calvin_scene_B", "calvin_scene_C")
# Authenticating the official archive reads and hashes more than 500 GiB.  A
# two-hour ceiling permits roughly 74 MiB/s sequential throughput while staying
# bounded.  It applies only to the authentication Gloo subgroup; the default
# NCCL timeout remains unchanged so it can continue to detect training hangs.
CALVIN_DATASET_AUTHENTICATION_TIMEOUT = timedelta(hours=2)
_INTEGER_MANIFEST_RUN_CONTRACT_FIELDS = frozenset(
    {
        "archive_bytes",
        "fixed_physical_prefix_width",
        "member_index_bytes",
        "physical_batch_size",
        "tensor_parallel_size",
        "data_parallel_size",
        "serving_batch_size",
        "global_batch_size",
    }
)
TRAIN_LOCK_SHA256 = "0b1fb188747ee99224078b3c40975ca7e6f8e082e22d2860f9b50ee679a67c46"
SINGLE_GPU_TRAIN_LOCK_SHA256 = "bea1c95bf906ba68bb8065f5329f7f65a67f7bbbca472d870e06bc208333b0c1"
EXPECTED_TRAIN_PYTHON = "3.11.15"
EXPECTED_TRAIN_PACKAGES = {
    "accelerate": "1.14.0",
    "huggingface-hub": "1.29.0",
    "numpy": "2.4.6",
    "peft": "0.20.0",
    "pillow": "12.3.0",
    "pyarrow": "20.0.0",
    "safetensors": "0.8.0",
    "tokenizers": "0.22.2",
    "torch": "2.13.0+cu126",
    "torchvision": "0.28.0+cu126",
    "transformers": "5.15.0",
}
SINGLE_GPU_EXPECTED_TRAIN_PACKAGES = {
    **EXPECTED_TRAIN_PACKAGES,
    "torch": "2.13.0+cu129",
    "torchvision": "0.28.0+cu129",
}
REQUIRED_TRAIN_ENVIRONMENT = {
    "BLIS_NUM_THREADS": "1",
    "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
    "CUDA_VISIBLE_DEVICES": "0,1",
    "HF_HUB_DISABLE_PROGRESS_BARS": "1",
    "HF_HUB_OFFLINE": "1",
    "HOME": "/root",
    "LANG": "C.UTF-8",
    "LC_ALL": "C.UTF-8",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "OMP_DYNAMIC": "FALSE",
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "PATH": "/usr/bin:/bin",
    "PYTHONNOUSERSITE": "1",
    "PYTHONPYCACHEPREFIX": "/dev/null",
    "PYTHONSAFEPATH": "1",
    "PYTHONDONTWRITEBYTECODE": "1",
    "RAYON_NUM_THREADS": "1",
    "TOKENIZERS_PARALLELISM": "false",
    "TORCH_NCCL_ASYNC_ERROR_HANDLING": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "TZ": "UTC",
    "VECLIB_MAXIMUM_THREADS": "1",
}
_RELEVANT_TRAIN_ENVIRONMENT_PREFIXES = (
    "BLIS_",
    "CUBLAS_",
    "CUDA_",
    "CUDNN_",
    "GCONV_PATH",
    "GLIBC_",
    "GOMP_",
    "KMP_",
    "LD_",
    "LOCPATH",
    "MALLOC_",
    "MKL_",
    "NCCL_",
    "NIX_",
    "NVIDIA_",
    "NUMEXPR_",
    "OMP_",
    "OPENBLAS_",
    "PYTORCH_",
    "RAYON_",
    "TORCH_",
    "VECLIB_",
)
CALVIN_CAMERA_SHAPES = {"rgb_static": [200, 200, 3], "rgb_gripper": [84, 84, 3]}
CALVIN_STATE_ADAPTER = "robot_obs[0:7]+robot_obs[14:15]"
CALVIN_ACTION_ADAPTER = "identity_official_scaled_rel_actions"
EXPERT_BATCH_ISOLATION = "sample_isolated_grouped_mm_v1"
PHYSICAL_BATCH_SIZE = 8
PINNED_CALVIN_SOURCE_REVISIONS = {
    "calvin": "fa03f01f19c65920e18cf37398a9ce859274af76",
    "calvin_env": "1431a46bd36bde5903fb6345e68b5ccc30def666",
    "tacto": "dd53360d9a8c186f0d6439372ec0be0fa5e21731",
}

_CALVIN_SOURCE_TREE_HASH_MAGIC = b"duo-vla-calvin-training-source-tree\x00v3\x00"
_CALVIN_SOURCE_EXPLICIT_RELATIVE_PATHS = (
    "configs/base.toml",
    "configs/calvin_abc_to_d.toml",
    "configs/calvin_abc_to_d_direct.toml",
    "scripts/run_calvin_train.sh",
    "scripts/train_calvin.py",
    "scripts/calvin/calvin_bridge.py",
    "scripts/calvin/prepare_archive_direct.py",
    "scripts/calvin/revisions.env",
    "scripts/calvin/run_policy_server.sh",
    "scripts/calvin/serve_policy.py",
    "pyproject.toml",
    "uv.lock",
)
_CALVIN_SINGLE_GPU_SOURCE_EXPLICIT_RELATIVE_PATHS = (
    *_CALVIN_SOURCE_EXPLICIT_RELATIVE_PATHS,
    *OPTIMIZED_SOURCE_PATHS,
    "configs/calvin_abc_to_d_single_gpu.toml",
    "configs/calvin_abc_to_d_direct_single_gpu.toml",
    "envs/train-single-gpu/pyproject.toml",
    "envs/train-single-gpu/uv.lock",
    "scripts/bootstrap_train_single_gpu_env.sh",
    "scripts/run_calvin_train_single_gpu.sh",
    "scripts/calvin/run_policy_server_single_gpu.sh",
)


def _train_runtime_pins(world_size: int) -> tuple[Path, str, dict[str, str], str]:
    if world_size == 1:
        return (
            Path("envs/train-single-gpu/uv.lock"),
            SINGLE_GPU_TRAIN_LOCK_SHA256,
            SINGLE_GPU_EXPECTED_TRAIN_PACKAGES,
            "12.9",
        )
    if world_size == 2:
        return Path("uv.lock"), TRAIN_LOCK_SHA256, EXPECTED_TRAIN_PACKAGES, "12.6"
    raise ValueError("training world size must be one or two")


_CALVIN_RUN_CONTRACT_FIELDS = frozenset(
    {
        "action_adapter",
        "archive_bytes",
        "archive_sha256",
        "calvin_env_revision",
        "calvin_revision",
        "calvin_source_revisions_sha256",
        "calvin_tacto_revision",
        "camera_shapes_sha256",
        "central_directory_sha256",
        "config_sha256",
        "dataset_manifest_file_sha256",
        "dataset_manifest_schema",
        "dataset_manifest_sha256",
        "execution_environment_sha256",
        "expert_batch_isolation",
        "experts_implementation",
        "fixed_physical_prefix_width",
        "member_index_bytes",
        "member_index_path",
        "member_index_schema",
        "member_index_sha256",
        "member_inventory_sha256",
        "metadata_sha256",
        "model_revision",
        "model_tree_sha256",
        "normalization_sha256",
        "optimizer_parameter_schema_sha256",
        "physical_batch_size",
        "policy_contract_sha256",
        "prefix_geometry_content_sha256",
        "protocol",
        "reader_schema",
        "run_uuid",
        "source_tree_sha256",
        "split_sha256",
        "state_adapter",
        "storage_identity_sha256",
        "storage_mode",
        "train_episode_sha256",
        "training_instruction_inventory_sha256",
        "validation_episode_sha256",
    }
)
_CALVIN_SINGLE_GPU_RUN_CONTRACT_FIELDS = _CALVIN_RUN_CONTRACT_FIELDS | {
    "execution_profile",
    "tensor_parallel_size",
}

_CALVIN_NORMALIZATION_DATASET_FIELDS = frozenset(
    {
        "archive_bytes",
        "archive_sha256",
        "central_directory_sha256",
        "dataset_manifest_file_sha256",
        "dataset_manifest_schema",
        "dataset_manifest_sha256",
        "member_index",
        "member_inventory_sha256",
        "metadata_files",
        "metadata_sha256",
        "name",
        "reader_schema",
        "split",
        "storage_identity_sha256",
        "storage_mode",
    }
)
_CALVIN_MEMBER_INDEX_FIELDS = frozenset({"bytes", "path", "schema", "sha256"})
_CALVIN_IDENTITY_FIELDS = frozenset(
    {
        "action_adapter",
        "archive_bytes",
        "archive_sha256",
        "calvin_source_revisions",
        "calvin_source_revisions_sha256",
        "camera_shapes",
        "camera_shapes_sha256",
        "central_directory_sha256",
        "dataset_manifest_file_sha256",
        "dataset_manifest_schema",
        "dataset_manifest_sha256",
        "member_index",
        "member_inventory_sha256",
        "metadata_files",
        "metadata_sha256",
        "normalization_sha256",
        "protocol",
        "reader_schema",
        "split",
        "split_sha256",
        "state_adapter",
        "storage_identity_sha256",
        "storage_mode",
    }
)
_CALVIN_STORAGE_RUN_CONTRACT_FIELDS = frozenset(
    {
        "archive_bytes",
        "archive_sha256",
        "central_directory_sha256",
        "dataset_manifest_file_sha256",
        "dataset_manifest_schema",
        "dataset_manifest_sha256",
        "member_index_bytes",
        "member_index_path",
        "member_index_schema",
        "member_index_sha256",
        "member_inventory_sha256",
        "metadata_sha256",
        "reader_schema",
        "storage_identity_sha256",
        "storage_mode",
    }
)


def _source_tree_sha256(root: Path, *, single_gpu: bool = False) -> str:
    """Hash an injectively framed, exact CALVIN training source inventory."""

    return calvin_source_tree_sha256(
        root,
        explicit_relative_paths=(
            _CALVIN_SINGLE_GPU_SOURCE_EXPLICIT_RELATIVE_PATHS if single_gpu else _CALVIN_SOURCE_EXPLICIT_RELATIVE_PATHS
        ),
        magic=_CALVIN_SOURCE_TREE_HASH_MAGIC,
    )


def _validate_resume_manifest_run_contract(
    manifest: dict[str, Any],
    run_contract: dict[str, str],
) -> None:
    """Compare the string rank-state contract with typed manifest fields."""

    for key, expected in run_contract.items():
        observed = manifest.get(key)
        if key in _INTEGER_MANIFEST_RUN_CONTRACT_FIELDS:
            matches = type(observed) is int and str(observed) == expected
        else:
            matches = observed == expected
        if not matches:
            raise ValueError(f"resume checkpoint run contract mismatch for {key}")


def _calvin_source_revisions(root: Path) -> dict[str, str]:
    path = root / "scripts/calvin/revisions.env"
    if not path.is_file():
        raise FileNotFoundError(f"pinned CALVIN revision file is missing: {path}")
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line and not line.startswith("#"):
            key, separator, value = line.partition("=")
            if not separator or not key or not value:
                raise ValueError(f"invalid CALVIN revision entry: {line!r}")
            values[key] = value
    observed = {
        "calvin": values.get("CALVIN_REVISION"),
        "calvin_env": values.get("CALVIN_ENV_REVISION"),
        "tacto": values.get("CALVIN_TACTO_REVISION"),
    }
    if observed != PINNED_CALVIN_SOURCE_REVISIONS:
        raise ValueError(
            f"CALVIN source revisions differ from the pinned training contract: "
            f"expected={PINNED_CALVIN_SOURCE_REVISIONS}, observed={observed}"
        )
    return dict(PINNED_CALVIN_SOURCE_REVISIONS)


def _indices_sha256(indices: tuple[int, ...]) -> str:
    return hashlib.sha256(",".join(map(str, indices)).encode()).hexdigest()


def _episode_indices_from_stats(
    stats_manifest: dict[str, Any],
    *,
    episode_count: int,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    split = stats_manifest.get("split")
    if not isinstance(split, dict):
        raise ValueError("CALVIN normalization artifact has no split identity")

    def parse(name: str) -> tuple[int, ...]:
        raw = split.get(name)
        invalid_value = isinstance(raw, list) and any(
            isinstance(value, bool) or not isinstance(value, int) for value in raw
        )
        if not isinstance(raw, list) or not raw or invalid_value:
            raise ValueError(f"CALVIN normalization artifact {name} must be a non-empty integer list")
        result = tuple(raw)
        if len(set(result)) != len(result) or tuple(sorted(result)) != result:
            raise ValueError(f"CALVIN normalization artifact {name} must be sorted and distinct")
        if result[0] < 0 or result[-1] >= episode_count:
            raise ValueError(f"CALVIN normalization artifact {name} is outside the training dataset")
        return result

    train = parse("train_episode_indices")
    validation = parse("validation_episode_indices")
    if set(train) & set(validation) or set(train) | set(validation) != set(range(episode_count)):
        raise ValueError("CALVIN normalization split must be a disjoint partition of all training episodes")
    expected_hashes = {
        "train_episode_sha256": _indices_sha256(train),
        "validation_episode_sha256": _indices_sha256(validation),
    }
    mismatches = [name for name, expected in expected_hashes.items() if split.get(name) != expected]
    if mismatches:
        raise ValueError(f"CALVIN normalization split hash mismatch: {mismatches}")
    return train, validation


def _validate_calvin_split_recipe(
    stats_manifest: dict[str, Any],
    dataset: CalvinNpzDataset,
    training_config: dict[str, Any],
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Require the artifact partition to be the config-declared deterministic split."""

    train, validation = _episode_indices_from_stats(
        stats_manifest,
        episode_count=len(dataset.episodes),
    )
    split = stats_manifest.get("split")
    assert isinstance(split, dict)
    configured_seed = int(training_config["validation_split_seed"])
    configured_fraction = float(training_config["validation_episode_fraction"])
    if split.get("seed") != configured_seed or split.get("validation_fraction") != configured_fraction:
        raise ValueError(
            "CALVIN normalization split recipe differs from the resolved training config: "
            f"artifact_seed={split.get('seed')}, config_seed={configured_seed}, "
            f"artifact_fraction={split.get('validation_fraction')}, config_fraction={configured_fraction}"
        )
    expected = make_calvin_episode_split(
        dataset.episodes,
        dataset.annotations,
        validation_fraction=configured_fraction,
        seed=configured_seed,
    )
    if train != expected.train_episode_indices or validation != expected.validation_episode_indices:
        raise ValueError("CALVIN normalization partition is not the deterministic config-declared episode split")
    return train, validation


def _annotations_for_task(
    annotations: tuple[CalvinAnnotation, ...],
    *,
    task: str | None,
) -> tuple[CalvinAnnotation, ...]:
    if task is None:
        return annotations
    selected = tuple(annotation for annotation in annotations if annotation.task == task)
    if not selected:
        raise ValueError(f"unknown CALVIN task: {task!r}")
    return selected


def _require_lower_sha256(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"CALVIN normalization artifact has no valid lowercase {field}")
    return value


def _calvin_data_identity(
    stats_manifest: dict[str, Any],
    *,
    source_revisions: dict[str, str],
) -> dict[str, Any]:
    dataset = stats_manifest.get("dataset")
    split = stats_manifest.get("split")
    state = stats_manifest.get("state")
    action = stats_manifest.get("action")
    algorithm = stats_manifest.get("algorithm")
    if not all(isinstance(value, dict) for value in (dataset, split, state, action, algorithm)):
        raise ValueError("CALVIN normalization artifact is missing a critical identity section")
    assert isinstance(dataset, dict)
    assert isinstance(split, dict)
    assert isinstance(state, dict)
    assert isinstance(action, dict)
    assert isinstance(algorithm, dict)
    if stats_manifest.get("schema") != CALVIN_STATS_SCHEMA:
        raise ValueError("CALVIN trainer requires the production normalization-v4 schema")
    if set(dataset) != _CALVIN_NORMALIZATION_DATASET_FIELDS:
        raise ValueError("CALVIN normalization dataset identity field inventory differs")
    member_index = dataset.get("member_index")
    if not isinstance(member_index, dict) or set(member_index) != _CALVIN_MEMBER_INDEX_FIELDS:
        raise ValueError("CALVIN normalization member-index identity field inventory differs")
    required = {
        "archive_bytes": (dataset.get("archive_bytes"), CALVIN_ABC_D_ARCHIVE_BYTES),
        "archive_sha256": (dataset.get("archive_sha256"), CALVIN_ABC_D_ARCHIVE_SHA256),
        "central_directory_sha256": (
            dataset.get("central_directory_sha256"),
            OFFICIAL_CENTRAL_DIRECTORY_SHA256,
        ),
        "dataset_manifest_schema": (dataset.get("dataset_manifest_schema"), CALVIN_DATASET_MANIFEST_SCHEMA),
        "member_index.path": (member_index.get("path"), CALVIN_INDEX_NAME),
        "member_index.schema": (member_index.get("schema"), CALVIN_MEMBER_INDEX_SCHEMA),
        "metadata_files": (dataset.get("metadata_files"), list(CALVIN_CRITICAL_TRAIN_METADATA)),
        "name": (dataset.get("name"), "task_ABC_D"),
        "reader_schema": (dataset.get("reader_schema"), CALVIN_ARCHIVE_READER_SCHEMA),
        "split": (dataset.get("split"), "training"),
        "storage_mode": (dataset.get("storage_mode"), CALVIN_STORAGE_MODE_ARCHIVE_DIRECT),
        "state.dimension": (state.get("dimension"), 8),
        "state.continuous_dimensions": (state.get("continuous_dimensions"), list(range(7))),
        "state.gripper_index": (state.get("gripper_index"), 7),
        "state.observed_gripper_values": (state.get("observed_gripper_values"), [-1.0, 1.0]),
        "action.dimension": (action.get("dimension"), 7),
        "action.continuous_dimensions": (action.get("continuous_dimensions"), list(range(6))),
        "action.gripper_index": (action.get("gripper_index"), 6),
        "action.observed_gripper_values": (action.get("observed_gripper_values"), [-1.0, 1.0]),
        "action.transform": (action.get("transform"), CALVIN_ACTION_ADAPTER),
        "algorithm.actions_re_normalized": (algorithm.get("actions_re_normalized"), False),
        "source_revisions": (source_revisions, PINNED_CALVIN_SOURCE_REVISIONS),
    }
    mismatches = [name for name, (observed, expected) in required.items() if observed != expected]
    if mismatches:
        raise ValueError(f"CALVIN normalization artifact data contract mismatch: {mismatches}")
    if type(member_index.get("bytes")) is not int or member_index["bytes"] <= 0:
        raise ValueError("CALVIN normalization member-index byte count is invalid")
    sha256_fields = {
        "central_directory_sha256": dataset.get("central_directory_sha256"),
        "dataset_manifest_file_sha256": dataset.get("dataset_manifest_file_sha256"),
        "dataset_manifest_sha256": dataset.get("dataset_manifest_sha256"),
        "member_index.sha256": member_index.get("sha256"),
        "member_inventory_sha256": dataset.get("member_inventory_sha256"),
        "metadata_sha256": dataset.get("metadata_sha256"),
        "normalization content_sha256": stats_manifest.get("content_sha256"),
        "storage_identity_sha256": dataset.get("storage_identity_sha256"),
    }
    validated_sha256 = {field: _require_lower_sha256(value, field=field) for field, value in sha256_fields.items()}
    identity = {
        "protocol": CALVIN_PROTOCOL,
        "archive_bytes": CALVIN_ABC_D_ARCHIVE_BYTES,
        "archive_sha256": CALVIN_ABC_D_ARCHIVE_SHA256,
        "central_directory_sha256": validated_sha256["central_directory_sha256"],
        "dataset_manifest_file_sha256": validated_sha256["dataset_manifest_file_sha256"],
        "dataset_manifest_schema": CALVIN_DATASET_MANIFEST_SCHEMA,
        "dataset_manifest_sha256": validated_sha256["dataset_manifest_sha256"],
        "member_index": {
            "bytes": member_index["bytes"],
            "path": member_index["path"],
            "schema": member_index["schema"],
            "sha256": validated_sha256["member_index.sha256"],
        },
        "member_inventory_sha256": validated_sha256["member_inventory_sha256"],
        "metadata_files": copy.deepcopy(dataset["metadata_files"]),
        "metadata_sha256": validated_sha256["metadata_sha256"],
        "reader_schema": CALVIN_ARCHIVE_READER_SCHEMA,
        "storage_identity_sha256": validated_sha256["storage_identity_sha256"],
        "storage_mode": CALVIN_STORAGE_MODE_ARCHIVE_DIRECT,
        "split": copy.deepcopy(split),
        "split_sha256": canonical_config_sha256(split),
        "normalization_sha256": validated_sha256["normalization content_sha256"],
        "camera_shapes": copy.deepcopy(CALVIN_CAMERA_SHAPES),
        "camera_shapes_sha256": canonical_config_sha256(CALVIN_CAMERA_SHAPES),
        "state_adapter": CALVIN_STATE_ADAPTER,
        "action_adapter": CALVIN_ACTION_ADAPTER,
        "calvin_source_revisions": dict(source_revisions),
        "calvin_source_revisions_sha256": canonical_config_sha256(source_revisions),
    }
    if set(identity) != _CALVIN_IDENTITY_FIELDS:
        raise RuntimeError("CALVIN permanent identity field inventory drifted")
    return identity


def _calvin_storage_run_contract(calvin_identity: dict[str, Any]) -> dict[str, str]:
    member_index = calvin_identity.get("member_index")
    if not isinstance(member_index, dict) or set(member_index) != _CALVIN_MEMBER_INDEX_FIELDS:
        raise ValueError("CALVIN permanent identity has an invalid member-index inventory")
    contract = {
        "archive_bytes": str(calvin_identity["archive_bytes"]),
        "archive_sha256": calvin_identity["archive_sha256"],
        "central_directory_sha256": calvin_identity["central_directory_sha256"],
        "dataset_manifest_file_sha256": calvin_identity["dataset_manifest_file_sha256"],
        "dataset_manifest_schema": calvin_identity["dataset_manifest_schema"],
        "dataset_manifest_sha256": calvin_identity["dataset_manifest_sha256"],
        "member_index_bytes": str(member_index["bytes"]),
        "member_index_path": member_index["path"],
        "member_index_schema": member_index["schema"],
        "member_index_sha256": member_index["sha256"],
        "member_inventory_sha256": calvin_identity["member_inventory_sha256"],
        "metadata_sha256": calvin_identity["metadata_sha256"],
        "reader_schema": calvin_identity["reader_schema"],
        "storage_identity_sha256": calvin_identity["storage_identity_sha256"],
        "storage_mode": calvin_identity["storage_mode"],
    }
    if set(contract) != _CALVIN_STORAGE_RUN_CONTRACT_FIELDS or not all(
        isinstance(value, str) for value in contract.values()
    ):
        raise RuntimeError("CALVIN storage run-contract inventory drifted")
    return contract


def _snapshot_identity(snapshot_root: Path, *, expected_revision: str) -> dict[str, Any]:
    return verify_huggingface_snapshot(
        snapshot_root,
        expected_revision=expected_revision,
    )


def _calvin_prefix_cameras() -> tuple[CameraGeometry, ...]:
    return tuple(
        CameraGeometry(name=name, height=CALVIN_CAMERA_SHAPES[name][0], width=CALVIN_CAMERA_SHAPES[name][1])
        for name in ("rgb_static", "rgb_gripper")
    )


def _validate_training_instruction_coverage(
    prefix_geometry: dict[str, Any],
    annotations: tuple[CalvinAnnotation, ...],
) -> str:
    """Require every authenticated A/B/C training instruction in the frozen inventory."""

    records = prefix_geometry["instruction_inventory"]["records"]
    frozen = {record["instruction"] for record in records}
    observed = tuple(sorted({annotation.instruction for annotation in annotations}))
    missing = [instruction for instruction in observed if instruction not in frozen]
    if missing:
        raise ValueError(f"CALVIN prefix geometry omits authenticated training instructions: {missing[:8]}")
    return instruction_inventory_sha256(observed)


def _configure_and_validate_training_runtime(
    project_root: Path,
    *,
    resolved_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Fail closed on the canonical launcher, lock, imports, and deterministic flags."""

    _inventory_project_source_root()
    _validate_project_module_origins({"duo_vla", "duo_vla.runtime_integrity"})
    if platform.python_version() != EXPECTED_TRAIN_PYTHON:
        raise RuntimeError(
            f"CALVIN training requires Python {EXPECTED_TRAIN_PYTHON}, found {platform.python_version()}"
        )
    project_root = project_root.resolve()
    cache_root = Path(os.environ.get("DUO_VLA_CACHE_ROOT", "/root/.cache/duo-vla")).resolve()
    expected_world_size = canonical_visible_cuda_world_size(os.environ)
    execution = (
        None if resolved_config is None else execution_from_config(resolved_config, world_size=expected_world_size)
    )
    software_tp_size = expected_world_size if execution is None else execution.tensor_parallel_size
    environment_name = "train-single-gpu" if software_tp_size == 1 else "train"
    expected_prefix = (cache_root / f"venvs/{environment_name}").resolve()
    if Path(sys.prefix).resolve() != expected_prefix:
        raise RuntimeError(f"CALVIN training requires the pinned train venv: {expected_prefix}")
    if sys.flags.safe_path != 1:
        raise RuntimeError("CALVIN training requires Python safe-path mode")
    if sys.flags.dont_write_bytecode != 1 or not sys.dont_write_bytecode:
        raise RuntimeError("CALVIN training requires -B")
    if sys.flags.no_user_site != 1 or site.ENABLE_USER_SITE:
        raise RuntimeError("CALVIN training requires the user site to be disabled")
    if sys.pycache_prefix != "/dev/null":
        raise RuntimeError("CALVIN training requires an impossible pycache lookup prefix")
    version = f"python{sys.version_info.major}.{sys.version_info.minor}"
    compact_version = f"python{sys.version_info.major}{sys.version_info.minor}"
    expected_sys_path = [
        str((project_root / "src").resolve()),
        str(Path(sys.base_prefix) / "lib" / f"{compact_version}.zip"),
        str(Path(sys.base_prefix) / "lib" / version),
        str(Path(sys.base_exec_prefix) / "lib" / version / "lib-dynload"),
        str(expected_prefix / "lib" / version / "site-packages"),
    ]
    if sys.path != expected_sys_path:
        raise RuntimeError(f"CALVIN training import search path differs: {sys.path}")
    python_hash_seed = os.environ.get("PYTHONHASHSEED")
    if python_hash_seed not in {"0", "1", "2"}:
        raise RuntimeError("CALVIN training requires PYTHONHASHSEED in {0,1,2}")
    expected_environment = {
        **REQUIRED_TRAIN_ENVIRONMENT,
        "CUDA_VISIBLE_DEVICES": os.environ["CUDA_VISIBLE_DEVICES"],
        "DUO_VLA_CACHE_ROOT": str(cache_root),
        "DUO_VLA_PROJECT_ROOT": str(project_root),
        "DUO_VLA_TRAIN_VENV": str(expected_prefix),
        "HF_HOME": str(Path(os.environ.get("HF_HOME", "/root/.cache/huggingface")).resolve()),
        "PYTHONHASHSEED": python_hash_seed,
    }
    forbidden = (
        "BASH_ENV",
        "ENV",
        "GLOBIGNORE",
        "LD_LIBRARY_PATH",
        "LD_PRELOAD",
        "PYTHONHOME",
        "PYTHONINSPECT",
        "PYTHONPATH",
        "PYTHONSTARTUP",
    )
    present_forbidden = [name for name in forbidden if os.environ.get(name)]
    allowed_relevant = set(expected_environment)
    unexpected_overrides = sorted(
        name
        for name in os.environ
        if name.startswith(_RELEVANT_TRAIN_ENVIRONMENT_PREFIXES) and name not in allowed_relevant
    )
    if present_forbidden or unexpected_overrides:
        raise RuntimeError(
            f"CALVIN training environment contains injection/algorithm overrides: "
            f"forbidden={present_forbidden}, overrides={unexpected_overrides}"
        )
    observed_environment = {name: os.environ.get(name) for name in expected_environment}
    if observed_environment != expected_environment:
        raise RuntimeError(f"CALVIN training environment differs from the canonical launcher: {observed_environment}")
    relative_lock_path, expected_lock_sha256, expected_packages, expected_cuda_runtime = _train_runtime_pins(
        software_tp_size
    )
    lock_path = project_root / relative_lock_path
    if not lock_path.is_file() or file_sha256(lock_path) != expected_lock_sha256:
        raise RuntimeError("CALVIN training lockfile SHA-256 mismatch")
    packages = {name: importlib.metadata.version(name) for name in expected_packages}
    if packages != expected_packages:
        raise RuntimeError(f"CALVIN training package pin mismatch: {packages}")
    if torch.version.cuda != expected_cuda_runtime:
        raise RuntimeError(
            f"CALVIN training CUDA runtime differs: expected={expected_cuda_runtime}, observed={torch.version.cuda}"
        )
    rank_environment = validate_torchrun_rank_environment(
        os.environ,
        required=any(name in os.environ for name in ("RANK", "LOCAL_RANK", "WORLD_SIZE")),
        expected_world_size=expected_world_size,
    )
    venv_identity = content_address_train_venv(expected_prefix)

    site_packages = expected_prefix / f"lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages"
    expected_module_roots = {
        "accelerate": site_packages,
        "duo_vla": project_root / "src",
        "huggingface_hub": site_packages,
        "numpy": site_packages,
        "peft": site_packages,
        "PIL": site_packages,
        "pyarrow": site_packages,
        "safetensors": site_packages,
        "tokenizers": site_packages,
        "torch": site_packages,
        "torchvision": site_packages,
        "transformers": site_packages,
    }
    module_origins: dict[str, str] = {}
    for name, expected_root in expected_module_roots.items():
        spec = importlib.util.find_spec(name)
        origin = None if spec is None else spec.origin
        if not isinstance(origin, str) or not Path(origin).resolve().is_relative_to(expected_root.resolve()):
            raise RuntimeError(f"CALVIN training module {name!r} resolves outside {expected_root}: {origin}")
        module_origins[name] = str(Path(origin).resolve())

    safe_roots = (
        project_root / "scripts",
        project_root / "src",
        Path(sys.base_prefix),
        expected_prefix,
    )
    effective_sys_path: list[str] = []
    for entry in sys.path:
        if not entry:
            raise RuntimeError("CALVIN training sys.path contains the current working directory")
        resolved = Path(entry).resolve()
        if not any(resolved.is_relative_to(root.resolve()) for root in safe_roots):
            raise RuntimeError(f"CALVIN training sys.path contains an untrusted entry: {resolved}")
        effective_sys_path.append(str(resolved))

    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    if not torch.are_deterministic_algorithms_enabled() or torch.is_deterministic_algorithms_warn_only_enabled():
        raise RuntimeError("CALVIN training could not enable strict deterministic algorithms")
    runtime_identity = {
        "environment": dict(sorted(observed_environment.items())),
        "lock_sha256": expected_lock_sha256,
        "module_origins": module_origins,
        "packages": packages,
        "python": platform.python_version(),
        "static_environment_sha256": static_environment_identity(observed_environment)["sha256"],
        "sys_path": effective_sys_path,
        "torchrun": rank_environment,
        "train_venv": venv_identity,
    }
    if software_tp_size == 1:
        runtime_identity["lock_path"] = relative_lock_path.as_posix()
    return runtime_identity


def _execution_environment(runtime_preflight: dict[str, Any]) -> dict[str, Any]:
    return {
        "authenticated_runtime": runtime_preflight,
        "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_tf32": torch.backends.cudnn.allow_tf32,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "deterministic_warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "gpu_capability": [list(torch.cuda.get_device_capability(index)) for index in range(torch.cuda.device_count())],
        "gpu_names": [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())],
        "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
        "peft": importlib.metadata.version("peft"),
        "python": sys.version.split()[0],
        "python_hash_seed": os.environ.get("PYTHONHASHSEED"),
        "torch": torch.__version__,
        "transformers": importlib.metadata.version("transformers"),
        "world_size": dist.get_world_size(),
    }


def _validate_and_build_interface_config(config: dict[str, Any]) -> ActionInterfaceConfig:
    model = config["model"]
    action = config["action"]
    lora = config["lora"]
    benchmark = config["benchmark"]
    optimization = config["optimization"]
    training = config["training"]
    reproducibility = config["reproducibility"]
    execution = execution_from_config(config, world_size=dist.get_world_size())
    expected_execution_profile = execution.profile
    required_declarations = {
        "model.experts_implementation": (model, "experts_implementation"),
        "model.expert_batch_isolation": (model, "expert_batch_isolation"),
        "optimization.physical_batch_size": (optimization, "physical_batch_size"),
        "benchmark.prefix_geometry_content_sha256": (benchmark, "prefix_geometry_content_sha256"),
        "benchmark.fixed_physical_prefix_width": (benchmark, "fixed_physical_prefix_width"),
    }
    missing = [name for name, (section, field) in required_declarations.items() if field not in section]
    if missing:
        raise ValueError(f"resolved config is missing required fixed-batch declarations: {missing}")
    required = {
        "protocol": (config["protocol"], CALVIN_PROTOCOL),
        "model.id": (model["id"], DEFAULT_DIFFUSION_GEMMA_SPEC.model_id),
        "model.revision": (model["revision"], DEFAULT_DIFFUSION_GEMMA_SPEC.revision),
        "model.dtype": (model["dtype"], "bfloat16"),
        "model.tensor_parallel_size": (int(model["tensor_parallel_size"]), execution.tensor_parallel_size),
        "model.attention_implementation": (model["attention_implementation"], "sdpa"),
        "model.experts_implementation": (
            model["experts_implementation"],
            GROUPED_MM_EXPERTS_IMPLEMENTATION,
        ),
        "model.expert_batch_isolation": (model["expert_batch_isolation"], execution.backend),
        "action.horizon": (int(action["horizon"]), 8),
        "action.dimension": (int(action["dimension"]), 7),
        "action.timestep_embedding_dimension": (int(action["timestep_embedding_dimension"]), 256),
        "action.timestep_scale": (float(action["timestep_scale"]), 1000.0),
        "action.timestep_max_period": (float(action["timestep_max_period"]), 10000.0),
        "action.conditioning_mlp_activation": (action["conditioning_mlp_activation"], "silu"),
        "action.output_head_initialization_std": (float(action["output_head_initialization_std"]), 1e-3),
        "lora.rank": (int(lora["rank"]), 16),
        "lora.alpha": (int(lora["alpha"]), 32),
        "lora.dropout": (float(lora["dropout"]), 0.0),
        "lora.projections": (list(lora["projections"]), ["q_proj", "k_proj", "v_proj", "o_proj"]),
        "lora.scope": (lora["scope"], "decoder_self_attention_only"),
        "benchmark.dataset": (benchmark["dataset"], "task_ABC_D"),
        "benchmark.train_split": (benchmark["train_split"], "training"),
        "benchmark.evaluation_split": (benchmark["evaluation_split"], "validation"),
        "benchmark.train_environments": (list(benchmark["train_environments"]), ["A", "B", "C"]),
        "benchmark.evaluation_environment": (benchmark["evaluation_environment"], "D"),
        "benchmark.state_dimension": (int(benchmark["state_dimension"]), 8),
        "benchmark.camera_order": (list(benchmark["camera_order"]), ["rgb_static", "rgb_gripper"]),
        "benchmark.execution_horizons": (list(benchmark["execution_horizons"]), [1, 4]),
        "benchmark.control_frequency_hz": (int(benchmark["control_frequency_hz"]), 30),
        "benchmark.num_sequences": (int(benchmark["num_sequences"]), 1000),
        "benchmark.subtasks_per_sequence": (int(benchmark["subtasks_per_sequence"]), 5),
        "benchmark.max_steps_per_subtask": (int(benchmark["max_steps_per_subtask"]), 360),
        "benchmark.evaluation_seed": (int(benchmark["evaluation_seed"]), 0),
        "optimization.optimizer": (optimization["optimizer"], "adamw"),
        "optimization.schedule": (optimization["schedule"], "linear_warmup_cosine_decay"),
        "optimization.tensor_parallel_gradient_norm": (
            optimization["tensor_parallel_gradient_norm"],
            "replicas_once_plus_all_shards",
        ),
        "optimization.loss_accumulation_dtype": (optimization["loss_accumulation_dtype"], "float32"),
        "optimization.global_batch_size": (int(optimization["global_batch_size"]), 64),
        "optimization.physical_batch_size": (
            int(optimization["physical_batch_size"]),
            execution.physical_batch,
        ),
        "optimization.microbatch_size": (int(optimization["microbatch_size"]), execution.physical_batch),
        "optimization.gradient_accumulation_steps": (
            int(optimization["gradient_accumulation_steps"]),
            execution.accumulation,
        ),
        "optimization.ema_decay": (float(optimization["ema_decay"]), 0.0),
        "training.data_loader_workers_per_rank": (int(training["data_loader_workers_per_rank"]), 0),
        "training.image_augmentation": (training["image_augmentation"], "none"),
        "training.seeds": (list(training["seeds"]), [0, 1, 2]),
        "reproducibility.deterministic_evaluation": (
            reproducibility["deterministic_evaluation"],
            True,
        ),
        "reproducibility.save_trainable_weights_only": (
            reproducibility["save_trainable_weights_only"],
            True,
        ),
    }
    mismatches = [name for name, (observed, expected) in required.items() if observed != expected]
    if config.get("execution_profile") != expected_execution_profile:
        mismatches.append("execution_profile")
    if mismatches:
        details = {name: required[name] for name in mismatches if name in required}
        if "execution_profile" in mismatches:
            details["execution_profile"] = (config.get("execution_profile"), expected_execution_profile)
        raise ValueError(f"resolved config is unsupported by this trainer: {details}")
    checkpoint_interval = training.get("checkpoint_interval")
    permanent_checkpoint_interval = training.get("permanent_checkpoint_interval")
    if (
        type(checkpoint_interval) is not int
        or checkpoint_interval <= 0
        or type(permanent_checkpoint_interval) is not int
        or permanent_checkpoint_interval <= 0
        or permanent_checkpoint_interval % checkpoint_interval
    ):
        raise ValueError(
            "training.permanent_checkpoint_interval must be a positive multiple of training.checkpoint_interval"
        )
    prefix_sha256 = benchmark["prefix_geometry_content_sha256"]
    fixed_width = benchmark["fixed_physical_prefix_width"]
    if not (
        isinstance(prefix_sha256, str)
        and len(prefix_sha256) == 64
        and all(character in "0123456789abcdef" for character in prefix_sha256)
    ):
        raise ValueError("benchmark.prefix_geometry_content_sha256 must be a lowercase SHA-256")
    if isinstance(fixed_width, bool) or not isinstance(fixed_width, int) or not 0 < fixed_width <= 1024 - 8:
        raise ValueError("benchmark.fixed_physical_prefix_width must leave room for all eight action positions")
    return ActionInterfaceConfig(
        hidden_size=DEFAULT_DIFFUSION_GEMMA_SPEC.expected_hidden_size,
        state_dim=int(benchmark["state_dimension"]),
        action_horizon=int(action["horizon"]),
        action_dim=int(action["dimension"]),
        timestep_embedding_dim=int(action["timestep_embedding_dimension"]),
        timestep_scale=float(action["timestep_scale"]),
        timestep_max_period=float(action["timestep_max_period"]),
        output_init_std=float(action["output_head_initialization_std"]),
    )


def _broadcast_rank0_error(error: str | None, *, group: dist.ProcessGroup | None = None) -> None:
    value = [error]
    if group is None:
        dist.broadcast_object_list(value, src=0)
    else:
        dist.broadcast_object_list(value, src=0, group=group)
    if value[0] is not None:
        raise RuntimeError(value[0])


def _authenticate_calvin_dataset_generation_distributed(
    training_root: Path,
) -> AuthenticatedCalvinDatasetGeneration:
    """Authenticate once and validate the capability on every rank."""

    authentication_group = dist.new_group(
        backend="gloo",
        timeout=CALVIN_DATASET_AUTHENTICATION_TIMEOUT,
    )
    generation_payload: list[dict[str, Any] | None] = [None]
    authentication_error: str | None = None
    generation: AuthenticatedCalvinDatasetGeneration | None = None
    try:
        if dist.get_rank() == 0:
            try:
                # scene_info.npy and auto_lang_ann.npy contain pickles.  Pin
                # their bytes and the source archive before either is loaded.
                _, generation = authenticate_calvin_dataset_generation(training_root)
                generation_payload[0] = generation.to_dict()
            except Exception as exc:
                authentication_error = f"CALVIN dataset authentication failed: {type(exc).__name__}: {exc}"
        _broadcast_rank0_error(authentication_error, group=authentication_group)
        dist.broadcast_object_list(generation_payload, src=0, group=authentication_group)

        local_generation_error: str | None = None
        try:
            generation = AuthenticatedCalvinDatasetGeneration.from_dict(generation_payload[0])
            verify_calvin_dataset_generation(training_root, generation)
        except Exception as exc:
            local_generation_error = f"rank {dist.get_rank()}: {type(exc).__name__}: {exc}"
        generation_errors: list[str | None] = [None] * dist.get_world_size()
        dist.all_gather_object(
            generation_errors,
            local_generation_error,
            group=authentication_group,
        )
        if any(error is not None for error in generation_errors):
            raise RuntimeError(f"CALVIN authenticated generation failed across TP ranks: {generation_errors}")
    finally:
        dist.destroy_process_group(authentication_group)
    if generation is None:  # Defensive: a successful rank-local validation must produce a capability.
        raise RuntimeError("CALVIN dataset authentication returned no generation capability")
    return generation


def _initialize_run_journal(
    output_dir: Path,
    *,
    config_sha256: str,
    resume: Path | None,
    recover_bootstrap: bool,
) -> tuple[str, str | None]:
    result: list[dict[str, Any] | None] = [None]
    if dist.get_rank() == 0:
        try:
            if resume is None and not recover_bootstrap:
                journal = create_run_journal(output_dir, config_sha256=config_sha256)
                latest_manifest_sha256 = None
            elif recover_bootstrap:
                if resume is not None:
                    raise ValueError("bootstrap recovery cannot select a resume checkpoint")
                try:
                    journal = load_run_journal(
                        output_dir,
                        expected_config_sha256=config_sha256,
                    )
                except FileNotFoundError:
                    journal = create_run_journal(output_dir, config_sha256=config_sha256)
                if journal.latest_checkpoint is not None:
                    raise ValueError("bootstrap recovery requires a journal without a checkpoint")
                recovery = quarantine_uncommitted_training_artifacts(output_dir)
                latest_manifest_sha256 = None
                if recovery.changed:
                    print(
                        json.dumps(
                            {
                                "recovery_quarantine": recovery.directory,
                                "moved_paths": recovery.moved_paths,
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )
            else:
                assert resume is not None
                journal = load_run_journal(
                    output_dir,
                    expected_config_sha256=config_sha256,
                )
                record = validate_resume_checkpoint(output_dir, resume)
                recovery = quarantine_uncommitted_training_artifacts(output_dir)
                retention = apply_checkpoint_retention(output_dir)
                reconcile_metrics_jsonl(output_dir)
                latest_manifest_sha256 = record.manifest_sha256
                if recovery.changed:
                    print(
                        json.dumps(
                            {
                                "recovery_quarantine": recovery.directory,
                                "moved_paths": recovery.moved_paths,
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )
                if retention.changed:
                    print(
                        json.dumps(
                            {
                                "checkpoint_retention_recovered": retention.recovered_transactions,
                                "checkpoint_retention_retired": retention.retired_paths,
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )
            result[0] = {
                "latest_manifest_sha256": latest_manifest_sha256,
                "run_uuid": journal.run_uuid,
            }
        except Exception as exc:
            result[0] = {"error": f"{type(exc).__name__}: {exc}"}
    dist.broadcast_object_list(result, src=0)
    payload = result[0]
    if not isinstance(payload, dict):
        raise RuntimeError("rank 0 did not broadcast a run journal result")
    if "error" in payload:
        raise RuntimeError(f"cannot initialize run journal: {payload['error']}")
    run_uuid = payload.get("run_uuid")
    latest_manifest_sha256 = payload.get("latest_manifest_sha256")
    if not isinstance(run_uuid, str) or not (latest_manifest_sha256 is None or isinstance(latest_manifest_sha256, str)):
        raise RuntimeError("rank 0 broadcast an invalid run journal result")
    return run_uuid, latest_manifest_sha256


def _prepare_output(
    output_dir: Path,
    *,
    resume: Path | None,
    config_sha256: str,
) -> bool:
    result: list[dict[str, Any] | None] = [None]
    if dist.get_rank() == 0:
        try:
            if resume is None:
                try:
                    output_dir.mkdir(parents=True, exist_ok=False)
                    descriptor = os.open(
                        output_dir.parent,
                        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
                    )
                    try:
                        os.fsync(descriptor)
                    finally:
                        os.close(descriptor)
                    recover_bootstrap = False
                except FileExistsError:
                    if output_dir.is_symlink() or not output_dir.is_dir():
                        raise ValueError(f"existing output path is not a real directory: {output_dir}") from None
                    journal_path = output_dir / "run_journal.json"
                    config_path = output_dir / "resolved_config.json"
                    if journal_path.exists():
                        journal = load_run_journal(
                            output_dir,
                            expected_config_sha256=config_sha256,
                        )
                        if journal.latest_checkpoint is not None:
                            raise FileExistsError(
                                "existing run has a committed checkpoint; pass --resume with the journal tip"
                            ) from None
                    if config_path.exists():
                        saved_config = json.loads(config_path.read_text(encoding="utf-8"))
                        if saved_config.get("config_sha256") != config_sha256:
                            raise ValueError("uncommitted run configuration differs from this launch") from None
                    unknown_entries = [
                        entry.name for entry in output_dir.iterdir() if not _is_known_uncommitted_run_entry(entry.name)
                    ]
                    if unknown_entries:
                        raise FileExistsError(
                            f"existing output directory contains unknown entries: {sorted(unknown_entries)}"
                        ) from None
                    recover_bootstrap = True
            elif not output_dir.is_dir():
                raise FileNotFoundError(f"resume output directory is missing: {output_dir}")
            else:
                recover_bootstrap = False
            result[0] = {"recover_bootstrap": recover_bootstrap}
        except Exception as exc:
            result[0] = {"error": f"cannot prepare run directory {output_dir}: {type(exc).__name__}: {exc}"}
    dist.broadcast_object_list(result, src=0)
    payload = result[0]
    if not isinstance(payload, dict):
        raise RuntimeError("rank 0 did not broadcast an output preparation result")
    if "error" in payload:
        raise RuntimeError(str(payload["error"]))
    recover_bootstrap = payload.get("recover_bootstrap")
    if not isinstance(recover_bootstrap, bool):
        raise RuntimeError("rank 0 broadcast an invalid output preparation result")
    dist.barrier()
    return recover_bootstrap


def _is_known_uncommitted_run_entry(name: str) -> bool:
    if name in {
        "checkpoints",
        "metrics.jsonl",
        "recovery_quarantine",
        "resolved_config.json",
        "run_journal.json",
    }:
        return True
    return bool(
        re.fullmatch(r"\.rank-state-update-[0-9]{6,}", name)
        or re.fullmatch(r"\.resolved_config\.json\.tmp-[0-9]+", name)
        or re.fullmatch(r"\.run_journal\.json\.tmp-[0-9a-f]{32}", name)
    )


def _acquire_run_lock(output_dir: Path):
    result: list[dict[str, str] | None] = [None]
    handle = None
    if dist.get_rank() == 0:
        try:
            output_dir.parent.mkdir(parents=True, exist_ok=True)
            identity = hashlib.sha256(str(output_dir).encode()).hexdigest()[:16]
            lock_path = output_dir.parent / f".{output_dir.name}.duo-vla-{identity}.lock"
            descriptor = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
            handle = os.fdopen(descriptor, "r+", encoding="utf-8")
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RuntimeError(f"another process owns this training output: {output_dir}") from exc
            handle.seek(0)
            handle.truncate()
            handle.write(json.dumps({"output_dir": str(output_dir), "pid": os.getpid()}) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
            result[0] = {"lock_path": str(lock_path)}
        except Exception as exc:
            if handle is not None:
                handle.close()
                handle = None
            result[0] = {"error": f"cannot acquire run lock: {type(exc).__name__}: {exc}"}
    dist.broadcast_object_list(result, src=0)
    payload = result[0]
    if not isinstance(payload, dict):
        raise RuntimeError("rank 0 did not broadcast a run-lock result")
    if "error" in payload:
        raise RuntimeError(str(payload["error"]))
    return handle


def _assert_fp32_trainables(
    lora_parameters: list[torch.nn.Parameter],
    interface_parameters: list[torch.nn.Parameter],
) -> None:
    invalid = [
        f"{group}[{index}]={parameter.dtype}"
        for group, parameters in (("lora", lora_parameters), ("interface", interface_parameters))
        for index, parameter in enumerate(parameters)
        if parameter.dtype != torch.float32
    ]
    _raise_if_rank_errors(
        "trainable dtype audit",
        None if not invalid else f"trainable tensors must remain FP32: {invalid[:8]}",
    )


def _raise_if_rank_errors(context: str, local_error: str | None) -> None:
    errors: list[str | None] = [None] * dist.get_world_size()
    dist.all_gather_object(errors, local_error)
    if any(error is not None for error in errors):
        raise RuntimeError(f"{context} failed across TP ranks: {errors}")


def _assert_fp32_optimizer_state(optimizer: torch.optim.Optimizer) -> None:
    invalid: list[str] = []
    floating_tensors = 0
    for parameter_index, state in enumerate(optimizer.state.values()):
        for name, value in state.items():
            if isinstance(value, torch.Tensor) and value.is_floating_point():
                floating_tensors += 1
                if value.dtype != torch.float32:
                    invalid.append(f"parameter[{parameter_index}].{name}={value.dtype}")
    local_error = None
    if floating_tensors == 0:
        local_error = "optimizer has no floating-point state after an update or resume"
    elif invalid:
        local_error = f"optimizer floating-point state must be FP32: {invalid[:8]}"
    _raise_if_rank_errors("optimizer dtype audit", local_error)


def _assert_replicated_optimizer_state(
    optimizer: torch.optim.Optimizer,
    named_parameters: list[tuple[str, torch.nn.Parameter]],
) -> str:
    local_hash: str | None = None
    local_error: str | None = None
    try:
        digest = hashlib.sha256()
        for name, parameter in sorted(named_parameters, key=lambda item: item[0]):
            state = optimizer.state.get(parameter)
            if not isinstance(state, dict) or not state:
                raise RuntimeError(f"replicated parameter has no optimizer state: {name}")
            digest.update(name.encode())
            for state_name, value in sorted(state.items()):
                digest.update(state_name.encode())
                if isinstance(value, torch.Tensor):
                    local = value.detach().cpu().contiguous()
                    digest.update(str(local.dtype).encode())
                    digest.update(str(tuple(local.shape)).encode())
                    digest.update(local.reshape(-1).view(torch.uint8).numpy().tobytes())
                else:
                    digest.update(repr(value).encode())
        local_hash = digest.hexdigest()
    except Exception as exc:
        local_error = f"rank {dist.get_rank()}: {type(exc).__name__}: {exc}"
    payloads: list[tuple[str | None, str | None] | None] = [None] * dist.get_world_size()
    dist.all_gather_object(payloads, (local_hash, local_error))
    errors = [payload[1] for payload in payloads if payload is not None and payload[1] is not None]
    if errors:
        raise RuntimeError(f"cannot hash replicated optimizer state: {errors}")
    hashes = [payload[0] for payload in payloads if payload is not None]
    assert local_hash is not None
    if any(value != local_hash for value in hashes):
        raise RuntimeError(f"replicated optimizer state diverged across TP ranks: {hashes}")
    return local_hash


def _assert_distributed_gradient_health(parameters: list[torch.nn.Parameter]) -> None:
    local = finite_gradient_audit(parameters)
    audits: list[tuple[int, int] | None] = [None] * dist.get_world_size()
    dist.all_gather_object(audits, local)
    if any(audit != (0, 0) for audit in audits):
        raise RuntimeError(f"gradient audit failed across TP ranks: {audits}")


def _assert_initial_gradients_present(
    lora_partition,
    interface_named_parameters: list[tuple[str, torch.nn.Parameter]],
) -> None:
    local_missing = [
        name for name, parameter in [*lora_partition.replicated, *interface_named_parameters] if parameter.grad is None
    ]
    missing_by_rank: list[list[str] | None] = [None] * dist.get_world_size()
    dist.all_gather_object(missing_by_rank, local_missing)
    if any(missing for missing in missing_by_rank):
        raise RuntimeError(f"initial replicated gradients are missing: {missing_by_rank}")
    for name, parameter in lora_partition.replicated:
        assert parameter.grad is not None
        assert_replicated_tensor(f"initial_gradient.{name}", parameter.grad)
    for name, parameter in interface_named_parameters:
        assert parameter.grad is not None
        assert_replicated_tensor(f"initial_interface_gradient.{name}", parameter.grad)


def _assert_distributed_trainer_state(
    trainer_state: TrainerState,
    checkpoint_manifest: dict[str, Any],
) -> None:
    serialized = trainer_state.to_dict()
    states: list[dict[str, Any] | None] = [None] * dist.get_world_size()
    dist.all_gather_object(states, serialized)
    if any(state != serialized for state in states):
        raise RuntimeError(f"TP ranks restored different trainer states: {states}")
    manifest_state = checkpoint_manifest.get("trainer_state")
    if manifest_state != serialized:
        raise ValueError("checkpoint manifest and rank-state trainer progress disagree")


def _fixed_distinct_anchors(
    sampler: CalvinTaskUniformAnchorSampler,
    *,
    count: int,
    seed: int,
) -> tuple[CalvinAnchor, ...]:
    if count <= 0:
        raise ValueError("count must be positive")
    if count > sampler.population_size:
        raise ValueError(f"cannot draw {count} distinct anchors from a population of {sampler.population_size}")
    generator = torch.Generator().manual_seed(seed)
    anchors: list[CalvinAnchor] = []
    seen: set[tuple[int, int]] = set()
    while len(anchors) < count:
        anchor = sampler.draw(generator)
        identity = (anchor.annotation_index, anchor.global_index)
        if identity not in seen:
            anchors.append(anchor)
            seen.add(identity)
    return tuple(anchors)


def _processor_inputs(
    processor,
    samples,
    device: torch.device,
    *,
    prefix_geometry: dict[str, Any],
    expected_batch_size: int = PHYSICAL_BATCH_SIZE,
):
    if expected_batch_size not in SUPPORTED_BATCHES or len(samples) != expected_batch_size:
        raise ValueError(f"CALVIN processor requires physical batch {expected_batch_size}, observed {len(samples)}")
    conversations = [
        [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": Image.fromarray(sample.observation.third_person)},
                    {"type": "image", "image": Image.fromarray(sample.observation.wrist)},
                    {"type": "text", "text": sample.instruction},
                ],
            }
        ]
        for sample in samples
    ]
    inputs = apply_fixed_prefix_chat_template(
        processor,
        conversations,
        fixed_physical_prefix_width=prefix_geometry["geometry"]["fixed_physical_prefix_width"],
        padding_side=prefix_geometry["tokenization"]["padding_side"],
        expected_batch_size=expected_batch_size,
        images_per_prefix=len(prefix_geometry["ordered_cameras"]),
    )
    expected_lengths = {
        record["instruction"]: record["valid_prefix_length"]
        for record in prefix_geometry["instruction_inventory"]["records"]
    }
    observed_lengths = tuple(int(value) for value in inputs["attention_mask"].sum(dim=1).tolist())
    required_lengths = tuple(expected_lengths.get(sample.instruction) for sample in samples)
    if None in required_lengths or observed_lengths != required_lengths:
        raise ValueError(
            "CALVIN processor valid-prefix lengths differ from the authenticated instruction inventory: "
            f"expected={required_lengths}, observed={observed_lengths}"
        )
    return inputs.to(device)


def _materialize_batch(
    dataset: CalvinNpzDataset,
    sampler: CalvinTaskUniformAnchorSampler,
    *,
    count: int,
    seed: int,
    state_normalizer,
) -> CalvinBatch:
    anchors = _fixed_distinct_anchors(sampler, count=count, seed=seed)
    samples = dataset.sample_many(anchors)
    return collate_calvin_samples(
        samples,
        state_normalizer=state_normalizer,
    )


def _materialize_plan_batches(dataset, sampler, plan_groups, *, state_normalizer):
    """Read one rank's planned anchors together; preserve B8 sampling and order."""
    plans = tuple(plan for group in plan_groups for plan in group)
    anchors_by_plan = [_fixed_distinct_anchors(sampler, count=8, seed=plan.data_seed) for plan in plans]
    # The reader requires unique anchors. Deduplicate IO only, then expand the
    # original multiplicity before per-plan collation; never resample duplicates.
    unique = {}
    for anchors in anchors_by_plan:
        for anchor in anchors:
            unique.setdefault((anchor.annotation_index, anchor.global_index), anchor)
    loaded = dataset.sample_many(tuple(unique.values()))
    by_identity = dict(zip(unique, loaded, strict=True))
    canonical = tuple(
        collate_calvin_samples(
            tuple(by_identity[(anchor.annotation_index, anchor.global_index)] for anchor in anchors),
            state_normalizer=state_normalizer,
        )
        for anchors in anchors_by_plan
    )
    result = []
    offset = 0
    for group in plan_groups:
        result.append(
            coalesce_calvin_batches(canonical[offset : offset + len(group)], physical_batch_size=len(group) * 8)
        )
        offset += len(group)
    return result


def _validation_is_due(*, next_update: int, total_updates: int, interval: int) -> bool:
    """Run validation only at its declared cadence and the configured final update.

    An invocation-local ``--stop-after-updates`` boundary is a resumability
    mechanism, not an experiment event, and must not silently add a costly
    validation pass or change the metric schedule.
    """

    if interval <= 0 or total_updates <= 0 or not 0 < next_update <= total_updates:
        raise ValueError("validation schedule values are invalid")
    return next_update % interval == 0 or next_update == total_updates


def _run_validation(
    *,
    dataset: CalvinNpzDataset,
    sampler: CalvinTaskUniformAnchorSampler,
    state_normalizer,
    processor,
    model,
    denoiser: DuoVLADenoiser,
    adapted,
    device: torch.device,
    samples: int,
    microbatch_size: int,
    validation_seed: int,
    policy_contract: PolicyContract,
    prefix_geometry: dict[str, Any],
    data_parallel_size: int = 1,
) -> float:
    denoiser.eval()
    adapted.eval()
    model.model.encoder.eval()
    numerators: list[float] = []
    element_count = 0
    if microbatch_size not in SUPPORTED_BATCHES or samples % (microbatch_size * data_parallel_size):
        raise ValueError("CALVIN validation requires complete physical batches on every rank")
    plans = tuple(make_microbatch_plan(validation_seed, 0, i) for i in range(samples // 8))
    groups = group_plans(
        plans,
        physical_batch=microbatch_size,
        rank=dist.get_rank() if data_parallel_size == 2 else 0,
        data_parallel_size=data_parallel_size,
    )
    with torch.no_grad():
        for group in groups:
            batch = _materialize_plan_batches(dataset, sampler, (group,), state_normalizer=state_normalizer)[0]
            state = batch.states.to(device)
            clean = batch.clean_actions.to(device)
            valid = batch.action_valid_mask.to(device)
            prefix_inputs = _processor_inputs(
                processor,
                batch.samples,
                device,
                prefix_geometry=prefix_geometry,
                expected_batch_size=microbatch_size,
            )
            prefix = encode_diffusion_gemma_prefix(model, dict(prefix_inputs))
            pair = canonical_training_pair(clean, policy_contract, group)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                prediction = denoiser(
                    pair.input_actions,
                    pair.timesteps,
                    state,
                    prefix_cache=prefix.past_key_values,
                    prefix_attention_mask=prefix.attention_mask,
                    action_valid_mask=valid,
                )
            component = masked_sse(prediction, pair.target, valid)
            numerators.append(float(component.squared_error_sum))
            element_count += component.element_count
    adapted.train()
    denoiser.train()
    model.model.encoder.eval()
    numerator = sum(numerators)
    if data_parallel_size == 2:
        numerator = reduce_scalar(numerator, device=device, dtype=torch.float64)
        element_count = reduce_scalar(element_count, device=device, dtype=torch.int64)
    loss = numerator / element_count
    replicated = torch.tensor(loss, device=device, dtype=torch.float64)
    assert_replicated_tensor("validation_loss", replicated)
    return loss


def _save_training_checkpoint(
    checkpoint_dir: Path,
    *,
    output_dir: Path,
    adapted,
    projector: ActionInputProjector,
    head: VelocityHead,
    optimizer: torch.optim.Optimizer,
    optimizer_named_parameters: Sequence[tuple[str, torch.nn.Parameter]],
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    trainer_state: TrainerState,
    run_contract: dict[str, str],
    normalization_artifact: Path,
    prefix_geometry_artifact: Path,
    resolved_config_path: Path,
    manifest: dict[str, Any],
    device: torch.device,
) -> None:
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    stage = output_dir / f".rank-state-update-{trainer_state.next_update:06d}"
    error: str | None = None
    if rank == 0:
        try:
            stage.mkdir(exist_ok=False)
        except OSError as exc:
            error = f"cannot create training-state staging directory: {exc}"
    _broadcast_rank0_error(error)
    dist.barrier()
    rank_path = stage / f"rank-{rank:03d}.pt"
    checkpoint_rng = capture_rng_state(device)
    try:
        local_hash: str | None = None
        local_error: str | None = None
        try:
            local_hash = save_training_rank_state(
                rank_path,
                rank=rank,
                world_size=world_size,
                trainer_state=trainer_state,
                optimizer=optimizer,
                named_parameters=optimizer_named_parameters,
                scheduler=scheduler,
                run_contract=run_contract,
                device=device,
            )
        except Exception as exc:
            local_error = f"rank {rank}: {type(exc).__name__}: {exc}"
        rank_errors: list[str | None] = [None] * world_size
        dist.all_gather_object(rank_errors, local_error)
        if any(rank_error is not None for rank_error in rank_errors):
            raise RuntimeError(f"cannot stage per-rank training state: {rank_errors}")
        assert local_hash is not None
        hashes: list[str | None] = [None] * world_size
        dist.all_gather_object(hashes, local_hash)
        dist.barrier()
        additional_artifacts: dict[str, str | Path] = {
            "normalization": normalization_artifact,
            "prefix_geometry": prefix_geometry_artifact,
            "resolved_config": resolved_config_path,
        }
        for index in range(world_size):
            additional_artifacts[f"training_rank_{index:03d}"] = stage / f"rank-{index:03d}.pt"
        completed = save_trainable_checkpoint(
            checkpoint_dir,
            adapted_model=adapted,
            interface_modules={"action_projector": projector, "velocity_head": head},
            additional_artifacts=additional_artifacts,
            manifest={
                **manifest,
                "trainer_state": trainer_state.to_dict(),
                "training_rank_state_sha256": hashes,
            },
        )
    finally:
        restore_rng_state(checkpoint_rng, device)
    dist.barrier()
    cleanup_error: str | None = None
    try:
        rank_path.unlink()
    except OSError as exc:
        cleanup_error = f"rank {rank}: cannot remove staged rank state: {exc}"
    cleanup_errors: list[str | None] = [None] * world_size
    dist.all_gather_object(cleanup_errors, cleanup_error)
    if any(error is not None for error in cleanup_errors):
        raise RuntimeError(f"cannot clean staged rank state: {cleanup_errors}")
    dist.barrier()
    stage_error: str | None = None
    if rank == 0:
        try:
            stage.rmdir()
            if completed is None:
                raise RuntimeError("rank 0 did not receive a completed checkpoint manifest")
        except Exception as exc:
            stage_error = f"cannot finalize rank-state staging cleanup: {type(exc).__name__}: {exc}"
    _broadcast_rank0_error(stage_error)


def _commit_training_checkpoint(
    output_dir: Path,
    checkpoint_dir: Path,
    *,
    update: int,
    parent_manifest_sha256: str | None,
    last_metrics: dict[str, Any],
) -> str:
    result: list[dict[str, str] | None] = [None]
    if dist.get_rank() == 0:
        try:
            manifest_sha256 = file_sha256(checkpoint_dir / "manifest.json")
            record_latest_checkpoint(
                output_dir,
                checkpoint=checkpoint_dir,
                update=update,
                manifest_sha256=manifest_sha256,
                parent_manifest_sha256=parent_manifest_sha256,
                last_metrics=last_metrics,
            )
            retention = apply_checkpoint_retention(output_dir)
            if retention.changed:
                print(
                    json.dumps(
                        {
                            "checkpoint_retention_recovered": retention.recovered_transactions,
                            "checkpoint_retention_retired": retention.retired_paths,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
            result[0] = {"manifest_sha256": manifest_sha256}
        except Exception as exc:
            result[0] = {"error": f"{type(exc).__name__}: {exc}"}
    dist.broadcast_object_list(result, src=0)
    payload = result[0]
    if not isinstance(payload, dict):
        raise RuntimeError("rank 0 did not broadcast a checkpoint commit result")
    if "error" in payload:
        raise RuntimeError(f"cannot commit checkpoint to the run journal: {payload['error']}")
    manifest_sha256 = payload.get("manifest_sha256")
    if not isinstance(manifest_sha256, str):
        raise RuntimeError("rank 0 broadcast an invalid checkpoint manifest SHA-256")
    return manifest_sha256


def _append_metric(metrics_path: Path, metric: dict[str, Any], *, should_log: bool) -> None:
    error: str | None = None
    if dist.get_rank() == 0:
        try:
            with metrics_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(metric, allow_nan=False, sort_keys=True) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            if should_log:
                print(json.dumps(metric, indent=2, sort_keys=True), flush=True)
        except Exception as exc:
            error = f"cannot append training metric: {type(exc).__name__}: {exc}"
    _broadcast_rank0_error(error)


def _load_training_state(
    checkpoint_dir: Path,
    *,
    optimizer: torch.optim.Optimizer,
    optimizer_named_parameters: Sequence[tuple[str, torch.nn.Parameter]],
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    run_contract: dict[str, str],
    device: torch.device,
) -> TrainerState:
    rank = dist.get_rank()
    trainer_state: TrainerState | None = None
    local_error: str | None = None
    try:
        manifest = load_checkpoint_manifest(checkpoint_dir)
        name = f"training_rank_{rank:03d}"
        artifact = manifest["artifacts"].get(name)
        recorded_hashes = manifest.get("training_rank_state_sha256")
        if not isinstance(artifact, dict) or not isinstance(recorded_hashes, list):
            raise ValueError("checkpoint has no resumable rank-state artifacts")
        if len(recorded_hashes) != dist.get_world_size() or not all(
            isinstance(value, str) and len(value) == 64 for value in recorded_hashes
        ):
            raise ValueError("checkpoint rank-state hash list does not match the TP topology")
        if artifact.get("sha256") != recorded_hashes[rank]:
            raise ValueError("checkpoint rank-state manifest hashes disagree")
        trainer_state = load_training_rank_state(
            checkpoint_dir / artifact["path"],
            expected_sha256=artifact["sha256"],
            rank=rank,
            world_size=dist.get_world_size(),
            optimizer=optimizer,
            named_parameters=optimizer_named_parameters,
            scheduler=scheduler,
            run_contract=run_contract,
            device=device,
        )
    except Exception as exc:
        local_error = f"rank {rank}: {type(exc).__name__}: {exc}"
    _raise_if_rank_errors("training-state restore", local_error)
    assert trainer_state is not None
    return trainer_state


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("training_root", type=Path)
    parser.add_argument("normalization_artifact", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--config", type=Path, default=Path("configs/calvin_abc_to_d.toml"))
    parser.add_argument(
        "--prefix-geometry-artifact",
        type=Path,
        help="Canonical prefix-geometry artifact; its expected SHA/P come only from --config.",
    )
    parser.add_argument("--task")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--total-updates", type=int)
    parser.add_argument("--warmup-updates", type=int)
    parser.add_argument("--microbatch-size", type=int)
    parser.add_argument("--gradient-accumulation-steps", type=int)
    parser.add_argument("--validation-interval", type=int)
    parser.add_argument("--validation-samples", type=int)
    parser.add_argument("--checkpoint-interval", type=int)
    parser.add_argument("--permanent-checkpoint-interval", type=int)
    parser.add_argument("--log-interval", type=int)
    parser.add_argument(
        "--max-cached-frames", type=int, help="Bounded decoded-frame LRU; defaults to the recipe or 512"
    )
    parser.add_argument(
        "--runtime-preflight-only",
        action="store_true",
        help="Validate the canonical train runtime without initializing distributed CUDA or reading data.",
    )
    parser.add_argument(
        "--stop-after-updates",
        type=int,
        help="Stop cleanly after this many updates in this invocation without changing the run contract.",
    )
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()

    args.training_root = args.training_root.resolve()
    args.normalization_artifact = args.normalization_artifact.resolve()
    args.output_dir = args.output_dir.resolve()
    args.config = args.config.resolve()
    if args.prefix_geometry_artifact is not None:
        args.prefix_geometry_artifact = args.prefix_geometry_artifact.resolve()
    if args.resume is not None:
        args.resume = args.resume.resolve() if args.resume.is_absolute() else (args.output_dir / args.resume).resolve()
        try:
            args.resume.relative_to(args.output_dir)
        except ValueError as exc:
            raise ValueError("resume checkpoint must be inside output_dir") from exc

    project_root = Path(__file__).resolve().parents[1]
    launch_config = load_resolved_toml(args.config)
    if args.max_cached_frames is None:
        args.max_cached_frames = launch_config.get("training", {}).get("max_cached_frames", 512)
    runtime_preflight = _configure_and_validate_training_runtime(project_root, resolved_config=launch_config)
    if args.runtime_preflight_only:
        print(json.dumps(runtime_preflight, indent=2, sort_keys=True))
        return
    if args.prefix_geometry_artifact is None:
        raise ValueError("CALVIN training requires --prefix-geometry-artifact")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", device_id=device)
    expected_world_size = canonical_visible_cuda_world_size(os.environ)
    if dist.get_world_size() != expected_world_size:
        raise RuntimeError(
            f"CALVIN training world size differs from the canonical visible-device topology: "
            f"expected={expected_world_size}, observed={dist.get_world_size()}"
        )
    rank = dist.get_rank()
    run_lock = None
    dataset: CalvinNpzDataset | None = None
    try:
        run_lock = _acquire_run_lock(args.output_dir)
        config = load_resolved_toml(args.config)
        config = copy.deepcopy(config)
        execution = execution_from_config(config, world_size=dist.get_world_size())
        is_data_parallel = execution.data_parallel_size == 2
        optimization = config["optimization"]
        training = config["training"]
        declared_batch_contract = {
            "global_batch_size": optimization.get("global_batch_size"),
            "microbatch_size": optimization.get("microbatch_size"),
            "gradient_accumulation_steps": optimization.get("gradient_accumulation_steps"),
            "physical_batch_size": optimization.get("physical_batch_size"),
        }
        expected_batch_contract = {
            "global_batch_size": 64,
            "microbatch_size": execution.physical_batch,
            "gradient_accumulation_steps": execution.accumulation,
            "physical_batch_size": execution.physical_batch,
        }
        if declared_batch_contract != expected_batch_contract:
            raise ValueError(
                "CALVIN config must declare the exact fixed-batch contract: "
                f"expected={expected_batch_contract}, observed={declared_batch_contract}"
            )
        if args.microbatch_size is not None and args.microbatch_size != declared_batch_contract["microbatch_size"]:
            raise ValueError("--microbatch-size cannot override the fixed physical batch")
        if (
            args.gradient_accumulation_steps is not None
            and args.gradient_accumulation_steps != declared_batch_contract["gradient_accumulation_steps"]
        ):
            raise ValueError("--gradient-accumulation-steps cannot override the fixed accumulation contract")
        run_seed = int(config["reproducibility"]["seed"] if args.seed is None else args.seed)
        if not 0 <= run_seed <= (1 << 32) - 1:
            raise ValueError("run seed must be in [0, 4294967295] for PYTHONHASHSEED and NumPy")
        total_updates = int(optimization["total_updates"] if args.total_updates is None else args.total_updates)
        if total_updates <= 0:
            raise ValueError("total updates must be positive")
        if args.warmup_updates is None:
            configured_warmup = int(optimization["warmup_updates"])
            warmup_updates = (
                configured_warmup if args.total_updates is None else min(configured_warmup, total_updates // 10)
            )
        else:
            warmup_updates = args.warmup_updates
        microbatch_size = int(optimization["microbatch_size"] if args.microbatch_size is None else args.microbatch_size)
        accumulation_steps = int(
            optimization["gradient_accumulation_steps"]
            if args.gradient_accumulation_steps is None
            else args.gradient_accumulation_steps
        )
        validation_interval = int(
            training["validation_interval"] if args.validation_interval is None else args.validation_interval
        )
        validation_samples = int(
            training["validation_samples"] if args.validation_samples is None else args.validation_samples
        )
        checkpoint_interval = int(
            training["checkpoint_interval"] if args.checkpoint_interval is None else args.checkpoint_interval
        )
        permanent_checkpoint_interval = int(
            training["permanent_checkpoint_interval"]
            if args.permanent_checkpoint_interval is None
            else args.permanent_checkpoint_interval
        )
        log_interval = int(training["log_interval"] if args.log_interval is None else args.log_interval)
        positive_values = {
            "microbatch_size": microbatch_size,
            "accumulation_steps": accumulation_steps,
            "validation_interval": validation_interval,
            "validation_samples": validation_samples,
            "checkpoint_interval": checkpoint_interval,
            "permanent_checkpoint_interval": permanent_checkpoint_interval,
            "log_interval": log_interval,
            "max_cached_frames": args.max_cached_frames,
        }
        if args.stop_after_updates is not None and args.stop_after_updates <= 0:
            raise ValueError("stop-after-updates must be positive")
        invalid = [name for name, value in positive_values.items() if value <= 0]
        if invalid or not 0 <= warmup_updates < total_updates:
            raise ValueError(f"invalid training schedule fields: {invalid}, warmup={warmup_updates}")
        if validation_samples % (execution.physical_batch * execution.data_parallel_size):
            raise ValueError("validation_samples must form full physical batches on all data-parallel ranks")
        if permanent_checkpoint_interval % checkpoint_interval:
            raise ValueError("permanent_checkpoint_interval must be a multiple of checkpoint_interval")
        optimization["total_updates"] = total_updates
        optimization["warmup_updates"] = warmup_updates
        optimization["microbatch_size"] = microbatch_size
        optimization["gradient_accumulation_steps"] = accumulation_steps
        optimization["global_batch_size"] = microbatch_size * accumulation_steps * execution.data_parallel_size
        training["validation_interval"] = validation_interval
        training["validation_samples"] = validation_samples
        training["checkpoint_interval"] = checkpoint_interval
        training["permanent_checkpoint_interval"] = permanent_checkpoint_interval
        training["log_interval"] = log_interval
        config["run"] = {"seed": run_seed, "task": args.task, "max_cached_frames": args.max_cached_frames}
        expected_hash_seed = str(run_seed)
        if os.environ.get("PYTHONHASHSEED") != expected_hash_seed:
            raise RuntimeError(
                f"launch training with PYTHONHASHSEED={expected_hash_seed} so Python/PEFT ordering is reproducible"
            )
        interface_config = _validate_and_build_interface_config(config)
        allowed_run_seeds = tuple(int(value) for value in training["seeds"])
        if run_seed not in allowed_run_seeds:
            raise ValueError(f"run seed {run_seed} is not one of the declared comparison seeds {allowed_run_seeds}")
        policy_contract = policy_contract_from_config(config)
        authenticated_generation = _authenticate_calvin_dataset_generation_distributed(args.training_root)
        dataset = CalvinNpzDataset(
            args.training_root,
            max_cached_frames=args.max_cached_frames,
            expected_scenes=CALVIN_EXPECTED_SCENES,
            authenticated_generation=authenticated_generation,
        )
        state_normalizer, stats_manifest = load_calvin_state_normalizer(
            args.normalization_artifact,
            expected_archive_sha256=CALVIN_ABC_D_ARCHIVE_SHA256,
            training_root=args.training_root,
            authenticated_generation=authenticated_generation,
        )
        train_indices, validation_indices = _validate_calvin_split_recipe(
            stats_manifest,
            dataset,
            training,
        )
        selected_annotations = _annotations_for_task(dataset.annotations, task=args.task)
        train_sampler = CalvinTaskUniformAnchorSampler(selected_annotations, train_indices)
        validation_sampler = CalvinTaskUniformAnchorSampler(selected_annotations, validation_indices)
        calvin_source_revisions = _calvin_source_revisions(project_root)
        calvin_identity = _calvin_data_identity(
            stats_manifest,
            source_revisions=calvin_source_revisions,
        )
        hf_home = Path(os.environ.get("HF_HOME", "/root/.cache/huggingface"))
        model_snapshot = (
            hf_home / "hub/models--google--diffusiongemma-26B-A4B-it/snapshots" / DEFAULT_DIFFUSION_GEMMA_SPEC.revision
        )
        model_identity: list[dict[str, Any] | None] = [None]
        if rank == 0:
            try:
                model_identity[0] = _snapshot_identity(
                    model_snapshot,
                    expected_revision=DEFAULT_DIFFUSION_GEMMA_SPEC.revision,
                )
            except Exception as exc:
                model_identity[0] = {"error": f"{type(exc).__name__}: {exc}"}
        dist.broadcast_object_list(model_identity, src=0)
        model_identity_payload = model_identity[0]
        if not isinstance(model_identity_payload, dict):
            raise RuntimeError("rank 0 did not broadcast a model snapshot identity")
        if "error" in model_identity_payload:
            raise RuntimeError(f"model snapshot authentication failed: {model_identity_payload['error']}")
        model_tree_sha256 = model_identity_payload["tree_metadata_sha256"]
        snapshot_identity = SnapshotTreeIdentity.from_huggingface_report(
            DEFAULT_DIFFUSION_GEMMA_SPEC.model_id,
            model_identity_payload,
        )
        prefix_geometry = load_prefix_geometry_contract(
            args.prefix_geometry_artifact,
            expected_content_sha256=config["benchmark"]["prefix_geometry_content_sha256"],
            expected_model_identity=snapshot_identity,
            expected_processor_identity=snapshot_identity,
            expected_ordered_cameras=_calvin_prefix_cameras(),
            expected_fixed_physical_prefix_width=config["benchmark"]["fixed_physical_prefix_width"],
        )
        training_instruction_inventory_sha256 = _validate_training_instruction_coverage(
            prefix_geometry,
            dataset.annotations,
        )
        execution_geometry = calvin_execution_geometry(config)
        execution_environment = _execution_environment(runtime_preflight)
        config["execution_environment"] = execution_environment
        config["artifact_trees"] = {"model_tree_sha256": model_tree_sha256}
        config["calvin_identity"] = calvin_identity
        config["execution_geometry"] = execution_geometry
        config["training_instruction_inventory_sha256"] = training_instruction_inventory_sha256
        source_sha256 = _source_tree_sha256(
            project_root,
            single_gpu=config.get("execution_profile") is not None,
        )
        config["source_tree_sha256"] = source_sha256
        config_sha256 = canonical_config_sha256(config)

        recover_bootstrap = _prepare_output(
            args.output_dir,
            resume=args.resume,
            config_sha256=config_sha256,
        )
        resolved_config_path = args.output_dir / "resolved_config.json"
        config_error: str | None = None
        if rank == 0:
            try:
                if args.resume is None and (not recover_bootstrap or not resolved_config_path.is_file()):
                    observed_config_hash = save_resolved_config(resolved_config_path, config)
                else:
                    _, observed_config_hash = load_verified_resolved_config(resolved_config_path)
                if observed_config_hash != config_sha256:
                    raise ValueError("resolved run configuration differs from the resume run")
            except Exception as exc:
                config_error = f"cannot validate the resolved run configuration: {type(exc).__name__}: {exc}"
        _broadcast_rank0_error(config_error)
        dist.barrier()
        run_uuid, parent_manifest_sha256 = _initialize_run_journal(
            args.output_dir,
            config_sha256=config_sha256,
            resume=args.resume,
            recover_bootstrap=recover_bootstrap,
        )
        dist.barrier()

        run_contract = {
            "config_sha256": config_sha256,
            "protocol": CALVIN_PROTOCOL,
            **_calvin_storage_run_contract(calvin_identity),
            "split_sha256": calvin_identity["split_sha256"],
            "train_episode_sha256": stats_manifest["split"]["train_episode_sha256"],
            "validation_episode_sha256": stats_manifest["split"]["validation_episode_sha256"],
            "normalization_sha256": calvin_identity["normalization_sha256"],
            "camera_shapes_sha256": calvin_identity["camera_shapes_sha256"],
            "state_adapter": CALVIN_STATE_ADAPTER,
            "action_adapter": CALVIN_ACTION_ADAPTER,
            "calvin_revision": calvin_source_revisions["calvin"],
            "calvin_env_revision": calvin_source_revisions["calvin_env"],
            "calvin_tacto_revision": calvin_source_revisions["tacto"],
            "calvin_source_revisions_sha256": calvin_identity["calvin_source_revisions_sha256"],
            "execution_environment_sha256": canonical_config_sha256(execution_environment),
            "model_revision": DEFAULT_DIFFUSION_GEMMA_SPEC.revision,
            "model_tree_sha256": model_tree_sha256,
            "prefix_geometry_content_sha256": prefix_geometry["content_sha256"],
            "fixed_physical_prefix_width": str(prefix_geometry["geometry"]["fixed_physical_prefix_width"]),
            "experts_implementation": GROUPED_MM_EXPERTS_IMPLEMENTATION,
            "expert_batch_isolation": execution.backend,
            "physical_batch_size": str(execution.physical_batch),
            "training_instruction_inventory_sha256": training_instruction_inventory_sha256,
            "policy_contract_sha256": canonical_config_sha256(policy_contract.to_dict()),
            "run_uuid": run_uuid,
            "source_tree_sha256": source_sha256,
        }
        if config.get("execution_profile") is not None:
            run_contract.update(
                execution_profile=str(config["execution_profile"]),
                tensor_parallel_size=str(execution.tensor_parallel_size),
            )
        if execution.optimized:
            run_contract.update({key: str(execution_geometry[key]) for key in EXTENDED_GEOMETRY_FIELDS})

        random.seed(2026 + run_seed)
        np.random.seed((2026 + run_seed) % (1 << 32))
        torch.manual_seed(2026 + run_seed)
        processor = AutoProcessor.from_pretrained(
            DEFAULT_DIFFUSION_GEMMA_SPEC.model_id,
            revision=DEFAULT_DIFFUSION_GEMMA_SPEC.revision,
            local_files_only=True,
        )
        if (
            getattr(getattr(processor, "tokenizer", None), "padding_side", None)
            != prefix_geometry["tokenization"]["padding_side"]
        ):
            raise RuntimeError("loaded processor padding side differs from the authenticated prefix geometry")
        model = load_diffusion_gemma_bf16_tp(
            local_files_only=True,
            tp_size=execution.tensor_parallel_size,
            **({"replica_mode": "data_parallel"} if is_data_parallel else {}),
        )
        install_experts, verify_experts = expert_backend_functions(execution.backend)
        installed_experts = install_experts(
            model,
            physical_batch_size=execution.physical_batch,
        )
        if (
            installed_experts.experts_implementation != GROUPED_MM_EXPERTS_IMPLEMENTATION
            or installed_experts.physical_batch_size != execution.physical_batch
        ):
            raise RuntimeError("installed expert execution differs from the resolved training contract")
        verify_experts(model, physical_batch_size=execution.physical_batch)
        backend = DiffusionGemmaActionDecoder.from_block_diffusion_model(model)
        torch.manual_seed(2027 + run_seed)
        resume_manifest = load_checkpoint_manifest(args.resume) if args.resume is not None else None
        if resume_manifest is not None:
            prefix_artifact = resume_manifest.get("artifacts", {}).get("prefix_geometry")
            if (
                not isinstance(prefix_artifact, dict)
                or prefix_artifact.get("path") != "artifacts/prefix_geometry.json"
                or prefix_artifact.get("sha256") != file_sha256(args.prefix_geometry_artifact)
            ):
                raise ValueError("resume checkpoint prefix-geometry artifact differs from this launch")
            if resume_manifest.get("execution_geometry") != execution_geometry:
                raise ValueError("resume checkpoint execution geometry mismatch")
            resumed_policy_contract = validate_manifest_policy_contract(resume_manifest, config)
            if resumed_policy_contract != policy_contract:
                raise ValueError("resume checkpoint policy contract mismatch")
            if resume_manifest.get("policy_contract_sha256") != run_contract["policy_contract_sha256"]:
                raise ValueError("resume checkpoint policy contract SHA-256 mismatch")
            validate_decoder_attention_lora_adapter_config(
                args.resume / "lora/adapter_config.json",
                rank=int(config["lora"]["rank"]),
                alpha=int(config["lora"]["alpha"]),
                dropout=float(config["lora"]["dropout"]),
            )
            validate_decoder_attention_lora_weights(
                args.resume / "lora/adapter_model.safetensors",
                rank=int(config["lora"]["rank"]),
            )
        if args.resume is None:
            adapted = apply_decoder_attention_lora(
                model,
                rank=int(config["lora"]["rank"]),
                alpha=int(config["lora"]["alpha"]),
                dropout=float(config["lora"]["dropout"]),
            )
        else:
            adapted, _ = load_lora_checkpoint(
                args.resume,
                model,
                is_trainable=True,
                validate_decoder_contract=True,
                expected_rank=int(config["lora"]["rank"]),
            )
        verify_experts(model, physical_batch_size=execution.physical_batch)
        projector = ActionInputProjector(interface_config).to(device)
        head = VelocityHead(
            interface_config.hidden_size,
            interface_config.action_dim,
            init_std=interface_config.output_init_std,
        ).to(device)
        if args.resume is not None:
            assert resume_manifest is not None
            interface_artifact = resume_manifest["artifacts"]["interface"]
            load_interface_state_dict(
                args.resume / interface_artifact["path"],
                {"action_projector": projector, "velocity_head": head},
            )
        denoiser = DuoVLADenoiser(projector, backend, head).train()
        adapted.train()
        model.model.encoder.eval()
        lora_partition = decoder_lora_parameter_partition(adapted)
        lora_named_parameters = [
            (name, parameter) for name, parameter in adapted.named_parameters() if parameter.requires_grad
        ]
        interface_named_parameters = [
            *((f"action_projector.{name}", parameter) for name, parameter in projector.named_parameters()),
            *((f"velocity_head.{name}", parameter) for name, parameter in head.named_parameters()),
        ]
        lora_parameters = [parameter for _, parameter in lora_named_parameters]
        interface_parameters = [parameter for _, parameter in interface_named_parameters]
        _assert_fp32_trainables(lora_parameters, interface_parameters)
        optimization_config = OptimizationConfig(
            lora_learning_rate=float(optimization["lora_learning_rate"]),
            interface_learning_rate=float(optimization["interface_learning_rate"]),
            beta1=float(optimization["adam_beta1"]),
            beta2=float(optimization["adam_beta2"]),
            epsilon=float(optimization["adam_epsilon"]),
            weight_decay=float(optimization["weight_decay"]),
            gradient_clip_norm=float(optimization["gradient_clip_norm"]),
            total_updates=total_updates,
            warmup_updates=warmup_updates,
            final_learning_rate_scale=float(optimization["final_learning_rate_scale"]),
        )
        optimizer, scheduler = create_optimizer_and_scheduler(
            lora_parameters,
            interface_parameters,
            optimization_config,
        )
        optimizer_named_parameters = [*lora_named_parameters, *interface_named_parameters]
        if is_data_parallel:
            if lora_partition.sharded or {id(p) for _, p in lora_partition.replicated} != {
                id(p) for p in lora_parameters
            }:
                raise RuntimeError("CALVIN DP2 cannot contain TP-sharded or unclassified trainables")
            assert_replicated_parameter_values(optimizer_named_parameters)
        run_contract["optimizer_parameter_schema_sha256"] = optimizer_parameter_schema_sha256(
            optimizer,
            optimizer_named_parameters,
        )
        expected_run_contract_fields = (
            _CALVIN_SINGLE_GPU_RUN_CONTRACT_FIELDS
            if config.get("execution_profile") is not None
            else _CALVIN_RUN_CONTRACT_FIELDS
        )
        if execution.optimized:
            expected_run_contract_fields = expected_run_contract_fields | EXTENDED_GEOMETRY_FIELDS
        if set(run_contract) != expected_run_contract_fields:
            raise RuntimeError("CALVIN training run-contract inventory drifted")
        if args.resume is None:
            trainer_state = TrainerState()
        else:
            assert resume_manifest is not None
            _validate_resume_manifest_run_contract(resume_manifest, run_contract)
            trainer_state = _load_training_state(
                args.resume,
                optimizer=optimizer,
                optimizer_named_parameters=optimizer_named_parameters,
                scheduler=scheduler,
                run_contract=run_contract,
                device=device,
            )
            _assert_fp32_optimizer_state(optimizer)
            validate_training_progress(
                trainer_state,
                optimizer=optimizer,
                scheduler=scheduler,
                examples_per_update=64,
                expected_learning_rates=[
                    optimization_config.lora_learning_rate
                    * learning_rate_scale(trainer_state.next_update, optimization_config),
                    optimization_config.interface_learning_rate
                    * learning_rate_scale(trainer_state.next_update, optimization_config),
                ],
            )
            _assert_distributed_trainer_state(trainer_state, resume_manifest)
        if trainer_state.next_update > total_updates:
            raise ValueError("resume checkpoint is beyond the configured training budget")
        parent_checkpoint = (
            None
            if parent_manifest_sha256 is None
            else {
                "relative_path": args.resume.relative_to(args.output_dir).as_posix(),
                "update": trainer_state.next_update,
            }
        )

        execution_end = total_updates
        if args.stop_after_updates is not None:
            execution_end = min(total_updates, trainer_state.next_update + args.stop_after_updates)

        invocation_start_update = trainer_state.next_update
        metrics_path = args.output_dir / "metrics.jsonl"
        validation_seed = run_seed ^ 0x5A17
        started = time.perf_counter()
        for update in range(trainer_state.next_update, execution_end):
            update_started = time.perf_counter()
            plans = make_update_plan(run_seed, update, gradient_accumulation_steps=8)
            plan_groups = group_plans(
                plans,
                physical_batch=microbatch_size,
                rank=rank if is_data_parallel else 0,
                data_parallel_size=execution.data_parallel_size,
            )
            batches = _materialize_plan_batches(dataset, train_sampler, plan_groups, state_normalizer=state_normalizer)
            total_elements = sum(
                masked_element_count(
                    batch.action_valid_mask,
                    action_dim=interface_config.action_dim,
                )
                for batch in batches
            )
            if is_data_parallel:
                total_elements = reduce_scalar(total_elements, device=device, dtype=torch.int64)
            optimizer.zero_grad(set_to_none=True)
            numerator = 0.0
            for microstep, (group, batch) in enumerate(zip(plan_groups, batches, strict=True)):
                state = batch.states.to(device)
                clean = batch.clean_actions.to(device)
                valid = batch.action_valid_mask.to(device)
                prefix_inputs = _processor_inputs(
                    processor,
                    batch.samples,
                    device,
                    prefix_geometry=prefix_geometry,
                    expected_batch_size=microbatch_size,
                )
                prefix = encode_diffusion_gemma_prefix(model, dict(prefix_inputs))
                pair = canonical_training_pair(clean, policy_contract, group)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    prediction = denoiser(
                        pair.input_actions,
                        pair.timesteps,
                        state,
                        prefix_cache=prefix.past_key_values,
                        prefix_attention_mask=prefix.attention_mask,
                        action_valid_mask=valid,
                    )
                component = masked_sse(prediction, pair.target, valid)
                component.loss_for_total(total_elements).backward()
                numerator += float(component.squared_error_sum.detach())
                if update == 0 and microstep == 0 and not is_data_parallel:
                    _assert_initial_gradients_present(
                        lora_partition,
                        interface_named_parameters,
                    )
            _assert_distributed_gradient_health([*lora_parameters, *interface_parameters])
            if is_data_parallel:
                sum_gradients_(optimizer_named_parameters)
            gradient_norm = clip_tensor_parallel_grad_norm_(
                [*(parameter for _, parameter in lora_partition.replicated), *interface_parameters],
                [parameter for _, parameter in lora_partition.sharded],
                max_norm=optimization_config.gradient_clip_norm,
                error_if_nonfinite=True,
            )
            applied_learning_rates = [group["lr"] for group in optimizer.param_groups]
            optimizer.step()
            if update == invocation_start_update:
                _assert_fp32_optimizer_state(optimizer)
            scheduler.step()
            trainer_state = trainer_state.advance(examples=64)
            if update == 0:
                assert_replicated_parameter_values(interface_named_parameters)
            if is_data_parallel:
                numerator = reduce_scalar(numerator, device=device, dtype=torch.float64)
                if update == invocation_start_update:
                    assert_replicated_parameter_values(optimizer_named_parameters)
            train_loss = numerator / total_elements
            assert_replicated_tensor(
                f"train_loss.{trainer_state.next_update}",
                torch.tensor(train_loss, device=device, dtype=torch.float64),
            )
            update_seconds = time.perf_counter() - update_started
            if is_data_parallel:
                update_seconds = reduce_scalar(update_seconds, device=device, dtype=torch.float64, maximum=True)
            metric: dict[str, Any] = {
                "examples_seen": trainer_state.examples_seen,
                "gradient_norm": float(gradient_norm),
                "interface_learning_rate": applied_learning_rates[1],
                "lora_learning_rate": applied_learning_rates[0],
                "objective": policy_contract.objective,
                "train_loss": train_loss,
                "update": trainer_state.next_update,
                "update_seconds": update_seconds,
            }
            if execution.optimized:
                metric.update(
                    physical_batch_size=microbatch_size,
                    data_parallel_size=execution.data_parallel_size,
                    global_batch_size=64,
                    canonical_microbatches=8,
                    physical_forwards_per_rank=len(plan_groups),
                )
            should_validate = _validation_is_due(
                next_update=trainer_state.next_update,
                total_updates=total_updates,
                interval=validation_interval,
            )
            if should_validate:
                metric["validation_loss"] = _run_validation(
                    dataset=dataset,
                    sampler=validation_sampler,
                    state_normalizer=state_normalizer,
                    processor=processor,
                    model=model,
                    denoiser=denoiser,
                    adapted=adapted,
                    device=device,
                    samples=validation_samples,
                    microbatch_size=microbatch_size,
                    validation_seed=validation_seed,
                    policy_contract=policy_contract,
                    prefix_geometry=prefix_geometry,
                    data_parallel_size=execution.data_parallel_size,
                )
            should_checkpoint = (
                trainer_state.next_update % checkpoint_interval == 0
                or trainer_state.next_update == total_updates
                or trainer_state.next_update == execution_end
            )
            if should_checkpoint:
                replicated_checkpoint_parameters = [
                    *lora_partition.replicated,
                    *interface_named_parameters,
                ]
                replicated_parameter_sha256 = assert_replicated_parameter_values(replicated_checkpoint_parameters)
                replicated_optimizer_sha256 = _assert_replicated_optimizer_state(
                    optimizer,
                    replicated_checkpoint_parameters,
                )
                checkpoint_dir = args.output_dir / "checkpoints" / f"update-{trainer_state.next_update:06d}"
                _save_training_checkpoint(
                    checkpoint_dir,
                    output_dir=args.output_dir,
                    adapted=adapted,
                    projector=projector,
                    head=head,
                    optimizer=optimizer,
                    optimizer_named_parameters=optimizer_named_parameters,
                    scheduler=scheduler,
                    trainer_state=trainer_state,
                    run_contract=run_contract,
                    normalization_artifact=args.normalization_artifact,
                    prefix_geometry_artifact=args.prefix_geometry_artifact,
                    resolved_config_path=resolved_config_path,
                    manifest={
                        **run_contract,
                        **(
                            {key: execution_geometry[key] for key in EXTENDED_GEOMETRY_FIELDS}
                            if execution.optimized
                            else {}
                        ),
                        "archive_bytes": calvin_identity["archive_bytes"],
                        "calvin_identity": calvin_identity,
                        "calvin_source_revisions": calvin_source_revisions,
                        "camera_shapes": CALVIN_CAMERA_SHAPES,
                        "complete": trainer_state.next_update == total_updates,
                        "configured_total_updates": total_updates,
                        "dataset": "task_ABC_D",
                        "dataset_split": "training",
                        "execution_environment": execution_environment,
                        "execution_geometry": execution_geometry,
                        "fixed_physical_prefix_width": execution_geometry["fixed_physical_prefix_width"],
                        "kind": "resumable-calvin-abc-to-d-training",
                        "last_metrics": metric,
                        "model_id": DEFAULT_DIFFUSION_GEMMA_SPEC.model_id,
                        "member_index_bytes": calvin_identity["member_index"]["bytes"],
                        "parent_manifest_sha256": parent_manifest_sha256,
                        "checkpoint_retention": make_checkpoint_retention_contract(
                            permanent_checkpoint_interval=permanent_checkpoint_interval,
                            parent_checkpoint=parent_checkpoint,
                        ),
                        "platform": platform.platform(),
                        "physical_batch_size": execution_geometry["physical_batch_size"],
                        "policy_contract": policy_contract.to_dict(),
                        **(
                            {"tensor_parallel_size": execution_geometry["tensor_parallel_size"]}
                            if "tensor_parallel_size" in execution_geometry
                            else {}
                        ),
                        "prefix_geometry": {
                            "instruction_inventory_sha256": prefix_geometry["instruction_inventory"]["sha256"],
                            "maximum_valid_prefix_length": prefix_geometry["geometry"]["maximum_valid_prefix_length"],
                            "padding_side": prefix_geometry["tokenization"]["padding_side"],
                        },
                        "replicated_optimizer_sha256": replicated_optimizer_sha256,
                        "replicated_parameter_sha256": replicated_parameter_sha256,
                        "run_seed": run_seed,
                        "split": stats_manifest["split"],
                        "task": args.task,
                        "train_episode_count": len(train_indices),
                        "validation_episode_count": len(validation_indices),
                    },
                    device=device,
                )
                parent_manifest_sha256 = _commit_training_checkpoint(
                    args.output_dir,
                    checkpoint_dir,
                    update=trainer_state.next_update,
                    parent_manifest_sha256=parent_manifest_sha256,
                    last_metrics=metric,
                )
                parent_checkpoint = {
                    "relative_path": checkpoint_dir.relative_to(args.output_dir).as_posix(),
                    "update": trainer_state.next_update,
                }
            _append_metric(
                metrics_path,
                metric,
                should_log=(
                    trainer_state.next_update % log_interval == 0
                    or trainer_state.next_update == total_updates
                    or trainer_state.next_update == execution_end
                ),
            )
        dist.barrier()
        if rank == 0:
            print(
                json.dumps(
                    {
                        "complete": trainer_state.next_update == total_updates,
                        "elapsed_seconds": time.perf_counter() - started,
                        "examples_seen": trainer_state.examples_seen,
                        "final_update": trainer_state.next_update,
                        "output_dir": str(args.output_dir),
                        "peak_memory_gib": torch.cuda.max_memory_allocated(device) / 2**30,
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
    finally:
        if dataset is not None:
            dataset.close()
        if run_lock is not None:
            run_lock.close()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
