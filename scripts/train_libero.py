#!/usr/bin/env python3
"""Resumable distributed Duo-VLA trainer for pinned LIBERO data."""

# ruff: noqa: E402 -- authenticate project sources before importing project code.

from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import importlib.metadata
import json
import os
import platform
import random
import re
import site
import stat
import sys
import time
from collections.abc import Mapping, Sequence
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
from duo_vla.backbones.loading import DEFAULT_DIFFUSION_GEMMA_SPEC, load_diffusion_gemma_bf16_tp
from duo_vla.backbones.sample_isolated_experts import (
    GROUPED_MM_EXPERTS_IMPLEMENTATION,
    install_sample_isolated_grouped_mm_experts,
    verify_sample_isolated_grouped_mm_experts,
)
from duo_vla.backbones.sample_isolated_experts_v2 import (
    SAMPLE_ISOLATED_GROUPED_MM_V2,
    install_sample_isolated_grouped_mm_experts_v2,
    verify_sample_isolated_grouped_mm_experts_v2,
)
from duo_vla.checkpointing import (
    load_checkpoint_manifest,
    load_interface_state_dict,
    load_lora_checkpoint,
    save_trainable_checkpoint,
)
from duo_vla.config import ActionInterfaceConfig
from duo_vla.data.batching import LiberoBatch, collate_libero_samples
from duo_vla.data.libero import LiberoParquetDataset
from duo_vla.data.libero_stats import LIBERO_DATASET_REVISION, load_libero_normalizers
from duo_vla.data.sampling import LiberoAnchor, TaskUniformAnchorSampler
from duo_vla.dp2_fork import (
    DP2_FORK_CHECKPOINT_CONTRACT_FIELDS,
    authenticate_dp2_child_environment,
    authenticate_dp2_fork_parent,
    dp2_fork_run_contract,
    dp2_semantic_recipe_sha256,
    has_dp2_checkpoint_lineage,
    load_published_dp2_fork_manifest,
    restore_tp1_fork_training_state,
    validate_dp2_checkpoint_lineage,
)
from duo_vla.hf_snapshot import verify_huggingface_snapshot
from duo_vla.modeling import DuoVLADenoiser
from duo_vla.objectives import PolicyTrainingPair, make_seeded_policy_training_pair
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
    load_prefix_geometry_contract,
    validate_prefix_geometry_contract,
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
from duo_vla.runtime_determinism import configure_strict_cuda_determinism, deterministic_torch_runtime
from duo_vla.runtime_integrity import (
    canonical_visible_cuda_world_size,
    content_address_train_venv,
    static_environment_identity,
    validate_torchrun_rank_environment,
)
from duo_vla.topology_fork import (
    TOPOLOGY_FORK_CHECKPOINT_FIELDS,
    authenticate_topology_child_environment,
    authenticate_topology_fork_parent,
    has_topology_fork_lineage,
    load_published_topology_fork_manifest,
    restore_topology_fork_training_state,
    topology_fork_run_contract,
    validate_topology_checkpoint_lineage,
)
from duo_vla.training import (
    MicrobatchPlan,
    TrainerState,
    make_microbatch_plan,
    make_update_plan,
    masked_element_count,
    masked_sse,
)
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

CANONICAL_STREAM_BATCH_SIZE = 8
CANONICAL_MICROSTEPS_PER_UPDATE = 8
GLOBAL_BATCH_SIZE = CANONICAL_STREAM_BATCH_SIZE * CANONICAL_MICROSTEPS_PER_UPDATE
SUPPORTED_PHYSICAL_BATCH_SIZES = frozenset({8, 16, 32, 64})
DATA_PARALLEL_STRATEGY = "data_parallel"
DATA_PARALLEL_PLAN_PARTITION = "contiguous-b8-chunks-by-rank"
DATA_PARALLEL_GRADIENT_REDUCTION = "sum_globally_normalized_sse_gradients"
DP2_DATA_PARALLEL_SIZE = 2
DP2_TENSOR_PARALLEL_SIZE = 1
DP2_RANK_PHYSICAL_BATCH_SIZE = 32
DP2_MAX_CACHED_FILES = 377
DP2_EXECUTION_PROFILE = "duovla-dp2-tp1-fused-v2-train-b32-serve-b8-v1"
_DP2_FORK_ONLY_RUN_CONTRACT_FIELDS = DP2_FORK_CHECKPOINT_CONTRACT_FIELDS
_DP2_FORK_RUN_CONTRACT_FIELDS = _DP2_FORK_ONLY_RUN_CONTRACT_FIELDS | {
    "data_parallel_size",
    "execution_profile",
    "max_cached_files",
    "run_uuid",
    "source_tree_sha256",
}
# Retain the established public name for callers that need the canonical B8
# stream width.  Runtime physical forwards may coalesce multiple such chunks.
PHYSICAL_BATCH_SIZE = CANONICAL_STREAM_BATCH_SIZE
DEFAULT_SERVING_BATCH_SIZE = CANONICAL_STREAM_BATCH_SIZE
EXPERT_BATCH_ISOLATION = "sample_isolated_grouped_mm_v1"
SUPPORTED_EXPERT_BATCH_ISOLATIONS = frozenset({EXPERT_BATCH_ISOLATION, SAMPLE_ISOLATED_GROUPED_MM_V2})
LIBERO_DATASET_TREE_SHA256 = "d9c14b4aff28bcc56f341b171c6a5a3b10510d4bd0378662891c5156d245add8"
LIBERO_DATASET_CONTENT_INVENTORY_SHA256 = "63fd7a951ebb397a33c43cad4a7c48c7c6911bd8d1481ff99b07da5f7890782c"
LIBERO_DATASET_FILES_VERIFIED = 382
LIBERO_DATASET_TOTAL_BYTES = 34_926_155_087
LIBERO_DATASET_AUTHENTICATION_TIMEOUT = timedelta(hours=1)
_INTEGER_MANIFEST_RUN_CONTRACT_FIELDS = frozenset(
    {
        "dataset_files_verified",
        "dataset_total_bytes",
        "data_parallel_size",
        "fixed_physical_prefix_width",
        "max_cached_files",
        "physical_batch_size",
        "rank_physical_batch_size",
        "serving_batch_size",
        "tensor_parallel_size",
        "world_size",
    }
)
LIBERO_PREFIX_CAMERAS = (
    CameraGeometry("agentview", 256, 256),
    CameraGeometry("eye_in_hand", 256, 256),
)
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
_ALGORITHM_ENVIRONMENT_PREFIXES = (
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
_ALLOWED_ALGORITHM_ENVIRONMENT = frozenset(
    {
        "BLIS_NUM_THREADS",
        "CUBLAS_WORKSPACE_CONFIG",
        "CUDA_DEVICE_ORDER",
        "CUDA_VISIBLE_DEVICES",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "OMP_DYNAMIC",
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "RAYON_NUM_THREADS",
        "TORCH_NCCL_ASYNC_ERROR_HANDLING",
        "VECLIB_MAXIMUM_THREADS",
    }
)


def _source_tree_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    paths: list[Path] = []
    for relative in ("src/duo_vla", "configs"):
        paths.extend(
            path for path in (root / relative).rglob("*") if path.is_file() and "__pycache__" not in path.parts
        )
    paths.extend(
        path
        for path in (
            root / "scripts/run_libero_train.sh",
            root / "scripts/run_libero_train_single_gpu.sh",
            root / "scripts/run_libero_train_dp2.sh",
            root / "scripts/create_libero_dp2_fork.py",
            root / "scripts/create_libero_topology_fork.py",
            root / "scripts/bootstrap_train_single_gpu_env.sh",
            root / "scripts/train_libero.py",
            root / "envs/train-single-gpu/pyproject.toml",
            root / "envs/train-single-gpu/uv.lock",
            root / "pyproject.toml",
            root / "uv.lock",
        )
        if path.is_file()
    )
    for path in sorted(paths):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


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


def _merge_dp2_fork_run_contract(
    run_contract: dict[str, str],
    fork_contract: dict[str, str],
) -> None:
    """Attach an authenticated fork contract without overriding child identity."""

    if set(fork_contract) != _DP2_FORK_RUN_CONTRACT_FIELDS or any(
        not isinstance(name, str) or not isinstance(value, str) for name, value in fork_contract.items()
    ):
        raise ValueError("DP2 fork run-contract field inventory differs")
    conflicts = {
        name: {"expected": run_contract[name], "observed": fork_contract[name]}
        for name in set(run_contract) & set(fork_contract)
        if run_contract[name] != fork_contract[name]
    }
    if conflicts:
        raise ValueError(f"DP2 fork conflicts with the child run contract: {conflicts}")
    run_contract.update(fork_contract)


def _merge_topology_fork_run_contract(
    run_contract: dict[str, str],
    fork_contract: dict[str, str],
) -> None:
    """Attach the generic topology lineage without overriding child identity."""

    if set(fork_contract) != TOPOLOGY_FORK_CHECKPOINT_FIELDS or any(
        not isinstance(name, str) or not isinstance(value, str) for name, value in fork_contract.items()
    ):
        raise ValueError("topology fork run-contract field inventory differs")
    conflicts = {
        name: {"expected": run_contract[name], "observed": fork_contract[name]}
        for name in set(run_contract) & set(fork_contract)
        if run_contract[name] != fork_contract[name]
    }
    if conflicts:
        raise ValueError(f"topology fork conflicts with the child run contract: {conflicts}")
    run_contract.update(fork_contract)


def _authenticate_dataset_snapshot_distributed(snapshot_root: Path) -> dict[str, Any]:
    """Authenticate every LIBERO dataset byte on rank zero before constructing a reader."""

    authentication_group = dist.new_group(backend="gloo", timeout=LIBERO_DATASET_AUTHENTICATION_TIMEOUT)
    result: list[dict[str, Any] | None] = [None]
    try:
        if dist.get_rank() == 0:
            try:
                report = verify_huggingface_snapshot(snapshot_root, expected_revision=LIBERO_DATASET_REVISION)
                required = {
                    "content_inventory_sha256": LIBERO_DATASET_CONTENT_INVENTORY_SHA256,
                    "files_verified": LIBERO_DATASET_FILES_VERIFIED,
                    "total_bytes": LIBERO_DATASET_TOTAL_BYTES,
                    "tree_metadata_sha256": LIBERO_DATASET_TREE_SHA256,
                }
                mismatches = {
                    name: {"expected": expected, "observed": report.get(name)}
                    for name, expected in required.items()
                    if report.get(name) != expected
                }
                if mismatches:
                    raise ValueError(f"LIBERO dataset snapshot differs from the qualified identity: {mismatches}")
                result[0] = report
            except Exception as exc:
                result[0] = {"error": f"{type(exc).__name__}: {exc}"}
        dist.broadcast_object_list(result, src=0, group=authentication_group)
    finally:
        dist.destroy_process_group(authentication_group)
    payload = result[0]
    if not isinstance(payload, dict):
        raise RuntimeError("rank 0 did not broadcast a LIBERO dataset authentication result")
    if "error" in payload:
        raise RuntimeError(f"LIBERO dataset authentication failed: {payload['error']}")
    required = {
        "content_inventory_sha256": LIBERO_DATASET_CONTENT_INVENTORY_SHA256,
        "files_verified": LIBERO_DATASET_FILES_VERIFIED,
        "total_bytes": LIBERO_DATASET_TOTAL_BYTES,
        "tree_metadata_sha256": LIBERO_DATASET_TREE_SHA256,
    }
    if any(payload.get(name) != expected for name, expected in required.items()):
        raise RuntimeError("broadcast LIBERO dataset identity differs from the qualified identity")
    return payload


def _training_environment_name(
    *,
    expected_world_size: int,
    resolved_config: dict[str, Any] | None,
) -> str:
    """Resolve the pinned venv from process and explicit parallel topology."""

    distributed = (
        None if resolved_config is None else _data_parallel_contract(resolved_config, world_size=expected_world_size)
    )
    return "train-single-gpu" if expected_world_size == 1 or distributed is not None else "train"


def _configure_and_validate_training_runtime(
    project_root: Path,
    *,
    resolved_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Reject inherited algorithm overrides and enable strict CUDA determinism.

    A two-process job is not sufficient to distinguish legacy TP2 from DP2.
    When the caller supplies the resolved recipe, validate its explicit
    distributed contract and select the TP1-compatible environment for DP2.
    """

    project_root = project_root.resolve()
    _inventory_project_source_root()
    _validate_project_module_origins({"duo_vla", "duo_vla.runtime_integrity"})
    cache_root = Path(os.environ.get("DUO_VLA_CACHE_ROOT", "/root/.cache/duo-vla")).resolve()
    expected_world_size = canonical_visible_cuda_world_size(os.environ)
    environment_name = _training_environment_name(
        expected_world_size=expected_world_size,
        resolved_config=resolved_config,
    )
    train_venv = Path(os.environ.get("DUO_VLA_TRAIN_VENV", "")).resolve()
    if train_venv != (cache_root / f"venvs/{environment_name}").resolve():
        raise RuntimeError("LIBERO training venv differs from the canonical execution topology")
    python_hash_seed = os.environ.get("PYTHONHASHSEED")
    if python_hash_seed not in {"0", "1", "2"}:
        raise RuntimeError("LIBERO training requires PYTHONHASHSEED in {0,1,2}")
    expected_environment = {
        **REQUIRED_TRAIN_ENVIRONMENT,
        "CUDA_VISIBLE_DEVICES": os.environ["CUDA_VISIBLE_DEVICES"],
        "DUO_VLA_CACHE_ROOT": str(cache_root),
        "DUO_VLA_PROJECT_ROOT": str(project_root),
        "DUO_VLA_TRAIN_VENV": str(train_venv),
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
    present_algorithm_overrides = sorted(
        name
        for name in os.environ
        if name.startswith(_ALGORITHM_ENVIRONMENT_PREFIXES) and name not in _ALLOWED_ALGORITHM_ENVIRONMENT
    )
    if present_forbidden or present_algorithm_overrides:
        raise RuntimeError(
            "LIBERO training environment contains injection/algorithm overrides: "
            f"forbidden={present_forbidden}, algorithm_overrides={present_algorithm_overrides}"
        )
    observed_environment = {name: os.environ.get(name) for name in expected_environment}
    if observed_environment != expected_environment:
        raise RuntimeError(f"LIBERO training environment differs from the canonical launcher: {observed_environment}")
    if Path(sys.prefix).resolve() != train_venv:
        raise RuntimeError(f"LIBERO training requires the pinned train venv: {train_venv}")
    if sys.flags.safe_path != 1:
        raise RuntimeError("LIBERO training requires Python safe-path mode")
    if sys.flags.dont_write_bytecode != 1 or not sys.dont_write_bytecode:
        raise RuntimeError("LIBERO training requires -B")
    if sys.flags.no_user_site != 1 or site.ENABLE_USER_SITE:
        raise RuntimeError("LIBERO training requires the user site to be disabled")
    if sys.pycache_prefix != "/dev/null":
        raise RuntimeError("LIBERO training requires an impossible pycache lookup prefix")
    version = f"python{sys.version_info.major}.{sys.version_info.minor}"
    compact_version = f"python{sys.version_info.major}{sys.version_info.minor}"
    expected_sys_path = [
        str((project_root / "src").resolve()),
        str(Path(sys.base_prefix) / "lib" / f"{compact_version}.zip"),
        str(Path(sys.base_prefix) / "lib" / version),
        str(Path(sys.base_exec_prefix) / "lib" / version / "lib-dynload"),
        str(train_venv / "lib" / version / "site-packages"),
    ]
    if sys.path != expected_sys_path:
        raise RuntimeError(f"LIBERO training import search path differs: {sys.path}")
    rank_environment = validate_torchrun_rank_environment(
        os.environ,
        required=any(name in os.environ for name in ("RANK", "LOCAL_RANK", "WORLD_SIZE")),
        expected_world_size=expected_world_size,
    )
    venv_identity = content_address_train_venv(train_venv)
    configure_strict_cuda_determinism(torch)
    return {
        "algorithm_override_environment": {},
        "environment": dict(sorted(observed_environment.items())),
        "nccl_environment": {},
        "static_environment_sha256": static_environment_identity(observed_environment)["sha256"],
        "torchrun": rank_environment,
        "train_venv": venv_identity,
    }


def _execution_environment(runtime_preflight: dict[str, Any]) -> dict[str, Any]:
    gpu_properties = [torch.cuda.get_device_properties(index) for index in range(torch.cuda.device_count())]
    return {
        "authenticated_runtime": runtime_preflight,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        **deterministic_torch_runtime(torch),
        "gpu_capability": [list(torch.cuda.get_device_capability(index)) for index in range(torch.cuda.device_count())],
        "gpu_names": [properties.name for properties in gpu_properties],
        "gpu_total_memory_bytes": [int(properties.total_memory) for properties in gpu_properties],
        "gpu_uuids": [str(properties.uuid) for properties in gpu_properties],
        "peft": importlib.metadata.version("peft"),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "transformers": importlib.metadata.version("transformers"),
        "world_size": dist.get_world_size(),
    }


def _prefix_geometry_pins(config: dict[str, Any]) -> tuple[str, int]:
    benchmark = config.get("benchmark")
    if not isinstance(benchmark, dict):
        raise ValueError("resolved config has no benchmark table")
    content_sha256 = benchmark.get("prefix_geometry_content_sha256")
    fixed_width = benchmark.get("fixed_physical_prefix_width")
    if not (
        isinstance(content_sha256, str)
        and len(content_sha256) == 64
        and all(character in "0123456789abcdef" for character in content_sha256)
    ):
        raise ValueError("benchmark.prefix_geometry_content_sha256 must be 64 lowercase hexadecimal characters")
    if isinstance(fixed_width, bool) or not isinstance(fixed_width, int) or fixed_width <= 0:
        raise ValueError("benchmark.fixed_physical_prefix_width must be a positive integer")
    return content_sha256, fixed_width


def _model_snapshot_identity(snapshot_report: dict[str, Any]) -> SnapshotTreeIdentity:
    return SnapshotTreeIdentity.from_huggingface_report(
        DEFAULT_DIFFUSION_GEMMA_SPEC.model_id,
        snapshot_report,
    )


def _authenticate_prefix_geometry(
    artifact_path: Path,
    *,
    expected_content_sha256: str,
    expected_fixed_physical_prefix_width: int,
    model_snapshot_report: dict[str, Any],
    instructions: tuple[str, ...],
) -> dict[str, Any]:
    """Authenticate the externally pinned LIBERO geometry before model construction."""

    model_identity = _model_snapshot_identity(model_snapshot_report)
    return load_prefix_geometry_contract(
        artifact_path,
        expected_content_sha256=expected_content_sha256,
        expected_model_identity=model_identity,
        expected_processor_identity=model_identity,
        expected_ordered_cameras=LIBERO_PREFIX_CAMERAS,
        expected_instructions=instructions,
        expected_fixed_physical_prefix_width=expected_fixed_physical_prefix_width,
    )


def _authenticate_prefix_geometry_distributed(
    artifact_path: Path,
    *,
    expected_content_sha256: str,
    expected_fixed_physical_prefix_width: int,
    model_snapshot: Path,
    instructions: tuple[str, ...],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Hash the large model snapshot once on rank zero, then validate on every rank."""

    result: list[dict[str, Any] | None] = [None]
    if dist.get_rank() == 0:
        try:
            snapshot_report = verify_huggingface_snapshot(
                model_snapshot,
                expected_revision=DEFAULT_DIFFUSION_GEMMA_SPEC.revision,
            )
            prefix_geometry = _authenticate_prefix_geometry(
                artifact_path,
                expected_content_sha256=expected_content_sha256,
                expected_fixed_physical_prefix_width=expected_fixed_physical_prefix_width,
                model_snapshot_report=snapshot_report,
                instructions=instructions,
            )
            result[0] = {
                "model_snapshot_report": snapshot_report,
                "prefix_geometry": prefix_geometry,
            }
        except Exception as exc:
            result[0] = {"error": f"{type(exc).__name__}: {exc}"}
    dist.broadcast_object_list(result, src=0)
    payload = result[0]
    if not isinstance(payload, dict):
        raise RuntimeError("rank 0 did not broadcast prefix geometry authentication")
    if "error" in payload:
        raise RuntimeError(f"prefix geometry authentication failed: {payload['error']}")
    snapshot_report = payload.get("model_snapshot_report")
    prefix_geometry = payload.get("prefix_geometry")
    if not isinstance(snapshot_report, dict) or not isinstance(prefix_geometry, dict):
        raise RuntimeError("rank 0 broadcast malformed prefix geometry authentication")
    model_identity = _model_snapshot_identity(snapshot_report)
    validated = validate_prefix_geometry_contract(
        prefix_geometry,
        expected_content_sha256=expected_content_sha256,
        expected_model_identity=model_identity,
        expected_processor_identity=model_identity,
        expected_ordered_cameras=LIBERO_PREFIX_CAMERAS,
        expected_instructions=instructions,
        expected_fixed_physical_prefix_width=expected_fixed_physical_prefix_width,
    )
    return validated, snapshot_report


def _plain_config_integer(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    return value


def _data_parallel_contract(config: dict[str, Any], *, world_size: int | None = None) -> dict[str, Any] | None:
    """Validate and return the explicit DP2 topology, leaving legacy TP recipes unchanged."""

    distributed = config.get("distributed")
    if distributed is None:
        return None
    if not isinstance(distributed, dict):
        raise ValueError("resolved config distributed section must be a table")
    expected: dict[str, object] = {
        "strategy": DATA_PARALLEL_STRATEGY,
        "world_size": DP2_DATA_PARALLEL_SIZE,
        "data_parallel_size": DP2_DATA_PARALLEL_SIZE,
        "tensor_parallel_size": DP2_TENSOR_PARALLEL_SIZE,
        "rank_physical_batch_size": DP2_RANK_PHYSICAL_BATCH_SIZE,
        "canonical_plan_partition": DATA_PARALLEL_PLAN_PARTITION,
        "gradient_reduction": DATA_PARALLEL_GRADIENT_REDUCTION,
    }
    if set(distributed) != set(expected):
        raise ValueError(
            "resolved config distributed fields differ from the DP2 contract: "
            f"expected={sorted(expected)}, observed={sorted(distributed)}"
        )
    mismatches = {
        name: {"expected": expected_value, "observed": distributed.get(name)}
        for name, expected_value in expected.items()
        if distributed.get(name) != expected_value
    }
    if world_size is not None and world_size != DP2_DATA_PARALLEL_SIZE:
        mismatches["runtime_world_size"] = {
            "expected": DP2_DATA_PARALLEL_SIZE,
            "observed": world_size,
        }
    if mismatches:
        raise ValueError(f"resolved config has an unsupported data-parallel topology: {mismatches}")
    training = config.get("training")
    if not isinstance(training, dict) or training.get("max_cached_files") != DP2_MAX_CACHED_FILES:
        raise ValueError(f"DP2 requires training.max_cached_files={DP2_MAX_CACHED_FILES} in the frozen recipe")
    return distributed


def _validate_dp2_fork_cli(args: argparse.Namespace) -> None:
    """Keep the frozen child recipe authoritative at the fork boundary."""

    contract_overrides = {
        name: getattr(args, name)
        for name in (
            "task",
            "seed",
            "total_updates",
            "warmup_updates",
            "microbatch_size",
            "gradient_accumulation_steps",
            "validation_interval",
            "validation_samples",
            "checkpoint_interval",
            "permanent_checkpoint_interval",
            "log_interval",
        )
        if getattr(args, name) is not None
    }
    if args.max_cached_files != DP2_MAX_CACHED_FILES:
        contract_overrides["max_cached_files"] = args.max_cached_files
    if contract_overrides:
        raise ValueError(
            f"--fork-from uses the frozen child recipe and rejects run-contract overrides: {sorted(contract_overrides)}"
        )


def _validate_dp2_fork_digest_argument(
    *,
    fork_from: Path | None,
    resume: Path | None,
    expected_sha256: list[str] | None,
) -> str | None:
    """Make the preregistered digest, rather than its adjacent sidecar, authoritative."""

    if fork_from is None:
        if expected_sha256 is not None:
            raise ValueError("--expected-fork-manifest-sha256 is valid only with --fork-from, never --resume")
        return None
    if resume is not None:  # argparse already prevents this; keep the helper fail-closed in isolation.
        raise ValueError("--fork-from and --resume are mutually exclusive")
    if (
        not isinstance(expected_sha256, list)
        or len(expected_sha256) != 1
        or re.fullmatch(r"[0-9a-f]{64}", expected_sha256[0]) is None
    ):
        raise ValueError("--fork-from requires one preregistered --expected-fork-manifest-sha256 64-hex digest")
    return expected_sha256[0]


def _validate_topology_fork_cli(args: argparse.Namespace, manifest: Mapping[str, Any]) -> None:
    """Keep the generic topology-fork child recipe authoritative."""

    child = manifest.get("child")
    if not isinstance(child, Mapping):
        raise ValueError("topology fork manifest has no child contract")
    contract_overrides = {
        name: getattr(args, name)
        for name in (
            "task",
            "total_updates",
            "warmup_updates",
            "microbatch_size",
            "gradient_accumulation_steps",
            "validation_interval",
            "validation_samples",
            "checkpoint_interval",
            "permanent_checkpoint_interval",
            "log_interval",
        )
        if getattr(args, name) is not None
    }
    if args.seed is not None:
        contract_overrides["seed"] = args.seed
    if args.max_cached_files != child.get("max_cached_files"):
        contract_overrides["max_cached_files"] = args.max_cached_files
    if contract_overrides:
        raise ValueError(
            "--topology-fork-from uses the frozen child recipe and rejects "
            f"run-contract overrides: {sorted(contract_overrides)}"
        )


def _validate_topology_fork_digest_argument(
    *,
    topology_fork_from: Path | None,
    resume: Path | None,
    legacy_fork_from: Path | None,
    expected_sha256: list[str] | None,
) -> str | None:
    if topology_fork_from is None:
        if expected_sha256 is not None:
            raise ValueError("--expected-topology-fork-manifest-sha256 is valid only with --topology-fork-from")
        return None
    if resume is not None or legacy_fork_from is not None:
        raise ValueError("topology fork, legacy fork, and resume modes are mutually exclusive")
    if (
        not isinstance(expected_sha256, list)
        or len(expected_sha256) != 1
        or re.fullmatch(r"[0-9a-f]{64}", expected_sha256[0]) is None
    ):
        raise ValueError(
            "--topology-fork-from requires one preregistered --expected-topology-fork-manifest-sha256 64-hex digest"
        )
    return expected_sha256[0]


def _validate_dp2_restore_mode(
    *,
    is_data_parallel: bool,
    has_fork_manifest: bool,
    has_resume_checkpoint: bool,
    has_topology_fork_manifest: bool = False,
) -> None:
    """Forbid an unlineaged DP2 start even when the trainer bypasses its launcher."""

    if has_fork_manifest and not is_data_parallel:
        raise ValueError("--fork-from requires the exact DP2 distributed recipe")
    if is_data_parallel and sum((has_fork_manifest, has_topology_fork_manifest, has_resume_checkpoint)) != 1:
        raise ValueError("DP2 requires exactly one authenticated legacy/topology fork or lineage-valid --resume")


def _validate_batch_contract(config: dict[str, Any]) -> tuple[int, int, int]:
    """Validate the fixed-global-batch geometry and return train/accum/serve sizes."""

    optimization = config.get("optimization")
    if not isinstance(optimization, dict):
        raise ValueError("resolved config has no optimization table")
    physical_batch_size = _plain_config_integer(
        optimization.get("physical_batch_size"),
        name="optimization.physical_batch_size",
    )
    microbatch_size = _plain_config_integer(
        optimization.get("microbatch_size"),
        name="optimization.microbatch_size",
    )
    accumulation_steps = _plain_config_integer(
        optimization.get("gradient_accumulation_steps"),
        name="optimization.gradient_accumulation_steps",
    )
    global_batch_size = _plain_config_integer(
        optimization.get("global_batch_size"),
        name="optimization.global_batch_size",
    )
    serving_batch_size = _plain_config_integer(
        optimization.get("serving_batch_size", DEFAULT_SERVING_BATCH_SIZE),
        name="optimization.serving_batch_size",
    )
    if physical_batch_size not in SUPPORTED_PHYSICAL_BATCH_SIZES:
        supported = ", ".join(str(value) for value in sorted(SUPPORTED_PHYSICAL_BATCH_SIZES))
        raise ValueError(f"optimization.physical_batch_size must be one of {{{supported}}}")
    if serving_batch_size not in SUPPORTED_PHYSICAL_BATCH_SIZES:
        supported = ", ".join(str(value) for value in sorted(SUPPORTED_PHYSICAL_BATCH_SIZES))
        raise ValueError(f"optimization.serving_batch_size must be one of {{{supported}}}")
    if serving_batch_size != DEFAULT_SERVING_BATCH_SIZE:
        raise ValueError(
            f"optimization.serving_batch_size must remain {DEFAULT_SERVING_BATCH_SIZE} "
            "until another singleton-serving geometry is qualified"
        )
    if microbatch_size != physical_batch_size:
        raise ValueError("optimization.microbatch_size must equal optimization.physical_batch_size")
    if global_batch_size != GLOBAL_BATCH_SIZE:
        raise ValueError(f"optimization.global_batch_size must equal {GLOBAL_BATCH_SIZE}")
    distributed = _data_parallel_contract(config)
    data_parallel_size = 1 if distributed is None else int(distributed["data_parallel_size"])
    if distributed is not None and physical_batch_size != int(distributed["rank_physical_batch_size"]):
        raise ValueError("optimization.physical_batch_size must equal distributed.rank_physical_batch_size")
    expected_accumulation = GLOBAL_BATCH_SIZE // (physical_batch_size * data_parallel_size)
    if accumulation_steps != expected_accumulation:
        raise ValueError(
            "optimization.gradient_accumulation_steps must equal "
            f"{expected_accumulation} for physical batch {physical_batch_size}"
        )
    if physical_batch_size % CANONICAL_STREAM_BATCH_SIZE:
        raise ValueError("optimization.physical_batch_size must coalesce complete canonical B8 stream chunks")
    return physical_batch_size, accumulation_steps, serving_batch_size


def _expected_execution_profile(
    *,
    world_size: int,
    expert_batch_isolation: str,
    physical_batch_size: int,
    serving_batch_size: int,
    data_parallel_size: int = 1,
) -> str | None:
    """Name the exact batch/backend topology without changing legacy profiles."""

    if expert_batch_isolation not in SUPPORTED_EXPERT_BATCH_ISOLATIONS:
        raise ValueError(f"unsupported expert batch-isolation backend: {expert_batch_isolation!r}")
    if data_parallel_size == DP2_DATA_PARALLEL_SIZE:
        if (
            world_size != DP2_DATA_PARALLEL_SIZE
            or expert_batch_isolation != SAMPLE_ISOLATED_GROUPED_MM_V2
            or physical_batch_size != DP2_RANK_PHYSICAL_BATCH_SIZE
            or serving_batch_size != DEFAULT_SERVING_BATCH_SIZE
        ):
            raise ValueError("DP=2 is qualified only for fused-v2 rank-B32/global-B64/serve-B8")
        return DP2_EXECUTION_PROFILE
    if data_parallel_size != 1:
        raise ValueError(f"unsupported data-parallel size: {data_parallel_size}")
    if world_size == 2:
        if (
            expert_batch_isolation != EXPERT_BATCH_ISOLATION
            or physical_batch_size != PHYSICAL_BATCH_SIZE
            or serving_batch_size != DEFAULT_SERVING_BATCH_SIZE
        ):
            raise ValueError("TP=2 is qualified only for the legacy v1/B8 execution geometry")
        return None
    if world_size != 1:
        raise ValueError(f"unsupported tensor-parallel world size: {world_size}")
    if (
        expert_batch_isolation == EXPERT_BATCH_ISOLATION
        and physical_batch_size == PHYSICAL_BATCH_SIZE
        and serving_batch_size == DEFAULT_SERVING_BATCH_SIZE
    ):
        return "duovla-single-gpu-tp1-v1"
    backend = "fused-v2" if expert_batch_isolation == SAMPLE_ISOLATED_GROUPED_MM_V2 else "sequential-v1"
    return f"duovla-single-gpu-tp1-{backend}-train-b{physical_batch_size}-serve-b{serving_batch_size}-v1"


def _install_expert_batch_isolation(model: torch.nn.Module, *, backend: str, physical_batch_size: int) -> None:
    if backend == EXPERT_BATCH_ISOLATION:
        install_sample_isolated_grouped_mm_experts(model, physical_batch_size=physical_batch_size)
    elif backend == SAMPLE_ISOLATED_GROUPED_MM_V2:
        install_sample_isolated_grouped_mm_experts_v2(model, physical_batch_size=physical_batch_size)
    else:  # pragma: no cover - config validation rejects this before model construction
        raise ValueError(f"unsupported expert batch-isolation backend: {backend!r}")


def _verify_expert_batch_isolation(model: torch.nn.Module, *, backend: str, physical_batch_size: int) -> None:
    if backend == EXPERT_BATCH_ISOLATION:
        verify_sample_isolated_grouped_mm_experts(model, physical_batch_size=physical_batch_size)
    elif backend == SAMPLE_ISOLATED_GROUPED_MM_V2:
        verify_sample_isolated_grouped_mm_experts_v2(model, physical_batch_size=physical_batch_size)
    else:  # pragma: no cover - config validation rejects this before model construction
        raise ValueError(f"unsupported expert batch-isolation backend: {backend!r}")


def _validate_and_build_interface_config(config: dict[str, Any]) -> ActionInterfaceConfig:
    model = config["model"]
    action = config["action"]
    lora = config["lora"]
    benchmark = config["benchmark"]
    optimization = config["optimization"]
    training = config["training"]
    physical_batch_size, accumulation_steps, serving_batch_size = _validate_batch_contract(config)
    distributed = _data_parallel_contract(config, world_size=dist.get_world_size())
    data_parallel_size = 1 if distributed is None else int(distributed["data_parallel_size"])
    tensor_parallel_size = dist.get_world_size() if distributed is None else int(distributed["tensor_parallel_size"])
    expert_batch_isolation = model.get("expert_batch_isolation")
    if not isinstance(expert_batch_isolation, str):
        raise ValueError("model.expert_batch_isolation must be a string")
    expected_execution_profile = _expected_execution_profile(
        world_size=dist.get_world_size(),
        expert_batch_isolation=expert_batch_isolation,
        physical_batch_size=physical_batch_size,
        serving_batch_size=serving_batch_size,
        data_parallel_size=data_parallel_size,
    )
    required = {
        "model.id": (model["id"], DEFAULT_DIFFUSION_GEMMA_SPEC.model_id),
        "model.revision": (model["revision"], DEFAULT_DIFFUSION_GEMMA_SPEC.revision),
        "model.dtype": (model["dtype"], "bfloat16"),
        "model.tensor_parallel_size": (int(model["tensor_parallel_size"]), tensor_parallel_size),
        "model.attention_implementation": (model["attention_implementation"], "sdpa"),
        "model.experts_implementation": (
            model["experts_implementation"],
            GROUPED_MM_EXPERTS_IMPLEMENTATION,
        ),
        "model.expert_batch_isolation": (model["expert_batch_isolation"], expert_batch_isolation),
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
        "benchmark.dataset_id": (benchmark["dataset_id"], "HuggingFaceVLA/libero"),
        "benchmark.dataset_revision": (benchmark["dataset_revision"], LIBERO_DATASET_REVISION),
        "benchmark.state_dimension": (int(benchmark["state_dimension"]), 8),
        "optimization.physical_batch_size": (int(optimization["physical_batch_size"]), physical_batch_size),
        "optimization.microbatch_size": (int(optimization["microbatch_size"]), physical_batch_size),
        "optimization.gradient_accumulation_steps": (
            int(optimization["gradient_accumulation_steps"]),
            accumulation_steps,
        ),
        "optimization.global_batch_size": (int(optimization["global_batch_size"]), GLOBAL_BATCH_SIZE),
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
    _prefix_geometry_pins(config)
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


def _execution_geometry(config: dict[str, Any]) -> dict[str, str | int]:
    content_sha256, fixed_width = _prefix_geometry_pins(config)
    physical_batch_size, _, serving_batch_size = _validate_batch_contract(config)
    geometry: dict[str, str | int] = {
        "experts_implementation": str(config["model"]["experts_implementation"]),
        "expert_batch_isolation": str(config["model"]["expert_batch_isolation"]),
        "physical_batch_size": physical_batch_size,
        "serving_batch_size": serving_batch_size,
        "fixed_physical_prefix_width": fixed_width,
        "prefix_geometry_content_sha256": content_sha256,
    }
    if config.get("execution_profile") is not None:
        geometry.update(
            execution_profile=str(config["execution_profile"]),
            tensor_parallel_size=int(config["model"]["tensor_parallel_size"]),
        )
    distributed = _data_parallel_contract(config)
    if distributed is not None:
        geometry.update(
            strategy=str(distributed["strategy"]),
            world_size=int(distributed["world_size"]),
            data_parallel_size=int(distributed["data_parallel_size"]),
            rank_physical_batch_size=int(distributed["rank_physical_batch_size"]),
            canonical_plan_partition=str(distributed["canonical_plan_partition"]),
            gradient_reduction=str(distributed["gradient_reduction"]),
        )
    if config["model"]["expert_batch_isolation"] == SAMPLE_ISOLATED_GROUPED_MM_V2:
        geometry["shared_weight_kernel_sha256"] = file_sha256(
            Path(__file__).resolve().parents[1] / "src/duo_vla/backbones/shared_weight_grouped_mm_triton.py"
        )
    return geometry


def _validate_checkpoint_execution_geometry(
    checkpoint_dir: Path,
    manifest: dict[str, Any],
    config: dict[str, Any],
) -> None:
    expected = _execution_geometry(config)
    mismatches = {
        name: {"expected": value, "observed": manifest.get(name)}
        for name, value in expected.items()
        if manifest.get(name) != value
    }
    if manifest.get("execution_geometry") != expected:
        mismatches["execution_geometry"] = {
            "expected": expected,
            "observed": manifest.get("execution_geometry"),
        }
    if mismatches:
        raise ValueError(f"checkpoint fixed-batch execution geometry differs: {mismatches}")
    artifact = manifest.get("artifacts", {}).get("prefix_geometry")
    if not isinstance(artifact, dict) or not isinstance(artifact.get("path"), str):
        raise ValueError("checkpoint has no prefix geometry artifact")
    artifact_path = checkpoint_dir / artifact["path"]
    try:
        artifact_path.resolve().relative_to(checkpoint_dir.resolve())
    except ValueError as exc:
        raise ValueError("checkpoint prefix geometry artifact escapes the checkpoint directory") from exc
    load_prefix_geometry_contract(
        artifact_path,
        expected_content_sha256=str(expected["prefix_geometry_content_sha256"]),
        expected_fixed_physical_prefix_width=int(expected["fixed_physical_prefix_width"]),
    )


def _broadcast_rank0_error(error: str | None) -> None:
    value = [error]
    dist.broadcast_object_list(value, src=0)
    if value[0] is not None:
        raise RuntimeError(value[0])


def _initialize_run_journal(
    output_dir: Path,
    *,
    config_sha256: str,
    resume: Path | None,
    recover_bootstrap: bool,
    new_run_uuid: str | None = None,
) -> tuple[str, str | None]:
    result: list[dict[str, Any] | None] = [None]
    if dist.get_rank() == 0:
        try:
            if resume is None and not recover_bootstrap:
                journal = create_run_journal(
                    output_dir,
                    config_sha256=config_sha256,
                    run_uuid=new_run_uuid,
                )
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
                    journal = create_run_journal(
                        output_dir,
                        config_sha256=config_sha256,
                        run_uuid=new_run_uuid,
                    )
                if new_run_uuid is not None and journal.run_uuid != new_run_uuid:
                    raise ValueError("bootstrap journal run UUID differs from the frozen fork manifest")
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
                if new_run_uuid is not None:
                    raise ValueError("ordinary resume cannot select a new run UUID")
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
        raise RuntimeError(f"{context} failed across distributed ranks: {errors}")


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
        raise RuntimeError(f"replicated optimizer state diverged across distributed ranks: {hashes}")
    return local_hash


def _assert_distributed_gradient_health(parameters: list[torch.nn.Parameter]) -> None:
    local = finite_gradient_audit(parameters)
    audits: list[tuple[int, int] | None] = [None] * dist.get_world_size()
    dist.all_gather_object(audits, local)
    if any(audit != (0, 0) for audit in audits):
        raise RuntimeError(f"gradient audit failed across distributed ranks: {audits}")


def _assert_initial_gradients_present(
    lora_partition,
    interface_named_parameters: list[tuple[str, torch.nn.Parameter]],
    *,
    data_parallel: bool = False,
) -> None:
    local_missing = [
        name for name, parameter in [*lora_partition.replicated, *interface_named_parameters] if parameter.grad is None
    ]
    missing_by_rank: list[list[str] | None] = [None] * dist.get_world_size()
    dist.all_gather_object(missing_by_rank, local_missing)
    if any(missing for missing in missing_by_rank):
        raise RuntimeError(f"initial replicated gradients are missing: {missing_by_rank}")
    # DP ranks intentionally own disjoint canonical plans, so their local
    # gradients differ until the explicit SUM collective below.
    if data_parallel:
        return
    for name, parameter in lora_partition.replicated:
        assert parameter.grad is not None
        assert_replicated_tensor(f"initial_gradient.{name}", parameter.grad)
    for name, parameter in interface_named_parameters:
        assert parameter.grad is not None
        assert_replicated_tensor(f"initial_interface_gradient.{name}", parameter.grad)


def _assert_data_parallel_trainables_are_replicated(
    lora_partition,
    lora_named_parameters: Sequence[tuple[str, torch.nn.Parameter]],
) -> None:
    """Fail closed if a TP shard accidentally enters the pure-DP optimizer."""

    if lora_partition.sharded:
        raise RuntimeError("DP2 requires every trainable LoRA parameter to be a full replica")
    expected = {id(parameter) for _, parameter in lora_named_parameters}
    observed = {id(parameter) for _, parameter in lora_partition.replicated}
    if observed != expected or len(observed) != len(tuple(lora_named_parameters)):
        raise RuntimeError("DP2 LoRA replica partition does not cover every trainable exactly once")


def _sum_data_parallel_gradients_(
    named_parameters: Sequence[tuple[str, torch.nn.Parameter]],
) -> None:
    """SUM globally-normalized full-replica gradients in one deterministic bucket.

    This deliberately does not use DDP: DDP's default division by world size
    would halve gradients that were already normalized by the global element
    count. Frozen backbone/expert tensors are absent from ``named_parameters``.
    """

    entries = tuple(sorted(named_parameters, key=lambda item: item[0]))
    if dist.get_world_size() != DP2_DATA_PARALLEL_SIZE:
        raise RuntimeError("DP2 gradient reduction requires exactly two ranks")
    if not entries or len({name for name, _ in entries}) != len(entries):
        raise ValueError("DP2 gradient reduction requires unique named trainables")
    if len({id(parameter) for _, parameter in entries}) != len(entries):
        raise ValueError("DP2 gradient reduction received duplicate trainable tensors")
    gradients = []
    for name, parameter in entries:
        gradient = parameter.grad
        if gradient is None:
            raise RuntimeError(f"DP2 trainable has no local gradient: {name}")
        if gradient.layout != torch.strided or gradient.is_sparse:
            raise RuntimeError(f"DP2 supports only dense strided gradients: {name}")
        gradients.append(gradient)
    reference = gradients[0]
    if any(gradient.device != reference.device or gradient.dtype != reference.dtype for gradient in gradients):
        raise RuntimeError("DP2 gradient bucket requires one device and dtype")
    flat = torch.cat([gradient.reshape(-1) for gradient in gradients])
    dist.all_reduce(flat, op=dist.ReduceOp.SUM)
    offset = 0
    for gradient in gradients:
        count = gradient.numel()
        gradient.copy_(flat.narrow(0, offset, count).view_as(gradient))
        offset += count


def _assert_replicated_gradient_values(
    named_parameters: Sequence[tuple[str, torch.nn.Parameter]],
) -> str:
    """Require one byte-identical post-reduction gradient bucket on every rank."""

    entries = sorted(named_parameters, key=lambda item: item[0])
    if not entries:
        raise ValueError("named gradient parameters must not be empty")
    digest = hashlib.sha256()
    for name, parameter in entries:
        if parameter.grad is None:
            raise RuntimeError(f"post-reduction gradient is missing: {name}")
        local = parameter.grad.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(local.dtype).encode())
        digest.update(str(tuple(local.shape)).encode())
        digest.update(local.reshape(-1).view(torch.uint8).numpy().tobytes())
    local_hash = digest.hexdigest()
    hashes: list[str | None] = [None] * dist.get_world_size()
    dist.all_gather_object(hashes, local_hash)
    if any(value != local_hash for value in hashes):
        raise RuntimeError(f"post-reduction gradients diverged across DP ranks: {hashes}")
    return local_hash


def _sum_data_parallel_scalar(value: float | int, *, device: torch.device, dtype: torch.dtype) -> float | int:
    """Return a DP2 SUM while preserving integer counts exactly."""

    if dist.get_world_size() != DP2_DATA_PARALLEL_SIZE:
        raise RuntimeError("DP2 scalar reduction requires exactly two ranks")
    reduced = torch.tensor(value, device=device, dtype=dtype)
    dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
    return int(reduced.item()) if not dtype.is_floating_point else float(reduced.item())


def _max_data_parallel_seconds(value: float, *, device: torch.device) -> float:
    if dist.get_world_size() != DP2_DATA_PARALLEL_SIZE:
        raise RuntimeError("DP2 timing reduction requires exactly two ranks")
    reduced = torch.tensor(value, device=device, dtype=torch.float64)
    dist.all_reduce(reduced, op=dist.ReduceOp.MAX)
    return float(reduced.item())


def _data_parallel_peak_memory_bytes(device: torch.device) -> list[int]:
    """Gather exact local allocator peaks in stable rank order."""

    if dist.get_world_size() != DP2_DATA_PARALLEL_SIZE:
        raise RuntimeError("DP2 memory reporting requires exactly two ranks")
    local_peak = int(torch.cuda.max_memory_allocated(device))
    peaks: list[int | None] = [None] * DP2_DATA_PARALLEL_SIZE
    dist.all_gather_object(peaks, local_peak)
    if any(type(value) is not int or value < 0 for value in peaks):
        raise RuntimeError(f"DP2 peak-memory gather returned invalid values: {peaks}")
    return [value for value in peaks if value is not None]


def _data_parallel_peak_memory_gib(device: torch.device) -> list[float]:
    return [value / 2**30 for value in _data_parallel_peak_memory_bytes(device)]


def _assert_distributed_trainer_state(
    trainer_state: TrainerState,
    checkpoint_manifest: dict[str, Any],
) -> None:
    serialized = trainer_state.to_dict()
    states: list[dict[str, Any] | None] = [None] * dist.get_world_size()
    dist.all_gather_object(states, serialized)
    if any(state != serialized for state in states):
        raise RuntimeError(f"distributed ranks restored different trainer states: {states}")
    manifest_state = checkpoint_manifest.get("trainer_state")
    if manifest_state != serialized:
        raise ValueError("checkpoint manifest and rank-state trainer progress disagree")


def _fixed_distinct_anchors(
    sampler: TaskUniformAnchorSampler,
    *,
    count: int,
    seed: int,
) -> tuple[LiberoAnchor, ...]:
    if count <= 0:
        raise ValueError("count must be positive")
    if count > sampler.population_size:
        raise ValueError(f"cannot draw {count} distinct anchors from a population of {sampler.population_size}")
    generator = torch.Generator().manual_seed(seed)
    anchors: list[LiberoAnchor] = []
    seen: set[tuple[int, int]] = set()
    while len(anchors) < count:
        anchor = sampler.draw(generator)
        identity = (anchor.episode_index, anchor.frame_index)
        if identity not in seen:
            anchors.append(anchor)
            seen.add(identity)
    return tuple(anchors)


def _processor_inputs(
    processor,
    samples,
    device: torch.device,
    prefix_geometry: dict[str, Any],
    *,
    expected_batch_size: int = PHYSICAL_BATCH_SIZE,
):
    if expected_batch_size not in SUPPORTED_PHYSICAL_BATCH_SIZES:
        raise ValueError("LIBERO processor received an unsupported physical batch size")
    if len(samples) != expected_batch_size:
        raise ValueError(f"LIBERO processor requires physical batch {expected_batch_size}, observed {len(samples)}")
    camera_arrays: list[tuple[np.ndarray, np.ndarray]] = []
    for index, sample in enumerate(samples):
        third_person = np.asarray(sample.observation.third_person)
        wrist = np.asarray(sample.observation.wrist)
        for camera, values in zip(LIBERO_PREFIX_CAMERAS, (third_person, wrist), strict=True):
            if values.shape != camera.shape or values.dtype != np.uint8:
                raise ValueError(
                    f"LIBERO sample {index} camera {camera.name!r} must be uint8{camera.shape}, "
                    f"observed dtype={values.dtype}, shape={values.shape}"
                )
        camera_arrays.append((third_person, wrist))
    conversations = [
        [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": Image.fromarray(third_person)},
                    {"type": "image", "image": Image.fromarray(wrist)},
                    {"type": "text", "text": sample.instruction},
                ],
            }
        ]
        for sample, (third_person, wrist) in zip(samples, camera_arrays, strict=True)
    ]
    fixed_width = int(prefix_geometry["geometry"]["fixed_physical_prefix_width"])
    padding_side = str(prefix_geometry["tokenization"]["padding_side"])
    inputs = apply_fixed_prefix_chat_template(
        processor,
        conversations,
        fixed_physical_prefix_width=fixed_width,
        padding_side=padding_side,
        expected_batch_size=expected_batch_size,
        images_per_prefix=len(LIBERO_PREFIX_CAMERAS),
    )
    expected_lengths = {
        record["instruction"]: int(record["valid_prefix_length"])
        for record in prefix_geometry["instruction_inventory"]["records"]
    }
    try:
        expected = tuple(expected_lengths[sample.instruction] for sample in samples)
    except KeyError as exc:
        raise ValueError(
            f"LIBERO instruction is absent from the authenticated prefix inventory: {exc.args[0]!r}"
        ) from exc
    observed = tuple(int(value) for value in inputs["attention_mask"].bool().sum(dim=1).tolist())
    if observed != expected:
        raise ValueError(
            f"processor valid-prefix lengths differ from the authenticated geometry: expected={expected}, "
            f"observed={observed}"
        )
    return inputs.to(device)


def _materialize_batch(
    dataset: LiberoParquetDataset,
    sampler: TaskUniformAnchorSampler,
    *,
    count: int,
    seed: int,
    state_normalizer,
    action_normalizer,
) -> LiberoBatch:
    anchors = _fixed_distinct_anchors(sampler, count=count, seed=seed)
    samples = dataset.sample_many(anchors)
    return collate_libero_samples(
        samples,
        state_normalizer=state_normalizer,
        action_normalizer=action_normalizer,
    )


def _materialize_canonical_batches(
    dataset: LiberoParquetDataset,
    sampler: TaskUniformAnchorSampler,
    plans: Sequence[MicrobatchPlan],
    *,
    state_normalizer,
    action_normalizer,
) -> tuple[LiberoBatch, ...]:
    """Materialize plans only as canonical B8 chunks, independent of forward width."""

    values = tuple(plans)
    if not values:
        raise ValueError("at least one canonical plan is required")
    return tuple(
        _materialize_batch(
            dataset,
            sampler,
            count=CANONICAL_STREAM_BATCH_SIZE,
            seed=plan.data_seed,
            state_normalizer=state_normalizer,
            action_normalizer=action_normalizer,
        )
        for plan in values
    )


def _group_canonical_plans(
    plans: Sequence[MicrobatchPlan],
    *,
    physical_batch_size: int,
) -> tuple[tuple[MicrobatchPlan, ...], ...]:
    if physical_batch_size not in SUPPORTED_PHYSICAL_BATCH_SIZES:
        raise ValueError("cannot group plans for an unsupported physical batch size")
    canonical_per_forward = physical_batch_size // CANONICAL_STREAM_BATCH_SIZE
    values = tuple(plans)
    if not values or len(values) % canonical_per_forward:
        raise ValueError("canonical plan count does not form complete physical forwards")
    return tuple(
        values[start : start + canonical_per_forward] for start in range(0, len(values), canonical_per_forward)
    )


def _canonical_update_plan_groups(
    seed: int,
    update: int,
    *,
    physical_batch_size: int,
) -> tuple[tuple[MicrobatchPlan, ...], ...]:
    """Keep the B8 seed stream invariant while changing physical-forward width."""

    canonical = make_update_plan(
        seed,
        update,
        gradient_accumulation_steps=CANONICAL_MICROSTEPS_PER_UPDATE,
    )
    return _group_canonical_plans(canonical, physical_batch_size=physical_batch_size)


def _partition_canonical_plans_by_rank(
    plans: Sequence[MicrobatchPlan],
    *,
    rank: int,
    data_parallel_size: int,
) -> tuple[MicrobatchPlan, ...]:
    """Assign one contiguous, non-overlapping portion of the canonical stream."""

    values = tuple(plans)
    if data_parallel_size != DP2_DATA_PARALLEL_SIZE:
        raise ValueError("canonical DP partition currently supports exactly two ranks")
    if isinstance(rank, bool) or not isinstance(rank, int) or not 0 <= rank < data_parallel_size:
        raise ValueError("data-parallel rank is out of range")
    if not values or len(values) % data_parallel_size:
        raise ValueError("canonical plan stream cannot be divided evenly across DP ranks")
    plans_per_rank = len(values) // data_parallel_size
    start = rank * plans_per_rank
    return values[start : start + plans_per_rank]


def _rank_canonical_update_plan_groups(
    seed: int,
    update: int,
    *,
    rank: int,
    data_parallel_size: int,
    physical_batch_size: int,
) -> tuple[tuple[MicrobatchPlan, ...], ...]:
    """Derive all eight plans globally, then select the rank's canonical half."""

    canonical = make_update_plan(
        seed,
        update,
        gradient_accumulation_steps=CANONICAL_MICROSTEPS_PER_UPDATE,
    )
    local = _partition_canonical_plans_by_rank(
        canonical,
        rank=rank,
        data_parallel_size=data_parallel_size,
    )
    return _group_canonical_plans(local, physical_batch_size=physical_batch_size)


def _coalesce_canonical_batches(
    batches: Sequence[LiberoBatch],
    *,
    physical_batch_size: int,
) -> LiberoBatch:
    values = tuple(batches)
    expected_chunks = physical_batch_size // CANONICAL_STREAM_BATCH_SIZE
    if physical_batch_size not in SUPPORTED_PHYSICAL_BATCH_SIZES or len(values) != expected_chunks:
        raise ValueError("physical forward must contain the exact number of canonical B8 batches")
    if any(batch.batch_size != CANONICAL_STREAM_BATCH_SIZE for batch in values):
        raise ValueError("only complete canonical B8 batches may be coalesced")
    samples = tuple(sample for batch in values for sample in batch.samples)
    states = torch.cat([batch.states for batch in values], dim=0)
    clean_actions = torch.cat([batch.clean_actions for batch in values], dim=0)
    action_valid_mask = torch.cat([batch.action_valid_mask for batch in values], dim=0)
    if (
        len(samples) != physical_batch_size
        or states.shape[0] != physical_batch_size
        or clean_actions.shape[0] != physical_batch_size
        or action_valid_mask.shape[0] != physical_batch_size
    ):
        raise ValueError("coalesced LIBERO batch has an invalid physical width")
    return LiberoBatch(
        samples=samples,
        states=states,
        clean_actions=clean_actions,
        action_valid_mask=action_valid_mask,
    )


def _canonical_training_pair(
    clean_actions: torch.Tensor,
    policy_contract: PolicyContract,
    plans: Sequence[MicrobatchPlan],
) -> PolicyTrainingPair:
    """Generate each canonical B8 objective slice before concatenating it."""

    values = tuple(plans)
    expected_batch_size = len(values) * CANONICAL_STREAM_BATCH_SIZE
    if not values or clean_actions.shape[0] != expected_batch_size:
        raise ValueError("clean actions do not match the canonical objective-plan slices")
    pairs = []
    for canonical_index, plan in enumerate(values):
        start = canonical_index * CANONICAL_STREAM_BATCH_SIZE
        stop = start + CANONICAL_STREAM_BATCH_SIZE
        pairs.append(
            make_seeded_policy_training_pair(
                clean_actions[start:stop],
                policy_contract,
                seed=plan.flow_seed,
            )
        )
    return PolicyTrainingPair(
        input_actions=torch.cat([pair.input_actions for pair in pairs], dim=0),
        timesteps=torch.cat([pair.timesteps for pair in pairs], dim=0),
        target=torch.cat([pair.target for pair in pairs], dim=0),
    )


def _validation_is_due(*, next_update: int, total_updates: int, interval: int) -> bool:
    """Keep validation on the declared cadence and configured final update only."""

    if next_update <= 0 or total_updates <= 0 or interval <= 0 or next_update > total_updates:
        raise ValueError("validation schedule values are invalid")
    return next_update % interval == 0 or next_update == total_updates


def _run_validation(
    *,
    dataset: LiberoParquetDataset,
    sampler: TaskUniformAnchorSampler,
    state_normalizer,
    action_normalizer,
    processor,
    model,
    denoiser: DuoVLADenoiser,
    adapted,
    device: torch.device,
    samples: int,
    physical_batch_size: int,
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
    if physical_batch_size not in SUPPORTED_PHYSICAL_BATCH_SIZES:
        raise ValueError("validation received an unsupported physical batch size")
    if data_parallel_size not in {1, DP2_DATA_PARALLEL_SIZE}:
        raise ValueError("validation received an unsupported data-parallel size")
    if data_parallel_size > 1 and dist.get_world_size() != data_parallel_size:
        raise RuntimeError("validation data-parallel size differs from the runtime world size")
    global_forward_width = physical_batch_size * data_parallel_size
    if samples % global_forward_width:
        raise ValueError(f"validation sample count must be divisible by {global_forward_width}")
    if samples % CANONICAL_STREAM_BATCH_SIZE:
        raise ValueError("validation samples must contain complete canonical B8 chunks")
    canonical_plans = tuple(
        make_microbatch_plan(validation_seed, 0, microstep)
        for microstep in range(samples // CANONICAL_STREAM_BATCH_SIZE)
    )
    local_plans = (
        canonical_plans
        if data_parallel_size == 1
        else _partition_canonical_plans_by_rank(
            canonical_plans,
            rank=dist.get_rank(),
            data_parallel_size=data_parallel_size,
        )
    )
    plan_groups = _group_canonical_plans(local_plans, physical_batch_size=physical_batch_size)
    with torch.no_grad():
        for plans in plan_groups:
            canonical_batches = _materialize_canonical_batches(
                dataset,
                sampler,
                plans,
                state_normalizer=state_normalizer,
                action_normalizer=action_normalizer,
            )
            batch = _coalesce_canonical_batches(
                canonical_batches,
                physical_batch_size=physical_batch_size,
            )
            state = batch.states.to(device)
            clean = batch.clean_actions.to(device)
            valid = batch.action_valid_mask.to(device)
            prefix_inputs = _processor_inputs(
                processor,
                batch.samples,
                device,
                prefix_geometry,
                expected_batch_size=physical_batch_size,
            )
            prefix = encode_diffusion_gemma_prefix(model, dict(prefix_inputs))
            pair = _canonical_training_pair(clean, policy_contract, plans)
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
    if data_parallel_size > 1:
        numerator = float(_sum_data_parallel_scalar(numerator, device=device, dtype=torch.float64))
        element_count = int(_sum_data_parallel_scalar(element_count, device=device, dtype=torch.int64))
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
    prefix_geometry_content_sha256: str,
    fixed_physical_prefix_width: int,
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
        prefix_validation_error: str | None = None
        if rank == 0:
            try:
                if not isinstance(completed, dict):
                    raise RuntimeError("rank 0 did not receive a completed checkpoint manifest")
                prefix_artifact = completed.get("artifacts", {}).get("prefix_geometry")
                if not isinstance(prefix_artifact, dict) or not isinstance(prefix_artifact.get("path"), str):
                    raise RuntimeError("completed checkpoint has no prefix geometry artifact")
                load_prefix_geometry_contract(
                    checkpoint_dir / prefix_artifact["path"],
                    expected_content_sha256=prefix_geometry_content_sha256,
                    expected_fixed_physical_prefix_width=fixed_physical_prefix_width,
                )
            except Exception as exc:
                prefix_validation_error = f"cannot authenticate copied prefix geometry: {type(exc).__name__}: {exc}"
        _broadcast_rank0_error(prefix_validation_error)
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
            raise ValueError("checkpoint rank-state hash list does not match the distributed topology")
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
    parser.add_argument("snapshot_root", type=Path)
    parser.add_argument("normalization_artifact", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--config", type=Path, default=Path("configs/libero.toml"))
    parser.add_argument(
        "--prefix-geometry-artifact",
        required=True,
        type=Path,
        help="Canonical artifact whose semantic SHA-256 and width are pinned by the benchmark config.",
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
    parser.add_argument("--max-cached-files", type=int, default=128)
    parser.add_argument(
        "--stop-after-updates",
        type=int,
        help="Stop cleanly after this many updates in this invocation without changing the run contract.",
    )
    restore_mode = parser.add_mutually_exclusive_group()
    restore_mode.add_argument("--resume", type=Path)
    restore_mode.add_argument(
        "--fork-from",
        type=Path,
        help=(
            "Start a new DP2 run from this frozen fork-manifest JSON; its adjacent "
            "<name>.json.sha256 sidecar authenticates the document."
        ),
    )
    restore_mode.add_argument(
        "--topology-fork-from",
        type=Path,
        help=("Start a new TP1 or DP2 run from an authenticated 2500-boundary topology-fork manifest."),
    )
    parser.add_argument(
        "--expected-fork-manifest-sha256",
        action="append",
        help="Preregistered raw SHA-256 of --fork-from; required for a new DP2 fork and forbidden on resume.",
    )
    parser.add_argument(
        "--expected-topology-fork-manifest-sha256",
        action="append",
        help=(
            "Preregistered raw SHA-256 of --topology-fork-from; required for a topology fork "
            "and forbidden in other restore modes."
        ),
    )
    args = parser.parse_args()

    args.snapshot_root = args.snapshot_root.resolve()
    args.normalization_artifact = args.normalization_artifact.resolve()
    args.output_dir = args.output_dir.resolve()
    args.config = args.config.resolve()
    args.prefix_geometry_artifact = Path(os.path.abspath(args.prefix_geometry_artifact))
    fork_manifest: dict[str, Any] | None = None
    fork_manifest_sha256: str | None = None
    fork_child_run_uuid: str | None = None
    topology_fork_manifest: dict[str, Any] | None = None
    topology_fork_manifest_sha256: str | None = None
    topology_fork_child_run_uuid: str | None = None
    expected_fork_manifest_sha256 = _validate_dp2_fork_digest_argument(
        fork_from=args.fork_from,
        resume=args.resume,
        expected_sha256=args.expected_fork_manifest_sha256,
    )
    if args.fork_from is not None:
        args.fork_from = Path(os.path.abspath(args.fork_from))
        _validate_dp2_fork_cli(args)
        fork_manifest, fork_manifest_sha256 = load_published_dp2_fork_manifest(args.fork_from)
        if fork_manifest_sha256 != expected_fork_manifest_sha256:
            raise ValueError("published fork manifest differs from --expected-fork-manifest-sha256")
        fork_child_run_uuid = str(fork_manifest["child"]["run_uuid"])
    expected_topology_fork_manifest_sha256 = _validate_topology_fork_digest_argument(
        topology_fork_from=args.topology_fork_from,
        resume=args.resume,
        legacy_fork_from=args.fork_from,
        expected_sha256=args.expected_topology_fork_manifest_sha256,
    )
    if args.topology_fork_from is not None:
        args.topology_fork_from = Path(os.path.abspath(args.topology_fork_from))
        topology_fork_manifest, topology_fork_manifest_sha256 = load_published_topology_fork_manifest(
            args.topology_fork_from
        )
        if topology_fork_manifest_sha256 != expected_topology_fork_manifest_sha256:
            raise ValueError("published topology fork differs from --expected-topology-fork-manifest-sha256")
        _validate_topology_fork_cli(args, topology_fork_manifest)
        topology_fork_child_run_uuid = str(topology_fork_manifest["child"]["run_uuid"])
    if args.resume is not None:
        args.resume = args.resume.resolve() if args.resume.is_absolute() else (args.output_dir / args.resume).resolve()
        try:
            args.resume.relative_to(args.output_dir)
        except ValueError as exc:
            raise ValueError("resume checkpoint must be inside output_dir") from exc

    project_root = Path(__file__).resolve().parents[1]
    # Runtime selection must use the authenticated recipe topology: both TP2
    # and DP2 expose two CUDA devices, but DP2 deliberately runs the qualified
    # TP1 (train-single-gpu) software environment on each replica.
    config = copy.deepcopy(load_resolved_toml(args.config))
    runtime_preflight = _configure_and_validate_training_runtime(
        project_root,
        resolved_config=config,
    )
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", device_id=device)
    expected_world_size = canonical_visible_cuda_world_size(os.environ)
    if dist.get_world_size() != expected_world_size:
        raise RuntimeError(
            f"LIBERO training world size differs from the canonical visible-device topology: "
            f"expected={expected_world_size}, observed={dist.get_world_size()}"
        )
    rank = dist.get_rank()
    run_lock = None
    try:
        run_lock = _acquire_run_lock(args.output_dir)
        optimization = config["optimization"]
        training = config["training"]
        # Reject an internally inconsistent committed recipe before applying
        # any explicitly requested runtime overrides.
        _validate_batch_contract(config)
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
            "max_cached_files": args.max_cached_files,
        }
        if args.stop_after_updates is not None and args.stop_after_updates <= 0:
            raise ValueError("stop-after-updates must be positive")
        invalid = [name for name, value in positive_values.items() if value <= 0]
        if invalid or not 0 <= warmup_updates < total_updates:
            raise ValueError(f"invalid training schedule fields: {invalid}, warmup={warmup_updates}")
        if permanent_checkpoint_interval % checkpoint_interval:
            raise ValueError("permanent_checkpoint_interval must be a multiple of checkpoint_interval")
        optimization["total_updates"] = total_updates
        optimization["warmup_updates"] = warmup_updates
        optimization["microbatch_size"] = microbatch_size
        optimization["gradient_accumulation_steps"] = accumulation_steps
        # In DP2 this is a global, not per-rank, batch. The explicit topology
        # below accounts for the second rank when validating the local geometry.
        optimization["global_batch_size"] = GLOBAL_BATCH_SIZE
        physical_batch_size, accumulation_steps, serving_batch_size = _validate_batch_contract(config)
        distributed = _data_parallel_contract(config, world_size=dist.get_world_size())
        data_parallel_size = 1 if distributed is None else int(distributed["data_parallel_size"])
        is_data_parallel = distributed is not None
        _validate_dp2_restore_mode(
            is_data_parallel=is_data_parallel,
            has_fork_manifest=fork_manifest is not None,
            has_resume_checkpoint=args.resume is not None,
            has_topology_fork_manifest=topology_fork_manifest is not None,
        )
        if is_data_parallel and args.max_cached_files != int(training["max_cached_files"]):
            raise ValueError(
                f"DP2 --max-cached-files must equal the frozen training.max_cached_files={training['max_cached_files']}"
            )
        expert_batch_isolation = str(config["model"]["expert_batch_isolation"])
        # Persist the serving geometry even for legacy TOMLs that predate its
        # separation from the training physical-forward width.
        optimization["serving_batch_size"] = serving_batch_size
        validation_forward_width = physical_batch_size * data_parallel_size
        if validation_samples % validation_forward_width:
            raise ValueError(f"validation samples must be divisible by global forward width {validation_forward_width}")
        training["validation_interval"] = validation_interval
        training["validation_samples"] = validation_samples
        training["checkpoint_interval"] = checkpoint_interval
        training["permanent_checkpoint_interval"] = permanent_checkpoint_interval
        training["log_interval"] = log_interval
        config["run"] = {"seed": run_seed, "task": args.task, "max_cached_files": args.max_cached_files}
        expected_hash_seed = str(run_seed)
        if os.environ.get("PYTHONHASHSEED") != expected_hash_seed:
            raise RuntimeError(
                f"launch training with PYTHONHASHSEED={expected_hash_seed} so Python/PEFT ordering is reproducible"
            )
        interface_config = _validate_and_build_interface_config(config)
        policy_contract = policy_contract_from_config(config)
        dataset_snapshot_report = _authenticate_dataset_snapshot_distributed(args.snapshot_root)
        dataset_tree_sha256 = str(dataset_snapshot_report["tree_metadata_sha256"])
        dataset_content_inventory_sha256 = str(dataset_snapshot_report["content_inventory_sha256"])
        dataset_files_verified = int(dataset_snapshot_report["files_verified"])
        dataset_total_bytes = int(dataset_snapshot_report["total_bytes"])
        hf_home = Path(os.environ.get("HF_HOME", "/root/.cache/huggingface"))
        model_snapshot = (
            hf_home / "hub/models--google--diffusiongemma-26B-A4B-it/snapshots" / DEFAULT_DIFFUSION_GEMMA_SPEC.revision
        )
        state_normalizer, action_normalizer, stats_manifest = load_libero_normalizers(
            args.normalization_artifact,
            expected_revision=LIBERO_DATASET_REVISION,
        )
        normalization_sha256 = stats_manifest["content_sha256"]
        dataset = LiberoParquetDataset(args.snapshot_root, max_cached_files=args.max_cached_files)
        canonical_instructions = tuple(dataset.task_by_index.values())
        prefix_geometry_content_sha256, fixed_physical_prefix_width = _prefix_geometry_pins(config)
        prefix_geometry, model_snapshot_report = _authenticate_prefix_geometry_distributed(
            args.prefix_geometry_artifact,
            expected_content_sha256=prefix_geometry_content_sha256,
            expected_fixed_physical_prefix_width=fixed_physical_prefix_width,
            model_snapshot=model_snapshot,
            instructions=canonical_instructions,
        )
        model_tree_sha256 = str(model_snapshot_report["tree_metadata_sha256"])
        execution_environment = _execution_environment(runtime_preflight)
        config["execution_environment"] = execution_environment
        execution_geometry = _execution_geometry(config)
        config["execution_geometry"] = execution_geometry
        config["artifact_trees"] = {
            "dataset_content_inventory_sha256": dataset_content_inventory_sha256,
            "dataset_files_verified": dataset_files_verified,
            "dataset_total_bytes": dataset_total_bytes,
            "dataset_tree_sha256": dataset_tree_sha256,
            "model_content_inventory_sha256": model_snapshot_report["content_inventory_sha256"],
            "model_files_verified": model_snapshot_report["files_verified"],
            "model_total_bytes": model_snapshot_report["total_bytes"],
            "model_tree_sha256": model_tree_sha256,
        }
        source_sha256 = _source_tree_sha256(project_root)
        config["source_tree_sha256"] = source_sha256
        config_sha256 = canonical_config_sha256(config)

        fork_parent_checkpoint: Path | None = None
        fork_parent_manifest: dict[str, Any] | None = None
        topology_fork_parent_checkpoint: Path | None = None
        topology_fork_parent_manifest: dict[str, Any] | None = None
        if fork_manifest is not None:
            assert fork_manifest_sha256 is not None
            assert fork_child_run_uuid is not None
            authenticate_dp2_child_environment(
                fork_manifest,
                fork_manifest_path=args.fork_from,
                fork_manifest_sha256=fork_manifest_sha256,
                project_root=project_root,
                config_path=args.config,
                child_output_dir=args.output_dir,
                child_run_uuid=fork_child_run_uuid,
                live_train_venv_identity=runtime_preflight["train_venv"],
                live_training_gpu_uuids=execution_environment["gpu_uuids"],
            )
            fork_parent_checkpoint, fork_parent_manifest, _ = authenticate_dp2_fork_parent(
                fork_manifest,
                fork_manifest_path=args.fork_from,
                fork_manifest_sha256=fork_manifest_sha256,
            )
        if topology_fork_manifest is not None:
            assert topology_fork_manifest_sha256 is not None
            assert topology_fork_child_run_uuid is not None
            authenticate_topology_child_environment(
                topology_fork_manifest,
                manifest_path=args.topology_fork_from,
                manifest_sha256=topology_fork_manifest_sha256,
                project_root=project_root,
                config_path=args.config,
                child_output_dir=args.output_dir,
                child_run_uuid=topology_fork_child_run_uuid,
                live_train_venv_identity=runtime_preflight["train_venv"],
                live_training_gpu_uuids=execution_environment["gpu_uuids"],
            )
            (
                topology_fork_parent_checkpoint,
                topology_fork_parent_manifest,
                _,
            ) = authenticate_topology_fork_parent(
                topology_fork_manifest,
                manifest_path=args.topology_fork_from,
                manifest_sha256=topology_fork_manifest_sha256,
            )

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
            new_run_uuid=fork_child_run_uuid or topology_fork_child_run_uuid,
        )
        dist.barrier()

        train_indices = tuple(int(index) for index in stats_manifest["split"]["train_episode_indices"])
        validation_indices = tuple(int(index) for index in stats_manifest["split"]["validation_episode_indices"])
        if args.task is not None:
            available_tasks = set(dataset.task_by_index.values())
            if args.task not in available_tasks:
                raise ValueError(f"unknown LIBERO task: {args.task!r}")
            train_indices = tuple(index for index in train_indices if dataset.episodes[index].task == args.task)
            validation_indices = tuple(
                index for index in validation_indices if dataset.episodes[index].task == args.task
            )
        train_sampler = TaskUniformAnchorSampler(dataset.episodes, train_indices)
        validation_sampler = TaskUniformAnchorSampler(dataset.episodes, validation_indices)
        run_contract = {
            "config_sha256": config_sha256,
            "dataset_content_inventory_sha256": dataset_content_inventory_sha256,
            "dataset_files_verified": str(dataset_files_verified),
            "dataset_revision": LIBERO_DATASET_REVISION,
            "dataset_total_bytes": str(dataset_total_bytes),
            "dataset_tree_sha256": dataset_tree_sha256,
            "expert_batch_isolation": expert_batch_isolation,
            "execution_environment_sha256": canonical_config_sha256(execution_environment),
            "experts_implementation": GROUPED_MM_EXPERTS_IMPLEMENTATION,
            "fixed_physical_prefix_width": str(fixed_physical_prefix_width),
            "model_content_inventory_sha256": str(model_snapshot_report["content_inventory_sha256"]),
            "model_revision": DEFAULT_DIFFUSION_GEMMA_SPEC.revision,
            "model_tree_sha256": model_tree_sha256,
            "normalization_sha256": normalization_sha256,
            "physical_batch_size": str(physical_batch_size),
            "policy_contract_sha256": canonical_config_sha256(policy_contract.to_dict()),
            "prefix_geometry_content_sha256": prefix_geometry_content_sha256,
            "run_uuid": run_uuid,
            "serving_batch_size": str(serving_batch_size),
            "source_tree_sha256": source_sha256,
        }
        if distributed is not None:
            run_contract.update(
                canonical_plan_partition=str(distributed["canonical_plan_partition"]),
                data_parallel_size=str(distributed["data_parallel_size"]),
                gradient_reduction=str(distributed["gradient_reduction"]),
                max_cached_files=str(args.max_cached_files),
                rank_physical_batch_size=str(distributed["rank_physical_batch_size"]),
                strategy=str(distributed["strategy"]),
                tensor_parallel_size=str(distributed["tensor_parallel_size"]),
                world_size=str(distributed["world_size"]),
            )
        if config.get("execution_profile") is not None:
            run_contract.update(
                execution_profile=str(config["execution_profile"]),
                tensor_parallel_size=str(config["model"]["tensor_parallel_size"]),
            )
        if "shared_weight_kernel_sha256" in execution_geometry:
            run_contract["shared_weight_kernel_sha256"] = str(execution_geometry["shared_weight_kernel_sha256"])

        fork_lineage: dict[str, Any] | None = None
        topology_lineage: dict[str, Any] | None = None
        if fork_manifest is not None:
            assert fork_manifest_sha256 is not None
            _merge_dp2_fork_run_contract(
                run_contract,
                dp2_fork_run_contract(
                    fork_manifest,
                    fork_manifest_sha256=fork_manifest_sha256,
                ),
            )
        if topology_fork_manifest is not None:
            assert topology_fork_manifest_sha256 is not None
            _merge_topology_fork_run_contract(
                run_contract,
                topology_fork_run_contract(
                    topology_fork_manifest,
                    manifest_sha256=topology_fork_manifest_sha256,
                ),
            )
        resume_manifest = load_checkpoint_manifest(args.resume) if args.resume is not None else None
        if resume_manifest is not None:
            if has_topology_fork_lineage(resume_manifest):
                topology_fields, topology_lineage = validate_topology_checkpoint_lineage(resume_manifest)
                if dp2_semantic_recipe_sha256(config) != topology_lineage["semantic_recipe_sha256"]:
                    raise ValueError("resume semantic recipe differs from its authenticated topology fork")
                _merge_topology_fork_run_contract(run_contract, topology_fields)
            elif is_data_parallel:
                fork_fields, fork_lineage = validate_dp2_checkpoint_lineage(resume_manifest)
                if dp2_semantic_recipe_sha256(config) != fork_lineage["semantic_recipe_sha256"]:
                    raise ValueError("DP2 resume semantic recipe differs from its authenticated fork lineage")
                _merge_dp2_fork_run_contract(
                    run_contract,
                    {
                        **fork_fields,
                        "data_parallel_size": run_contract["data_parallel_size"],
                        "execution_profile": run_contract["execution_profile"],
                        "max_cached_files": run_contract["max_cached_files"],
                        "run_uuid": run_contract["run_uuid"],
                        "source_tree_sha256": run_contract["source_tree_sha256"],
                    },
                )
            elif has_dp2_checkpoint_lineage(resume_manifest):
                raise ValueError("non-DP resume checkpoint unexpectedly carries DP2 fork lineage")
            _validate_checkpoint_execution_geometry(args.resume, resume_manifest, config)
            resumed_policy_contract = validate_manifest_policy_contract(resume_manifest, config)
            if resumed_policy_contract != policy_contract:
                raise ValueError("resume checkpoint policy contract mismatch")
            if resume_manifest.get("policy_contract_sha256") != run_contract["policy_contract_sha256"]:
                raise ValueError("resume checkpoint policy contract SHA-256 mismatch")
        elif fork_parent_manifest is not None:
            parent_policy_contract = validate_manifest_policy_contract(fork_parent_manifest, config)
            if parent_policy_contract != policy_contract:
                raise ValueError("fork parent checkpoint policy contract mismatch")
            if fork_parent_manifest.get("policy_contract_sha256") != run_contract["policy_contract_sha256"]:
                raise ValueError("fork parent checkpoint policy contract SHA-256 mismatch")
        elif topology_fork_parent_manifest is not None:
            parent_policy_contract = validate_manifest_policy_contract(topology_fork_parent_manifest, config)
            if parent_policy_contract != policy_contract:
                raise ValueError("topology fork parent checkpoint policy contract mismatch")
            if topology_fork_parent_manifest.get("policy_contract_sha256") != run_contract["policy_contract_sha256"]:
                raise ValueError("topology fork parent checkpoint policy contract SHA-256 mismatch")

        random.seed(2026 + run_seed)
        np.random.seed((2026 + run_seed) % (1 << 32))
        torch.manual_seed(2026 + run_seed)
        processor = AutoProcessor.from_pretrained(
            DEFAULT_DIFFUSION_GEMMA_SPEC.model_id,
            revision=DEFAULT_DIFFUSION_GEMMA_SPEC.revision,
            local_files_only=True,
        )
        loader_kwargs: dict[str, Any] = {}
        if is_data_parallel:
            loader_kwargs["replica_mode"] = DATA_PARALLEL_STRATEGY
        model = load_diffusion_gemma_bf16_tp(
            local_files_only=True,
            tp_size=int(config["model"]["tensor_parallel_size"]),
            **loader_kwargs,
        )
        _install_expert_batch_isolation(
            model,
            backend=expert_batch_isolation,
            physical_batch_size=physical_batch_size,
        )
        _verify_expert_batch_isolation(
            model,
            backend=expert_batch_isolation,
            physical_batch_size=physical_batch_size,
        )
        backend = DiffusionGemmaActionDecoder.from_block_diffusion_model(model)
        torch.manual_seed(2027 + run_seed)
        trainable_checkpoint = (
            args.resume if args.resume is not None else topology_fork_parent_checkpoint or fork_parent_checkpoint
        )
        trainable_manifest = (
            resume_manifest if resume_manifest is not None else topology_fork_parent_manifest or fork_parent_manifest
        )
        if trainable_checkpoint is None:
            adapted = apply_decoder_attention_lora(
                model,
                rank=int(config["lora"]["rank"]),
                alpha=int(config["lora"]["alpha"]),
                dropout=float(config["lora"]["dropout"]),
            )
        else:
            adapted, _ = load_lora_checkpoint(
                trainable_checkpoint,
                model,
                is_trainable=True,
                validate_decoder_contract=True,
                expected_rank=int(config["lora"]["rank"]),
            )
        _verify_expert_batch_isolation(
            model,
            backend=expert_batch_isolation,
            physical_batch_size=physical_batch_size,
        )
        projector = ActionInputProjector(interface_config).to(device)
        head = VelocityHead(
            interface_config.hidden_size,
            interface_config.action_dim,
            init_std=interface_config.output_init_std,
        ).to(device)
        if trainable_checkpoint is not None:
            assert trainable_manifest is not None
            interface_artifact = trainable_manifest["artifacts"]["interface"]
            load_interface_state_dict(
                trainable_checkpoint / interface_artifact["path"],
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
        if is_data_parallel:
            _assert_data_parallel_trainables_are_replicated(lora_partition, lora_named_parameters)
            assert_replicated_parameter_values([*lora_named_parameters, *interface_named_parameters])
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
        run_contract["optimizer_parameter_schema_sha256"] = optimizer_parameter_schema_sha256(
            optimizer,
            optimizer_named_parameters,
        )
        if topology_fork_manifest is not None:
            assert topology_fork_manifest_sha256 is not None
            topology_restore = restore_topology_fork_training_state(
                topology_fork_manifest,
                manifest_path=args.topology_fork_from,
                manifest_sha256=topology_fork_manifest_sha256,
                rank=rank,
                world_size=dist.get_world_size(),
                optimizer=optimizer,
                named_parameters=optimizer_named_parameters,
                scheduler=scheduler,
                child_run_contract=run_contract,
                device=device,
            )
            trainer_state = topology_restore.trainer_state
            topology_lineage = topology_restore.lineage
            if topology_fork_child_run_uuid != run_uuid:
                raise RuntimeError("restored topology fork child identity differs from the active run")
        elif fork_manifest is not None:
            assert fork_manifest_sha256 is not None
            fork_restore = restore_tp1_fork_training_state(
                fork_manifest,
                fork_manifest_path=args.fork_from,
                fork_manifest_sha256=fork_manifest_sha256,
                rank=rank,
                world_size=dist.get_world_size(),
                optimizer=optimizer,
                named_parameters=optimizer_named_parameters,
                scheduler=scheduler,
                child_run_contract=run_contract,
                device=device,
            )
            trainer_state = fork_restore.trainer_state
            fork_lineage = fork_restore.lineage_dict()
            if (
                fork_restore.child_run_uuid != run_uuid
                or fork_restore.child_run_seed != run_seed
                or fork_restore.parent_run_seed != run_seed
                or fork_restore.child_source_tree_sha256 != source_sha256
            ):
                raise RuntimeError("restored DP2 fork identity differs from the active child run")
        elif args.resume is None:
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
        if trainable_checkpoint is not None:
            _assert_fp32_optimizer_state(optimizer)
            validate_training_progress(
                trainer_state,
                optimizer=optimizer,
                scheduler=scheduler,
                examples_per_update=GLOBAL_BATCH_SIZE,
                expected_learning_rates=[
                    optimization_config.lora_learning_rate
                    * learning_rate_scale(trainer_state.next_update, optimization_config),
                    optimization_config.interface_learning_rate
                    * learning_rate_scale(trainer_state.next_update, optimization_config),
                ],
            )
            assert trainable_manifest is not None
            _assert_distributed_trainer_state(trainer_state, trainable_manifest)
            if is_data_parallel:
                _assert_replicated_optimizer_state(optimizer, optimizer_named_parameters)
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
            plan_groups = (
                _rank_canonical_update_plan_groups(
                    run_seed,
                    update,
                    rank=rank,
                    data_parallel_size=data_parallel_size,
                    physical_batch_size=physical_batch_size,
                )
                if is_data_parallel
                else _canonical_update_plan_groups(
                    run_seed,
                    update,
                    physical_batch_size=physical_batch_size,
                )
            )
            canonical_plans = tuple(plan for group in plan_groups for plan in group)
            canonical_batches = _materialize_canonical_batches(
                dataset,
                train_sampler,
                canonical_plans,
                state_normalizer=state_normalizer,
                action_normalizer=action_normalizer,
            )
            chunks_per_forward = physical_batch_size // CANONICAL_STREAM_BATCH_SIZE
            batches = tuple(
                _coalesce_canonical_batches(
                    canonical_batches[start : start + chunks_per_forward],
                    physical_batch_size=physical_batch_size,
                )
                for start in range(0, len(canonical_batches), chunks_per_forward)
            )
            if len(plan_groups) != len(batches) or len(batches) != accumulation_steps:
                raise RuntimeError("physical update grouping differs from the resolved accumulation contract")
            local_total_elements = sum(
                masked_element_count(
                    batch.action_valid_mask,
                    action_dim=interface_config.action_dim,
                )
                for batch in batches
            )
            total_elements = (
                int(
                    _sum_data_parallel_scalar(
                        local_total_elements,
                        device=device,
                        dtype=torch.int64,
                    )
                )
                if is_data_parallel
                else local_total_elements
            )
            optimizer.zero_grad(set_to_none=True)
            numerator = 0.0
            for microstep, (plans, batch) in enumerate(zip(plan_groups, batches, strict=True)):
                state = batch.states.to(device)
                clean = batch.clean_actions.to(device)
                valid = batch.action_valid_mask.to(device)
                prefix_inputs = _processor_inputs(
                    processor,
                    batch.samples,
                    device,
                    prefix_geometry,
                    expected_batch_size=physical_batch_size,
                )
                prefix = encode_diffusion_gemma_prefix(model, dict(prefix_inputs))
                pair = _canonical_training_pair(clean, policy_contract, plans)
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
                if update == invocation_start_update and microstep == 0:
                    _assert_initial_gradients_present(
                        lora_partition,
                        interface_named_parameters,
                        data_parallel=is_data_parallel,
                    )
            _assert_distributed_gradient_health([*lora_parameters, *interface_parameters])
            if is_data_parallel:
                _sum_data_parallel_gradients_(optimizer_named_parameters)
                _assert_distributed_gradient_health([*lora_parameters, *interface_parameters])
                if update == invocation_start_update:
                    _assert_replicated_gradient_values(optimizer_named_parameters)
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
            trainer_state = trainer_state.advance(examples=GLOBAL_BATCH_SIZE)
            if is_data_parallel and update == invocation_start_update:
                assert_replicated_parameter_values(optimizer_named_parameters)
            elif update == 0:
                assert_replicated_parameter_values(interface_named_parameters)
            global_numerator = (
                float(_sum_data_parallel_scalar(numerator, device=device, dtype=torch.float64))
                if is_data_parallel
                else numerator
            )
            train_loss = global_numerator / total_elements
            assert_replicated_tensor(
                f"train_loss.{trainer_state.next_update}",
                torch.tensor(train_loss, device=device, dtype=torch.float64),
            )
            update_seconds = time.perf_counter() - update_started
            if is_data_parallel:
                update_seconds = _max_data_parallel_seconds(update_seconds, device=device)
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
            if is_data_parallel:
                metric.update(
                    data_parallel_size=data_parallel_size,
                    global_batch_size=GLOBAL_BATCH_SIZE,
                    rank_physical_batch_size=physical_batch_size,
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
                    action_normalizer=action_normalizer,
                    processor=processor,
                    model=model,
                    denoiser=denoiser,
                    adapted=adapted,
                    device=device,
                    samples=validation_samples,
                    physical_batch_size=physical_batch_size,
                    validation_seed=validation_seed,
                    policy_contract=policy_contract,
                    prefix_geometry=prefix_geometry,
                    data_parallel_size=data_parallel_size,
                )
            should_checkpoint = (
                trainer_state.next_update % checkpoint_interval == 0
                or trainer_state.next_update == total_updates
                or trainer_state.next_update == execution_end
            )
            if should_checkpoint:
                allocator_peak_memory_bytes_by_rank = (
                    _data_parallel_peak_memory_bytes(device) if is_data_parallel else None
                )
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
                    prefix_geometry_content_sha256=prefix_geometry_content_sha256,
                    fixed_physical_prefix_width=fixed_physical_prefix_width,
                    resolved_config_path=resolved_config_path,
                    manifest={
                        "config_sha256": config_sha256,
                        "dataset_content_inventory_sha256": dataset_content_inventory_sha256,
                        "dataset_files_verified": dataset_files_verified,
                        "dataset_id": "HuggingFaceVLA/libero",
                        "dataset_revision": LIBERO_DATASET_REVISION,
                        "dataset_total_bytes": dataset_total_bytes,
                        "dataset_tree_sha256": dataset_tree_sha256,
                        "expert_batch_isolation": expert_batch_isolation,
                        "execution_environment": execution_environment,
                        "execution_environment_sha256": run_contract["execution_environment_sha256"],
                        "experts_implementation": GROUPED_MM_EXPERTS_IMPLEMENTATION,
                        "execution_geometry": execution_geometry,
                        "fixed_physical_prefix_width": fixed_physical_prefix_width,
                        "kind": "resumable-libero-training",
                        "last_metrics": metric,
                        "model_id": DEFAULT_DIFFUSION_GEMMA_SPEC.model_id,
                        "model_content_inventory_sha256": model_snapshot_report["content_inventory_sha256"],
                        "model_files_verified": model_snapshot_report["files_verified"],
                        "model_revision": DEFAULT_DIFFUSION_GEMMA_SPEC.revision,
                        "model_total_bytes": model_snapshot_report["total_bytes"],
                        "model_tree_sha256": model_tree_sha256,
                        "normalization_sha256": normalization_sha256,
                        "optimizer_parameter_schema_sha256": run_contract["optimizer_parameter_schema_sha256"],
                        "parent_manifest_sha256": parent_manifest_sha256,
                        "checkpoint_retention": make_checkpoint_retention_contract(
                            permanent_checkpoint_interval=permanent_checkpoint_interval,
                            parent_checkpoint=parent_checkpoint,
                        ),
                        "physical_batch_size": physical_batch_size,
                        "platform": platform.platform(),
                        "policy_contract": policy_contract.to_dict(),
                        "policy_contract_sha256": run_contract["policy_contract_sha256"],
                        "prefix_geometry_content_sha256": prefix_geometry_content_sha256,
                        "replicated_optimizer_sha256": replicated_optimizer_sha256,
                        "replicated_parameter_sha256": replicated_parameter_sha256,
                        "run_seed": run_seed,
                        "run_uuid": run_uuid,
                        "serving_batch_size": serving_batch_size,
                        "source_tree_sha256": source_sha256,
                        "task": args.task,
                        "train_episode_count": len(train_indices),
                        "validation_episode_count": len(validation_indices),
                        **(
                            {
                                "fork_lineage": fork_lineage,
                                **{name: run_contract[name] for name in sorted(_DP2_FORK_ONLY_RUN_CONTRACT_FIELDS)},
                            }
                            if fork_lineage is not None
                            else {}
                        ),
                        **(
                            {
                                "topology_fork_lineage": topology_lineage,
                                **{name: run_contract[name] for name in sorted(TOPOLOGY_FORK_CHECKPOINT_FIELDS)},
                            }
                            if topology_lineage is not None
                            else {}
                        ),
                        **(
                            {
                                "cuda_allocator_peak_memory_bytes_by_rank": allocator_peak_memory_bytes_by_rank,
                                "cuda_allocator_peak_memory_bytes_max": max(allocator_peak_memory_bytes_by_rank),
                                "canonical_plan_partition": distributed["canonical_plan_partition"],
                                "data_parallel_size": distributed["data_parallel_size"],
                                "gradient_reduction": distributed["gradient_reduction"],
                                "max_cached_files": args.max_cached_files,
                                "rank_physical_batch_size": distributed["rank_physical_batch_size"],
                                "strategy": distributed["strategy"],
                                "world_size": distributed["world_size"],
                            }
                            if distributed is not None
                            else {}
                        ),
                        **(
                            {"shared_weight_kernel_sha256": execution_geometry["shared_weight_kernel_sha256"]}
                            if "shared_weight_kernel_sha256" in execution_geometry
                            else {}
                        ),
                        **(
                            {
                                "execution_profile": run_contract["execution_profile"],
                                "tensor_parallel_size": int(config["model"]["tensor_parallel_size"]),
                            }
                            if "execution_profile" in run_contract
                            else {}
                        ),
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
        peak_memory_gib_by_rank = _data_parallel_peak_memory_gib(device) if is_data_parallel else None
        if rank == 0:
            completion: dict[str, Any] = {
                "complete": trainer_state.next_update == total_updates,
                "elapsed_seconds": time.perf_counter() - started,
                "examples_seen": trainer_state.examples_seen,
                "final_update": trainer_state.next_update,
                "output_dir": str(args.output_dir),
                "peak_memory_gib": (
                    max(peak_memory_gib_by_rank)
                    if peak_memory_gib_by_rank is not None
                    else torch.cuda.max_memory_allocated(device) / 2**30
                ),
            }
            if peak_memory_gib_by_rank is not None:
                completion.update(
                    data_parallel_size=data_parallel_size,
                    global_batch_size=GLOBAL_BATCH_SIZE,
                    peak_memory_gib_by_rank=peak_memory_gib_by_rank,
                    rank_physical_batch_size=physical_batch_size,
                )
            print(
                json.dumps(
                    completion,
                    indent=2,
                    sort_keys=True,
                )
            )
    finally:
        if run_lock is not None:
            run_lock.close()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
