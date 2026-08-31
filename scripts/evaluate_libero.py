#!/usr/bin/env python3
"""Deterministic LIBERO rollout evaluator backed by the persistent Duo-VLA policy socket."""

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
import site
import stat
import statistics
import subprocess
import sys
import time
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np

_EVALUATOR_SCRIPT_DIR = Path(__file__).resolve().parent


def _activate_project_source_root() -> Path:
    source_root = _EVALUATOR_SCRIPT_DIR.parent / "src"
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
                    opened = os.fstat(child)
                    if not same_identity(opened, observed):
                        raise RuntimeError(f"project source directory changed while opening: {context}")
                    walk(child, (*prefix, name))
                finally:
                    os.close(child)
            elif stat.S_ISREG(observed.st_mode) and name.endswith(".py"):
                continue
            else:
                raise RuntimeError(f"project source import entry is unsafe: {context}")
        after = os.fstat(directory)
        if names != sorted(os.listdir(directory)) or not same_identity(before, after):
            raise RuntimeError(f"project source directory changed during inventory: {'/'.join(prefix) or '.'}")

    root_descriptor = os.open(
        source_root,
        os.O_RDONLY | os.O_NONBLOCK | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
    )
    try:
        if os.listdir(root_descriptor) != ["duo_vla"]:
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


def _bootstrap_source_file_identity(path: Path) -> dict[str, Any]:
    """Capture evaluator sources before importing any repository-local dependency."""

    if not hasattr(os, "O_NOFOLLOW"):
        raise RuntimeError("official source authentication requires O_NOFOLLOW")
    descriptor = os.open(str(path), os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0))
    digest = hashlib.sha256()
    size = 0
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise RuntimeError(f"official evaluator source is not regular: {path}")
        with os.fdopen(descriptor, "rb") as source:
            descriptor = -1
            while True:
                block = source.read(1024 * 1024)
                if not block:
                    break
                size += len(block)
                digest.update(block)
            after = os.fstat(source.fileno())
        stable = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
        if any(getattr(before, field) != getattr(after, field) for field in stable) or size != after.st_size:
            raise RuntimeError(f"official evaluator source changed during import: {path}")
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    return {
        "bytes": size,
        "device": after.st_dev,
        "inode": after.st_ino,
        "path": str(path.resolve()),
        "sha256": digest.hexdigest(),
    }


_IMPORT_EVALUATOR_SOURCE_IDENTITIES = {
    name: _bootstrap_source_file_identity(_EVALUATOR_SCRIPT_DIR / name)
    for name in ("evaluate_libero.py", "libero_bridge.py", "preflight_libero_env.py")
}


def _load_local_module(name: str, path: Path) -> Any:
    existing = sys.modules.get(name)
    if existing is not None:
        return existing
    specification = importlib.util.spec_from_file_location(name, path)
    if specification is None or specification.loader is None:
        raise RuntimeError(f"cannot construct import specification for {path}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    try:
        specification.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


_libero_bridge = _load_local_module("libero_bridge", _EVALUATOR_SCRIPT_DIR / "libero_bridge.py")
from libero_bridge import (  # noqa: E402
    ACTION_DIM,
    IMAGE_SHAPE,
    LIBERO_EXECUTION_GEOMETRY,
    PROTOCOL,
    SUITES,
    PolicyClient,
    validate_execution_geometry,
)

_bridge_file = getattr(_libero_bridge, "__file__", None)
_bridge_origin = getattr(getattr(_libero_bridge, "__spec__", None), "origin", None)
if (
    not isinstance(_bridge_file, str)
    or not isinstance(_bridge_origin, str)
    or Path(_bridge_file).resolve() != Path(_IMPORT_EVALUATOR_SOURCE_IDENTITIES["libero_bridge.py"]["path"])
    or Path(_bridge_origin).resolve() != Path(_IMPORT_EVALUATOR_SOURCE_IDENTITIES["libero_bridge.py"]["path"])
):
    raise RuntimeError("imported LIBERO bridge has an unexpected module origin")

from duo_vla.benchmarks.libero_dev_states import (  # noqa: E402
    canonical_official_state_hashes,
    load_bank,
    sequence_root_sha256,
    state_sha256,
)
from duo_vla.libero_replay_evidence import (  # noqa: E402
    ORIGINAL_HDF5_CONTENT_SHA256,
    ORIGINAL_HDF5_FILE_COUNT,
    ORIGINAL_HDF5_REPOSITORY_ID,
    ORIGINAL_HDF5_REVISION,
    ORIGINAL_HDF5_TOTAL_BYTES,
)
from duo_vla.run_config import load_resolved_toml  # noqa: E402
from duo_vla.run_journal import validate_resume_checkpoint  # noqa: E402
from duo_vla.runtime_integrity import (  # noqa: E402
    BASE_PYTHON_RUNTIME_IDENTITY_SCHEMA,  # noqa: F401 -- exported for protocol fixtures.
    EVAL_VENV_IDENTITY_SCHEMA,
    TRAIN_VENV_IDENTITY_SCHEMA,
    require_matching_eval_venv,
    require_matching_train_venv,
)

_REQUIRED_PROJECT_MODULES = {
    "duo_vla",
    "duo_vla.benchmarks.libero_dev_states",
    "duo_vla.libero_replay_evidence",
    "duo_vla.run_config",
    "duo_vla.run_journal",
    "duo_vla.runtime_integrity",
}
_validate_project_module_origins(_REQUIRED_PROJECT_MODULES)

CAMERA_NAMES = ("agentview", "robot0_eye_in_hand")
CAMERA_KEYS = ("agentview_image", "robot0_eye_in_hand_image")
ENVIRONMENT_SEED = 7
SETTLE_STEPS = 10
POLICY_BUDGETS = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
}
MODEL_REVISION = "f7f5b7f5fa82ffc52addd066915886d497f5517b"
DATASET_REVISION = "86958911c0f959db2bbbdb107eb3e17c5f9c798e"
DATASET_TREE_METADATA_SHA256 = "d9c14b4aff28bcc56f341b171c6a5a3b10510d4bd0378662891c5156d245add8"
DATASET_CONTENT_INVENTORY_SHA256 = "63fd7a951ebb397a33c43cad4a7c48c7c6911bd8d1481ff99b07da5f7890782c"
DATASET_SNAPSHOT_FILES_VERIFIED = 382
DATASET_SNAPSHOT_TOTAL_BYTES = 34_926_155_087
NORMALIZATION_SHA256 = "a972b5d95a8aaa8ae7582bafcbc071261979cb46c2a3515b4da7a7cf0156ac73"
TASK_INVENTORY_SHA256 = "d00c211a09f34003089ba5a4dbbbb0e11af2543f4bba9cb1901a04a2a25e0117"
ORIGINAL_HDF5_INVENTORY_RAW_SHA256 = "3b5f9b164434c91c41a3ff1e9681d91ab856969699c40fd7247ba93ed909cd7c"
OPEN_GRIPPER_NOOP = np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0], dtype=np.float32)
PREREGISTRATION_SCHEMA = "duo-vla-libero-official-preregistration-v5"
OFFICIAL_RUN_SCHEMA = "duo-vla-libero-official-run-v3"
OFFICIAL_SUMMARY_SCHEMA = "duo-vla-libero-official-summary-v2"
OUTPUT_ROOTS_SCHEMA = "duo-vla-libero-official-output-roots-v1"
OUTPUT_CLAIM_SCHEMA = "duo-vla-libero-official-output-claim-v1"
CLAIM_RECORD_SCHEMA = "duo-vla-libero-official-claim-record-v1"
COMPLETION_SCHEMA = "duo-vla-libero-official-completion-v1"
SIMULATOR_ATTESTATION_SCHEMA = "duo-vla-libero-simulator-attestation-v3"
AGGREGATION_PYTHON_VERSION = "3.12.13"
CONTAMINATION_SCHEMA = "duo-vla-evaluation-contamination-v1"
CONTAMINATION_LEDGER_SHA256 = "014b3db5dfc8bce9027a20f6deb25cd701a41e6f4204d7cc458416cbc53a2992"
FINAL_CHECKPOINT_UPDATE = 30_000
OFFICIAL_GLOBAL_BATCH_SIZE = 64
OFFICIAL_TRAIN_EPISODES = 1_525
OFFICIAL_VALIDATION_EPISODES = 168
OFFICIAL_TRAINING_EXAMPLES = FINAL_CHECKPOINT_UPDATE * OFFICIAL_GLOBAL_BATCH_SIZE
OFFICIAL_TRAIN_SEEDS = (0, 1, 2)
OFFICIAL_EXECUTION_HORIZONS = (1, 4)
OFFICIAL_FLOW_NFES = (1, 5, 10)
OFFICIAL_FLOW_CHECKPOINT_NFE = 10
OFFICIAL_RESETS_PER_TASK = 50
OFFICIAL_FULL_EPISODES = 2_000
OFFICIAL_PRIMARY_EPISODES = 1_999
OFFICIAL_POLICY_WARMUP_CALLS = 2
OFFICIAL_POLICY_WARMUP_REPLAN_ID = max(POLICY_BUDGETS.values())
OFFICIAL_EXCLUDED_EPISODE = {"reset_id": 0, "suite": "libero_goal", "task_id": 7}
_PREREGISTRATION_FIELDS = {
    "aggregation_python_version",
    "aggregator_sha256",
    "benchmark_protocol",
    "cells",
    "contamination",
    "episode_count",
    "episode_matrix_sha256",
    "episodes",
    "evaluation_seed",
    "expert_replay_qualification",
    "execution_horizons",
    "final_checkpoint_update",
    "final_freeze_token_sha256",
    "official_resets_per_task",
    "official_output_roots",
    "policy_warmup_calls",
    "schema",
    "simulator_attestation_sha256",
    "suites",
    "task_ids",
    "training_seeds",
}
_CELL_FIELDS = {
    "cell_id",
    "checkpoint",
    "episode_matrix_sha256",
    "execution_geometry",
    "execution_horizon",
    "inference_seed_behavior",
    "latency_runtime_sha256",
    "nfe",
    "objective",
    "output_claim",
    "policy_contract_sha256",
    "policy_warmup_calls",
    "sampler",
    "serving_policy_sha256",
    "serving_runtime_sha256",
    "train_seed",
}
_CELL_CREATOR_FIELDS = _CELL_FIELDS - {"episode_matrix_sha256", "output_claim", "serving_policy_sha256"}
_OUTPUT_ROOTS_FIELDS = {"claims", "runs", "schema"}
_OUTPUT_ROOT_IDENTITY_FIELDS = {"device", "inode", "path"}
_OUTPUT_CLAIM_FIELDS = {"claim_id_sha256", "claim_path", "output_dir", "schema"}
_CLAIM_RECORD_FIELDS = {
    "cell_id",
    "claim_id_sha256",
    "created_utc",
    "final_freeze_token_sha256",
    "output_dir",
    "preregistration_sha256",
    "schema",
}
_COMPLETION_FIELDS = {
    "cell_id",
    "claim_json_sha256",
    "episode_records",
    "episodes_jsonl_sha256",
    "preregistration_sha256",
    "run_json_sha256",
    "schema",
    "summary_json_sha256",
}
_CHECKPOINT_FIELDS = {
    "dataset_content_inventory_sha256",
    "dataset_tree_sha256",
    "manifest_sha256",
    "source_tree_sha256",
    "update",
}
_EXPERT_REPLAY_QUALIFICATION_FIELDS = {
    "config_file_sha256",
    "content_sha256",
    "dataset_content_inventory_sha256",
    "dataset_snapshot_files_verified",
    "dataset_snapshot_total_bytes",
    "dataset_tree_metadata_sha256",
    "demonstration_count",
    "evidence_manifest_raw_sha256",
    "gates_sha256",
    "kind",
    "normalization_content_sha256",
    "normalization_raw_sha256",
    "original_hdf5_file_count",
    "original_hdf5_inventory_content_sha256",
    "original_hdf5_inventory_raw_sha256",
    "original_hdf5_repository_id",
    "original_hdf5_revision",
    "original_hdf5_total_bytes",
    "project_source_tree_sha256",
    "raw_evidence_root_sha256",
    "report_sha256",
    "schema",
    "simulator_attestation_raw_sha256",
    "simulator_attestation_sha256",
    "simulator_runtime_sha256",
    "status",
    "successful_demonstration_count",
    "successful_task_count",
    "task_count",
    "task_inventory_sha256",
    "train_venv_identity",
    "validator_runtime_identity",
}
_TRAIN_VENV_IDENTITY_FIELDS = {
    "base_python_runtime",
    "content_inventory_sha256",
    "files_verified",
    "root",
    "root_sha256",
    "schema",
    "startup_hooks",
    "startup_hooks_sha256",
    "symlinks_verified",
    "total_bytes",
    "tree_metadata_sha256",
}
_VALIDATOR_RUNTIME_IDENTITY_FIELDS = {
    "distribution_records",
    "eval_venv_identity",
    "installed_distributions",
    "module_origins",
    "packages",
    "process",
    "project_sources",
    "schema",
    "simulator_runtime_sha256",
    "site_packages",
}
VALIDATOR_RUNTIME_IDENTITY_SCHEMA = "duo-vla-libero-expert-replay-validator-runtime-v1"
_SIMULATOR_RUNTIME_IDENTITY_FIELDS = (
    "assets",
    "backend",
    "eval_venv_identity",
    "distribution_records",
    "egl_device",
    "evaluator_lock_sha256",
    "installed_distributions",
    "manifest_sha256",
    "module_origins",
    "opengl",
    "packages",
    "process",
    "project_sources",
    "site_packages",
    "source",
    "torch",
)
_CONTAMINATION_FIELDS = {
    "excluded_episodes",
    "full_episode_count",
    "ledger_schema",
    "ledger_sha256",
    "primary_episode_count",
    "report_label",
}
_REQUIRED_EVALUATOR_ENVIRONMENT = {
    "MKL_NUM_THREADS": "1",
    "MUJOCO_EGL_DEVICE_ID": "0",
    "MUJOCO_GL": "egl",
    "NUMEXPR_NUM_THREADS": "1",
    "OMP_DYNAMIC": "FALSE",
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "PYOPENGL_PLATFORM": "egl",
    "PYTHONHASHSEED": "0",
    "PYTHONNOUSERSITE": "1",
    "PYTHONSAFEPATH": "1",
    "PYTHONDONTWRITEBYTECODE": "1",
}
_EVALUATOR_ALGORITHM_PREFIXES = (
    "CUBLAS_",
    "CUDA_",
    "CUDNN_",
    "MKL_",
    "MUJOCO_",
    "NCCL_",
    "NPY_",
    "NUMEXPR_",
    "OMP_",
    "OPENBLAS_",
    "PYOPENGL_",
    "PYTORCH_",
    "TORCH_",
)
_ALLOWED_EVALUATOR_ALGORITHM_ENVIRONMENT = frozenset(_REQUIRED_EVALUATOR_ENVIRONMENT)
_EVALUATOR_INJECTION_PREFIXES = ("LD_", "MALLOC_", "OPENSSL_", "PYTHON")
_ALLOWED_EVALUATOR_INJECTION_ENVIRONMENT = {
    "PYTHONHASHSEED",
    "PYTHONNOUSERSITE",
    "PYTHONSAFEPATH",
    "PYTHONDONTWRITEBYTECODE",
}


@dataclass(frozen=True, slots=True)
class DevelopmentResetBank:
    manifest: dict[str, Any]
    manifest_sha256: str
    states_by_task: dict[tuple[str, int], np.ndarray]
    task_records: dict[tuple[str, int], dict[str, Any]]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"value is not finite canonical JSON: {exc}") from exc


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _source_file_identity(path: Path) -> dict[str, Any]:
    """Hash one stable, non-symlink source file and bind its inode."""

    require(hasattr(os, "O_NOFOLLOW"), "official source authentication requires O_NOFOLLOW")
    flags = os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
    try:
        descriptor = os.open(str(path), flags)
    except OSError as exc:
        raise RuntimeError(f"cannot open official evaluator source: {path}") from exc
    digest = hashlib.sha256()
    size = 0
    try:
        before = os.fstat(descriptor)
        require(stat.S_ISREG(before.st_mode), f"official evaluator source is not regular: {path}")
        with os.fdopen(descriptor, "rb") as source:
            descriptor = -1
            while True:
                block = source.read(1024 * 1024)
                if not block:
                    break
                size += len(block)
                digest.update(block)
            after = os.fstat(source.fileno())
        stable = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
        require(
            all(getattr(before, name) == getattr(after, name) for name in stable) and size == after.st_size,
            f"official evaluator source changed while being hashed: {path}",
        )
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    return {
        "bytes": size,
        "device": after.st_dev,
        "inode": after.st_ino,
        "path": str(path.resolve()),
        "sha256": digest.hexdigest(),
    }


def capture_evaluator_source_identities(project_root: Path) -> dict[str, dict[str, Any]]:
    scripts = project_root.resolve() / "scripts"
    return {
        name: _source_file_identity(scripts / name)
        for name in ("evaluate_libero.py", "libero_bridge.py", "preflight_libero_env.py")
    }


def require_evaluator_sources_unchanged(
    expected: Mapping[str, Mapping[str, Any]],
    *,
    project_root: Path,
) -> None:
    observed = capture_evaluator_source_identities(project_root)
    require(
        canonical_json_bytes(observed) == canonical_json_bytes(expected),
        "official evaluator sources changed after startup",
    )


def _real_directory_identity(path: Path, *, name: str, require_empty: bool = False) -> dict[str, Any]:
    """Bind a canonical, non-symlink directory to its live device and inode."""

    require(hasattr(os, "O_NOFOLLOW"), "official output roots require O_NOFOLLOW")
    candidate = Path(path)
    require(candidate.is_absolute(), f"{name} must be an absolute path")
    try:
        canonical = candidate.resolve(strict=True)
        before = os.lstat(str(candidate))
    except OSError as exc:
        raise RuntimeError(f"cannot resolve {name}: {candidate}") from exc
    require(canonical == candidate, f"{name} must be a canonical real path")
    require(stat.S_ISDIR(before.st_mode), f"{name} must be a real directory")
    flags = os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_DIRECTORY", 0)
    try:
        descriptor = os.open(str(candidate), flags)
    except OSError as exc:
        raise RuntimeError(f"cannot open {name}: {candidate}") from exc
    try:
        opened = os.fstat(descriptor)
        require(
            stat.S_ISDIR(opened.st_mode) and opened.st_dev == before.st_dev and opened.st_ino == before.st_ino,
            f"{name} identity changed while being opened",
        )
        if require_empty:
            require(not os.listdir(descriptor), f"{name} must be empty when pre-registration is created")
        after = os.fstat(descriptor)
        require(
            after.st_dev == opened.st_dev and after.st_ino == opened.st_ino,
            f"{name} identity changed while being inspected",
        )
    finally:
        os.close(descriptor)
    return {"device": opened.st_dev, "inode": opened.st_ino, "path": str(candidate)}


def capture_official_output_roots(
    runs_root: Path,
    claims_root: Path,
    *,
    require_empty: bool = False,
) -> dict[str, Any]:
    runs = _real_directory_identity(runs_root, name="official run root", require_empty=require_empty)
    claims = _real_directory_identity(claims_root, name="official claim root", require_empty=require_empty)
    require(runs["path"] != claims["path"], "official run and claim roots must be distinct")
    common = os.path.commonpath((runs["path"], claims["path"]))
    require(
        common not in (runs["path"], claims["path"]),
        "official run and claim roots must not contain one another",
    )
    require(
        (runs["device"], runs["inode"]) != (claims["device"], claims["inode"]),
        "official run and claim roots must identify different directories",
    )
    return {"claims": claims, "runs": runs, "schema": OUTPUT_ROOTS_SCHEMA}


def validate_official_output_roots(value: Any, *, require_live: bool = False) -> dict[str, Any]:
    require(isinstance(value, Mapping), "official output roots must be an object")
    _require_exact_keys(value, _OUTPUT_ROOTS_FIELDS, "official output roots")
    require(value["schema"] == OUTPUT_ROOTS_SCHEMA, "official output roots schema mismatch")
    checked: dict[str, dict[str, Any]] = {}
    for key in ("runs", "claims"):
        identity = value[key]
        require(isinstance(identity, Mapping), f"official {key} root identity must be an object")
        _require_exact_keys(identity, _OUTPUT_ROOT_IDENTITY_FIELDS, f"official {key} root identity")
        path = identity["path"]
        require(isinstance(path, str) and bool(path), f"official {key} root path is invalid")
        candidate = Path(path)
        require(
            candidate.is_absolute() and str(candidate) == path and candidate.resolve(strict=False) == candidate,
            f"official {key} root path is not canonical",
        )
        for field in ("device", "inode"):
            require(type(identity[field]) is int and identity[field] >= 0, f"official {key} root {field} is invalid")
        checked[key] = dict(identity)
    require(checked["runs"]["path"] != checked["claims"]["path"], "official output roots must be distinct")
    common = os.path.commonpath((checked["runs"]["path"], checked["claims"]["path"]))
    require(
        common not in (checked["runs"]["path"], checked["claims"]["path"]),
        "official output roots must not contain one another",
    )
    require(
        (checked["runs"]["device"], checked["runs"]["inode"])
        != (checked["claims"]["device"], checked["claims"]["inode"]),
        "official output roots must identify different directories",
    )
    if require_live:
        for key in ("runs", "claims"):
            observed = _real_directory_identity(Path(checked[key]["path"]), name=f"official {key} root")
            require(observed == checked[key], f"official {key} root identity drifted after pre-registration")
    return {"claims": checked["claims"], "runs": checked["runs"], "schema": OUTPUT_ROOTS_SCHEMA}


def derive_output_claim(
    cell_id: str,
    roots: Mapping[str, Any],
    final_freeze_token_sha256: str,
) -> dict[str, Any]:
    require(isinstance(cell_id, str) and bool(cell_id), "output claim cell ID is invalid")
    require(Path(cell_id).name == cell_id and cell_id not in {".", ".."}, "output claim cell ID is not filename-safe")
    checked = validate_official_output_roots(roots)
    _require_sha256(final_freeze_token_sha256, "output claim freeze-token SHA-256")
    output_dir = str(Path(checked["runs"]["path"]) / cell_id)
    claim_path = str(Path(checked["claims"]["path"]) / f"{cell_id}.json")
    identity = {
        "cell_id": cell_id,
        "claim_path": claim_path,
        "final_freeze_token_sha256": final_freeze_token_sha256,
        "output_dir": output_dir,
        "protocol": PROTOCOL,
    }
    return {
        "claim_id_sha256": canonical_sha256(identity),
        "claim_path": claim_path,
        "output_dir": output_dir,
        "schema": OUTPUT_CLAIM_SCHEMA,
    }


def validate_output_claim(
    value: Any,
    *,
    cell_id: str,
    roots: Mapping[str, Any],
    final_freeze_token_sha256: str,
) -> dict[str, Any]:
    require(isinstance(value, Mapping), "cell output claim must be an object")
    _require_exact_keys(value, _OUTPUT_CLAIM_FIELDS, "cell output claim")
    require(value["schema"] == OUTPUT_CLAIM_SCHEMA, "cell output claim schema mismatch")
    expected = derive_output_claim(cell_id, roots, final_freeze_token_sha256)
    require(canonical_json_bytes(value) == canonical_json_bytes(expected), "cell output claim is not canonical")
    return dict(value)


def build_claim_record(
    output_claim: Mapping[str, Any],
    *,
    cell_id: str,
    preregistration_sha256: str,
    final_freeze_token_sha256: str,
    created_utc: str,
) -> dict[str, Any]:
    _require_sha256(preregistration_sha256, "claim pre-registration SHA-256")
    _require_sha256(final_freeze_token_sha256, "claim freeze-token SHA-256")
    require(isinstance(created_utc, str) and bool(created_utc), "claim creation timestamp is invalid")
    require(isinstance(output_claim, Mapping), "output claim must be an object")
    _require_exact_keys(output_claim, _OUTPUT_CLAIM_FIELDS, "output claim")
    return {
        "cell_id": cell_id,
        "claim_id_sha256": output_claim["claim_id_sha256"],
        "created_utc": created_utc,
        "final_freeze_token_sha256": final_freeze_token_sha256,
        "output_dir": output_claim["output_dir"],
        "preregistration_sha256": preregistration_sha256,
        "schema": CLAIM_RECORD_SCHEMA,
    }


def validate_claim_record(
    value: Any,
    *,
    output_claim: Mapping[str, Any],
    cell_id: str,
    preregistration_sha256: str,
    final_freeze_token_sha256: str,
) -> dict[str, Any]:
    require(isinstance(value, Mapping), "official claim record must be an object")
    _require_exact_keys(value, _CLAIM_RECORD_FIELDS, "official claim record")
    require(value["schema"] == CLAIM_RECORD_SCHEMA, "official claim record schema mismatch")
    require(isinstance(value["created_utc"], str) and bool(value["created_utc"]), "claim timestamp is invalid")
    expected = build_claim_record(
        output_claim,
        cell_id=cell_id,
        preregistration_sha256=preregistration_sha256,
        final_freeze_token_sha256=final_freeze_token_sha256,
        created_utc=value["created_utc"],
    )
    require(canonical_json_bytes(value) == canonical_json_bytes(expected), "official claim record binding mismatch")
    return dict(value)


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


def validate_train_venv_identity(value: Any, *, name: str = "train-venv identity") -> dict[str, Any]:
    """Validate the complete content-addressed train-venv identity."""

    require(isinstance(value, Mapping), f"{name} must be an object")
    _require_exact_keys(value, _TRAIN_VENV_IDENTITY_FIELDS, name)
    require(value["schema"] == TRAIN_VENV_IDENTITY_SCHEMA, f"{name} schema mismatch")
    for field in (
        "content_inventory_sha256",
        "root_sha256",
        "startup_hooks_sha256",
        "tree_metadata_sha256",
    ):
        _require_sha256(value[field], f"{name} {field}")
    for field in ("files_verified", "symlinks_verified", "total_bytes"):
        require(type(value[field]) is int and value[field] >= 0, f"{name} {field} is invalid")
    root = value["root"]
    require(
        isinstance(root, str) and bool(root) and Path(root).is_absolute() and str(Path(root)) == root,
        f"{name} root is not a canonical absolute path",
    )
    hooks = value["startup_hooks"]
    require(isinstance(hooks, list), f"{name} startup hooks must be a list")
    checked_hooks: list[str] = []
    for hook in hooks:
        require(isinstance(hook, str) and bool(hook), f"{name} startup hook is invalid")
        relative = PurePosixPath(hook)
        require(not relative.is_absolute() and ".." not in relative.parts, f"{name} startup hook path escapes")
        require(str(relative) == hook, f"{name} startup hook path is not canonical")
        checked_hooks.append(hook)
    require(checked_hooks == sorted(set(checked_hooks)), f"{name} startup hook inventory is not sorted and unique")
    checked = json.loads(canonical_json_bytes(value).decode("ascii"))
    try:
        require_matching_train_venv(checked, dict(checked))
    except RuntimeError as exc:
        raise RuntimeError(f"{name} is invalid: {exc}") from exc
    return checked


def validate_eval_venv_identity(value: Any, *, name: str = "eval-venv identity") -> dict[str, Any]:
    """Validate the exact eval-venv identity, including its base Python runtime."""

    require(isinstance(value, Mapping), f"{name} must be an object")
    _require_exact_keys(value, _TRAIN_VENV_IDENTITY_FIELDS, name)
    require(value["schema"] == EVAL_VENV_IDENTITY_SCHEMA, f"{name} schema mismatch")
    checked = json.loads(canonical_json_bytes(value).decode("ascii"))
    try:
        require_matching_eval_venv(checked, dict(checked))
    except RuntimeError as exc:
        raise RuntimeError(f"{name} is invalid: {exc}") from exc
    return checked


def validate_validator_runtime_identity(
    value: Any,
    *,
    name: str = "expert replay validator runtime identity",
) -> dict[str, Any]:
    """Validate the complete runtime identity copied from a qualified report."""

    require(isinstance(value, Mapping), f"{name} must be an object")
    _require_exact_keys(value, _VALIDATOR_RUNTIME_IDENTITY_FIELDS, name)
    require(value["schema"] == VALIDATOR_RUNTIME_IDENTITY_SCHEMA, f"{name} schema mismatch")
    _require_sha256(value["simulator_runtime_sha256"], f"{name} simulator_runtime_sha256")
    validate_eval_venv_identity(value["eval_venv_identity"], name=f"{name} eval-venv identity")
    for field in (
        "distribution_records",
        "installed_distributions",
        "module_origins",
        "packages",
        "process",
        "project_sources",
        "site_packages",
    ):
        require(isinstance(value[field], Mapping), f"{name} {field} must be an object")
    process = value["process"]
    _require_exact_keys(
        process,
        {
            "environment",
            "python_base_exec_prefix",
            "python_base_prefix",
            "python_executable",
            "python_flags",
            "python_invocation_flags",
            "python_prefix",
            "python_pycache_prefix",
            "python_version",
            "sys_path",
        },
        f"{name} process identity",
    )
    environment = process["environment"]
    require(isinstance(environment, Mapping), f"{name} process environment must be an object")
    require(
        "PYTHONPATH" not in environment
        and environment.get("PYTHONSAFEPATH") == "1"
        and environment.get("PYTHONDONTWRITEBYTECODE") == "1",
        f"{name} process does not enforce safe imports",
    )
    require(
        process["python_flags"] == {"dont_write_bytecode": True, "no_user_site": True, "safe_path": True}
        and process["python_invocation_flags"] == ["-P", "-B", "-X", "pycache_prefix=/dev/null"]
        and process["python_pycache_prefix"] == "/dev/null",
        f"{name} Python startup controls differ",
    )
    sys_path = process["sys_path"]
    require(
        isinstance(sys_path, list)
        and len(sys_path) == 5
        and len(set(sys_path)) == 5
        and all(isinstance(path, str) and Path(path).is_absolute() for path in sys_path),
        f"{name} import search path is invalid",
    )
    _require_exact_keys(
        value["project_sources"],
        {
            "bridge",
            "evaluator",
            "launcher",
            "preflight",
            "preflight_launcher",
            "replay_binder_launcher",
            "replay_collector_launcher",
            "replay_contract",
            "replay_qualification",
            "replay_qualification_launcher",
        },
        f"{name} project sources",
    )
    for source_name, digest in value["project_sources"].items():
        _require_sha256(digest, f"{name} project source {source_name}")
    _require_exact_keys(
        value["site_packages"],
        {"declared_files", "startup_files", "symlinks", "unregistered_files"},
        f"{name} site-packages identity",
    )
    _require_exact_keys(
        value["installed_distributions"],
        {"count", "inventory_sha256", "packages"},
        f"{name} installed-distribution identity",
    )
    return json.loads(canonical_json_bytes(value).decode("ascii"))


def validate_validator_runtime_against_attestation(
    value: Any,
    attestation: Mapping[str, Any],
) -> dict[str, Any]:
    """Require the qualified validator runtime to be the attestation-v3 runtime subset."""

    checked = validate_validator_runtime_identity(value)
    runtime_fields = _VALIDATOR_RUNTIME_IDENTITY_FIELDS - {"schema", "simulator_runtime_sha256"}
    require(
        all(checked[name] == attestation.get(name) for name in runtime_fields),
        "expert replay validator runtime differs from the simulator attestation",
    )
    simulator_runtime = {name: attestation.get(name) for name in _SIMULATOR_RUNTIME_IDENTITY_FIELDS}
    require(
        all(item is not None for item in simulator_runtime.values()),
        "simulator attestation runtime identity is incomplete",
    )
    require(
        checked["simulator_runtime_sha256"] == canonical_sha256(simulator_runtime),
        "expert replay validator simulator-runtime digest mismatch",
    )
    return checked


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, value in pairs:
        if name in result:
            raise ValueError(f"duplicate JSON field {name!r}")
        result[name] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant {value}")


def _read_strict_json(
    path: Path,
    *,
    name: str,
    expected_sha256: str | None = None,
) -> tuple[dict[str, Any], str]:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise RuntimeError(f"cannot read {name}: {path}") from exc
    digest = hashlib.sha256(raw).hexdigest()
    if expected_sha256 is not None:
        _require_sha256(expected_sha256, f"expected {name} SHA-256")
        require(hmac.compare_digest(digest, expected_sha256), f"{name} differs from its externally supplied SHA-256")
    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_unique_json_object,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, ValueError) as exc:
        raise RuntimeError(f"{name} is not strict finite UTF-8 JSON") from exc
    require(isinstance(value, dict), f"{name} root must be an object")
    return value, digest


def validate_evaluator_process_environment(project_root: Path) -> dict[str, str]:
    _activate_project_source_root()
    _validate_project_module_origins(_REQUIRED_PROJECT_MODULES)
    project_root = project_root.resolve()
    cache_root = Path(os.environ.get("DUO_VLA_CACHE_ROOT", "/root/.cache/duo-vla")).resolve()
    forbidden = ("GCONV_PATH", "GLIBC_TUNABLES", "LOCPATH")
    present_forbidden = [name for name in forbidden if os.environ.get(name)]
    injection_overrides = sorted(
        name
        for name in os.environ
        if name.startswith(_EVALUATOR_INJECTION_PREFIXES) and name not in _ALLOWED_EVALUATOR_INJECTION_ENVIRONMENT
    )
    algorithm_overrides = sorted(
        name
        for name in os.environ
        if name.startswith(_EVALUATOR_ALGORITHM_PREFIXES) and name not in _ALLOWED_EVALUATOR_ALGORITHM_ENVIRONMENT
    )
    require(
        not present_forbidden and not injection_overrides and not algorithm_overrides,
        "LIBERO evaluator environment contains injection/algorithm overrides: "
        f"forbidden={present_forbidden}, injection_overrides={injection_overrides}, "
        f"algorithm_overrides={algorithm_overrides}",
    )
    observed = {name: os.environ.get(name) for name in _REQUIRED_EVALUATOR_ENVIRONMENT}
    require(observed == _REQUIRED_EVALUATOR_ENVIRONMENT, f"LIBERO evaluator environment differs: {observed}")
    expected_prefix = (cache_root / "venvs/libero-eval").resolve()
    require(Path(sys.prefix).resolve() == expected_prefix, f"LIBERO evaluator must run from {expected_prefix}")
    require(sys.flags.safe_path == 1, "LIBERO evaluator requires Python safe-path mode")
    require(sys.flags.dont_write_bytecode == 1 and sys.dont_write_bytecode, "LIBERO evaluator requires -B")
    require(sys.flags.no_user_site == 1 and not site.ENABLE_USER_SITE, "LIBERO evaluator requires no user site")
    require(sys.pycache_prefix == "/dev/null", "LIBERO evaluator requires an impossible pycache lookup prefix")
    version = f"python{sys.version_info.major}.{sys.version_info.minor}"
    compact_version = f"python{sys.version_info.major}{sys.version_info.minor}"
    expected_sys_path = [
        str((project_root / "src").resolve()),
        str(Path(sys.base_prefix) / "lib" / f"{compact_version}.zip"),
        str(Path(sys.base_prefix) / "lib" / version),
        str(Path(sys.base_exec_prefix) / "lib" / version / "lib-dynload"),
        str(expected_prefix / "lib" / version / "site-packages"),
    ]
    require(sys.path == expected_sys_path, f"LIBERO evaluator import search path differs: {sys.path}")
    expected_path = f"{expected_prefix / 'bin'}:/usr/bin:/bin"
    require(os.environ.get("PATH") == expected_path, f"LIBERO evaluator requires PATH={expected_path}")
    require(os.environ.get("HF_HOME") == "/root/.cache/huggingface", "LIBERO evaluator requires pinned HF_HOME")
    expected_libero_config = str((cache_root / "simulators/libero/config").resolve())
    require(
        os.environ.get("LIBERO_CONFIG_PATH") == expected_libero_config,
        f"LIBERO evaluator requires LIBERO_CONFIG_PATH={expected_libero_config}",
    )
    expected_environment = {
        **_REQUIRED_EVALUATOR_ENVIRONMENT,
        "DUO_VLA_CACHE_ROOT": str(cache_root),
        "HF_HOME": "/root/.cache/huggingface",
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "LIBERO_CONFIG_PATH": expected_libero_config,
        "PATH": expected_path,
    }
    require(
        dict(os.environ) == expected_environment,
        "LIBERO evaluator process environment is not the exact closed launcher allowlist",
    )
    return expected_environment


def load_contamination_contract(project_root: Path) -> dict[str, Any]:
    """Authenticate the immutable exposure ledger and return its scoring rule."""

    ledger_path = project_root / "docs/evaluation_contamination.json"
    companion_path = project_root / "docs/evaluation_contamination.sha256"
    ledger, digest = _read_strict_json(
        ledger_path,
        name="LIBERO contamination ledger",
        expected_sha256=CONTAMINATION_LEDGER_SHA256,
    )
    try:
        companion = companion_path.read_text(encoding="ascii").strip().split()
    except OSError as exc:
        raise RuntimeError(f"cannot read contamination SHA-256 companion: {companion_path}") from exc
    require(
        companion == [CONTAMINATION_LEDGER_SHA256, "docs/evaluation_contamination.json"],
        "contamination SHA-256 companion differs from the frozen ledger",
    )
    _require_exact_keys(ledger, {"primary_reporting_rule", "records", "schema"}, "contamination ledger")
    require(ledger["schema"] == CONTAMINATION_SCHEMA, "contamination ledger schema mismatch")
    require(
        isinstance(ledger["records"], list) and bool(ledger["records"]),
        "contamination ledger has no exposure records",
    )
    rule = ledger["primary_reporting_rule"]
    require(isinstance(rule, dict), "contamination primary reporting rule must be an object")
    _require_exact_keys(
        rule,
        {
            "clean_holdout_episodes_per_policy_seed_k",
            "exclude_symmetrically_across",
            "excluded_cells",
            "official_2000_episode_result_label",
        },
        "contamination primary reporting rule",
    )
    require(
        rule["clean_holdout_episodes_per_policy_seed_k"] == OFFICIAL_PRIMARY_EPISODES,
        "contamination clean-holdout denominator changed",
    )
    require(
        rule["exclude_symmetrically_across"] == ["policy_objective", "flow_nfe", "execution_horizon", "training_seed"],
        "contamination symmetry factors changed",
    )
    excluded = rule["excluded_cells"]
    require(isinstance(excluded, list) and len(excluded) == 1, "contamination ledger must exclude exactly one cell")
    cell = excluded[0]
    require(isinstance(cell, dict), "contamination excluded cell must be an object")
    _require_exact_keys(cell, {"init_state_id", "suite", "task_id", "task_name"}, "contamination excluded cell")
    require(
        cell
        == {
            "init_state_id": 0,
            "suite": "libero_goal",
            "task_id": 7,
            "task_name": "turn_on_the_stove",
        },
        "contamination excluded episode changed",
    )
    require(
        rule["official_2000_episode_result_label"] == "non_blind_full_set_comparability_only",
        "contamination full-set label changed",
    )
    return {
        "excluded_episodes": [dict(OFFICIAL_EXCLUDED_EPISODE)],
        "full_episode_count": OFFICIAL_FULL_EPISODES,
        "ledger_schema": CONTAMINATION_SCHEMA,
        "ledger_sha256": digest,
        "primary_episode_count": OFFICIAL_PRIMARY_EPISODES,
        "report_label": "primary_clean_holdout",
    }


def official_episode_matrix(contamination: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Build the canonical 1,999-episode primary matrix in published order."""

    _require_exact_keys(contamination, _CONTAMINATION_FIELDS, "contamination contract")
    excluded = {(item["suite"], item["task_id"], item["reset_id"]) for item in contamination["excluded_episodes"]}
    episodes = [
        {"reset_id": reset_id, "suite": suite, "task_id": task_id}
        for suite in SUITES
        for task_id in range(10)
        for reset_id in range(OFFICIAL_RESETS_PER_TASK)
        if (suite, task_id, reset_id) not in excluded
    ]
    require(len(episodes) == OFFICIAL_PRIMARY_EPISODES, "canonical LIBERO primary episode count changed")
    require(
        len(episodes) + len(excluded) == OFFICIAL_FULL_EPISODES,
        "canonical LIBERO full episode count changed",
    )
    return episodes


def official_cell_id(train_seed: int, objective: str, nfe: int, execution_horizon: int) -> str:
    label = "flow" if objective == "rectified_flow" else "direct"
    return f"seed-{train_seed}-{label}-nfe-{nfe}-k-{execution_horizon}"


def _official_factor_matrix() -> set[tuple[int, str, int, int]]:
    return {
        (seed, objective, nfe, execution_horizon)
        for seed in OFFICIAL_TRAIN_SEEDS
        for objective, nfes in (("rectified_flow", OFFICIAL_FLOW_NFES), ("direct_regression", (1,)))
        for nfe in nfes
        for execution_horizon in OFFICIAL_EXECUTION_HORIZONS
    }


def _validate_registered_cell(
    cell: Any,
    *,
    episode_matrix_sha256: str,
    roots: Mapping[str, Any],
    final_freeze_token_sha256: str,
) -> dict[str, Any]:
    require(isinstance(cell, dict), "pre-registration cell must be an object")
    _require_exact_keys(cell, _CELL_FIELDS, "pre-registration cell")
    checkpoint = cell["checkpoint"]
    require(isinstance(checkpoint, dict), "pre-registration checkpoint identity must be an object")
    _require_exact_keys(checkpoint, _CHECKPOINT_FIELDS, "pre-registration checkpoint identity")
    _require_sha256(checkpoint["manifest_sha256"], "checkpoint manifest_sha256")
    _require_sha256(checkpoint["source_tree_sha256"], "checkpoint source_tree_sha256")
    _require_sha256(checkpoint["dataset_tree_sha256"], "checkpoint dataset_tree_sha256")
    _require_sha256(
        checkpoint["dataset_content_inventory_sha256"],
        "checkpoint dataset_content_inventory_sha256",
    )
    require(
        checkpoint["dataset_tree_sha256"] == DATASET_TREE_METADATA_SHA256
        and checkpoint["dataset_content_inventory_sha256"] == DATASET_CONTENT_INVENTORY_SHA256,
        "checkpoint dataset identity differs from the pinned LIBERO snapshot",
    )
    require(
        type(checkpoint["update"]) is int and checkpoint["update"] == FINAL_CHECKPOINT_UPDATE,
        f"official checkpoint update must equal {FINAL_CHECKPOINT_UPDATE}",
    )
    _require_sha256(cell["policy_contract_sha256"], "policy_contract_sha256")
    _require_sha256(cell["latency_runtime_sha256"], "latency_runtime_sha256")
    _require_sha256(cell["serving_policy_sha256"], "serving_policy_sha256")
    _require_sha256(cell["serving_runtime_sha256"], "serving_runtime_sha256")
    _require_sha256(cell["episode_matrix_sha256"], "episode_matrix_sha256")
    require(
        hmac.compare_digest(cell["episode_matrix_sha256"], episode_matrix_sha256),
        "cell episode matrix differs from the frozen primary matrix",
    )
    require(cell["execution_geometry"] == LIBERO_EXECUTION_GEOMETRY, "cell execution geometry mismatch")
    train_seed = cell["train_seed"]
    objective = cell["objective"]
    nfe = cell["nfe"]
    execution_horizon = cell["execution_horizon"]
    require(type(train_seed) is int and train_seed in OFFICIAL_TRAIN_SEEDS, "cell train_seed must be one of {0,1,2}")
    require(
        type(execution_horizon) is int and execution_horizon in OFFICIAL_EXECUTION_HORIZONS,
        "cell execution_horizon must be one of {1,4}",
    )
    require(
        type(cell["policy_warmup_calls"]) is int and cell["policy_warmup_calls"] == OFFICIAL_POLICY_WARMUP_CALLS,
        f"official policy warm-up count must equal {OFFICIAL_POLICY_WARMUP_CALLS}",
    )
    if objective == "rectified_flow":
        require(type(nfe) is int and nfe in OFFICIAL_FLOW_NFES, "flow cell NFE must be one of {1,5,10}")
        require(cell["sampler"] == "euler_uniform", "flow cell sampler mismatch")
        require(
            cell["inference_seed_behavior"] == "episode_identity_gaussian_noise",
            "flow cell inference seed behavior mismatch",
        )
    elif objective == "direct_regression":
        require(type(nfe) is int and nfe == 1, "direct cell NFE must equal one")
        require(cell["sampler"] == "single_forward", "direct cell sampler mismatch")
        require(
            cell["inference_seed_behavior"] == "episode_identity_echo_only",
            "direct cell inference seed behavior mismatch",
        )
    else:
        raise RuntimeError(f"unsupported pre-registration objective {objective!r}")
    selected_policy = {name: cell[name] for name in ("inference_seed_behavior", "nfe", "objective", "sampler")}
    require(
        hmac.compare_digest(cell["serving_policy_sha256"], canonical_sha256(selected_policy)),
        "selected serving-policy SHA-256 mismatch",
    )
    require(
        cell["cell_id"] == official_cell_id(train_seed, objective, nfe, execution_horizon),
        "pre-registration cell_id is not canonical",
    )
    validate_output_claim(
        cell["output_claim"],
        cell_id=cell["cell_id"],
        roots=roots,
        final_freeze_token_sha256=final_freeze_token_sha256,
    )
    return dict(cell)


def validate_preregistration_manifest(
    manifest: Any,
    *,
    contamination: Mapping[str, Any],
    simulator_attestation_sha256: str,
) -> list[dict[str, Any]]:
    require(isinstance(manifest, dict), "pre-registration manifest must be an object")
    _require_exact_keys(manifest, _PREREGISTRATION_FIELDS, "pre-registration manifest")
    require(manifest["schema"] == PREREGISTRATION_SCHEMA, "pre-registration schema mismatch")
    roots = validate_official_output_roots(manifest["official_output_roots"])
    require(
        manifest["aggregation_python_version"] == AGGREGATION_PYTHON_VERSION,
        f"official aggregation requires Python {AGGREGATION_PYTHON_VERSION}",
    )
    _require_sha256(manifest["aggregator_sha256"], "pre-registration aggregator_sha256")
    require(manifest["benchmark_protocol"] == PROTOCOL, "pre-registration protocol mismatch")
    require(manifest["training_seeds"] == list(OFFICIAL_TRAIN_SEEDS), "pre-registration training seeds changed")
    require(
        manifest["execution_horizons"] == list(OFFICIAL_EXECUTION_HORIZONS),
        "pre-registration execution horizons changed",
    )
    require(manifest["suites"] == list(SUITES), "pre-registration suite order changed")
    require(manifest["task_ids"] == list(range(10)), "pre-registration task set changed")
    require(
        manifest["official_resets_per_task"] == OFFICIAL_RESETS_PER_TASK,
        "pre-registration official reset count changed",
    )
    require(
        manifest["policy_warmup_calls"] == OFFICIAL_POLICY_WARMUP_CALLS,
        f"pre-registration policy warm-up count must equal {OFFICIAL_POLICY_WARMUP_CALLS}",
    )
    require(
        manifest["final_checkpoint_update"] == FINAL_CHECKPOINT_UPDATE,
        "pre-registration final checkpoint update changed",
    )
    require(
        type(manifest["evaluation_seed"]) is int and 0 <= manifest["evaluation_seed"] < 2**63,
        "pre-registration evaluation_seed is invalid",
    )
    _require_sha256(manifest["final_freeze_token_sha256"], "final_freeze_token_sha256")
    expert_replay = manifest["expert_replay_qualification"]
    require(isinstance(expert_replay, Mapping), "expert replay qualification identity must be an object")
    _require_exact_keys(
        expert_replay,
        _EXPERT_REPLAY_QUALIFICATION_FIELDS,
        "expert replay qualification identity",
    )
    config_file_sha256 = expert_replay["config_file_sha256"]
    require(isinstance(config_file_sha256, Mapping), "expert replay qualification config identity is invalid")
    _require_exact_keys(
        config_file_sha256,
        {"direct_regression", "rectified_flow"},
        "expert replay qualification config identity",
    )
    for name, value in config_file_sha256.items():
        _require_sha256(value, f"expert replay qualification config {name}")
    for name in (
        "content_sha256",
        "dataset_content_inventory_sha256",
        "dataset_tree_metadata_sha256",
        "evidence_manifest_raw_sha256",
        "gates_sha256",
        "normalization_content_sha256",
        "normalization_raw_sha256",
        "original_hdf5_inventory_content_sha256",
        "original_hdf5_inventory_raw_sha256",
        "project_source_tree_sha256",
        "raw_evidence_root_sha256",
        "report_sha256",
        "simulator_attestation_raw_sha256",
        "simulator_attestation_sha256",
        "simulator_runtime_sha256",
        "task_inventory_sha256",
    ):
        _require_sha256(expert_replay[name], f"expert replay qualification {name}")
    require(
        expert_replay["schema"] == "duo-vla-libero-expert-replay-qualification-v1"
        and expert_replay["kind"] == "libero-40-task-regenerated-expert-replay"
        and expert_replay["status"] == "passed",
        "expert replay qualification schema/kind/status mismatch",
    )
    require(
        expert_replay["task_count"] == expert_replay["successful_task_count"] == 40
        and expert_replay["demonstration_count"] == expert_replay["successful_demonstration_count"] >= 40,
        "expert replay qualification is not a 40/40 successful replay",
    )
    require(
        expert_replay["dataset_tree_metadata_sha256"] == DATASET_TREE_METADATA_SHA256
        and expert_replay["dataset_content_inventory_sha256"] == DATASET_CONTENT_INVENTORY_SHA256,
        "expert replay qualification dataset identity mismatch",
    )
    require(
        expert_replay["dataset_snapshot_files_verified"] == DATASET_SNAPSHOT_FILES_VERIFIED
        and expert_replay["dataset_snapshot_total_bytes"] == DATASET_SNAPSHOT_TOTAL_BYTES,
        "expert replay qualification dataset snapshot counts mismatch",
    )
    validate_train_venv_identity(
        expert_replay["train_venv_identity"],
        name="expert replay qualification train-venv identity",
    )
    validator_runtime = validate_validator_runtime_identity(expert_replay["validator_runtime_identity"])
    require(
        validator_runtime["simulator_runtime_sha256"] == expert_replay["simulator_runtime_sha256"],
        "expert replay validator runtime differs from its simulator runtime identity",
    )
    require(
        expert_replay["normalization_content_sha256"] == NORMALIZATION_SHA256,
        "expert replay qualification normalization identity mismatch",
    )
    require(
        expert_replay["original_hdf5_repository_id"] == ORIGINAL_HDF5_REPOSITORY_ID
        and expert_replay["original_hdf5_revision"] == ORIGINAL_HDF5_REVISION,
        "expert replay qualification original HDF5 repository identity mismatch",
    )
    require(
        expert_replay["original_hdf5_inventory_content_sha256"] == ORIGINAL_HDF5_CONTENT_SHA256,
        "expert replay qualification original HDF5 inventory identity mismatch",
    )
    require(
        expert_replay["original_hdf5_inventory_raw_sha256"] == ORIGINAL_HDF5_INVENTORY_RAW_SHA256,
        "expert replay qualification original HDF5 raw inventory identity mismatch",
    )
    require(
        expert_replay["original_hdf5_file_count"] == ORIGINAL_HDF5_FILE_COUNT
        and expert_replay["original_hdf5_total_bytes"] == ORIGINAL_HDF5_TOTAL_BYTES,
        "expert replay qualification original HDF5 size identity mismatch",
    )
    require(
        expert_replay["task_inventory_sha256"] == TASK_INVENTORY_SHA256,
        "expert replay qualification task inventory identity mismatch",
    )
    _require_sha256(manifest["simulator_attestation_sha256"], "simulator_attestation_sha256")
    _require_sha256(simulator_attestation_sha256, "current simulator attestation SHA-256")
    require(
        hmac.compare_digest(manifest["simulator_attestation_sha256"], simulator_attestation_sha256),
        "current simulator attestation differs from pre-registration",
    )
    require(
        expert_replay["simulator_attestation_sha256"] == manifest["simulator_attestation_sha256"],
        "expert replay qualification simulator attestation mismatch",
    )
    require(manifest["contamination"] == dict(contamination), "pre-registration contamination contract changed")
    episodes = official_episode_matrix(contamination)
    require(manifest["episode_count"] == OFFICIAL_PRIMARY_EPISODES, "pre-registration episode count changed")
    require(manifest["episodes"] == episodes, "pre-registration episode matrix changed")
    episode_sha256 = canonical_sha256(episodes)
    _require_sha256(manifest["episode_matrix_sha256"], "pre-registration episode_matrix_sha256")
    require(
        hmac.compare_digest(manifest["episode_matrix_sha256"], episode_sha256),
        "pre-registration episode matrix SHA-256 mismatch",
    )
    cells = manifest["cells"]
    require(isinstance(cells, list), "pre-registration cells must be a list")
    checked = [
        _validate_registered_cell(
            cell,
            episode_matrix_sha256=episode_sha256,
            roots=roots,
            final_freeze_token_sha256=manifest["final_freeze_token_sha256"],
        )
        for cell in cells
    ]
    ids = [cell["cell_id"] for cell in checked]
    require(len(ids) == len(set(ids)), "pre-registration cell_id values must be unique")
    factors = {(cell["train_seed"], cell["objective"], cell["nfe"], cell["execution_horizon"]) for cell in checked}
    require(factors == _official_factor_matrix(), "pre-registration does not contain the exact 24-cell factor matrix")
    require(len(checked) == len(factors) == 24, "pre-registration must contain exactly 24 policy cells")
    require(
        len({cell["output_claim"]["output_dir"] for cell in checked}) == 24,
        "pre-registration must assign one output directory per cell",
    )
    require(
        len({cell["output_claim"]["claim_path"] for cell in checked}) == 24,
        "pre-registration must assign one claim path per cell",
    )
    require(
        len({cell["output_claim"]["claim_id_sha256"] for cell in checked}) == 24,
        "pre-registration must assign one claim identity per cell",
    )

    for seed in OFFICIAL_TRAIN_SEEDS:
        for objective in ("rectified_flow", "direct_regression"):
            group = [cell for cell in checked if cell["train_seed"] == seed and cell["objective"] == objective]
            checkpoint_identities = {canonical_json_bytes(cell["checkpoint"]) for cell in group}
            contract_ids = {cell["policy_contract_sha256"] for cell in group}
            runtime_ids = {cell["serving_runtime_sha256"] for cell in group}
            require(len(checkpoint_identities) == 1, "NFE/K cells must share one final checkpoint per seed/objective")
            require(len(contract_ids) == 1, "NFE/K cells must share one policy contract per seed/objective")
            require(len(runtime_ids) == 1, "NFE/K cells must share one serving runtime per seed/objective")
        flow_manifest = next(
            cell["checkpoint"]["manifest_sha256"]
            for cell in checked
            if cell["train_seed"] == seed and cell["objective"] == "rectified_flow"
        )
        direct_manifest = next(
            cell["checkpoint"]["manifest_sha256"]
            for cell in checked
            if cell["train_seed"] == seed and cell["objective"] == "direct_regression"
        )
        require(flow_manifest != direct_manifest, "flow and direct cells cannot share a checkpoint manifest")
    require(
        len({cell["checkpoint"]["source_tree_sha256"] for cell in checked}) == 1,
        "all official checkpoints must share one qualified source tree",
    )
    require(
        {
            (
                cell["checkpoint"]["source_tree_sha256"],
                cell["checkpoint"]["dataset_tree_sha256"],
                cell["checkpoint"]["dataset_content_inventory_sha256"],
            )
            for cell in checked
        }
        == {
            (
                expert_replay["project_source_tree_sha256"],
                expert_replay["dataset_tree_metadata_sha256"],
                expert_replay["dataset_content_inventory_sha256"],
            )
        },
        "official checkpoints differ from the expert-replay-qualified source/data identity",
    )
    require(
        len({cell["latency_runtime_sha256"] for cell in checked}) == 1,
        "all latency-reporting cells must share one hardware/software runtime identity",
    )
    require(
        len({cell["checkpoint"]["manifest_sha256"] for cell in checked}) == 6,
        "official matrix must contain one distinct final checkpoint per seed/objective",
    )
    contract_ids_by_objective = {
        objective: {cell["policy_contract_sha256"] for cell in checked if cell["objective"] == objective}
        for objective in ("rectified_flow", "direct_regression")
    }
    require(
        all(len(values) == 1 for values in contract_ids_by_objective.values()),
        "each objective must share one policy contract across all training seeds",
    )
    require(
        len(set.union(*contract_ids_by_objective.values())) == 2,
        "flow and direct objectives must have distinct policy contracts",
    )
    return checked


def load_preregistration(
    path: Path,
    *,
    cell_id: str,
    execution_horizon: int,
    evaluation_seed: int,
    final_freeze_token: str,
    preregistration_sha256: str,
    simulator_attestation_sha256: str,
    contamination: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], str]:
    manifest, manifest_sha256 = _read_strict_json(
        path,
        name="LIBERO pre-registration manifest",
        expected_sha256=preregistration_sha256,
    )
    cells = validate_preregistration_manifest(
        manifest,
        contamination=contamination,
        simulator_attestation_sha256=simulator_attestation_sha256,
    )
    require(manifest["evaluation_seed"] == evaluation_seed, "CLI evaluation seed differs from pre-registration")
    require(isinstance(final_freeze_token, str) and bool(final_freeze_token.strip()), "final freeze token is empty")
    token_sha256 = hashlib.sha256(final_freeze_token.encode("utf-8")).hexdigest()
    require(
        hmac.compare_digest(token_sha256, manifest["final_freeze_token_sha256"]),
        "final freeze token does not match pre-registration",
    )
    matching = [cell for cell in cells if cell["cell_id"] == cell_id]
    require(len(matching) == 1, "selected --cell-id is absent from pre-registration")
    selected = matching[0]
    require(
        selected["execution_horizon"] == execution_horizon,
        "CLI execution horizon differs from the pre-registered cell",
    )
    return dict(manifest), selected, manifest_sha256


def load_development_reset_bank(bank_dir: Path) -> DevelopmentResetBank:
    manifest, artifacts = load_bank(bank_dir)
    runtime_manifest_path = (
        Path(os.environ.get("DUO_VLA_CACHE_ROOT", "/root/.cache/duo-vla")) / "simulators/libero/manifest.json"
    )
    try:
        runtime_manifest = json.loads(runtime_manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot authenticate current LIBERO runtime: {runtime_manifest_path}") from exc
    expected_simulator = {
        "assets_revision": runtime_manifest["assets"]["revision"],
        "egl_device": str(int(os.environ.get("MUJOCO_EGL_DEVICE_ID", "0"))),
        "packages": runtime_manifest["environment"]["packages"],
        "python": runtime_manifest["environment"]["python"],
        "schema": runtime_manifest["schema"],
        "source_revision": runtime_manifest["source"]["revision"],
    }
    require(manifest["simulator"] == expected_simulator, "development reset bank simulator identity mismatch")
    states_by_task: dict[tuple[str, int], np.ndarray] = {}
    task_records: dict[tuple[str, int], dict[str, Any]] = {}
    for record in manifest["tasks"]:
        identity = (record["suite"], record["task_id"])
        path = record["artifact"]["path"]
        require(path in artifacts, f"development reset bank is missing {path}")
        states_by_task[identity] = artifacts[path]
        task_records[identity] = record
    manifest_path = bank_dir / "manifest.json"
    return DevelopmentResetBank(
        manifest=manifest,
        manifest_sha256=sha256_file(manifest_path),
        states_by_task=states_by_task,
        task_records=task_records,
    )


def rotate_eval_rgb_180(image: np.ndarray) -> np.ndarray:
    values = np.asarray(image)
    require(values.shape == IMAGE_SHAPE, f"camera image must have shape {IMAGE_SHAPE}, got {values.shape}")
    require(values.dtype == np.uint8, f"camera image must have uint8 dtype, got {values.dtype}")
    return np.flip(values, axis=(0, 1)).copy()


def libero_state(observation: dict[str, Any]) -> np.ndarray:
    from robosuite.utils.transform_utils import quat2axisangle

    position = np.asarray(observation["robot0_eef_pos"], dtype=np.float32)
    quaternion = np.asarray(observation["robot0_eef_quat"], dtype=np.float64).copy()
    gripper = np.asarray(observation["robot0_gripper_qpos"], dtype=np.float32)
    require(position.shape == (3,), f"unexpected EEF position shape: {position.shape}")
    require(quaternion.shape == (4,), f"unexpected EEF quaternion shape: {quaternion.shape}")
    require(gripper.shape == (2,), f"unexpected gripper shape: {gripper.shape}")
    axis_angle = np.asarray(quat2axisangle(quaternion), dtype=np.float32)
    state = np.concatenate((position, axis_angle, gripper)).astype(np.float32, copy=False)
    require(state.shape == (8,) and bool(np.isfinite(state).all()), "constructed LIBERO state is invalid")
    return state.copy()


def libero_env_action(action: np.ndarray) -> tuple[np.ndarray, int]:
    values = np.asarray(action, dtype=np.float32).reshape(-1)
    require(
        values.shape == (ACTION_DIM,) and bool(np.isfinite(values).all()),
        "policy action must contain seven finite values",
    )
    clip_count = int(np.count_nonzero(np.abs(values[:6]) > 1.0))
    output = values.copy()
    output[:6] = np.clip(output[:6], -1.0, 1.0)
    output[6] = 1.0 if output[6] >= 0 else -1.0
    return output, clip_count


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    return float(np.percentile(np.asarray(values, dtype=np.float64), q, method="linear"))


def wilson95(successes: int, trials: int) -> list[float] | None:
    if trials == 0:
        return None
    require(0 <= successes <= trials, "invalid success counts")
    z = 1.959963984540054
    proportion = successes / trials
    denominator = 1.0 + z * z / trials
    center = (proportion + z * z / (2.0 * trials)) / denominator
    radius = z * math.sqrt(proportion * (1.0 - proportion) / trials + z * z / (4.0 * trials * trials)) / denominator
    return [max(0.0, center - radius), min(1.0, center + radius)]


def parse_selection(text: str, *, upper: int, name: str) -> tuple[int, ...]:
    if text == "all":
        return tuple(range(upper))
    selected: set[int] = set()
    for component in text.split(","):
        component = component.strip()
        if not component:
            raise ValueError(f"empty component in {name}")
        if "-" in component:
            fields = component.split("-")
            if len(fields) != 2:
                raise ValueError(f"invalid range {component!r} in {name}")
            start, stop = map(int, fields)
            if stop < start:
                raise ValueError(f"descending range {component!r} in {name}")
            selected.update(range(start, stop + 1))
        else:
            selected.add(int(component))
    if not selected or min(selected) < 0 or max(selected) >= upper:
        raise ValueError(f"{name} must select values in [0, {upper})")
    return tuple(sorted(selected))


def run_exact_simulator_preflight(project_root: Path, *, construct_environment: bool) -> dict[str, Any]:
    script = project_root / "scripts/preflight_libero_env.py"
    command = [sys.executable, "-P", "-B", "-X", "pycache_prefix=/dev/null", str(script)]
    if not construct_environment:
        command.append("--imports-only")
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode:
        raise RuntimeError(
            f"exact LIBERO simulator preflight failed ({completed.returncode}):\n{completed.stdout}{completed.stderr}"
        )
    lines = completed.stdout.splitlines()
    json_start = next((index for index, line in enumerate(lines) if line.startswith("{")), None)
    require(json_start is not None, f"simulator preflight did not emit JSON: {completed.stdout}")
    report = json.loads("\n".join(lines[json_start:]))
    require(report.get("status") == "ok", "simulator preflight did not report success")
    require(report.get("schema") == SIMULATOR_ATTESTATION_SCHEMA, "simulator attestation schema mismatch")
    require(
        report.get("environment_constructed") is construct_environment,
        "simulator preflight environment-construction status mismatch",
    )
    return report


def validate_policy_health(health: dict[str, Any], *, allow_fake_policy: bool) -> None:
    mode = health.get("mode")
    require(mode in {"fake", "real"}, "policy server reported an unknown mode")
    require(health.get("prefix_cache_scope") == "request", "policy prefix cache scope mismatch")
    if mode == "fake":
        require(allow_fake_policy, "refusing test-only fake policy without --allow-fake-policy")
        for name in (
            "checkpoint",
            "execution_geometry",
            "latency_runtime_sha256",
            "model_revision",
            "normalization_content_sha256",
            "serving_runtime_sha256",
        ):
            require(health.get(name) is None, f"fake policy cannot claim {name}")
        return
    require(health.get("model_revision") == MODEL_REVISION, "policy model revision mismatch")
    require(health.get("dataset_revision") == DATASET_REVISION, "policy dataset revision mismatch")
    require(
        health.get("normalization_content_sha256") == NORMALIZATION_SHA256,
        "policy normalization content hash mismatch",
    )
    serving_runtime_sha256 = health.get("serving_runtime_sha256")
    require(
        isinstance(serving_runtime_sha256, str)
        and len(serving_runtime_sha256) == 64
        and all(character in "0123456789abcdef" for character in serving_runtime_sha256),
        "policy serving runtime hash is invalid",
    )
    latency_runtime_sha256 = health.get("latency_runtime_sha256")
    require(
        isinstance(latency_runtime_sha256, str)
        and len(latency_runtime_sha256) == 64
        and all(character in "0123456789abcdef" for character in latency_runtime_sha256),
        "policy latency runtime hash is invalid",
    )
    checkpoint = health.get("checkpoint")
    require(isinstance(checkpoint, dict), "real policy health has no checkpoint identity")
    require(checkpoint.get("kind") == "resumable-libero-training", "real policy checkpoint kind mismatch")
    require(health.get("train_seed") in OFFICIAL_TRAIN_SEEDS, "real policy train seed must be one of {0,1,2}")
    require(checkpoint.get("train_seed") == health.get("train_seed"), "checkpoint and policy train seeds disagree")
    execution_geometry = validate_execution_geometry(health.get("execution_geometry"))
    require(
        checkpoint.get("execution_geometry") == execution_geometry == LIBERO_EXECUTION_GEOMETRY,
        "checkpoint and live execution geometry disagree",
    )
    require(
        isinstance(checkpoint.get("manifest_sha256"), str) and len(checkpoint["manifest_sha256"]) == 64,
        "policy checkpoint manifest hash is invalid",
    )
    require(
        isinstance(checkpoint.get("policy_contract_sha256"), str) and len(checkpoint["policy_contract_sha256"]) == 64,
        "policy checkpoint contract hash is invalid",
    )
    checkpoint_contract = checkpoint.get("policy_contract")
    require(isinstance(checkpoint_contract, dict), "real policy checkpoint has no policy contract")
    wire_contract = {name: health[name] for name in ("objective", "sampler", "nfe", "inference_seed_behavior")}
    for name in ("objective", "sampler", "inference_seed_behavior"):
        require(checkpoint_contract.get(name) == wire_contract[name], "checkpoint and policy-server contracts disagree")
    if wire_contract["objective"] == "rectified_flow":
        require(wire_contract["nfe"] in {1, 5, 10}, "unsupported rectified-flow serving NFE")
    else:
        require(checkpoint_contract.get("nfe") == wire_contract["nfe"], "direct-regression NFE was overridden")


def _validate_health_train_venv(
    checkpoint: Mapping[str, Any],
    expected_train_venv_identity: Mapping[str, Any],
) -> dict[str, Any]:
    expected = validate_train_venv_identity(
        expected_train_venv_identity,
        name="pre-registered expert replay train-venv identity",
    )
    observed = validate_train_venv_identity(
        checkpoint.get("train_venv"),
        name="policy health checkpoint train-venv identity",
    )
    require(
        canonical_json_bytes(observed) == canonical_json_bytes(expected),
        "policy health checkpoint train-venv differs from expert replay qualification",
    )
    training_environment = checkpoint.get("training_execution_environment")
    require(isinstance(training_environment, Mapping), "policy health has no authenticated training environment")
    _require_sha256(
        checkpoint.get("training_execution_environment_sha256"),
        "policy health training execution environment SHA-256",
    )
    require(
        canonical_sha256(training_environment) == checkpoint["training_execution_environment_sha256"],
        "policy health training execution environment digest mismatch",
    )
    authenticated_runtime = training_environment.get("authenticated_runtime")
    require(
        isinstance(authenticated_runtime, Mapping), "policy health training environment has no authenticated runtime"
    )
    nested = validate_train_venv_identity(
        authenticated_runtime.get("train_venv"),
        name="policy health authenticated-runtime train-venv identity",
    )
    require(
        canonical_json_bytes(nested) == canonical_json_bytes(expected),
        "policy health authenticated train-venv differs from expert replay qualification",
    )
    return expected


def _authenticate_final_checkpoint(
    health: Mapping[str, Any],
    selected_cell: Mapping[str, Any],
    *,
    expected_train_venv_identity: Mapping[str, Any],
) -> dict[str, Any]:
    checkpoint = health["checkpoint"]
    checked_train_venv = _validate_health_train_venv(checkpoint, expected_train_venv_identity)
    registered = selected_cell["checkpoint"]
    require(isinstance(checkpoint, Mapping), "policy health checkpoint identity must be an object")
    checkpoint_dir = Path(checkpoint["path"]).resolve()
    require(checkpoint_dir.parent.name == "checkpoints", "checkpoint is not inside a canonical checkpoints directory")
    run_root = checkpoint_dir.parent.parent
    record = validate_resume_checkpoint(run_root, checkpoint_dir)
    require(record.update == registered["update"], "run-journal checkpoint update differs from pre-registration")
    require(
        hmac.compare_digest(record.manifest_sha256, registered["manifest_sha256"]),
        "run-journal checkpoint manifest differs from pre-registration",
    )
    manifest, manifest_sha256 = _read_strict_json(
        checkpoint_dir / "manifest.json",
        name="final checkpoint manifest",
        expected_sha256=registered["manifest_sha256"],
    )
    require(manifest.get("schema") == "duo-vla-checkpoint-v1", "final checkpoint schema mismatch")
    require(manifest.get("kind") == "resumable-libero-training", "final checkpoint kind mismatch")
    require(manifest.get("run_seed") == selected_cell["train_seed"], "final checkpoint train seed mismatch")
    require(
        manifest.get("source_tree_sha256") == registered["source_tree_sha256"],
        "final checkpoint source tree differs from pre-registration",
    )
    require(
        manifest.get("dataset_tree_sha256") == registered["dataset_tree_sha256"]
        and manifest.get("dataset_content_inventory_sha256") == registered["dataset_content_inventory_sha256"],
        "final checkpoint dataset identity differs from pre-registration",
    )
    require(
        manifest.get("policy_contract_sha256") == selected_cell["policy_contract_sha256"],
        "final checkpoint policy contract differs from pre-registration",
    )
    require(manifest.get("task") is None, "official checkpoint must be the full 40-task multitask run")
    require(
        manifest.get("train_episode_count") == OFFICIAL_TRAIN_EPISODES,
        f"official checkpoint train episode count must equal {OFFICIAL_TRAIN_EPISODES}",
    )
    require(
        manifest.get("validation_episode_count") == OFFICIAL_VALIDATION_EPISODES,
        f"official checkpoint validation episode count must equal {OFFICIAL_VALIDATION_EPISODES}",
    )

    artifacts = manifest.get("artifacts")
    require(isinstance(artifacts, Mapping), "final checkpoint has no artifact inventory")
    resolved_artifact = artifacts.get("resolved_config")
    require(isinstance(resolved_artifact, Mapping), "final checkpoint has no resolved configuration artifact")
    _require_exact_keys(resolved_artifact, {"bytes", "path", "sha256"}, "resolved configuration artifact")
    relative_path = resolved_artifact["path"]
    expected_bytes = resolved_artifact["bytes"]
    expected_artifact_sha256 = resolved_artifact["sha256"]
    require(isinstance(relative_path, str) and bool(relative_path), "resolved configuration path is invalid")
    require(
        type(expected_bytes) is int and expected_bytes > 0,
        "resolved configuration byte count is invalid",
    )
    _require_sha256(expected_artifact_sha256, "resolved configuration artifact SHA-256")
    resolved_config_path = (checkpoint_dir / relative_path).resolve()
    require(
        resolved_config_path.is_relative_to(checkpoint_dir),
        "resolved configuration artifact escapes the checkpoint directory",
    )
    require(
        resolved_config_path.is_file() and resolved_config_path.stat().st_size == expected_bytes,
        "resolved configuration artifact byte count differs from the checkpoint manifest",
    )
    resolved_envelope, _ = _read_strict_json(
        resolved_config_path,
        name="final checkpoint resolved configuration",
        expected_sha256=expected_artifact_sha256,
    )
    _require_exact_keys(resolved_envelope, {"config", "config_sha256"}, "resolved configuration envelope")
    resolved_config = resolved_envelope["config"]
    resolved_config_sha256 = resolved_envelope["config_sha256"]
    require(isinstance(resolved_config, dict), "resolved configuration payload must be an object")
    _require_sha256(resolved_config_sha256, "resolved configuration semantic SHA-256")
    require(
        canonical_sha256(resolved_config) == resolved_config_sha256 == manifest.get("config_sha256"),
        "resolved configuration semantic identity differs from the final checkpoint",
    )
    resolved_training_environment = resolved_config.get("execution_environment")
    require(
        isinstance(resolved_training_environment, Mapping)
        and resolved_training_environment == manifest.get("execution_environment")
        and resolved_training_environment == checkpoint.get("training_execution_environment"),
        "checkpoint, resolved config, and policy health training environments differ",
    )
    require(
        canonical_sha256(resolved_training_environment)
        == manifest.get("execution_environment_sha256")
        == checkpoint.get("training_execution_environment_sha256"),
        "checkpoint training execution environment identity mismatch",
    )
    resolved_authenticated_runtime = resolved_training_environment.get("authenticated_runtime")
    require(
        isinstance(resolved_authenticated_runtime, Mapping),
        "resolved training environment has no authenticated runtime",
    )
    resolved_train_venv = validate_train_venv_identity(
        resolved_authenticated_runtime.get("train_venv"),
        name="resolved training environment train-venv identity",
    )
    require(
        canonical_json_bytes(resolved_train_venv) == canonical_json_bytes(checked_train_venv),
        "checkpoint authenticated train-venv differs from expert replay qualification",
    )

    objective = selected_cell["objective"]
    canonical_name = "libero_direct_regression.toml" if objective == "direct_regression" else "libero.toml"
    canonical_recipe = load_resolved_toml(Path(__file__).resolve().parents[1] / "configs" / canonical_name)
    dynamic_fields = {"artifact_trees", "execution_environment", "run", "source_tree_sha256"}
    require(
        set(resolved_config) == set(canonical_recipe) | dynamic_fields,
        "official resolved configuration top-level fields differ from the canonical recipe",
    )
    for name, expected in canonical_recipe.items():
        require(
            resolved_config.get(name) == expected,
            f"official resolved {name} recipe differs from {canonical_name}",
        )
    require(
        resolved_config.get("source_tree_sha256") == manifest.get("source_tree_sha256"),
        "resolved configuration source tree differs from the final checkpoint",
    )
    artifact_trees = resolved_config.get("artifact_trees")
    require(isinstance(artifact_trees, Mapping), "official resolved configuration has no artifact tree identities")
    require(
        artifact_trees.get("dataset_tree_sha256") == registered["dataset_tree_sha256"]
        and artifact_trees.get("dataset_content_inventory_sha256") == registered["dataset_content_inventory_sha256"],
        "resolved configuration dataset identity differs from pre-registration",
    )
    run_config = resolved_config.get("run")
    require(isinstance(run_config, Mapping), "official resolved configuration has no run table")
    _require_exact_keys(run_config, {"max_cached_files", "seed", "task"}, "official resolved run table")
    require(run_config["seed"] == selected_cell["train_seed"], "official resolved run seed changed")
    require(run_config["task"] is None, "official resolved run must use the full 40-task multitask dataset")
    require(run_config["max_cached_files"] == 128, "official resolved max-cached-files setting changed")

    trainer_state = manifest.get("trainer_state")
    require(isinstance(trainer_state, dict), "final checkpoint has no trainer state")
    require(
        trainer_state.get("next_update") == registered["update"] == FINAL_CHECKPOINT_UPDATE,
        "final checkpoint trainer update mismatch",
    )
    require(
        trainer_state
        == {
            "examples_seen": OFFICIAL_TRAINING_EXAMPLES,
            "next_update": FINAL_CHECKPOINT_UPDATE,
            "schema": "duo-vla-trainer-state-v1",
        },
        "final checkpoint trainer state is not the complete canonical 30,000-update run",
    )
    last_metrics = manifest.get("last_metrics")
    require(
        isinstance(last_metrics, dict) and last_metrics.get("update") == registered["update"],
        "final checkpoint metric update mismatch",
    )
    require(
        last_metrics.get("examples_seen") == OFFICIAL_TRAINING_EXAMPLES,
        "final checkpoint metric example count mismatch",
    )
    require(last_metrics.get("objective") == objective, "final checkpoint metric objective mismatch")
    for name in ("gradient_norm", "train_loss", "update_seconds", "validation_loss"):
        value = last_metrics.get(name)
        require(
            isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)),
            f"final checkpoint metric {name} must be finite",
        )
    manifest_contract = manifest.get("policy_contract")
    require(isinstance(manifest_contract, dict), "final checkpoint has no policy contract")
    for name in ("inference_seed_behavior", "objective", "sampler"):
        require(
            manifest_contract.get(name) == selected_cell[name],
            f"final checkpoint {name} differs from pre-registration",
        )
    expected_checkpoint_nfe = 1 if selected_cell["objective"] == "direct_regression" else OFFICIAL_FLOW_CHECKPOINT_NFE
    require(
        manifest_contract.get("nfe") == expected_checkpoint_nfe,
        f"final checkpoint NFE must equal {expected_checkpoint_nfe} for {selected_cell['objective']}",
    )
    return {
        "manifest_sha256": manifest_sha256,
        "path": str(checkpoint_dir),
        "run_journal_latest": True,
        "run_root": str(run_root),
        "source_tree_sha256": manifest["source_tree_sha256"],
        "train_venv": checked_train_venv,
        "update": record.update,
    }


def validate_official_policy_health(
    health: dict[str, Any],
    *,
    selected_cell: Mapping[str, Any],
    execution_horizon: int,
    expert_replay_qualification: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind live policy/runtime/checkpoint identity to one frozen official cell."""

    validate_policy_health(health, allow_fake_policy=False)
    require(health["mode"] == "real", "official-score refuses fake policy health")
    require(execution_horizon == selected_cell["execution_horizon"], "official execution horizon changed")
    require(health["train_seed"] == selected_cell["train_seed"], "live train seed differs from pre-registration")
    checkpoint = health["checkpoint"]
    require(
        hmac.compare_digest(checkpoint["manifest_sha256"], selected_cell["checkpoint"]["manifest_sha256"]),
        "live checkpoint manifest differs from pre-registration",
    )
    require(
        hmac.compare_digest(checkpoint["source_tree_sha256"], selected_cell["checkpoint"]["source_tree_sha256"]),
        "live checkpoint source tree differs from pre-registration",
    )
    require(
        hmac.compare_digest(checkpoint["policy_contract_sha256"], selected_cell["policy_contract_sha256"]),
        "live checkpoint policy contract differs from pre-registration",
    )
    require(
        hmac.compare_digest(health["serving_runtime_sha256"], selected_cell["serving_runtime_sha256"]),
        "live serving runtime differs from pre-registration",
    )
    require(
        hmac.compare_digest(health["latency_runtime_sha256"], selected_cell["latency_runtime_sha256"]),
        "live latency runtime differs from pre-registration",
    )
    require(
        health["execution_geometry"] == selected_cell["execution_geometry"] == LIBERO_EXECUTION_GEOMETRY,
        "live execution geometry differs from pre-registration",
    )
    selected_policy = {name: health[name] for name in ("inference_seed_behavior", "nfe", "objective", "sampler")}
    registered_policy = {
        name: selected_cell[name] for name in ("inference_seed_behavior", "nfe", "objective", "sampler")
    }
    require(selected_policy == registered_policy, "live serving policy differs from pre-registration")
    require(
        hmac.compare_digest(canonical_sha256(selected_policy), selected_cell["serving_policy_sha256"]),
        "live serving-policy SHA-256 differs from pre-registration",
    )
    require(
        isinstance(expert_replay_qualification, Mapping),
        "official policy validation requires expert replay qualification identity",
    )
    return _authenticate_final_checkpoint(
        health,
        selected_cell,
        expected_train_venv_identity=expert_replay_qualification.get("train_venv_identity"),
    )


def run_episode(
    environment: Any,
    *,
    initial_state: np.ndarray,
    client: PolicyClient,
    train_seed: int,
    evaluation_seed: int,
    suite: str,
    task_id: int,
    task_name: str,
    instruction: str,
    init_state_id: int,
    execution_horizon: int,
    policy_budget: int,
    reset_source: str = "official",
    reset_state_sha256: str | None = None,
    environment_seed: int = ENVIRONMENT_SEED,
    expected_settled_state_sha256: str | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    require(reset_source in {"official", "clean-dev"}, "unknown LIBERO reset source")
    if reset_source == "official":
        require(reset_state_sha256 is None, "official reset must not carry a state hash")
        require(expected_settled_state_sha256 is None, "official reset must not carry a settled-state hash")
    else:
        require(reset_state_sha256 is not None, "clean-dev reset requires a state hash")
        require(expected_settled_state_sha256 is not None, "clean-dev reset requires a settled-state hash")
    environment.seed(environment_seed)
    environment.reset()
    observation = environment.set_init_state(initial_state)
    action_queue: deque[np.ndarray] = deque()
    policy_latencies: list[float] = []
    server_latencies: list[float] = []
    normalized_clip_fractions: list[float] = []
    policy_steps = 0
    policy_calls = 0
    settle_executed = 0
    environment_clip_count = 0
    success = False
    simulator_done = False

    while settle_executed < SETTLE_STEPS and not success:
        observation, _, done, _ = environment.step(OPEN_GRIPPER_NOOP.copy())
        simulator_done = simulator_done or bool(done)
        settle_executed += 1
        success = bool(environment.check_success())

    if expected_settled_state_sha256 is not None:
        observed_settled_hash = state_sha256(environment.get_sim_state())
        require(
            observed_settled_hash == expected_settled_state_sha256,
            "clean-dev mandatory settle trajectory differs from the authenticated bank",
        )

    while policy_steps < policy_budget and not success:
        if not action_queue:
            agentview = rotate_eval_rgb_180(np.asarray(observation[CAMERA_KEYS[0]]))
            wrist = rotate_eval_rgb_180(np.asarray(observation[CAMERA_KEYS[1]]))
            state = libero_state(observation)
            request_started = time.perf_counter()
            actions, response = client.predict(
                suite=suite,
                task_id=task_id,
                reset_source=reset_source,
                reset_id=init_state_id,
                reset_state_sha256=reset_state_sha256,
                replan_id=policy_calls,
                execution_horizon=execution_horizon,
                train_seed=train_seed,
                evaluation_seed=evaluation_seed,
                instruction=instruction,
                agentview_rgb=agentview,
                wrist_rgb=wrist,
                state=state,
            )
            client_seconds = time.perf_counter() - request_started
            require(response["evaluation_seed"] == evaluation_seed, "policy response evaluation seed drifted")
            require(response["reset_source"] == reset_source, "policy response reset source drifted")
            require(response["reset_id"] == init_state_id, "policy response reset id drifted")
            require(
                response["reset_state_sha256"] == reset_state_sha256,
                "policy response reset state hash drifted",
            )
            server_seconds = float(response["policy_seconds"])
            require(
                math.isfinite(client_seconds)
                and math.isfinite(server_seconds)
                and 0.0 <= server_seconds <= client_seconds,
                "client round-trip latency must be finite and at least the server latency",
            )
            policy_latencies.append(client_seconds)
            server_latencies.append(server_seconds)
            normalized_clip_fractions.append(float(response["normalized_clip_fraction"]))
            action_queue.extend(action.copy() for action in actions[:execution_horizon])
            policy_calls += 1

        raw_action = action_queue.popleft()
        action, clip_count = libero_env_action(raw_action)
        environment_clip_count += clip_count
        observation, _, done, _ = environment.step(action)
        simulator_done = simulator_done or bool(done)
        policy_steps += 1
        success = bool(environment.check_success())
        if success:
            action_queue.clear()

    successful_steps = policy_steps if success else None
    return {
        "action_clipped_channels": environment_clip_count,
        "action_clip_fraction": environment_clip_count / (policy_steps * 6) if policy_steps else 0.0,
        "action_continuous_channels": policy_steps * 6,
        "elapsed_seconds": time.perf_counter() - started,
        "environment_seed": environment_seed,
        "evaluation_seed": evaluation_seed,
        "execution_horizon": execution_horizon,
        "init_state_id": init_state_id if reset_source == "official" else None,
        "normalized_action_clip_fraction": (
            statistics.fmean(normalized_clip_fractions) if normalized_clip_fractions else 0.0
        ),
        "policy_budget": policy_budget,
        "policy_calls": policy_calls,
        "policy_latency_p50_seconds": percentile(policy_latencies, 50),
        "policy_latency_p95_seconds": percentile(policy_latencies, 95),
        "policy_latency_seconds": policy_latencies,
        "policy_steps": policy_steps,
        "reset_id": init_state_id,
        "reset_source": reset_source,
        "reset_state_sha256": reset_state_sha256,
        "server_latency_p50_seconds": percentile(server_latencies, 50),
        "server_latency_p95_seconds": percentile(server_latencies, 95),
        "server_latency_seconds": server_latencies,
        "settle_steps": settle_executed,
        "simulator_done": bool(simulator_done),
        "steps_to_success": successful_steps,
        "success": success,
        "suite": suite,
        "task_id": task_id,
        "task_name": task_name,
    }


def summarize_episodes(episodes: list[dict[str, Any]], *, execution_horizon: int) -> dict[str, Any]:
    require(episodes, "cannot summarize an empty rollout")
    tasks: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for episode in episodes:
        tasks.setdefault((episode["suite"], episode["task_id"]), []).append(episode)
    task_metrics: list[dict[str, Any]] = []
    for (suite, task_id), values in sorted(tasks.items()):
        successes = sum(bool(value["success"]) for value in values)
        task_metrics.append(
            {
                "episodes": len(values),
                "success_rate": successes / len(values),
                "successes": successes,
                "suite": suite,
                "task_id": task_id,
                "task_name": values[0]["task_name"],
                "wilson95": wilson95(successes, len(values)),
            }
        )

    suites: list[dict[str, Any]] = []
    for suite in SUITES:
        suite_tasks = [value for value in task_metrics if value["suite"] == suite]
        if not suite_tasks:
            continue
        suite_episodes = [value for value in episodes if value["suite"] == suite]
        successes = sum(bool(value["success"]) for value in suite_episodes)
        suites.append(
            {
                "episodes": len(suite_episodes),
                "pooled_success_rate": successes / len(suite_episodes),
                "pooled_wilson95": wilson95(successes, len(suite_episodes)),
                "suite": suite,
                "suite_task_macro_success": statistics.fmean(value["success_rate"] for value in suite_tasks),
                "successes": successes,
                "tasks": len(suite_tasks),
            }
        )

    successful_steps = [float(value["steps_to_success"]) for value in episodes if value["steps_to_success"] is not None]
    latency_values = [float(latency) for value in episodes for latency in value["policy_latency_seconds"]]
    server_latency_values = [float(latency) for value in episodes for latency in value["server_latency_seconds"]]
    episode_elapsed_seconds = sum(float(value["elapsed_seconds"]) for value in episodes)
    require(
        math.isfinite(episode_elapsed_seconds) and episode_elapsed_seconds > 0.0,
        "episode elapsed time must be finite and positive",
    )
    clipped_channels = sum(int(value["action_clipped_channels"]) for value in episodes)
    continuous_channels = sum(int(value["action_continuous_channels"]) for value in episodes)
    policy_calls = sum(int(value["policy_calls"]) for value in episodes)
    complete_40_task_macro = len(task_metrics) == 40 and {
        (item["suite"], item["task_id"]) for item in task_metrics
    } == {(suite, task_id) for suite in SUITES for task_id in range(10)}
    successes = sum(bool(value["success"]) for value in episodes)
    return {
        "action_clip_fraction": clipped_channels / continuous_channels if continuous_channels else 0.0,
        "complete_40_task_macro": complete_40_task_macro,
        "episode_elapsed_seconds": episode_elapsed_seconds,
        "episode_throughput_per_hour": len(episodes) * 3600.0 / episode_elapsed_seconds,
        "episodes": len(episodes),
        "execution_horizon": execution_horizon,
        "normalized_action_clip_fraction": (
            sum(float(value["normalized_action_clip_fraction"]) * int(value["policy_calls"]) for value in episodes)
            / policy_calls
            if policy_calls
            else 0.0
        ),
        "overall_40_task_macro_success": (
            statistics.fmean(value["success_rate"] for value in task_metrics) if complete_40_task_macro else None
        ),
        "overall_pooled_success_rate": successes / len(episodes),
        "overall_pooled_wilson95": wilson95(successes, len(episodes)),
        "policy_calls": policy_calls,
        "policy_call_throughput_per_second": policy_calls / episode_elapsed_seconds,
        "policy_latency_p50_seconds": percentile(latency_values, 50),
        "policy_latency_p95_seconds": percentile(latency_values, 95),
        "server_latency_p50_seconds": percentile(server_latency_values, 50),
        "server_latency_p95_seconds": percentile(server_latency_values, 95),
        "steps_to_success_mean": statistics.fmean(successful_steps) if successful_steps else None,
        "steps_to_success_p50": percentile(successful_steps, 50),
        "steps_to_success_p95": percentile(successful_steps, 95),
        "suites": suites,
        "task_metrics": task_metrics,
    }


def bind_official_summary(
    summary: dict[str, Any],
    *,
    contamination: Mapping[str, Any],
    episode_matrix_sha256: str,
) -> dict[str, Any]:
    require(summary["complete_40_task_macro"] is True, "official summary does not contain all 40 tasks")
    require(summary["episodes"] == OFFICIAL_PRIMARY_EPISODES, "official summary denominator is not 1,999")
    require(summary["policy_calls"] > 0, "official latency report has no measured episode policy calls")
    task_counts = {(item["suite"], item["task_id"]): item["episodes"] for item in summary["task_metrics"]}
    expected_counts = {(suite, task_id): OFFICIAL_RESETS_PER_TASK for suite in SUITES for task_id in range(10)}
    expected_counts[("libero_goal", 7)] = OFFICIAL_RESETS_PER_TASK - 1
    require(task_counts == expected_counts, "official task denominators do not implement the contamination exclusion")
    result = dict(summary)
    result["reporting"] = {
        "contamination_ledger_sha256": contamination["ledger_sha256"],
        "denominator": OFFICIAL_PRIMARY_EPISODES,
        "episode_matrix_sha256": episode_matrix_sha256,
        "excluded_episode_count": 1,
        "excluded_episodes": contamination["excluded_episodes"],
        "full_official_episode_count": OFFICIAL_FULL_EPISODES,
        "label": contamination["report_label"],
        "non_blind_full_set_reported": False,
        "policy_warmup_calls": OFFICIAL_POLICY_WARMUP_CALLS,
        "policy_warmup_included_in_latency": False,
        "latency_scope": "episode_policy_calls_only",
    }
    result["schema"] = OFFICIAL_SUMMARY_SCHEMA
    return result


def _construct_environment(task: Any) -> Any:
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    bddl_file = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    return OffScreenRenderEnv(
        bddl_file_name=str(bddl_file),
        camera_heights=256,
        camera_widths=256,
        camera_names=list(CAMERA_NAMES),
        control_freq=20,
        render_gpu_device_id=int(os.environ.get("MUJOCO_EGL_DEVICE_ID", "0")),
    )


def _development_task_resets(
    bank: DevelopmentResetBank,
    *,
    suite_name: str,
    task_id: int,
    task: Any,
    official_states: np.ndarray,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    from libero.libero import get_libero_path

    identity = (suite_name, task_id)
    require(identity in bank.task_records, f"development reset bank has no task {suite_name}:{task_id}")
    record = bank.task_records[identity]
    require(record["task_name"] == task.name, "development reset task name changed")
    require(record["instruction"] == task.language, "development reset task instruction changed")
    bddl = record["bddl"]
    require(bddl.get("file") == task.bddl_file, "development reset BDDL filename changed")
    require(bddl.get("problem_folder") == task.problem_folder, "development reset BDDL folder changed")
    bddl_path = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    require(sha256_file(bddl_path) == bddl.get("sha256"), "development reset BDDL SHA-256 changed")
    state_size, official_hashes = canonical_official_state_hashes(official_states)
    official = record["official_states"]
    require(official["state_size"] == state_size, "development reset official-state size changed")
    require(official["sha256"] == list(official_hashes), "development reset official-state hashes changed")
    require(
        official["root_sha256"] == sequence_root_sha256(official_hashes),
        "development reset official-state root changed",
    )
    states = bank.states_by_task[identity]
    entries = record["entries"]
    require(len(states) == len(entries) == bank.manifest["states_per_task"], "development reset rows changed")
    return states, entries


def run_rollouts(
    *,
    client: PolicyClient,
    train_seed: int,
    evaluation_seed: int,
    suites: tuple[str, ...],
    task_ids: tuple[int, ...],
    init_state_ids: tuple[int, ...],
    execution_horizon: int,
    episode_sink: Any,
    reset_source: str = "official",
    development_bank: DevelopmentResetBank | None = None,
    official_episode_identities: frozenset[tuple[str, int, int]] | None = None,
) -> list[dict[str, Any]]:
    from libero.libero import benchmark

    benchmark_map = benchmark.get_benchmark_dict()
    if official_episode_identities is not None:
        require(reset_source == "official", "an official episode matrix cannot select development resets")
    episodes: list[dict[str, Any]] = []
    for suite_name in suites:
        suite = benchmark_map[suite_name]()
        require(suite.n_tasks == 10, f"{suite_name} task count changed")
        for task_id in task_ids:
            task = suite.get_task(task_id)
            official_states = np.asarray(suite.get_task_init_states(task_id))
            require(len(official_states) == 50, f"{suite_name} task {task_id} fixed-state count changed")
            if reset_source == "official":
                require(development_bank is None, "official rollouts must not use a development bank")
                states = official_states
                entries: list[dict[str, Any] | None] = [None] * len(states)
            else:
                require(development_bank is not None, "clean-dev rollouts require a development bank")
                states, clean_entries = _development_task_resets(
                    development_bank,
                    suite_name=suite_name,
                    task_id=task_id,
                    task=task,
                    official_states=official_states,
                )
                entries = list(clean_entries)
            environment = _construct_environment(task)
            try:
                for init_state_id in init_state_ids:
                    require(init_state_id < len(states), f"reset id {init_state_id} is outside the selected bank")
                    if (
                        official_episode_identities is not None
                        and (suite_name, task_id, init_state_id) not in official_episode_identities
                    ):
                        continue
                    entry = entries[init_state_id]
                    episode = run_episode(
                        environment,
                        initial_state=states[init_state_id],
                        client=client,
                        train_seed=train_seed,
                        evaluation_seed=evaluation_seed,
                        suite=suite_name,
                        task_id=task_id,
                        task_name=task.name,
                        instruction=task.language,
                        init_state_id=init_state_id,
                        execution_horizon=execution_horizon,
                        policy_budget=POLICY_BUDGETS[suite_name],
                        reset_source=reset_source,
                        reset_state_sha256=None if entry is None else entry["state_sha256"],
                        environment_seed=ENVIRONMENT_SEED if entry is None else entry["sampler_seed"],
                        expected_settled_state_sha256=(None if entry is None else entry["settled_state_sha256"]),
                    )
                    episodes.append(episode)
                    episode_sink.write(json.dumps(episode, allow_nan=False, sort_keys=True) + "\n")
                    episode_sink.flush()
                    os.fsync(episode_sink.fileno())
                    print(json.dumps(episode, allow_nan=False, sort_keys=True), flush=True)
            finally:
                environment.close()
    if official_episode_identities is not None:
        observed = {(episode["suite"], episode["task_id"], episode["reset_id"]) for episode in episodes}
        require(
            observed == set(official_episode_identities),
            "official rollout did not consume the exact episode matrix",
        )
        require(
            len(episodes) == len(official_episode_identities),
            "official rollout emitted duplicate episode identities",
        )
    return episodes


def dry_run(
    client: PolicyClient,
    health: dict[str, Any],
    *,
    execution_horizon: int,
    evaluation_seed: int,
    replan_id: int = 0,
) -> dict[str, Any]:
    pixels = np.arange(math.prod(IMAGE_SHAPE), dtype=np.uint32).reshape(IMAGE_SHAPE)
    agentview = (pixels % 251).astype(np.uint8)
    wrist = ((pixels * 7 + 3) % 251).astype(np.uint8)
    actions, response = client.predict(
        suite="libero_spatial",
        task_id=0,
        reset_source="official",
        reset_id=0,
        reset_state_sha256=None,
        replan_id=replan_id,
        execution_horizon=execution_horizon,
        train_seed=health["train_seed"],
        evaluation_seed=evaluation_seed,
        instruction="pick up the black bowl between the plate and the ramekin and place it on the plate",
        agentview_rgb=agentview,
        wrist_rgb=wrist,
        state=np.zeros(8, dtype=np.float32),
    )
    return {
        "actions_sha256": hashlib.sha256(actions.tobytes()).hexdigest(),
        "evaluation_seed": response["evaluation_seed"],
        "execution_horizon": execution_horizon,
        "health": health,
        "inference_seed": response["inference_seed"],
        "inference_seed_behavior": response["inference_seed_behavior"],
        "nfe": response["nfe"],
        "objective": response["objective"],
        "policy_seconds": response["policy_seconds"],
        "replan_id": replan_id,
        "reset_id": response["reset_id"],
        "reset_source": response["reset_source"],
        "reset_state_sha256": response["reset_state_sha256"],
        "sampler": response["sampler"],
        "status": "ok",
    }


def run_policy_warmups(
    client: PolicyClient,
    health: dict[str, Any],
    *,
    count: int,
    execution_horizon: int,
    evaluation_seed: int,
    attempt_callback: Callable[[int, Mapping[str, Any]], None] | None = None,
    report_callback: Callable[[Sequence[Mapping[str, Any]]], None] | None = None,
) -> list[dict[str, Any]]:
    require(count > 0, "rollout mode requires at least one discarded policy warm-up call")
    warmup_horizons = (
        OFFICIAL_EXECUTION_HORIZONS
        if count == OFFICIAL_POLICY_WARMUP_CALLS
        else tuple(execution_horizon for _ in range(count))
    )
    reports: list[dict[str, Any]] = []
    for warmup_index, warmup_horizon in enumerate(warmup_horizons):
        intent = {
            "evaluation_seed": evaluation_seed,
            "execution_horizon": warmup_horizon,
            "replan_id": OFFICIAL_POLICY_WARMUP_REPLAN_ID,
            "reset_id": 0,
            "reset_source": "official",
            "warmup_index": warmup_index,
        }
        if attempt_callback is not None:
            attempt_callback(warmup_index, intent)
        report = dry_run(
            client,
            health,
            execution_horizon=warmup_horizon,
            evaluation_seed=evaluation_seed,
            replan_id=OFFICIAL_POLICY_WARMUP_REPLAN_ID,
        )
        report.pop("health")
        report["warmup_index"] = warmup_index
        reports.append(report)
        if report_callback is not None:
            report_callback(reports)
    deterministic = [warmup_k_independent_response(report) for report in reports]
    require(
        len({canonical_json_bytes(value) for value in deterministic}) == 1,
        "reserved-identity warm-up response was not K-independent",
    )
    return reports


def warmup_k_independent_response(report: Mapping[str, Any]) -> dict[str, Any]:
    """Return the deterministic warm-up identity, excluding K and timing."""

    return {
        name: value
        for name, value in report.items()
        if name not in {"execution_horizon", "policy_seconds", "warmup_index"}
    }


def _read_published_target_identity(path: Path, *, name: str) -> dict[str, Any]:
    require(hasattr(os, "O_NOFOLLOW"), "publication verification requires O_NOFOLLOW")
    flags = os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
    try:
        descriptor = os.open(str(path), flags)
    except OSError as exc:
        raise RuntimeError(f"cannot open published {name}: {path}") from exc
    digest = hashlib.sha256()
    size = 0
    try:
        before = os.fstat(descriptor)
        require(stat.S_ISREG(before.st_mode), f"published {name} is not a regular file")
        with os.fdopen(descriptor, "rb") as source:
            descriptor = -1
            while True:
                block = source.read(1024 * 1024)
                if not block:
                    break
                size += len(block)
                digest.update(block)
            after = os.fstat(source.fileno())
        stable = ("st_dev", "st_ino", "st_nlink", "st_size", "st_mtime_ns", "st_ctime_ns")
        require(
            all(getattr(before, field) == getattr(after, field) for field in stable) and size == after.st_size,
            f"published {name} changed while being verified",
        )
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    return {
        "bytes": size,
        "device": after.st_dev,
        "inode": after.st_ino,
        "links": after.st_nlink,
        "sha256": digest.hexdigest(),
    }


def _publication_identity(value: Mapping[str, Any]) -> dict[str, Any]:
    return {name: value[name] for name in ("bytes", "device", "inode", "sha256")}


def _verify_published_target(path: Path, expected: Mapping[str, Any], *, links: int, name: str) -> None:
    observed = _read_published_target_identity(path, name=name)
    require(
        _publication_identity(observed) == dict(expected) and observed["links"] == links,
        f"published {name} target identity, content, or link count changed",
    )


def _unlink_created_path(path: Path, expected: Mapping[str, Any] | None) -> bool:
    if expected is None:
        return False
    try:
        observed = os.stat(str(path), follow_symlinks=False)
    except OSError:
        return False
    if (
        not stat.S_ISREG(observed.st_mode)
        or observed.st_dev != expected.get("device")
        or observed.st_ino != expected.get("inode")
    ):
        return False
    try:
        path.unlink()
    except OSError:
        return False
    return True


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(str(path), os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def publish_bytes_and_sha256_exclusive(
    path: Path,
    payload: bytes,
    *,
    commit_guard: Callable[[], None] | None = None,
) -> str:
    """Durably publish an immutable payload/sidecar pair without overwriting."""

    require(isinstance(payload, bytes), "exclusive publication payload must be bytes")
    parent = path.parent
    require(parent.is_dir(), "exclusive publication parent must already exist")
    companion = path.with_suffix(path.suffix + ".sha256")
    digest = hashlib.sha256(payload).hexdigest()
    companion_payload = f"{digest}  {path.name}\n".encode("ascii")
    nonce = f"{os.getpid()}-{time.time_ns()}-{secrets.token_hex(8)}"
    temporary_payload = parent / f".{path.name}.tmp-{nonce}"
    temporary_companion = parent / f".{companion.name}.tmp-{nonce}"
    payload_identity: dict[str, Any] | None = None
    companion_identity: dict[str, Any] | None = None
    final_payload = False
    final_companion = False

    def write_temporary(candidate: Path, content: bytes, name: str) -> dict[str, Any]:
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW
        descriptor = os.open(str(candidate), flags, 0o600)
        with os.fdopen(descriptor, "wb") as sink:
            sink.write(content)
            sink.flush()
            os.fsync(sink.fileno())
        observed = _read_published_target_identity(candidate, name=name)
        require(
            observed["bytes"] == len(content)
            and observed["sha256"] == hashlib.sha256(content).hexdigest()
            and observed["links"] == 1,
            f"temporary published {name} identity is invalid",
        )
        return _publication_identity(observed)

    try:
        payload_identity = write_temporary(temporary_payload, payload, "payload")
        companion_identity = write_temporary(temporary_companion, companion_payload, "SHA-256 companion")
        if commit_guard is not None:
            commit_guard()
        _verify_published_target(temporary_payload, payload_identity, links=1, name="payload temporary")
        _verify_published_target(temporary_companion, companion_identity, links=1, name="SHA-256 temporary")
        os.link(str(temporary_companion), str(companion), follow_symlinks=False)
        final_companion = True
        if commit_guard is not None:
            commit_guard()
        _verify_published_target(temporary_payload, payload_identity, links=1, name="payload temporary")
        _verify_published_target(companion, companion_identity, links=2, name="SHA-256 companion")
        os.link(str(temporary_payload), str(path), follow_symlinks=False)
        final_payload = True
        _fsync_directory(parent)
        if commit_guard is not None:
            commit_guard()
        _verify_published_target(path, payload_identity, links=2, name="payload")
        _verify_published_target(companion, companion_identity, links=2, name="SHA-256 companion")
        require(_unlink_created_path(temporary_payload, payload_identity), "payload temporary changed before cleanup")
        require(
            _unlink_created_path(temporary_companion, companion_identity),
            "SHA-256 temporary changed before cleanup",
        )
        _fsync_directory(parent)
        _verify_published_target(path, payload_identity, links=1, name="payload")
        _verify_published_target(companion, companion_identity, links=1, name="SHA-256 companion")
        return digest
    except BaseException:
        if final_payload:
            _unlink_created_path(path, payload_identity)
        if final_companion:
            _unlink_created_path(companion, companion_identity)
        _unlink_created_path(temporary_payload, payload_identity)
        _unlink_created_path(temporary_companion, companion_identity)
        with contextlib.suppress(OSError):
            _fsync_directory(parent)
        raise
    finally:
        _unlink_created_path(temporary_payload, payload_identity)
        _unlink_created_path(temporary_companion, companion_identity)


def _capture_exclusive_pair(
    path: Path,
    *,
    expected_sha256: str,
    name: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    _require_sha256(expected_sha256, f"{name} SHA-256")
    payload = _read_published_target_identity(path, name=name)
    require(payload["links"] == 1 and payload["sha256"] == expected_sha256, f"published {name} is invalid")
    companion_path = path.with_suffix(path.suffix + ".sha256")
    companion = _read_published_target_identity(companion_path, name=f"{name} SHA-256 companion")
    expected = f"{expected_sha256}  {path.name}\n".encode("ascii")
    require(
        companion["links"] == 1
        and companion["bytes"] == len(expected)
        and companion["sha256"] == hashlib.sha256(expected).hexdigest(),
        f"published {name} SHA-256 companion is invalid",
    )
    return _publication_identity(payload), _publication_identity(companion)


def write_json_atomic(path: Path, value: Mapping[str, Any]) -> dict[str, Any]:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}-{time.time_ns()}-{secrets.token_hex(4)}")
    payload = (json.dumps(value, allow_nan=False, ensure_ascii=True, indent=2, sort_keys=True) + "\n").encode("ascii")
    identity: dict[str, Any] | None = None
    try:
        descriptor = os.open(str(temporary), os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        with os.fdopen(descriptor, "wb") as sink:
            sink.write(payload)
            sink.flush()
            os.fsync(sink.fileno())
        observed = _read_published_target_identity(temporary, name="atomic JSON temporary")
        require(observed["links"] == 1 and observed["bytes"] == len(payload), "atomic JSON temporary is invalid")
        identity = _publication_identity(observed)
        os.replace(str(temporary), str(path))
        _fsync_directory(path.parent)
        _verify_published_target(path, identity, links=1, name="atomic JSON")
        return identity
    finally:
        _unlink_created_path(temporary, identity)


class EvaluationJournal:
    """Durable lifecycle journal for one pre-claimed official LIBERO cell."""

    def __init__(
        self,
        output_dir: Path,
        run_manifest: Mapping[str, Any],
        *,
        claim_path: Path,
        claim_json_sha256: str,
        commit_guard: Callable[[], None] | None = None,
    ) -> None:
        self.output_dir = output_dir
        self.run_path = output_dir / "run.json"
        self.episodes_path = output_dir / "episodes.jsonl"
        self.summary_path = output_dir / "summary.json"
        self.completion_path = output_dir / "completion.json"
        self.run_manifest = dict(run_manifest)
        self.claim_path = claim_path
        self.claim_json_sha256 = claim_json_sha256
        self._commit_guard = commit_guard
        self.episode_sink: Any | None = None
        self._output_identity: dict[str, Any] | None = None
        self._claim_targets = _capture_exclusive_pair(
            claim_path,
            expected_sha256=claim_json_sha256,
            name="official claim JSON",
        )

    def _guard(self, targets: Sequence[tuple[Path, Mapping[str, Any], str]] = ()) -> None:
        all_targets = (
            (self.claim_path, self._claim_targets[0], "official claim JSON"),
            (
                self.claim_path.with_suffix(self.claim_path.suffix + ".sha256"),
                self._claim_targets[1],
                "official claim SHA-256 companion",
            ),
            *targets,
        )
        for path, identity, name in all_targets:
            _verify_published_target(path, identity, links=1, name=name)
        if self._output_identity is not None:
            require(
                _real_directory_identity(self.output_dir, name="official run directory") == self._output_identity,
                "official run directory identity changed",
            )
        if self._commit_guard is not None:
            self._commit_guard()
        if self._output_identity is not None:
            require(
                _real_directory_identity(self.output_dir, name="official run directory") == self._output_identity,
                "official run directory identity changed",
            )
        for path, identity, name in all_targets:
            _verify_published_target(path, identity, links=1, name=name)

    def start(self) -> None:
        claimed = False
        try:
            self._guard()
            self.output_dir.mkdir(parents=False, exist_ok=False, mode=0o700)
            claimed = True
            self._output_identity = _real_directory_identity(self.output_dir, name="official run directory")
            self._guard()
            self.run_manifest["status"] = "running"
            write_json_atomic(self.run_path, self.run_manifest)
            descriptor = os.open(
                str(self.episodes_path),
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                0o600,
            )
            self.episode_sink = os.fdopen(descriptor, "w", encoding="utf-8")
            self.episode_sink.flush()
            os.fsync(self.episode_sink.fileno())
            _fsync_directory(self.output_dir)
            self._guard()
        except BaseException as exc:
            if claimed:
                try:
                    self.fail(exc)
                except BaseException as fail_exc:
                    raise RuntimeError(
                        "evaluation journal start failed and failure journaling also failed: "
                        f"{type(fail_exc).__name__}: {fail_exc}"
                    ) from exc
            raise

    def update_running(self, values: Mapping[str, Any]) -> None:
        require(self.run_manifest.get("status") == "running", "evaluation journal is not running")
        self.run_manifest.update(dict(values))
        self._guard()
        write_json_atomic(self.run_path, self.run_manifest)
        self._guard()

    def _close_and_identify_episodes(self) -> dict[str, Any]:
        if self.episode_sink is not None:
            sink = self.episode_sink
            try:
                sink.flush()
                os.fsync(sink.fileno())
            finally:
                sink.close()
                self.episode_sink = None
        observed = _read_published_target_identity(self.episodes_path, name="episodes JSONL")
        require(observed["links"] == 1, "episodes JSONL is linked")
        return _publication_identity(observed)

    def complete(self, summary: Mapping[str, Any], *, episode_records: int) -> None:
        try:
            episodes_identity = self._close_and_identify_episodes()
            require(episode_records == OFFICIAL_PRIMARY_EPISODES, "official score episode record count is incomplete")
            episode_target = (self.episodes_path, episodes_identity, "episodes JSONL")
            self._guard((episode_target,))
            summary_identity = write_json_atomic(self.summary_path, summary)
            summary_target = (self.summary_path, summary_identity, "summary JSON")
            self._guard((episode_target, summary_target))
            self.run_manifest.update(
                {
                    "episode_records": episode_records,
                    "episodes_jsonl_sha256": episodes_identity["sha256"],
                    "finished_utc": datetime.now(UTC).isoformat(),
                    "status": "complete",
                    "summary_json_sha256": summary_identity["sha256"],
                }
            )
            run_identity = write_json_atomic(self.run_path, self.run_manifest)
            run_target = (self.run_path, run_identity, "complete run JSON")
            self._guard((episode_target, summary_target, run_target))
            completion = {
                "cell_id": self.run_manifest["cell"]["cell_id"],
                "claim_json_sha256": self.claim_json_sha256,
                "episode_records": episode_records,
                "episodes_jsonl_sha256": episodes_identity["sha256"],
                "preregistration_sha256": self.run_manifest["preregistration_sha256"],
                "run_json_sha256": run_identity["sha256"],
                "schema": COMPLETION_SCHEMA,
                "summary_json_sha256": summary_identity["sha256"],
            }
            completion_text = json.dumps(
                completion,
                allow_nan=False,
                ensure_ascii=True,
                indent=2,
                sort_keys=True,
            )
            payload = (completion_text + "\n").encode("ascii")

            def completion_guard() -> None:
                self._guard((episode_target, summary_target, run_target))

            publish_bytes_and_sha256_exclusive(self.completion_path, payload, commit_guard=completion_guard)
        except BaseException as exc:
            self.fail(exc)
            raise

    def fail(self, error: BaseException) -> None:
        partial_sha256: str | None = None
        try:
            partial_sha256 = self._close_and_identify_episodes()["sha256"]
        except BaseException:
            if self.episode_sink is not None:
                with contextlib.suppress(BaseException):
                    self.episode_sink.close()
                self.episode_sink = None
        self.run_manifest.update(
            {
                "error": {"message": str(error), "type": type(error).__name__},
                "failed_utc": datetime.now(UTC).isoformat(),
                "partial_episodes_jsonl_sha256": partial_sha256,
                "status": "failed",
            }
        )
        write_json_atomic(self.run_path, self.run_manifest)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("development", "official-score"), default="development")
    parser.add_argument(
        "--socket",
        type=Path,
        default=Path(os.environ.get("DUO_VLA_CACHE_ROOT", "/root/.cache/duo-vla")) / "run/libero-policy.sock",
    )
    parser.add_argument("--suite", choices=(*SUITES, "all"), default="libero_spatial")
    parser.add_argument("--task-ids", default="0", help="sorted task selection such as 0,2-4 or all")
    parser.add_argument("--init-state-ids", default="0", help="selected reset indices such as 0,2-9 or all")
    parser.add_argument(
        "--evaluation-seed", type=int, required=True, help="policy-noise seed shared across comparisons"
    )
    parser.add_argument(
        "--reset-source",
        choices=("official", "clean-dev"),
        default="official",
        help="published fixed states or an authenticated non-official development bank",
    )
    parser.add_argument("--dev-state-bank", type=Path, help="required with --reset-source clean-dev")
    parser.add_argument("--execution-horizon", type=int, choices=(1, 4), required=True)
    parser.add_argument("--policy-timeout-seconds", type=float, default=300.0)
    parser.add_argument(
        "--policy-warmup-calls",
        type=int,
        default=2,
        help="synthetic policy calls discarded before rollout latency measurement",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--dry-run", action="store_true", help="one synthetic IPC call; do not construct MuJoCo")
    parser.add_argument(
        "--allow-fake-policy", action="store_true", help="explicitly allow test-only fake server health"
    )
    parser.add_argument("--preregistration-manifest", type=Path)
    parser.add_argument("--preregistration-sha256")
    parser.add_argument("--cell-id")
    parser.add_argument(
        "--final-freeze-token",
        help="explicit string whose SHA-256 must match pre-registration; the token is never written",
    )
    return parser.parse_args(argv)


def validate_mode_arguments(args: argparse.Namespace) -> None:
    require(0 <= args.evaluation_seed < 2**63, "evaluation-seed must be in [0, 2^63)")
    require(
        math.isfinite(args.policy_timeout_seconds) and args.policy_timeout_seconds > 0,
        "policy timeout must be positive",
    )
    require(args.policy_warmup_calls >= 0, "policy-warmup-calls must be nonnegative")
    preregistration_values = (
        args.preregistration_manifest,
        args.preregistration_sha256,
        args.cell_id,
        args.final_freeze_token,
    )
    if args.mode == "development":
        require(
            all(value is None for value in preregistration_values),
            "development mode cannot receive a pre-registration",
        )
        if args.dry_run:
            require(args.output_dir is None, "development dry-run does not create rollout artifacts")
        else:
            require(args.reset_source == "clean-dev", "development rollout must use authenticated clean-dev resets")
            require(args.output_dir is not None, "development rollout requires --output-dir")
            require(args.policy_warmup_calls > 0, "development rollout requires at least one warm-up call")
        return
    require(not args.dry_run, "official-score cannot use --dry-run")
    require(not args.allow_fake_policy, "official-score cannot allow a fake policy")
    require(args.reset_source == "official", "official-score requires published official resets")
    require(args.dev_state_bank is None, "official-score cannot receive a development reset bank")
    require(args.suite == "all", "official-score requires --suite all")
    require(args.task_ids == "all", "official-score requires --task-ids all")
    require(args.init_state_ids == "all", "official-score requires --init-state-ids all")
    require(args.output_dir is not None, "official-score requires --output-dir")
    require(
        args.policy_warmup_calls == OFFICIAL_POLICY_WARMUP_CALLS,
        f"official-score requires exactly {OFFICIAL_POLICY_WARMUP_CALLS} policy warm-up calls",
    )
    require(args.preregistration_manifest is not None, "official-score requires --preregistration-manifest")
    _require_sha256(args.preregistration_sha256, "official-score --preregistration-sha256")
    require(isinstance(args.cell_id, str) and bool(args.cell_id), "official-score requires --cell-id")
    require(
        isinstance(args.final_freeze_token, str) and bool(args.final_freeze_token.strip()),
        "official-score requires an explicit --final-freeze-token",
    )


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    validate_mode_arguments(args)
    project_root = Path(__file__).resolve().parents[1]
    evaluator_environment = validate_evaluator_process_environment(project_root)
    require(platform.python_version().startswith("3.12."), "LIBERO evaluator requires Python 3.12")
    startup_sources = _IMPORT_EVALUATOR_SOURCE_IDENTITIES
    require_evaluator_sources_unchanged(startup_sources, project_root=project_root)

    if args.mode == "official-score":
        assert args.preregistration_manifest is not None
        assert args.preregistration_sha256 is not None
        assert args.cell_id is not None
        assert args.final_freeze_token is not None
        assert args.output_dir is not None
        contamination = load_contamination_contract(project_root)
        recorded_manifest, _ = _read_strict_json(
            args.preregistration_manifest.resolve(),
            name="LIBERO pre-registration manifest",
            expected_sha256=args.preregistration_sha256,
        )
        recorded_attestation_sha256 = recorded_manifest.get("simulator_attestation_sha256")
        _require_sha256(recorded_attestation_sha256, "recorded simulator attestation SHA-256")
        manifest, selected_cell, preregistration_sha256 = load_preregistration(
            args.preregistration_manifest.resolve(),
            cell_id=args.cell_id,
            execution_horizon=args.execution_horizon,
            evaluation_seed=args.evaluation_seed,
            final_freeze_token=args.final_freeze_token,
            preregistration_sha256=args.preregistration_sha256,
            simulator_attestation_sha256=recorded_attestation_sha256,
            contamination=contamination,
        )
        roots = validate_official_output_roots(manifest["official_output_roots"], require_live=True)
        output_claim = selected_cell["output_claim"]
        expected_output_dir = Path(output_claim["output_dir"])
        expected_claim_path = Path(output_claim["claim_path"])
        require(
            args.output_dir.is_absolute() and args.output_dir == expected_output_dir,
            "official-score --output-dir differs from the pre-registered canonical output directory",
        )
        official_episodes = official_episode_matrix(contamination)
        created_utc = datetime.now(UTC).isoformat()

        def output_commit_guard() -> None:
            require_evaluator_sources_unchanged(startup_sources, project_root=project_root)
            validate_official_output_roots(roots, require_live=True)

        claim_record = build_claim_record(
            output_claim,
            cell_id=selected_cell["cell_id"],
            preregistration_sha256=preregistration_sha256,
            final_freeze_token_sha256=manifest["final_freeze_token_sha256"],
            created_utc=created_utc,
        )
        claim_payload = (
            json.dumps(claim_record, allow_nan=False, ensure_ascii=True, indent=2, sort_keys=True) + "\n"
        ).encode("ascii")
        # The immutable external claim consumes this cell before any policy
        # connection, warm-up call, simulator construction, or scoring.
        claim_json_sha256 = publish_bytes_and_sha256_exclusive(
            expected_claim_path,
            claim_payload,
            commit_guard=output_commit_guard,
        )
        run_manifest = {
            "cell": selected_cell,
            "claim_json_sha256": claim_json_sha256,
            "contamination": contamination,
            "created_utc": created_utc,
            "episode_count": len(official_episodes),
            "episode_matrix_sha256": canonical_sha256(official_episodes),
            "evaluation_seed": args.evaluation_seed,
            "evaluator_environment": evaluator_environment,
            "execution_horizon": args.execution_horizon,
            "final_freeze_token_sha256": manifest["final_freeze_token_sha256"],
            "init_state_ids": list(range(OFFICIAL_RESETS_PER_TASK)),
            "mode": "official-score",
            "output_claim": output_claim,
            "policy_socket": str(args.socket),
            "policy_warmup": {
                "attempted_count": 0,
                "completed_count": 0,
                "current_request": None,
                "expected_count": args.policy_warmup_calls,
                "included_in_episode_latency": False,
                "reports": [],
                "status": "pending",
            },
            "preregistration_manifest": str(args.preregistration_manifest.resolve()),
            "preregistration_sha256": preregistration_sha256,
            "protocol": PROTOCOL,
            "reset_identity": {
                "bank": None,
                "id_field": "published_init_state_id",
                "source": "official",
                "state_sha256": None,
            },
            "reset_source": "official",
            "schema": OFFICIAL_RUN_SCHEMA,
            "suites": list(SUITES),
            "task_ids": list(range(10)),
            "validator_runtime_identity": manifest["expert_replay_qualification"]["validator_runtime_identity"],
        }
        journal = EvaluationJournal(
            expected_output_dir,
            run_manifest,
            claim_path=expected_claim_path,
            claim_json_sha256=claim_json_sha256,
            commit_guard=output_commit_guard,
        )
        journal.start()
        try:
            simulator_preflight = run_exact_simulator_preflight(project_root, construct_environment=True)
            simulator_attestation_sha256 = canonical_sha256(simulator_preflight)
            require(
                hmac.compare_digest(simulator_attestation_sha256, recorded_attestation_sha256),
                "current simulator attestation differs from pre-registration",
            )
            qualified_validator_runtime = validate_validator_runtime_against_attestation(
                manifest["expert_replay_qualification"]["validator_runtime_identity"],
                simulator_preflight,
            )
            live_eval_venv = validate_eval_venv_identity(
                simulator_preflight.get("eval_venv_identity"),
                name="current simulator eval-venv identity",
            )
            require(
                canonical_json_bytes(live_eval_venv)
                == canonical_json_bytes(qualified_validator_runtime["eval_venv_identity"]),
                "current evaluator venv differs from the expert replay validator runtime",
            )
            project_sources = simulator_preflight.get("project_sources")
            require(isinstance(project_sources, Mapping), "simulator attestation has no project sources")
            for name, attested_name in (("evaluate_libero.py", "evaluator"), ("libero_bridge.py", "bridge")):
                require(
                    project_sources.get(attested_name) == startup_sources[name]["sha256"],
                    f"simulator-attested {attested_name} source differs from evaluator startup",
                )
            journal.update_running(
                {
                    "simulator_attestation_sha256": simulator_attestation_sha256,
                    "simulator_preflight": simulator_preflight,
                }
            )
            partial_reports: list[dict[str, Any]] = []

            def record_warmup_attempt(index: int, intent: Mapping[str, Any]) -> None:
                journal.update_running(
                    {
                        "policy_warmup": {
                            "attempted_count": index + 1,
                            "completed_count": len(partial_reports),
                            "current_request": dict(intent),
                            "expected_count": args.policy_warmup_calls,
                            "included_in_episode_latency": False,
                            "reports": list(partial_reports),
                            "status": "running",
                        }
                    }
                )

            def record_warmup_report(reports: Sequence[Mapping[str, Any]]) -> None:
                partial_reports[:] = [dict(report) for report in reports]
                journal.update_running(
                    {
                        "policy_warmup": {
                            "attempted_count": len(reports),
                            "completed_count": len(reports),
                            "current_request": None,
                            "expected_count": args.policy_warmup_calls,
                            "included_in_episode_latency": False,
                            "reports": list(partial_reports),
                            "status": "running",
                        }
                    }
                )

            with PolicyClient(args.socket, timeout_seconds=args.policy_timeout_seconds) as client:
                health = client.health()
                checkpoint_identity = validate_official_policy_health(
                    health,
                    selected_cell=selected_cell,
                    execution_horizon=args.execution_horizon,
                    expert_replay_qualification=manifest["expert_replay_qualification"],
                )
                journal.update_running({"checkpoint": checkpoint_identity, "policy_health": health})
                warmup_reports = run_policy_warmups(
                    client,
                    health,
                    count=args.policy_warmup_calls,
                    execution_horizon=args.execution_horizon,
                    evaluation_seed=args.evaluation_seed,
                    attempt_callback=record_warmup_attempt,
                    report_callback=record_warmup_report,
                )
                policy_warmup = {
                    "count": args.policy_warmup_calls,
                    "included_in_episode_latency": False,
                    "k_independent_response_sha256": canonical_sha256(warmup_k_independent_response(warmup_reports[0])),
                    "replan_id": OFFICIAL_POLICY_WARMUP_REPLAN_ID,
                    "reports": warmup_reports,
                    "validated_before_scoring": True,
                }
                journal.update_running({"policy_warmup": policy_warmup})
                require(journal.episode_sink is not None, "official episode journal is unavailable")
                episodes = run_rollouts(
                    client=client,
                    train_seed=health["train_seed"],
                    evaluation_seed=args.evaluation_seed,
                    suites=SUITES,
                    task_ids=tuple(range(10)),
                    init_state_ids=tuple(range(OFFICIAL_RESETS_PER_TASK)),
                    execution_horizon=args.execution_horizon,
                    episode_sink=journal.episode_sink,
                    reset_source="official",
                    development_bank=None,
                    official_episode_identities=frozenset(
                        (episode["suite"], episode["task_id"], episode["reset_id"]) for episode in official_episodes
                    ),
                )
            observed_order = [
                {"reset_id": episode["reset_id"], "suite": episode["suite"], "task_id": episode["task_id"]}
                for episode in episodes
            ]
            require(observed_order == official_episodes, "official episode order differs from pre-registration")
            summary = bind_official_summary(
                summarize_episodes(episodes, execution_horizon=args.execution_horizon),
                contamination=contamination,
                episode_matrix_sha256=canonical_sha256(official_episodes),
            )
            journal.complete(summary, episode_records=len(episodes))
            print(json.dumps(summary, allow_nan=False, indent=2, sort_keys=True))
        except BaseException as exc:
            if journal.run_manifest.get("status") != "failed":
                journal.fail(exc)
            raise
        return

    simulator_preflight = run_exact_simulator_preflight(project_root, construct_environment=not args.dry_run)
    simulator_attestation_sha256 = canonical_sha256(simulator_preflight)
    if args.reset_source == "official":
        development_bank = None
        reset_count = OFFICIAL_RESETS_PER_TASK
    else:
        assert args.dev_state_bank is not None
        development_bank = load_development_reset_bank(args.dev_state_bank.resolve())
        reset_count = int(development_bank.manifest["states_per_task"])
    suites = SUITES if args.suite == "all" else (args.suite,)
    task_ids = parse_selection(args.task_ids, upper=10, name="task ids")
    init_state_ids = parse_selection(args.init_state_ids, upper=reset_count, name="reset ids")
    with PolicyClient(args.socket, timeout_seconds=args.policy_timeout_seconds) as client:
        health = client.health()
        validate_policy_health(health, allow_fake_policy=args.allow_fake_policy)
        if args.dry_run:
            print(
                json.dumps(
                    dry_run(
                        client,
                        health,
                        execution_horizon=args.execution_horizon,
                        evaluation_seed=args.evaluation_seed,
                    ),
                    indent=2,
                    sort_keys=True,
                )
            )
            return
        assert args.output_dir is not None
        policy_warmup = run_policy_warmups(
            client,
            health,
            count=args.policy_warmup_calls,
            execution_horizon=args.execution_horizon,
            evaluation_seed=args.evaluation_seed,
        )
        output_dir = args.output_dir.resolve()
        output_dir.mkdir(parents=True, exist_ok=False)
        bank_identity = {
            "base_seed": development_bank.manifest["base_seed"],
            "manifest_sha256": development_bank.manifest_sha256,
            "root_sha256": development_bank.manifest["root_sha256"],
            "schema": development_bank.manifest["schema"],
            "states_per_task": development_bank.manifest["states_per_task"],
        }
        run_manifest = {
            "created_utc": datetime.now(UTC).isoformat(),
            "evaluator_environment": evaluator_environment,
            "evaluation_seed": args.evaluation_seed,
            "execution_horizon": args.execution_horizon,
            "init_state_ids": list(init_state_ids),
            "mode": "development",
            "policy_health": health,
            "policy_socket": str(args.socket),
            "policy_warmup": {
                "count": args.policy_warmup_calls,
                "included_in_episode_latency": False,
                "reports": policy_warmup,
            },
            "protocol": PROTOCOL,
            "reset_identity": {
                "bank": bank_identity,
                "id_field": "clean_dev_reset_id",
                "source": "clean-dev",
                "state_sha256": "per-episode-manifest-entry",
            },
            "reset_source": "clean-dev",
            "simulator_attestation_sha256": simulator_attestation_sha256,
            "simulator_preflight": simulator_preflight,
            "suites": list(suites),
            "task_ids": list(task_ids),
        }
        (output_dir / "run.json").write_text(
            json.dumps(run_manifest, allow_nan=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        with (output_dir / "episodes.jsonl").open("x", encoding="utf-8") as episode_sink:
            episodes = run_rollouts(
                client=client,
                train_seed=health["train_seed"],
                evaluation_seed=args.evaluation_seed,
                suites=suites,
                task_ids=task_ids,
                init_state_ids=init_state_ids,
                execution_horizon=args.execution_horizon,
                episode_sink=episode_sink,
                reset_source="clean-dev",
                development_bank=development_bank,
            )
        summary = summarize_episodes(episodes, execution_horizon=args.execution_horizon)
        (output_dir / "summary.json").write_text(
            json.dumps(summary, allow_nan=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(summary, allow_nan=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
