#!/usr/bin/env python3
"""Fail-closed CPU comparison of two canonical CALVIN update-2 training runs.

The qualification compares a run that stopped after update one and resumed with
an independently-created uninterrupted run.  The only logical exclusions are
the two run UUIDs and wall-clock ``update_seconds``.  Checkpoint manifest
normalization is limited to fields whose bytes necessarily bind those excluded
values or the independently authenticated checkpoint lineage.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import io
import json
import math
import os
import platform
import stat
import struct
import sys
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

_BOOTSTRAP_CAPABILITY_GLOBAL = "__duo_vla_qualification_bootstrap_capability__"
try:
    _BOOTSTRAP_CAPABILITY_RAW = globals().pop(_BOOTSTRAP_CAPABILITY_GLOBAL)
except KeyError as exc:
    raise RuntimeError("use the closed isolated qualification launcher") from exc

# This snapshot is deliberately constructed before importing torch or any
# production duo_vla module.  The closed launcher additionally compiles the
# exact comparator bytes whose digest it passes in the environment, avoiding a
# self-attestation gap in which this file had already been imported before its
# first identity was recorded.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_LAUNCHER_PATH = _PROJECT_ROOT / "scripts/calvin/run_compare_training_reproducibility.sh"
_DIRECT_PRODUCTION_SOURCE_RELATIVE_PATHS = {
    "checkpointing": "src/duo_vla/checkpointing.py",
    "policy_contract": "src/duo_vla/policy_contract.py",
    "run_config": "src/duo_vla/run_config.py",
    "run_journal": "src/duo_vla/run_journal.py",
    "training": "src/duo_vla/training.py",
    "training_checkpoint": "src/duo_vla/training_checkpoint.py",
}
_TRAINING_SOURCE_EXPLICIT_RELATIVE_PATHS = (
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
_CALVIN_SOURCE_TREE_HASH_MAGIC = b"duo-vla-calvin-training-source-tree\x00v3\x00"
_LAUNCHER_BOOTSTRAP_MODE = "launcher-exact-comparator-bytes-isolated-v2"
_EXPECTED_INTERPRETER_FLAGS = {
    "dont_write_bytecode": 1,
    "ignore_environment": 1,
    "isolated": 1,
    "no_site": 1,
    "no_user_site": 1,
    "safe_path": True,
}
_FORBIDDEN_BOOTSTRAP_MODULES = ("_cuda_bindings_redirector", "_distutils_hack", "_virtualenv")


def _bootstrap_stable_regular_file_bytes(path: Path) -> bytes:
    before = os.stat(path, follow_symlinks=False)
    if not stat.S_ISREG(before.st_mode):
        raise RuntimeError(f"qualification source must be a real file: {path}")
    descriptor = os.open(path, os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW | os.O_CLOEXEC)
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode) or (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino):
            raise RuntimeError(f"qualification source changed while opening: {path}")
        blocks: list[bytes] = []
        while block := os.read(descriptor, 1024 * 1024):
            blocks.append(block)
        after_descriptor = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    after_path = os.stat(path, follow_symlinks=False)
    identity_fields = ("st_dev", "st_ino", "st_mode", "st_size", "st_mtime_ns", "st_ctime_ns", "st_nlink")
    if any(getattr(opened, field) != getattr(after_descriptor, field) for field in identity_fields) or any(
        getattr(opened, field) != getattr(after_path, field) for field in identity_fields
    ):
        raise RuntimeError(f"qualification source changed while reading: {path}")
    raw = b"".join(blocks)
    if len(raw) != opened.st_size:
        raise RuntimeError(f"qualification source size changed while reading: {path}")
    return raw


def _bootstrap_sha256_file(path: Path) -> str:
    return hashlib.sha256(_bootstrap_stable_regular_file_bytes(path)).hexdigest()


_BOOTSTRAP_DIRECTORY_OPEN_FLAGS = os.O_RDONLY | os.O_NONBLOCK | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
_BOOTSTRAP_FILE_OPEN_FLAGS = os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW | os.O_CLOEXEC
_BOOTSTRAP_IDENTITY_FIELDS = ("st_dev", "st_ino", "st_mode", "st_size", "st_mtime_ns", "st_ctime_ns", "st_nlink")


def _bootstrap_same_identity(left: os.stat_result, right: os.stat_result) -> bool:
    return all(getattr(left, field) == getattr(right, field) for field in _BOOTSTRAP_IDENTITY_FIELDS)


def _bootstrap_relative_parts(relative_text: str) -> tuple[str, ...]:
    relative = PurePosixPath(relative_text)
    if (
        not relative_text
        or "\\" in relative_text
        or relative.is_absolute()
        or relative.as_posix() != relative_text
        or relative_text == "."
        or ".." in relative.parts
    ):
        raise RuntimeError(f"CALVIN source-tree path is not canonical: {relative_text!r}")
    return relative.parts


def _bootstrap_open_directory(parent_fd: int, name: str, *, context: str) -> tuple[int, os.stat_result]:
    before = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    if not stat.S_ISDIR(before.st_mode):
        raise RuntimeError(f"CALVIN source-tree ancestor must be a real directory: {context}")
    descriptor = os.open(name, _BOOTSTRAP_DIRECTORY_OPEN_FLAGS, dir_fd=parent_fd)
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISDIR(opened.st_mode) or not _bootstrap_same_identity(before, opened):
            raise RuntimeError(f"CALVIN source-tree ancestor changed while opening: {context}")
        return descriptor, opened
    except BaseException:
        os.close(descriptor)
        raise


def _bootstrap_verify_directory(
    parent_fd: int,
    name: str,
    descriptor: int,
    expected: os.stat_result,
    *,
    context: str,
) -> None:
    after_descriptor = os.fstat(descriptor)
    after_path = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    if not _bootstrap_same_identity(expected, after_descriptor) or not _bootstrap_same_identity(expected, after_path):
        raise RuntimeError(f"CALVIN source-tree ancestor changed while reading: {context}")


def _bootstrap_open_absolute_directory_chain(
    path: Path,
) -> tuple[list[int], list[tuple[int, str, int, os.stat_result, str]]]:
    if not path.is_absolute():
        raise RuntimeError(f"CALVIN source root must be absolute: {path}")
    descriptors = [os.open("/", _BOOTSTRAP_DIRECTORY_OPEN_FLAGS)]
    bindings: list[tuple[int, str, int, os.stat_result, str]] = []
    try:
        for index, name in enumerate(path.parts[1:]):
            context = "/" + "/".join(path.parts[1 : index + 2])
            child_fd, child_identity = _bootstrap_open_directory(descriptors[-1], name, context=context)
            bindings.append((descriptors[-1], name, child_fd, child_identity, context))
            descriptors.append(child_fd)
        return descriptors, bindings
    except BaseException:
        for descriptor in reversed(descriptors):
            os.close(descriptor)
        raise


def _bootstrap_verify_absolute_directory_chain(
    descriptors: Sequence[int],
    bindings: Sequence[tuple[int, str, int, os.stat_result, str]],
) -> None:
    if not _bootstrap_same_identity(os.fstat(descriptors[0]), os.stat("/", follow_symlinks=False)):
        raise RuntimeError("CALVIN source-tree filesystem root changed while reading")
    for parent_fd, name, child_fd, expected, context in reversed(bindings):
        _bootstrap_verify_directory(parent_fd, name, child_fd, expected, context=context)


def _bootstrap_walk_source(descriptor: int, prefix: tuple[str, ...], output: list[str]) -> None:
    directory_before = os.fstat(descriptor)
    names_before = sorted(os.listdir(descriptor))
    for name in names_before:
        context = "/".join((*prefix, name))
        observed = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
        if stat.S_ISDIR(observed.st_mode):
            if name == "__pycache__":
                continue
            child_fd, child_identity = _bootstrap_open_directory(descriptor, name, context=context)
            try:
                _bootstrap_walk_source(child_fd, (*prefix, name), output)
                _bootstrap_verify_directory(
                    descriptor,
                    name,
                    child_fd,
                    child_identity,
                    context=context,
                )
            finally:
                os.close(child_fd)
        elif stat.S_ISREG(observed.st_mode):
            output.append(context)
        else:
            raise RuntimeError(f"CALVIN source-tree entry must be a regular file or real directory: {context}")
    if names_before != sorted(os.listdir(descriptor)) or not _bootstrap_same_identity(
        directory_before, os.fstat(descriptor)
    ):
        raise RuntimeError(f"CALVIN source-tree directory changed during inventory: {'/'.join(prefix)}")


def _bootstrap_read_relative_file(root_fd: int, relative_text: str) -> bytes:
    parts = _bootstrap_relative_parts(relative_text)
    directory_fds = [os.dup(root_fd)]
    bindings: list[tuple[int, str, int, os.stat_result, str]] = []
    file_descriptor: int | None = None
    try:
        for index, name in enumerate(parts[:-1]):
            context = "/".join(parts[: index + 1])
            child_fd, child_identity = _bootstrap_open_directory(directory_fds[-1], name, context=context)
            bindings.append((directory_fds[-1], name, child_fd, child_identity, context))
            directory_fds.append(child_fd)
        file_name = parts[-1]
        try:
            before = os.stat(file_name, dir_fd=directory_fds[-1], follow_symlinks=False)
        except FileNotFoundError as exc:
            raise RuntimeError(f"required CALVIN source-tree entry is missing: {relative_text}") from exc
        if not stat.S_ISREG(before.st_mode):
            raise RuntimeError(f"CALVIN source-tree entry must be a regular file: {relative_text}")
        file_descriptor = os.open(file_name, _BOOTSTRAP_FILE_OPEN_FLAGS, dir_fd=directory_fds[-1])
        opened = os.fstat(file_descriptor)
        if not stat.S_ISREG(opened.st_mode) or not _bootstrap_same_identity(before, opened):
            raise RuntimeError(f"CALVIN source-tree entry changed while opening: {relative_text}")
        blocks: list[bytes] = []
        while block := os.read(file_descriptor, 1024 * 1024):
            blocks.append(block)
        after_descriptor = os.fstat(file_descriptor)
        after_path = os.stat(file_name, dir_fd=directory_fds[-1], follow_symlinks=False)
        if not _bootstrap_same_identity(opened, after_descriptor) or not _bootstrap_same_identity(opened, after_path):
            raise RuntimeError(f"CALVIN source-tree entry changed while reading: {relative_text}")
        for parent_fd, name, child_fd, expected, context in reversed(bindings):
            _bootstrap_verify_directory(parent_fd, name, child_fd, expected, context=context)
        raw = b"".join(blocks)
        if len(raw) != opened.st_size:
            raise RuntimeError(f"CALVIN source-tree entry size changed while reading: {relative_text}")
        return raw
    finally:
        if file_descriptor is not None:
            os.close(file_descriptor)
        for descriptor in reversed(directory_fds):
            os.close(descriptor)


def _bootstrap_production_source_tree_sha256(root: Path) -> str:
    """Repeat train_calvin's injectively framed source identity."""

    root_path = Path(os.path.abspath(root))
    root_descriptors, root_bindings = _bootstrap_open_absolute_directory_chain(root_path)
    root_fd = root_descriptors[-1]
    try:
        root_opened = os.fstat(root_fd)
        if not stat.S_ISDIR(root_opened.st_mode):
            raise RuntimeError(f"CALVIN source root must be a real directory: {root_path}")
        src_fd, src_identity = _bootstrap_open_directory(root_fd, "src", context="src")
        try:
            duo_fd, duo_identity = _bootstrap_open_directory(src_fd, "duo_vla", context="src/duo_vla")
            try:
                discovered: list[str] = []
                _bootstrap_walk_source(duo_fd, ("src", "duo_vla"), discovered)
                _bootstrap_verify_directory(src_fd, "duo_vla", duo_fd, duo_identity, context="src/duo_vla")
            finally:
                os.close(duo_fd)
            _bootstrap_verify_directory(root_fd, "src", src_fd, src_identity, context="src")
        finally:
            os.close(src_fd)
        paths = [*discovered, *_TRAINING_SOURCE_EXPLICIT_RELATIVE_PATHS]
        for relative in paths:
            _bootstrap_relative_parts(relative)
        if len(set(paths)) != len(paths):
            raise RuntimeError("CALVIN source-tree inventory contains duplicate relative paths")
        ordered_paths = tuple(sorted(paths))
        digest = hashlib.sha256()
        digest.update(_CALVIN_SOURCE_TREE_HASH_MAGIC)
        digest.update(struct.pack(">Q", len(ordered_paths)))
        for relative_text in ordered_paths:
            relative = relative_text.encode("utf-8")
            raw = _bootstrap_read_relative_file(root_fd, relative_text)
            digest.update(struct.pack(">Q", len(relative)))
            digest.update(relative)
            digest.update(struct.pack(">Q", len(raw)))
            digest.update(raw)
        root_after_descriptor = os.fstat(root_fd)
        if not _bootstrap_same_identity(root_opened, root_after_descriptor):
            raise RuntimeError(f"CALVIN source root changed while hashing: {root_path}")
        _bootstrap_verify_absolute_directory_chain(root_descriptors, root_bindings)
        return digest.hexdigest()
    finally:
        for descriptor in reversed(root_descriptors):
            os.close(descriptor)


def _bootstrap_source_identity() -> dict[str, str]:
    files = {
        "comparator_sha256": _bootstrap_sha256_file(Path(__file__).resolve()),
        "launcher_sha256": _bootstrap_sha256_file(_LAUNCHER_PATH),
        "production_source_tree_sha256": _bootstrap_production_source_tree_sha256(_PROJECT_ROOT),
        **{
            f"production_{name}_sha256": _bootstrap_sha256_file(_PROJECT_ROOT / relative)
            for name, relative in sorted(_DIRECT_PRODUCTION_SOURCE_RELATIVE_PATHS.items())
        },
    }
    encoded = json.dumps(files, allow_nan=False, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return {**files, "qualification_source_sha256": hashlib.sha256(encoded).hexdigest()}


def _bootstrap_launcher_binding(identity: Mapping[str, str], capability: object) -> dict[str, Any]:
    if not isinstance(capability, dict):
        raise RuntimeError("qualification launcher bootstrap capability is invalid")
    expected_fields = {
        "comparator_sha256",
        "expected_train_venv",
        "forbidden_modules_preloaded",
        "interpreter_flags",
        "launcher_sha256",
        "mode",
        "project_src",
        "pyvenv_cfg_sha256",
        "site_packages",
        "sys_path",
    }
    if set(capability) != expected_fields:
        raise RuntimeError("qualification launcher bootstrap capability fields differ")
    string_fields = {
        "comparator_sha256",
        "expected_train_venv",
        "launcher_sha256",
        "mode",
        "project_src",
        "pyvenv_cfg_sha256",
        "site_packages",
    }
    if not all(isinstance(capability[field], str) for field in string_fields):
        raise RuntimeError("qualification launcher bootstrap capability types differ")
    if (
        not isinstance(capability["interpreter_flags"], dict)
        or not isinstance(capability["forbidden_modules_preloaded"], list)
        or not isinstance(capability["sys_path"], list)
        or not all(isinstance(path, str) for path in capability["sys_path"])
    ):
        raise RuntimeError("qualification launcher bootstrap capability types differ")
    if capability["mode"] != _LAUNCHER_BOOTSTRAP_MODE:
        raise RuntimeError("qualification launcher bootstrap mode is invalid")
    if capability["comparator_sha256"] != identity["comparator_sha256"]:
        raise RuntimeError("executed comparator bytes differ from the pre-import source snapshot")
    if capability["launcher_sha256"] != identity["launcher_sha256"]:
        raise RuntimeError("executed launcher bytes differ from the pre-import source snapshot")
    if capability["interpreter_flags"] != _EXPECTED_INTERPRETER_FLAGS:
        raise RuntimeError("qualification isolated interpreter flags differ")
    if capability["forbidden_modules_preloaded"] != []:
        raise RuntimeError("qualification bootstrap preloaded a forbidden site hook")
    expected_venv = Path(capability["expected_train_venv"]).resolve(strict=True)
    expected_site_packages = expected_venv / "lib/python3.11/site-packages"
    if Path(capability["site_packages"]) != expected_site_packages:
        raise RuntimeError("qualification bootstrap site-packages path differs")
    if Path(capability["project_src"]) != (_PROJECT_ROOT / "src").resolve(strict=True):
        raise RuntimeError("qualification bootstrap project source path differs")
    if capability["sys_path"][-2:] != [str((_PROJECT_ROOT / "src").resolve(strict=True)), str(expected_site_packages)]:
        raise RuntimeError("qualification bootstrap sys.path additions differ")
    if capability["pyvenv_cfg_sha256"] != _bootstrap_sha256_file(expected_venv / "pyvenv.cfg"):
        raise RuntimeError("qualification bootstrap pyvenv identity differs")
    return copy.deepcopy(capability)


_IMPORTED_QUALIFICATION_SOURCE_IDENTITY = _bootstrap_source_identity()
_QUALIFICATION_BOOTSTRAP_IDENTITY = _bootstrap_launcher_binding(
    _IMPORTED_QUALIFICATION_SOURCE_IDENTITY,
    _BOOTSTRAP_CAPABILITY_RAW,
)
del _BOOTSTRAP_CAPABILITY_RAW
_LAUNCHER_BOOTSTRAP_AUTHENTICATED = True

import torch  # noqa: E402 - source snapshot must precede every non-stdlib import
from torch.distributed.tensor import DTensor  # noqa: E402

from duo_vla.checkpointing import SCHEMA_VERSION as CHECKPOINT_SCHEMA  # noqa: E402
from duo_vla.checkpointing import load_checkpoint_manifest  # noqa: E402
from duo_vla.policy_contract import validate_manifest_policy_contract  # noqa: E402
from duo_vla.run_config import canonical_config_sha256, load_resolved_toml, load_verified_resolved_config  # noqa: E402
from duo_vla.run_journal import RUN_JOURNAL_SCHEMA, load_run_journal, validate_resume_checkpoint  # noqa: E402
from duo_vla.training import TrainerState  # noqa: E402
from duo_vla.training_checkpoint import (  # noqa: E402
    OPTIMIZER_PARAMETER_SCHEMA,
    TRAINING_RANK_STATE_SCHEMA,
    optimizer_parameter_inventory_sha256,
    validate_optimizer_state_dict,
)

REPORT_SCHEMA = "duo-vla-calvin-training-reproducibility-qualification-v3"
QUALIFICATION_KIND = "calvin-abc-to-d-strict-update-2-training-reproducibility"
CALVIN_PROTOCOL = "duovla-calvin-abc-to-d-v1"
EXPECTED_UPDATE = 2
EXPECTED_WORLD_SIZE = 2
EXPECTED_TOTAL_UPDATES = 30_000
EXPECTED_WARMUP_UPDATES = 1_000
EXPECTED_CHECKPOINT_INTERVAL = 1
EXPECTED_PHYSICAL_BATCH_SIZE = 8
EXPECTED_GRADIENT_ACCUMULATION_STEPS = 8
EXPECTED_GLOBAL_BATCH_SIZE = EXPECTED_PHYSICAL_BATCH_SIZE * EXPECTED_GRADIENT_ACCUMULATION_STEPS
EXPECTED_MAX_CACHED_FRAMES = 512
EXPECTED_PYTHON_VERSION = "3.11.15"
EXPECTED_TORCH_VERSION = "2.13.0+cu126"
EXPECTED_MODEL_ID = "google/diffusiongemma-26B-A4B-it"
EXPECTED_MODEL_REVISION = "f7f5b7f5fa82ffc52addd066915886d497f5517b"
EXPECTED_MODEL_HIDDEN_SIZE = 2816
EXPECTED_MODEL_DECODER_LAYERS = 30
EXPECTED_MODEL_SLIDING_ATTENTION_WIDTH = 4096
EXPECTED_MODEL_FULL_ATTENTION_WIDTH = 8192
EXPECTED_MODEL_SLIDING_KV_WIDTH = 2048
EXPECTED_MODEL_FULL_KV_WIDTH = 1024
EXPECTED_MODEL_SLIDING_ATTENTION_LAYERS = (
    0,
    1,
    2,
    3,
    4,
    6,
    7,
    8,
    9,
    10,
    12,
    13,
    14,
    15,
    16,
    18,
    19,
    20,
    21,
    22,
    24,
    25,
    26,
    27,
    28,
)
EXPECTED_CALVIN_ARCHIVE_BYTES = 555_309_812_705
EXPECTED_CALVIN_ARCHIVE_SHA256 = "c2036c67eb4c06966af1d1e1665bdb572c69e1404f5e77ffd46b384ff2b79f74"
EXPECTED_CALVIN_CENTRAL_DIRECTORY_SHA256 = "b4f79bda7f6b966b51aa419badd0f7db7a8972a7b58d6d342af60aceff0ea31b"
EXPECTED_CALVIN_MANIFEST_SCHEMA = "duo-vla-calvin-dataset-manifest-v4"
EXPECTED_CALVIN_MEMBER_INDEX_PATH = "task_ABC_D.members-v2.sqlite3"
EXPECTED_CALVIN_MEMBER_INDEX_SCHEMA = "duo-vla-calvin-member-index-v2"
EXPECTED_CALVIN_READER_SCHEMA = "duo-vla-calvin-archive-reader-v1"
EXPECTED_CALVIN_STORAGE_MODE = "archive-direct"
EXPECTED_CALVIN_METADATA_FILES = [
    "ep_start_end_ids.npy",
    "lang_annotations/auto_lang_ann.npy",
    "scene_info.npy",
    ".hydra/merged_config.yaml",
]
EXPECTED_STATE_ADAPTER = "robot_obs[0:7]+robot_obs[14:15]"
EXPECTED_ACTION_ADAPTER = "identity_official_scaled_rel_actions"
EXPECTED_CALVIN_SOURCE_REVISIONS = {
    "calvin": "fa03f01f19c65920e18cf37398a9ce859274af76",
    "calvin_env": "1431a46bd36bde5903fb6345e68b5ccc30def666",
    "tacto": "dd53360d9a8c186f0d6439372ec0be0fa5e21731",
}
EXPECTED_TRAIN_PACKAGES = {
    "accelerate": "1.14.0",
    "huggingface-hub": "1.29.0",
    "numpy": "2.4.6",
    "peft": "0.20.0",
    "pillow": "12.3.0",
    "pyarrow": "20.0.0",
    "safetensors": "0.8.0",
    "tokenizers": "0.22.2",
    "torch": EXPECTED_TORCH_VERSION,
    "torchvision": "0.28.0+cu126",
    "transformers": "5.15.0",
}
EXPECTED_TRAIN_ENVIRONMENT = {
    "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
    "CUDA_VISIBLE_DEVICES": "0,1",
    "HF_HUB_OFFLINE": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "OMP_DYNAMIC": "FALSE",
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "PYTHONNOUSERSITE": "1",
    "TOKENIZERS_PARALLELISM": "false",
    "TRANSFORMERS_OFFLINE": "1",
}
EXPECTED_RUN_CONTRACT_FIELDS = frozenset(
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
EXPECTED_ARTIFACT_PATHS = {
    "interface": "interface.safetensors",
    "lora_config": "lora/adapter_config.json",
    "lora_weights": "lora/adapter_model.safetensors",
    "normalization": "artifacts/normalization.json",
    "prefix_geometry": "artifacts/prefix_geometry.json",
    "resolved_config": "artifacts/resolved_config.json",
    "training_rank_000": "artifacts/training_rank_000.pt",
    "training_rank_001": "artifacts/training_rank_001.pt",
}
EXACT_TIP_ARTIFACTS = ("lora_weights", "lora_config", "interface", "resolved_config")
INTEGER_RUN_CONTRACT_FIELDS = frozenset(
    {"archive_bytes", "fixed_physical_prefix_width", "member_index_bytes", "physical_batch_size"}
)
MANIFEST_NORMALIZED_FIELDS = (
    "run_uuid",
    "parent_manifest_sha256",
    "last_metrics.update_seconds",
    "training_rank_state_sha256[*]",
    "artifacts.training_rank_000.sha256",
    "artifacts.training_rank_001.sha256",
)
METRIC_IGNORED_FIELDS = ("update_seconds",)
RANK_STATE_IGNORED_FIELDS = ("run_contract.run_uuid",)
_MAX_JSON_BYTES = 32 * 1024 * 1024


class QualificationError(RuntimeError):
    """Raised before publication when any strict qualification invariant fails."""


@dataclass(frozen=True)
class CheckpointAudit:
    path: Path
    update: int
    manifest: dict[str, Any]
    manifest_bytes: int
    manifest_identity: tuple[int, ...]
    manifest_sha256: str
    artifacts: dict[str, Path]
    artifact_identities: dict[str, tuple[int, ...]]


@dataclass(frozen=True)
class RunAudit:
    role: str
    root: Path
    run_uuid: str
    config_sha256: str
    config: dict[str, Any]
    resolved_config_path: Path
    resolved_config_identity: tuple[int, ...]
    resolved_config_raw_sha256: str
    journal_sha256: str
    journal_bytes: int
    journal_identity: tuple[int, ...]
    metrics_sha256: str
    metrics_bytes: int
    metrics_identity: tuple[int, ...]
    metrics: tuple[dict[str, Any], ...]
    checkpoints: tuple[CheckpointAudit, CheckpointAudit]


def require(condition: object, message: str) -> None:
    if not condition:
        raise QualificationError(message)


_STABLE_IDENTITY_FIELDS = ("st_dev", "st_ino", "st_mode", "st_size", "st_mtime_ns", "st_ctime_ns", "st_nlink")


def _stable_identity(value: os.stat_result) -> tuple[int, ...]:
    return tuple(getattr(value, field) for field in _STABLE_IDENTITY_FIELDS)


def _read_pinned_regular_descriptor(
    descriptor: int,
    *,
    context: str,
    expected_identity: tuple[int, int] | None = None,
    expected_nlink: int | None = None,
    expected_bytes: bytes | None = None,
) -> tuple[bytes, os.stat_result]:
    before = os.fstat(descriptor)
    require(stat.S_ISREG(before.st_mode), f"{context} descriptor is not a regular file")
    if expected_identity is not None:
        require((before.st_dev, before.st_ino) == expected_identity, f"{context} inode identity drifted")
    if expected_nlink is not None:
        require(before.st_nlink == expected_nlink, f"{context} link count drifted")
    blocks: list[bytes] = []
    offset = 0
    while offset < before.st_size:
        block = os.pread(descriptor, min(1024 * 1024, before.st_size - offset), offset)
        require(block, f"{context} ended before its recorded byte length")
        blocks.append(block)
        offset += len(block)
    require(not os.pread(descriptor, 1, offset), f"{context} grew beyond its recorded byte length")
    after = os.fstat(descriptor)
    require(
        all(getattr(before, field) == getattr(after, field) for field in _STABLE_IDENTITY_FIELDS),
        f"{context} changed while reading its pinned descriptor",
    )
    raw = b"".join(blocks)
    require(len(raw) == before.st_size, f"{context} byte length drifted")
    if expected_bytes is not None:
        require(raw == expected_bytes, f"{context} bytes drifted")
    return raw, after


def _stable_regular_file_bytes(
    path: Path,
    *,
    context: str,
    require_single_link: bool = False,
) -> tuple[bytes, os.stat_result]:
    try:
        before = os.stat(path, follow_symlinks=False)
    except FileNotFoundError as exc:
        raise QualificationError(f"{context} is missing: {path}") from exc
    require(stat.S_ISREG(before.st_mode), f"{context} must be a regular non-symlink file: {path}")
    if require_single_link:
        require(before.st_nlink == 1, f"{context} must have exactly one link")
    descriptor = os.open(path, os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW | os.O_CLOEXEC)
    try:
        raw, opened = _read_pinned_regular_descriptor(
            descriptor,
            context=context,
            expected_identity=(before.st_dev, before.st_ino),
            expected_nlink=1 if require_single_link else before.st_nlink,
        )
    finally:
        os.close(descriptor)
    after = os.stat(path, follow_symlinks=False)
    require(
        all(getattr(opened, field) == getattr(after, field) for field in _STABLE_IDENTITY_FIELDS),
        f"{context} path identity changed while reading",
    )
    return raw, opened


def _verify_published_entry(
    directory_fd: int,
    name: str,
    descriptor: int,
    *,
    expected_identity: tuple[int, int],
    expected_nlink: int,
    expected_bytes: bytes,
    context: str,
) -> None:
    observed_path_before = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
    require(stat.S_ISREG(observed_path_before.st_mode), f"{context} path is not a regular file")
    raw, observed_descriptor = _read_pinned_regular_descriptor(
        descriptor,
        context=context,
        expected_identity=expected_identity,
        expected_nlink=expected_nlink,
        expected_bytes=expected_bytes,
    )
    observed_path_after = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
    require(len(raw) == len(expected_bytes), f"{context} byte length drifted")
    require(
        (observed_path_before.st_dev, observed_path_before.st_ino) == expected_identity,
        f"{context} path inode identity drifted",
    )
    require(observed_path_before.st_nlink == expected_nlink, f"{context} path link count drifted")
    require(observed_path_before.st_size == len(expected_bytes), f"{context} path byte length drifted")
    require(
        all(
            getattr(observed_path_before, field)
            == getattr(observed_descriptor, field)
            == getattr(observed_path_after, field)
            for field in _STABLE_IDENTITY_FIELDS
        ),
        f"{context} path and pinned descriptor differ",
    )


def _valid_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _require_sha256(value: object, name: str) -> str:
    require(_valid_sha256(value), f"{name} must be a lowercase hexadecimal SHA-256")
    assert isinstance(value, str)
    return value


def _require_uuid(value: object, name: str) -> str:
    require(isinstance(value, str), f"{name} must be a canonical UUID")
    try:
        parsed = uuid.UUID(value)
    except (AttributeError, ValueError) as exc:
        raise QualificationError(f"{name} must be a canonical UUID") from exc
    require(str(parsed) == value, f"{name} must be a canonical UUID")
    assert isinstance(value, str)
    return value


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    raw, _ = _stable_regular_file_bytes(path, context=f"authenticated file {path}")
    return _sha256_bytes(raw)


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, allow_nan=False, ensure_ascii=False, separators=(",", ":"), sort_keys=True) + "\n"
    ).encode("utf-8")


def _canonical_pretty_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _source_identity_payload(value: Mapping[str, str]) -> dict[str, str]:
    require(
        set(value) == {*_IMPORTED_QUALIFICATION_SOURCE_IDENTITY.keys()},
        "qualification source identity fields differ",
    )
    payload = {key: item for key, item in value.items() if key != "qualification_source_sha256"}
    require(all(_valid_sha256(item) for item in payload.values()), "qualification source identity has an invalid hash")
    expected = hashlib.sha256(
        json.dumps(payload, allow_nan=False, separators=(",", ":"), sort_keys=True).encode("utf-8")
    ).hexdigest()
    require(
        value.get("qualification_source_sha256") == expected,
        "qualification source aggregate SHA-256 is invalid",
    )
    return payload


def require_qualification_source_unchanged(
    expected: Mapping[str, str],
    observed: Mapping[str, str],
    *,
    context: str,
) -> dict[str, str]:
    """Reject source mutation from the pre-production-import snapshot onward."""

    _source_identity_payload(expected)
    _source_identity_payload(observed)
    require(
        canonical_json_bytes(dict(expected)) == canonical_json_bytes(dict(observed)),
        f"{context} qualification source identity changed",
    )
    return dict(observed)


def _expected_train_venv() -> Path:
    candidate = Path(_QUALIFICATION_BOOTSTRAP_IDENTITY["expected_train_venv"])
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as exc:
        raise QualificationError(f"canonical train environment is unavailable: {candidate}: {exc}") from exc
    require(resolved.is_dir(), f"canonical train environment is not a directory: {resolved}")
    return resolved


def qualification_runtime_identity() -> dict[str, Any]:
    expected_venv = _expected_train_venv()
    return {
        "bootstrap_forbidden_modules_preloaded": list(_QUALIFICATION_BOOTSTRAP_IDENTITY["forbidden_modules_preloaded"]),
        "cuda_initialized": torch.cuda.is_initialized(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "expected_train_venv": str(expected_venv),
        "interpreter_flags": {name: getattr(sys.flags, name) for name in sorted(_EXPECTED_INTERPRETER_FLAGS)},
        "project_src": _QUALIFICATION_BOOTSTRAP_IDENTITY["project_src"],
        "pyvenv_cfg_sha256": _bootstrap_sha256_file(expected_venv / "pyvenv.cfg"),
        "python_base_prefix": str(Path(sys.base_prefix).resolve(strict=True)),
        "python_executable": os.path.abspath(sys.executable),
        "python_implementation": platform.python_implementation(),
        "python_prefix": str(Path(sys.prefix).resolve(strict=True)),
        "python_version": platform.python_version(),
        "site_packages": _QUALIFICATION_BOOTSTRAP_IDENTITY["site_packages"],
        "sys_path": list(sys.path),
        "torch_version": str(torch.__version__),
    }


def require_canonical_cpu_runtime(value: Mapping[str, Any], *, context: str) -> dict[str, Any]:
    expected_fields = {
        "bootstrap_forbidden_modules_preloaded",
        "cuda_initialized",
        "cuda_visible_devices",
        "expected_train_venv",
        "interpreter_flags",
        "project_src",
        "pyvenv_cfg_sha256",
        "python_base_prefix",
        "python_executable",
        "python_implementation",
        "python_prefix",
        "python_version",
        "site_packages",
        "sys_path",
        "torch_version",
    }
    require(set(value) == expected_fields, f"{context} runtime identity fields differ")
    require(value["python_version"] == EXPECTED_PYTHON_VERSION, f"{context} requires Python {EXPECTED_PYTHON_VERSION}")
    require(value["python_implementation"] == "CPython", f"{context} requires CPython")
    require(value["torch_version"] == EXPECTED_TORCH_VERSION, f"{context} requires torch {EXPECTED_TORCH_VERSION}")
    require(value["interpreter_flags"] == _EXPECTED_INTERPRETER_FLAGS, f"{context} isolated interpreter flags drifted")
    require(value["bootstrap_forbidden_modules_preloaded"] == [], f"{context} preloaded a forbidden site hook")
    require(type(value["cuda_initialized"]) is bool and not value["cuda_initialized"], f"{context} initialized CUDA")
    require(value["cuda_visible_devices"] == "", f"{context} requires CUDA_VISIBLE_DEVICES to be empty")
    expected_venv = _expected_train_venv()
    require(value["expected_train_venv"] == str(expected_venv), f"{context} expected train venv drifted")
    require(
        value["python_prefix"] == value["python_base_prefix"],
        f"{context} unexpectedly processed the virtualenv site configuration",
    )
    require(
        value["python_executable"] == str(expected_venv / "bin/python"),
        f"{context} did not use the canonical train Python entry point",
    )
    require(
        value["pyvenv_cfg_sha256"] == _QUALIFICATION_BOOTSTRAP_IDENTITY["pyvenv_cfg_sha256"],
        f"{context} pyvenv identity drifted",
    )
    require(value["project_src"] == _QUALIFICATION_BOOTSTRAP_IDENTITY["project_src"], f"{context} source path drifted")
    require(
        value["site_packages"] == _QUALIFICATION_BOOTSTRAP_IDENTITY["site_packages"],
        f"{context} site-packages path drifted",
    )
    require(value["sys_path"] == _QUALIFICATION_BOOTSTRAP_IDENTITY["sys_path"], f"{context} sys.path drifted")
    return dict(value)


def require_runtime_unchanged(
    expected: Mapping[str, Any],
    observed: Mapping[str, Any],
    *,
    context: str,
) -> dict[str, Any]:
    checked_expected = require_canonical_cpu_runtime(expected, context=context)
    checked_observed = require_canonical_cpu_runtime(observed, context=context)
    require(
        canonical_json_bytes(checked_expected) == canonical_json_bytes(checked_observed),
        f"{context} runtime identity changed",
    )
    return checked_observed


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _reject_nonfinite_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant: {value}")


def _validate_json_value(value: Any, path: str = "$") -> None:
    if value is None or type(value) in {bool, int, str}:
        return
    if type(value) is float:
        require(math.isfinite(value), f"non-finite JSON number at {path}")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_json_value(item, f"{path}[{index}]")
        return
    if isinstance(value, dict):
        require(all(isinstance(key, str) for key in value), f"non-string JSON object key at {path}")
        for key, item in value.items():
            _validate_json_value(item, f"{path}.{key}")
        return
    raise QualificationError(f"unsupported JSON value type at {path}: {type(value).__name__}")


def _strict_json_from_bytes(raw: bytes, *, source: Path) -> Any:
    require(len(raw) <= _MAX_JSON_BYTES, f"JSON input exceeds {_MAX_JSON_BYTES} bytes: {source}")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise QualificationError(f"JSON input is not UTF-8: {source}") from exc
    try:
        value = json.loads(
            text,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonfinite_constant,
        )
    except (json.JSONDecodeError, ValueError) as exc:
        raise QualificationError(f"malformed strict JSON: {source}: {exc}") from exc
    _validate_json_value(value)
    return value


def _require_real_directory(path: Path, name: str) -> Path:
    try:
        mode = path.lstat().st_mode
    except FileNotFoundError as exc:
        raise QualificationError(f"{name} is missing: {path}") from exc
    require(stat.S_ISDIR(mode) and not path.is_symlink(), f"{name} must be a real directory: {path}")
    return path.resolve(strict=True)


def _require_real_file(path: Path, name: str) -> Path:
    try:
        mode = path.lstat().st_mode
    except FileNotFoundError as exc:
        raise QualificationError(f"{name} is missing: {path}") from exc
    require(stat.S_ISREG(mode) and not path.is_symlink(), f"{name} must be a real file: {path}")
    return path


def _contained_real_file(root: Path, relative_text: str, name: str) -> Path:
    require(relative_text and "\\" not in relative_text, f"{name} has a non-canonical relative path")
    relative = PurePosixPath(relative_text)
    require(
        not relative.is_absolute()
        and relative.as_posix() == relative_text
        and relative_text != "."
        and ".." not in relative.parts,
        f"{name} has a non-contained relative path",
    )
    current = root
    for part in relative.parts[:-1]:
        current = current / part
        _require_real_directory(current, f"{name} parent")
    path = root.joinpath(*relative.parts)
    _require_real_file(path, name)
    resolved = path.resolve(strict=True)
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise QualificationError(f"{name} escapes its checkpoint directory") from exc
    return resolved


def _read_strict_json(path: Path, name: str) -> tuple[bytes, Any, os.stat_result]:
    try:
        raw, identity = _stable_regular_file_bytes(path, context=name)
    except OSError as exc:
        raise QualificationError(f"cannot read {name}: {path}: {exc}") from exc
    return raw, _strict_json_from_bytes(raw, source=path), identity


def _dtensor_metadata(value: torch.Tensor) -> dict[str, Any] | None:
    if not isinstance(value, DTensor):
        return None
    mesh = value.device_mesh
    mesh_tensor = mesh.mesh.detach().cpu().contiguous()
    return {
        "device_type": mesh.device_type,
        "mesh": mesh_tensor.reshape(-1).tolist(),
        "mesh_dim_names": list(mesh.mesh_dim_names) if mesh.mesh_dim_names is not None else None,
        "mesh_shape": list(mesh_tensor.shape),
        "placements": [
            {
                "repr": repr(placement),
                "type": f"{type(placement).__module__}.{type(placement).__qualname__}",
            }
            for placement in value.placements
        ],
    }


def _local_tensor(value: torch.Tensor) -> torch.Tensor:
    local = value.to_local() if isinstance(value, DTensor) else value
    require(
        isinstance(local, torch.Tensor) and local.device.type == "cpu" and local.layout == torch.strided,
        "logical comparison requires CPU strided local tensors",
    )
    return local


def _tensor_byte_view(value: torch.Tensor) -> torch.Tensor:
    """Return per-rank logical values as a flat CPU byte view, including DTensors."""

    contiguous = _local_tensor(value).detach().contiguous()
    if contiguous.ndim == 0:
        contiguous = contiguous.reshape(1)
    return contiguous.view(torch.uint8).reshape(-1)


def _assert_equal(left: Any, right: Any, *, context: str, path: str = "$") -> None:
    """Compare loaded state without Python's bool/int or +0/-0 coercions."""

    if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
        require(
            isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor),
            f"{context} differs at {path}: tensor/non-tensor type mismatch",
        )
        require(left.device.type == right.device.type == "cpu", f"{context} has a non-CPU tensor at {path}")
        require(left.layout == right.layout == torch.strided, f"{context} has unsupported tensor layout at {path}")
        require(
            isinstance(left, DTensor) is isinstance(right, DTensor),
            f"{context} differs at {path}: distributed/local tensor type",
        )
        require(left.dtype == right.dtype, f"{context} differs at {path}: tensor dtype")
        require(tuple(left.shape) == tuple(right.shape), f"{context} differs at {path}: tensor shape")
        require(left.stride() == right.stride(), f"{context} differs at {path}: tensor stride")
        require(_dtensor_metadata(left) == _dtensor_metadata(right), f"{context} differs at {path}: DTensor metadata")
        left_local = _local_tensor(left)
        right_local = _local_tensor(right)
        require(
            tuple(left_local.shape) == tuple(right_local.shape),
            f"{context} differs at {path}: local tensor shape",
        )
        require(left_local.stride() == right_local.stride(), f"{context} differs at {path}: local tensor stride")
        left_bytes = _tensor_byte_view(left)
        right_bytes = _tensor_byte_view(right)
        require(torch.equal(left_bytes, right_bytes), f"{context} differs at {path}: tensor bytes")
        return

    require(type(left) is type(right), f"{context} differs at {path}: {type(left).__name__}/{type(right).__name__}")
    if isinstance(left, Mapping):
        left_keys = {(type(key), key) for key in left}
        right_keys = {(type(key), key) for key in right}
        require(left_keys == right_keys, f"{context} differs at {path}: mapping keys")
        for _, key in sorted(left_keys, key=lambda item: (item[0].__name__, repr(item[1]))):
            _assert_equal(left[key], right[key], context=context, path=f"{path}.{key}")
        return
    if isinstance(left, (list, tuple)):
        require(len(left) == len(right), f"{context} differs at {path}: sequence length")
        for index, (left_item, right_item) in enumerate(zip(left, right, strict=True)):
            _assert_equal(left_item, right_item, context=context, path=f"{path}[{index}]")
        return
    if type(left) is float:
        require(
            math.isfinite(left) and math.isfinite(right),
            f"{context} has a non-finite float at {path}",
        )
        require(struct.pack(">d", left) == struct.pack(">d", right), f"{context} differs at {path}: float bytes")
        return
    if left is None or type(left) in {bool, int, str, bytes}:
        require(left == right, f"{context} differs at {path}: {left!r} != {right!r}")
        return
    raise QualificationError(f"{context} contains unsupported type at {path}: {type(left).__name__}")


def _logical_digest(value: Any) -> str:
    digest = hashlib.sha256()

    def update(item: Any) -> None:
        if isinstance(item, torch.Tensor):
            require(
                item.device.type == "cpu" and item.layout == torch.strided,
                "logical digest requires CPU strided tensors",
            )
            raw = _tensor_byte_view(item).numpy().tobytes()
            local = _local_tensor(item)
            metadata = canonical_json_bytes(
                {
                    "distributed": _dtensor_metadata(item),
                    "dtype": str(item.dtype),
                    "local_shape": list(local.shape),
                    "local_stride": list(local.stride()),
                    "shape": list(item.shape),
                    "stride": list(item.stride()),
                }
            )
            digest.update(b"tensor\0" + metadata + len(raw).to_bytes(8, "big") + raw)
        elif isinstance(item, Mapping):
            digest.update(f"mapping:{type(item).__module__}.{type(item).__qualname__}\0".encode())
            keys = sorted(item, key=lambda key: (type(key).__name__, repr(key)))
            for key in keys:
                update(key)
                update(item[key])
            digest.update(b"mapping-end\0")
        elif isinstance(item, (list, tuple)):
            digest.update(f"sequence:{type(item).__name__}:{len(item)}\0".encode())
            for child in item:
                update(child)
            digest.update(b"sequence-end\0")
        elif type(item) is float:
            require(math.isfinite(item), "logical digest refuses non-finite floats")
            digest.update(b"float\0" + struct.pack(">d", item))
        elif item is None:
            digest.update(b"none\0")
        elif type(item) is bool:
            digest.update(b"bool\0" + bytes([item]))
        elif type(item) is int:
            encoded = str(item).encode()
            digest.update(b"int\0" + len(encoded).to_bytes(8, "big") + encoded)
        elif isinstance(item, (str, bytes)):
            encoded = item.encode("utf-8") if isinstance(item, str) else item
            digest.update(type(item).__name__.encode() + b"\0" + len(encoded).to_bytes(8, "big") + encoded)
        else:
            raise QualificationError(f"logical digest contains unsupported type: {type(item).__name__}")

    update(value)
    return digest.hexdigest()


def _tensor_stats(value: Any) -> tuple[int, int]:
    tensors = 0
    tensor_bytes = 0

    def visit(item: Any) -> None:
        nonlocal tensors, tensor_bytes
        if isinstance(item, torch.Tensor):
            local = _local_tensor(item)
            tensors += 1
            tensor_bytes += local.numel() * local.element_size()
        elif isinstance(item, Mapping):
            for key, child in item.items():
                visit(key)
                visit(child)
        elif isinstance(item, (list, tuple)):
            for child in item:
                visit(child)

    visit(value)
    return tensors, tensor_bytes


def _validate_calvin_split(value: Any) -> dict[str, Any]:
    require(isinstance(value, dict), "resolved config CALVIN split identity is invalid")
    assert isinstance(value, dict)
    expected_fields = {
        "algorithm",
        "seed",
        "train_episode_indices",
        "train_episode_sha256",
        "validation_episode_indices",
        "validation_episode_sha256",
        "validation_fraction",
    }
    require(set(value) == expected_fields, "resolved config CALVIN split fields differ")
    require(
        value["algorithm"] == "scene-grouped stable sha256 whole-episode ordering with all-task coverage assertion",
        "resolved config CALVIN split algorithm drifted",
    )
    require(value["seed"] == 1729 and type(value["seed"]) is int, "resolved config CALVIN split seed drifted")
    require(
        type(value["validation_fraction"]) is float and value["validation_fraction"] == 0.1,
        "resolved config CALVIN split fraction drifted",
    )

    def indices(name: str) -> list[int]:
        observed = value[name]
        require(
            isinstance(observed, list)
            and observed
            and all(type(index) is int and index >= 0 for index in observed)
            and observed == sorted(set(observed)),
            f"resolved config CALVIN {name} is invalid",
        )
        return observed

    train = indices("train_episode_indices")
    validation = indices("validation_episode_indices")
    require(not set(train) & set(validation), "resolved config CALVIN split overlaps")
    combined = sorted((*train, *validation))
    require(combined == list(range(combined[-1] + 1)), "resolved config CALVIN split is not a full episode partition")
    for name, observed in (("train", train), ("validation", validation)):
        expected_sha256 = hashlib.sha256(",".join(map(str, observed)).encode()).hexdigest()
        require(value[f"{name}_episode_sha256"] == expected_sha256, f"resolved config CALVIN {name} split hash drifted")
    return value


def _validate_calvin_identity(value: Any) -> dict[str, Any]:
    require(isinstance(value, dict), "resolved config CALVIN identity is invalid")
    assert isinstance(value, dict)
    expected_fields = {
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
    require(set(value) == expected_fields, "resolved config CALVIN identity fields differ")
    require(value["protocol"] == CALVIN_PROTOCOL, "resolved config CALVIN identity protocol drifted")
    require(
        value["archive_bytes"] == EXPECTED_CALVIN_ARCHIVE_BYTES,
        "resolved config CALVIN archive byte length drifted",
    )
    require(value["archive_sha256"] == EXPECTED_CALVIN_ARCHIVE_SHA256, "resolved config CALVIN archive drifted")
    require(
        value["central_directory_sha256"] == EXPECTED_CALVIN_CENTRAL_DIRECTORY_SHA256,
        "resolved config CALVIN central-directory identity drifted",
    )
    require(
        value["dataset_manifest_schema"] == EXPECTED_CALVIN_MANIFEST_SCHEMA,
        "resolved config CALVIN dataset-manifest schema drifted",
    )
    require(value["storage_mode"] == EXPECTED_CALVIN_STORAGE_MODE, "resolved config CALVIN storage mode drifted")
    require(value["reader_schema"] == EXPECTED_CALVIN_READER_SCHEMA, "resolved config CALVIN reader schema drifted")
    require(
        value["metadata_files"] == EXPECTED_CALVIN_METADATA_FILES,
        "resolved config CALVIN metadata-file inventory drifted",
    )
    require(value["state_adapter"] == EXPECTED_STATE_ADAPTER, "resolved config CALVIN state adapter drifted")
    require(value["action_adapter"] == EXPECTED_ACTION_ADAPTER, "resolved config CALVIN action adapter drifted")
    for name in (
        "dataset_manifest_file_sha256",
        "dataset_manifest_sha256",
        "member_inventory_sha256",
        "metadata_sha256",
        "normalization_sha256",
        "storage_identity_sha256",
    ):
        _require_sha256(value[name], f"resolved config CALVIN {name}")
    member_index = value["member_index"]
    require(
        isinstance(member_index, dict)
        and set(member_index) == {"bytes", "path", "schema", "sha256"}
        and type(member_index.get("bytes")) is int
        and member_index["bytes"] > 0
        and member_index.get("path") == EXPECTED_CALVIN_MEMBER_INDEX_PATH
        and member_index.get("schema") == EXPECTED_CALVIN_MEMBER_INDEX_SCHEMA,
        "resolved config CALVIN member-index identity drifted",
    )
    _require_sha256(member_index["sha256"], "resolved config CALVIN member-index SHA-256")
    expected_camera_shapes = {"rgb_gripper": [84, 84, 3], "rgb_static": [200, 200, 3]}
    _assert_equal(value["camera_shapes"], expected_camera_shapes, context="resolved config CALVIN camera shapes")
    require(
        value["camera_shapes_sha256"] == canonical_config_sha256(expected_camera_shapes),
        "resolved config CALVIN camera shape hash drifted",
    )
    _assert_equal(
        value["calvin_source_revisions"],
        EXPECTED_CALVIN_SOURCE_REVISIONS,
        context="resolved config CALVIN source revisions",
    )
    require(
        value["calvin_source_revisions_sha256"] == canonical_config_sha256(EXPECTED_CALVIN_SOURCE_REVISIONS),
        "resolved config CALVIN source revision hash drifted",
    )
    split = _validate_calvin_split(value["split"])
    require(value["split_sha256"] == canonical_config_sha256(split), "resolved config CALVIN split identity drifted")
    return value


def _validate_training_execution_environment(value: Any, *, run_seed: int) -> dict[str, Any]:
    require(isinstance(value, dict), "resolved config execution environment is invalid")
    assert isinstance(value, dict)
    expected_fields = {
        "authenticated_runtime",
        "cublas_workspace_config",
        "cuda_runtime",
        "cudnn",
        "cudnn_benchmark",
        "cudnn_deterministic",
        "cudnn_tf32",
        "deterministic_algorithms",
        "deterministic_warn_only",
        "float32_matmul_precision",
        "gpu_capability",
        "gpu_names",
        "matmul_tf32",
        "peft",
        "python",
        "python_hash_seed",
        "torch",
        "transformers",
        "world_size",
    }
    require(set(value) == expected_fields, "resolved config execution environment fields differ")
    expected_scalars = {
        "cublas_workspace_config": ":4096:8",
        "cuda_runtime": "12.6",
        "cudnn_benchmark": False,
        "cudnn_deterministic": True,
        "cudnn_tf32": False,
        "deterministic_algorithms": True,
        "deterministic_warn_only": False,
        "float32_matmul_precision": "highest",
        "matmul_tf32": False,
        "peft": EXPECTED_TRAIN_PACKAGES["peft"],
        "python": EXPECTED_PYTHON_VERSION,
        "python_hash_seed": str(run_seed),
        "torch": EXPECTED_TORCH_VERSION,
        "transformers": EXPECTED_TRAIN_PACKAGES["transformers"],
        "world_size": EXPECTED_WORLD_SIZE,
    }
    for name, expected in expected_scalars.items():
        require(
            type(value[name]) is type(expected) and value[name] == expected, f"execution environment {name} drifted"
        )
    require(type(value["cudnn"]) is int and value["cudnn"] > 0, "execution environment cuDNN identity is invalid")
    capabilities = value["gpu_capability"]
    names = value["gpu_names"]
    require(
        isinstance(capabilities, list)
        and len(capabilities) == EXPECTED_WORLD_SIZE
        and all(
            isinstance(capability, list)
            and len(capability) == 2
            and all(type(component) is int and component >= 0 for component in capability)
            for capability in capabilities
        ),
        "execution environment GPU capability inventory is invalid",
    )
    require(
        isinstance(names, list)
        and len(names) == EXPECTED_WORLD_SIZE
        and all(isinstance(name, str) and name for name in names),
        "execution environment GPU name inventory is invalid",
    )
    runtime = value["authenticated_runtime"]
    require(isinstance(runtime, dict), "authenticated training runtime is invalid")
    assert isinstance(runtime, dict)
    require(
        set(runtime) == {"environment", "lock_sha256", "module_origins", "packages", "python", "sys_path"},
        "authenticated training runtime fields differ",
    )
    expected_environment = {**EXPECTED_TRAIN_ENVIRONMENT, "PYTHONPATH": str((_PROJECT_ROOT / "src").resolve())}
    _assert_equal(runtime["environment"], expected_environment, context="authenticated training environment")
    require(
        runtime["lock_sha256"] == "0b1fb188747ee99224078b3c40975ca7e6f8e082e22d2860f9b50ee679a67c46",
        "training lock identity drifted",
    )
    _assert_equal(runtime["packages"], EXPECTED_TRAIN_PACKAGES, context="authenticated training package pins")
    require(runtime["python"] == EXPECTED_PYTHON_VERSION, "authenticated training Python drifted")
    origins = runtime["module_origins"]
    expected_origin_names = {
        "PIL",
        "accelerate",
        "duo_vla",
        "huggingface_hub",
        "numpy",
        "peft",
        "pyarrow",
        "safetensors",
        "tokenizers",
        "torch",
        "torchvision",
        "transformers",
    }
    require(
        isinstance(origins, dict) and set(origins) == expected_origin_names, "training module-origin inventory drifted"
    )
    train_venv = _expected_train_venv()
    site_packages = train_venv / "lib/python3.11/site-packages"
    for name, origin in origins.items():
        require(isinstance(origin, str) and Path(origin).is_absolute(), f"training module origin {name} is invalid")
        expected_root = _PROJECT_ROOT / "src" if name == "duo_vla" else site_packages
        require(
            Path(origin).resolve().is_relative_to(expected_root.resolve()), f"training module origin {name} drifted"
        )
    sys_path = runtime["sys_path"]
    require(
        isinstance(sys_path, list)
        and sys_path
        and len(sys_path) == len(set(sys_path))
        and all(isinstance(path, str) and Path(path).is_absolute() for path in sys_path),
        "authenticated training sys.path is invalid",
    )
    safe_roots = (_PROJECT_ROOT / "scripts", _PROJECT_ROOT / "src", Path(sys.base_prefix), train_venv)
    for path in sys_path:
        require(
            any(Path(path).resolve().is_relative_to(root.resolve()) for root in safe_roots),
            f"authenticated training sys.path entry is outside canonical roots: {path}",
        )
    return value


def _validate_config(
    config: dict[str, Any],
    *,
    config_sha256: str,
    source_identity: Mapping[str, str],
) -> None:
    require(config.get("protocol") == CALVIN_PROTOCOL, "resolved config is not the canonical CALVIN protocol")
    benchmark = config.get("benchmark")
    optimization = config.get("optimization")
    training = config.get("training")
    model = config.get("model")
    action = config.get("action")
    run = config.get("run")
    require(
        all(isinstance(value, dict) for value in (benchmark, optimization, training, model, action, run)),
        "resolved config omits canonical tables",
    )
    assert isinstance(benchmark, dict)
    assert isinstance(optimization, dict)
    assert isinstance(training, dict)
    assert isinstance(model, dict)
    assert isinstance(action, dict)
    assert isinstance(run, dict)
    require(benchmark.get("dataset") == "task_ABC_D", "resolved config dataset is not task_ABC_D")
    require(benchmark.get("train_split") == "training", "resolved config train split is not training")
    require(benchmark.get("train_environments") == ["A", "B", "C"], "resolved config train environments drifted")
    require(benchmark.get("evaluation_environment") == "D", "resolved config evaluation environment drifted")
    require(benchmark.get("camera_order") == ["rgb_static", "rgb_gripper"], "resolved config camera order drifted")
    require(model.get("tensor_parallel_size") == EXPECTED_WORLD_SIZE, "resolved config tensor-parallel size is not two")
    require(
        optimization.get("physical_batch_size") == EXPECTED_PHYSICAL_BATCH_SIZE,
        "resolved config physical batch is not eight",
    )
    require(
        optimization.get("microbatch_size") == EXPECTED_PHYSICAL_BATCH_SIZE,
        "resolved config microbatch size is not the fixed physical batch",
    )
    require(
        optimization.get("gradient_accumulation_steps") == EXPECTED_GRADIENT_ACCUMULATION_STEPS,
        "resolved config gradient accumulation is not eight",
    )
    require(
        optimization.get("global_batch_size") == EXPECTED_GLOBAL_BATCH_SIZE,
        "resolved config global batch is not 64",
    )
    require(
        optimization.get("total_updates") == EXPECTED_TOTAL_UPDATES,
        "resolved config must retain the canonical 30,000-update training budget",
    )
    require(
        optimization.get("warmup_updates") == EXPECTED_WARMUP_UPDATES,
        "resolved config must retain the canonical 1,000-update warmup",
    )
    require(
        type(training.get("checkpoint_interval")) is int
        and training["checkpoint_interval"] == EXPECTED_CHECKPOINT_INTERVAL
        and type(training.get("permanent_checkpoint_interval")) is int
        and training["permanent_checkpoint_interval"] == EXPECTED_CHECKPOINT_INTERVAL,
        "reproducibility qualification requires checkpoint and permanent intervals of one",
    )
    require(action.get("horizon") == 8 and action.get("dimension") == 7, "resolved config action geometry drifted")
    _assert_equal(training.get("seeds"), [0, 1, 2], context="resolved config training seed inventory")
    require(type(run.get("seed")) is int and run["seed"] in {0, 1, 2}, "resolved config run seed is not canonical")
    require(run.get("task") is None, "CALVIN ABC-to-D reproducibility qualification requires the full task inventory")
    require(
        type(run.get("max_cached_frames")) is int and run["max_cached_frames"] == EXPECTED_MAX_CACHED_FRAMES,
        "resolved config max-cached-frames is not canonical",
    )
    prefix_sha256 = _require_sha256(
        benchmark.get("prefix_geometry_content_sha256"),
        "resolved config prefix geometry SHA-256",
    )
    fixed_width = benchmark.get("fixed_physical_prefix_width")
    require(
        type(fixed_width) is int and 0 < fixed_width <= 1024 - 8,
        "resolved config fixed physical prefix width is invalid",
    )
    calvin_identity = _validate_calvin_identity(config.get("calvin_identity"))
    execution_environment = _validate_training_execution_environment(
        config.get("execution_environment"),
        run_seed=run["seed"],
    )
    artifact_trees = config.get("artifact_trees")
    require(
        isinstance(artifact_trees, dict)
        and set(artifact_trees) == {"model_tree_sha256"}
        and _valid_sha256(artifact_trees["model_tree_sha256"]),
        "resolved config model artifact-tree identity is invalid",
    )
    execution_geometry = config.get("execution_geometry")
    expected_execution_geometry = {
        "expert_batch_isolation": "sample_isolated_grouped_mm_v1",
        "experts_implementation": "grouped_mm",
        "fixed_physical_prefix_width": fixed_width,
        "physical_batch_size": EXPECTED_PHYSICAL_BATCH_SIZE,
        "prefix_geometry_content_sha256": prefix_sha256,
    }
    _assert_equal(execution_geometry, expected_execution_geometry, context="resolved config execution geometry")
    _require_sha256(
        config.get("training_instruction_inventory_sha256"),
        "resolved config training instruction inventory SHA-256",
    )
    source_tree_sha256 = _require_sha256(config.get("source_tree_sha256"), "resolved config source-tree SHA-256")
    require(
        source_tree_sha256 == source_identity.get("production_source_tree_sha256"),
        "resolved config training source tree differs from the live production source tree",
    )
    canonical = load_resolved_toml(_PROJECT_ROOT / "configs/calvin_abc_to_d.toml")
    expected_config = copy.deepcopy(canonical)
    expected_config["benchmark"]["prefix_geometry_content_sha256"] = prefix_sha256
    expected_config["benchmark"]["fixed_physical_prefix_width"] = fixed_width
    expected_config["training"]["checkpoint_interval"] = EXPECTED_CHECKPOINT_INTERVAL
    expected_config["training"]["permanent_checkpoint_interval"] = EXPECTED_CHECKPOINT_INTERVAL
    expected_config["run"] = {
        "max_cached_frames": EXPECTED_MAX_CACHED_FRAMES,
        "seed": run["seed"],
        "task": None,
    }
    expected_config["artifact_trees"] = copy.deepcopy(artifact_trees)
    expected_config["calvin_identity"] = copy.deepcopy(calvin_identity)
    expected_config["execution_environment"] = copy.deepcopy(execution_environment)
    expected_config["execution_geometry"] = copy.deepcopy(execution_geometry)
    expected_config["training_instruction_inventory_sha256"] = config["training_instruction_inventory_sha256"]
    expected_config["source_tree_sha256"] = source_tree_sha256
    _assert_equal(config, expected_config, context="exact canonical CALVIN smoke resolved config")
    require(canonical_config_sha256(config) == config_sha256, "resolved config semantic SHA-256 drifted")
    validate_manifest_policy_contract({"policy_contract": _policy_contract(config)}, config)


def _policy_contract(config: Mapping[str, Any]) -> dict[str, Any]:
    # Validation and construction share the production contract implementation.
    from duo_vla.policy_contract import policy_contract_from_config

    return policy_contract_from_config(config).to_dict()


def _validate_metric(metric: dict[str, Any], *, update: int, examples_seen: int, objective: str) -> None:
    require(type(metric.get("update")) is int and metric["update"] == update, f"metric update {update} is invalid")
    require(
        type(metric.get("examples_seen")) is int and metric["examples_seen"] == examples_seen,
        f"metric update {update} examples_seen drifted",
    )
    seconds = metric.get("update_seconds")
    require(
        type(seconds) is float and math.isfinite(seconds) and seconds > 0.0,
        f"metric update {update} has invalid update_seconds",
    )
    require(metric.get("objective") == objective, f"metric update {update} objective drifted")


def _checkpoint_inventory(root: Path) -> tuple[Path, Path]:
    checkpoints = _require_real_directory(root / "checkpoints", "checkpoint root")
    children = list(checkpoints.iterdir())
    expected_names = {"update-000001", "update-000002"}
    require(
        {child.name for child in children} == expected_names,
        "checkpoint inventory must contain exactly updates one and two",
    )
    ordered: list[Path] = []
    for name in sorted(expected_names):
        ordered.append(_require_real_directory(checkpoints / name, f"checkpoint {name}"))
    for child in root.iterdir():
        require(
            not child.name.startswith(".rank-state-update-"),
            f"uncommitted rank-state staging directory remains in run: {child.name}",
        )
    return ordered[0], ordered[1]


def _audit_checkpoint(
    path: Path,
    *,
    update: int,
    run_uuid: str,
    config_sha256: str,
    parent_manifest_sha256: str | None,
    config: dict[str, Any],
) -> CheckpointAudit:
    manifest_path = path / "manifest.json"
    raw, parsed, manifest_identity = _read_strict_json(manifest_path, f"checkpoint update {update} manifest")
    require(isinstance(parsed, dict), f"checkpoint update {update} manifest is not an object")
    require(raw == _canonical_pretty_json_bytes(parsed), f"checkpoint update {update} manifest is not canonical JSON")
    loaded = load_checkpoint_manifest(path, verify_hashes=True)
    _assert_equal(parsed, loaded, context=f"checkpoint update {update} strict manifest parse")
    manifest = parsed
    require(manifest.get("schema") == CHECKPOINT_SCHEMA, f"checkpoint update {update} schema drifted")
    require(
        manifest.get("kind") == "resumable-calvin-abc-to-d-training",
        f"checkpoint update {update} kind is not canonical CALVIN training",
    )
    require(manifest.get("dataset") == "task_ABC_D", f"checkpoint update {update} dataset drifted")
    require(manifest.get("dataset_split") == "training", f"checkpoint update {update} split drifted")
    require(manifest.get("protocol") == CALVIN_PROTOCOL, f"checkpoint update {update} protocol drifted")
    require(manifest.get("run_uuid") == run_uuid, f"checkpoint update {update} run_uuid differs from journal")
    require(
        manifest.get("config_sha256") == config_sha256,
        f"checkpoint update {update} config SHA differs from journal",
    )
    require(
        manifest.get("source_tree_sha256") == config["source_tree_sha256"],
        f"checkpoint update {update} source-tree SHA differs from resolved config",
    )
    require(
        manifest.get("parent_manifest_sha256") == parent_manifest_sha256,
        f"checkpoint update {update} parent manifest lineage is invalid",
    )
    trainer_state = TrainerState.from_dict(manifest.get("trainer_state"))
    require(trainer_state.next_update == update, f"checkpoint update {update} trainer progress is invalid")
    global_batch = config["optimization"]["global_batch_size"]
    require(trainer_state.examples_seen == update * global_batch, f"checkpoint update {update} example count drifted")
    policy_contract = validate_manifest_policy_contract(manifest, config)
    require(
        manifest.get("policy_contract_sha256") == canonical_config_sha256(policy_contract.to_dict()),
        f"checkpoint update {update} policy contract hash drifted",
    )
    metric = manifest.get("last_metrics")
    require(isinstance(metric, dict), f"checkpoint update {update} has no last_metrics object")
    _validate_metric(
        metric,
        update=update,
        examples_seen=trainer_state.examples_seen,
        objective=policy_contract.objective,
    )
    total_updates = config["optimization"]["total_updates"]
    require(manifest.get("configured_total_updates") == total_updates, f"checkpoint update {update} budget drifted")
    require(
        manifest.get("complete") is (update == total_updates), f"checkpoint update {update} completion flag drifted"
    )
    require(
        type(manifest.get("run_seed")) is int and manifest["run_seed"] == config["run"]["seed"],
        f"checkpoint update {update} run seed drifted",
    )
    require(manifest.get("task") == config["run"].get("task"), f"checkpoint update {update} task drifted")
    _assert_equal(
        manifest.get("calvin_identity"),
        config["calvin_identity"],
        context=f"checkpoint update {update} CALVIN identity",
    )
    _assert_equal(
        manifest.get("calvin_source_revisions"),
        config["calvin_identity"]["calvin_source_revisions"],
        context=f"checkpoint update {update} CALVIN source revisions",
    )
    _assert_equal(
        manifest.get("camera_shapes"),
        config["calvin_identity"]["camera_shapes"],
        context=f"checkpoint update {update} camera shapes",
    )
    _assert_equal(
        manifest.get("split"), config["calvin_identity"]["split"], context=f"checkpoint update {update} split"
    )
    _assert_equal(
        manifest.get("execution_environment"),
        config["execution_environment"],
        context=f"checkpoint update {update} execution environment",
    )
    _assert_equal(
        manifest.get("execution_geometry"),
        config["execution_geometry"],
        context=f"checkpoint update {update} execution geometry",
    )
    require(manifest.get("model_id") == EXPECTED_MODEL_ID, f"checkpoint update {update} model id drifted")

    artifact_table = manifest.get("artifacts")
    require(isinstance(artifact_table, dict), f"checkpoint update {update} artifact table is invalid")
    require(
        set(artifact_table) == set(EXPECTED_ARTIFACT_PATHS), f"checkpoint update {update} artifact inventory drifted"
    )
    artifacts: dict[str, Path] = {}
    artifact_identities: dict[str, tuple[int, ...]] = {}
    observed_paths: set[str] = set()
    for name, expected_path in EXPECTED_ARTIFACT_PATHS.items():
        entry = artifact_table[name]
        require(
            isinstance(entry, dict) and set(entry) == {"path", "bytes", "sha256"}, f"artifact {name} schema drifted"
        )
        require(entry["path"] == expected_path, f"artifact {name} path drifted")
        require(type(entry["bytes"]) is int and entry["bytes"] > 0, f"artifact {name} byte count is invalid")
        digest = _require_sha256(entry["sha256"], f"artifact {name} SHA-256")
        require(expected_path not in observed_paths, f"duplicate checkpoint artifact path: {expected_path}")
        observed_paths.add(expected_path)
        artifact_path = _contained_real_file(path, expected_path, f"artifact {name}")
        artifact_raw, artifact_stat = _stable_regular_file_bytes(
            artifact_path,
            context=f"checkpoint update {update} artifact {name}",
            require_single_link=True,
        )
        require(artifact_stat.st_size == entry["bytes"], f"artifact {name} byte count mismatch")
        require(_sha256_bytes(artifact_raw) == digest, f"artifact {name} hash mismatch")
        artifacts[name] = artifact_path
        artifact_identities[name] = _stable_identity(artifact_stat)

    rank_hashes = manifest.get("training_rank_state_sha256")
    require(
        isinstance(rank_hashes, list)
        and len(rank_hashes) == EXPECTED_WORLD_SIZE
        and all(_valid_sha256(value) for value in rank_hashes),
        f"checkpoint update {update} rank-state hash inventory drifted",
    )
    for rank in range(EXPECTED_WORLD_SIZE):
        require(
            rank_hashes[rank] == artifact_table[f"training_rank_{rank:03d}"]["sha256"],
            f"checkpoint update {update} rank {rank} hashes disagree",
        )
    return CheckpointAudit(
        path=path,
        update=update,
        manifest=manifest,
        manifest_bytes=len(raw),
        manifest_identity=_stable_identity(manifest_identity),
        manifest_sha256=_sha256_bytes(raw),
        artifacts=artifacts,
        artifact_identities=artifact_identities,
    )


def _read_metrics(path: Path) -> tuple[bytes, tuple[dict[str, Any], ...], os.stat_result]:
    raw, identity = _stable_regular_file_bytes(path, context="metrics history")
    require(len(raw) <= _MAX_JSON_BYTES, "metrics history is unreasonably large")
    require(raw.endswith(b"\n"), "metrics history must end at a committed newline")
    lines = raw.splitlines(keepends=True)
    require(len(lines) == EXPECTED_UPDATE, "metrics history must contain exactly updates one and two")
    metrics: list[dict[str, Any]] = []
    for index, line in enumerate(lines, start=1):
        require(
            line.endswith(b"\n") and not line.endswith(b"\r\n"), f"metrics line {index} has a non-canonical terminator"
        )
        value = _strict_json_from_bytes(line[:-1], source=path)
        require(isinstance(value, dict), f"metrics line {index} is not an object")
        expected = (json.dumps(value, allow_nan=False, sort_keys=True) + "\n").encode("utf-8")
        require(line == expected, f"metrics line {index} is not canonical JSON")
        metrics.append(value)
    return raw, tuple(metrics), identity


def _audit_run(
    role: str,
    root_value: str | Path,
    *,
    expected_journal_sha256: str,
    expected_run_uuid: str,
    expected_config_sha256: str,
    source_identity: Mapping[str, str],
) -> RunAudit:
    root = _require_real_directory(Path(root_value), f"{role} run root")
    expected_journal_sha256 = _require_sha256(expected_journal_sha256, f"{role} expected journal SHA-256")
    expected_run_uuid = _require_uuid(expected_run_uuid, f"{role} expected run UUID")
    expected_config_sha256 = _require_sha256(expected_config_sha256, "expected config SHA-256")

    journal_path = root / "run_journal.json"
    journal_raw, journal_value, journal_identity = _read_strict_json(journal_path, f"{role} run journal")
    require(_sha256_bytes(journal_raw) == expected_journal_sha256, f"{role} run journal external SHA-256 mismatch")
    require(isinstance(journal_value, dict), f"{role} run journal is not an object")
    require(journal_value.get("schema") == RUN_JOURNAL_SCHEMA, f"{role} run journal schema drifted")
    require(journal_raw == canonical_json_bytes(journal_value), f"{role} run journal is not canonical JSON")
    journal = load_run_journal(root, expected_config_sha256=expected_config_sha256)
    _assert_equal(journal_value, journal.to_dict(), context=f"{role} run journal strict parse")
    require(journal.run_uuid == expected_run_uuid, f"{role} run journal UUID differs from its independent pin")
    require(journal.config_sha256 == expected_config_sha256, f"{role} run config SHA differs from its pin")
    tip = journal.latest_checkpoint
    require(tip is not None and tip.update == EXPECTED_UPDATE, f"{role} journal tip is not update two")
    assert tip is not None
    require(tip.relative_path == "checkpoints/update-000002", f"{role} journal tip path is not canonical")

    resolved_path = root / "resolved_config.json"
    resolved_raw, resolved_value, resolved_identity = _read_strict_json(resolved_path, f"{role} resolved config")
    require(isinstance(resolved_value, dict), f"{role} resolved config envelope is invalid")
    require(
        resolved_raw == _canonical_pretty_json_bytes(resolved_value), f"{role} resolved config is not canonical JSON"
    )
    config, observed_config_sha256 = load_verified_resolved_config(
        resolved_path,
        expected_sha256=expected_config_sha256,
    )
    require(observed_config_sha256 == expected_config_sha256, f"{role} resolved config hash drifted")
    _validate_config(
        config,
        config_sha256=expected_config_sha256,
        source_identity=source_identity,
    )

    checkpoint_one_path, checkpoint_two_path = _checkpoint_inventory(root)
    checkpoint_one = _audit_checkpoint(
        checkpoint_one_path,
        update=1,
        run_uuid=expected_run_uuid,
        config_sha256=expected_config_sha256,
        parent_manifest_sha256=None,
        config=config,
    )
    checkpoint_two = _audit_checkpoint(
        checkpoint_two_path,
        update=2,
        run_uuid=expected_run_uuid,
        config_sha256=expected_config_sha256,
        parent_manifest_sha256=checkpoint_one.manifest_sha256,
        config=config,
    )
    record = validate_resume_checkpoint(root, checkpoint_two_path)
    require(record == tip, f"{role} journal tip failed production resume authentication")
    require(tip.manifest_sha256 == checkpoint_two.manifest_sha256, f"{role} journal tip manifest hash drifted")
    require(
        tip.parent_manifest_sha256 == checkpoint_one.manifest_sha256,
        f"{role} journal parent lineage drifted",
    )

    for checkpoint in (checkpoint_one, checkpoint_two):
        config_artifact = checkpoint.artifacts["resolved_config"]
        config_artifact_raw, _ = _stable_regular_file_bytes(
            config_artifact,
            context=f"{role} update {checkpoint.update} resolved config artifact",
            require_single_link=True,
        )
        require(
            config_artifact_raw == resolved_raw,
            f"{role} update {checkpoint.update} checkpoint resolved-config bytes drifted",
        )

    metrics_raw, metrics, metrics_identity = _read_metrics(root / "metrics.jsonl")
    policy_objective = _policy_contract(config)["objective"]
    global_batch = config["optimization"]["global_batch_size"]
    for update, (metric, checkpoint) in enumerate(zip(metrics, (checkpoint_one, checkpoint_two), strict=True), start=1):
        _validate_metric(metric, update=update, examples_seen=update * global_batch, objective=policy_objective)
        _assert_equal(metric, checkpoint.manifest["last_metrics"], context=f"{role} update {update} metric/manifest")
    _assert_equal(metrics[-1], tip.last_metrics, context=f"{role} final metric/journal")

    return RunAudit(
        role=role,
        root=root,
        run_uuid=expected_run_uuid,
        config_sha256=expected_config_sha256,
        config=config,
        resolved_config_path=resolved_path,
        resolved_config_identity=_stable_identity(resolved_identity),
        resolved_config_raw_sha256=_sha256_bytes(resolved_raw),
        journal_sha256=expected_journal_sha256,
        journal_bytes=len(journal_raw),
        journal_identity=_stable_identity(journal_identity),
        metrics_sha256=_sha256_bytes(metrics_raw),
        metrics_bytes=len(metrics_raw),
        metrics_identity=_stable_identity(metrics_identity),
        metrics=metrics,
        checkpoints=(checkpoint_one, checkpoint_two),
    )


def _normalized_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    value = copy.deepcopy(manifest)
    value["run_uuid"] = "<independently-authenticated-run-uuid>"
    value["parent_manifest_sha256"] = "<independently-authenticated-parent-lineage>"
    metrics = value.get("last_metrics")
    require(isinstance(metrics, dict) and "update_seconds" in metrics, "manifest has no normalizable update_seconds")
    metrics["update_seconds"] = "<wall-clock-seconds>"
    rank_hashes = value.get("training_rank_state_sha256")
    require(isinstance(rank_hashes, list) and len(rank_hashes) == EXPECTED_WORLD_SIZE, "manifest rank hashes invalid")
    value["training_rank_state_sha256"] = [f"<rank-{rank}-state-bytes>" for rank in range(EXPECTED_WORLD_SIZE)]
    artifacts = value.get("artifacts")
    require(isinstance(artifacts, dict), "manifest artifact table invalid during normalization")
    for rank in range(EXPECTED_WORLD_SIZE):
        artifacts[f"training_rank_{rank:03d}"]["sha256"] = f"<rank-{rank}-state-bytes>"
    return value


def _normalized_metrics(metrics: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized = copy.deepcopy(list(metrics))
    for update, metric in enumerate(normalized, start=1):
        require(set(METRIC_IGNORED_FIELDS).issubset(metric), f"metric update {update} lacks update_seconds")
        for field in METRIC_IGNORED_FIELDS:
            del metric[field]
    return normalized


def _files_are_exact(left: Path, right: Path, *, context: str) -> tuple[int, str]:
    left_raw, left_stat = _stable_regular_file_bytes(left, context=f"{context} left", require_single_link=True)
    right_raw, right_stat = _stable_regular_file_bytes(right, context=f"{context} right", require_single_link=True)
    require(left_stat.st_size == right_stat.st_size, f"{context} byte lengths differ")
    require(left_raw == right_raw, f"{context} bytes differ")
    return left_stat.st_size, _sha256_bytes(left_raw)


def _expected_learning_rate_scale(update: int, config: Mapping[str, Any]) -> float:
    optimization = config["optimization"]
    warmup = optimization["warmup_updates"]
    total = optimization["total_updates"]
    final_scale = optimization["final_learning_rate_scale"]
    if warmup and update < warmup:
        initial = 1.0 / (warmup + 1)
        return initial + (1.0 - initial) * update / warmup
    progress = min(1.0, (update - warmup) / (total - 1 - warmup))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return final_scale + (1.0 - final_scale) * cosine


_EXPECTED_INTERFACE_PARAMETER_NAMES = (
    "action_projector.horizon_embedding",
    "action_projector.action_type_embedding",
    "action_projector.action_projection.weight",
    "action_projector.action_projection.bias",
    "action_projector.timestep_mlp.net.0.weight",
    "action_projector.timestep_mlp.net.0.bias",
    "action_projector.timestep_mlp.net.2.weight",
    "action_projector.timestep_mlp.net.2.bias",
    "action_projector.state_mlp.net.0.weight",
    "action_projector.state_mlp.net.0.bias",
    "action_projector.state_mlp.net.2.weight",
    "action_projector.state_mlp.net.2.bias",
    "velocity_head.projection.weight",
    "velocity_head.projection.bias",
)


def _expected_lora_parameter_shapes(rank: int) -> dict[str, list[int]]:
    sliding_layers = set(EXPECTED_MODEL_SLIDING_ATTENTION_LAYERS)
    shapes: dict[str, list[int]] = {}
    for layer in range(EXPECTED_MODEL_DECODER_LAYERS):
        sliding = layer in sliding_layers
        for projection in ("q_proj", "k_proj", "v_proj", "o_proj"):
            if projection == "v_proj" and not sliding:
                continue
            if projection == "q_proj":
                input_features = EXPECTED_MODEL_HIDDEN_SIZE
                output_features = (
                    EXPECTED_MODEL_SLIDING_ATTENTION_WIDTH if sliding else EXPECTED_MODEL_FULL_ATTENTION_WIDTH
                )
            elif projection in {"k_proj", "v_proj"}:
                input_features = EXPECTED_MODEL_HIDDEN_SIZE
                output_features = EXPECTED_MODEL_SLIDING_KV_WIDTH if sliding else EXPECTED_MODEL_FULL_KV_WIDTH
            else:
                input_features = (
                    EXPECTED_MODEL_SLIDING_ATTENTION_WIDTH if sliding else EXPECTED_MODEL_FULL_ATTENTION_WIDTH
                )
                output_features = EXPECTED_MODEL_HIDDEN_SIZE
            prefix = f"base_model.model.model.decoder.layers.{layer}.self_attn.{projection}"
            shapes[f"{prefix}.lora_A.default.weight"] = [rank, input_features]
            shapes[f"{prefix}.lora_B.default.weight"] = [output_features, rank]
    expected_parameters = 2 * (3 * EXPECTED_MODEL_DECODER_LAYERS + len(sliding_layers))
    require(len(shapes) == expected_parameters, "pinned DiffusionGemma LoRA parameter topology is invalid")
    return shapes


def _expected_interface_parameter_shapes(config: Mapping[str, Any], hidden_size: int) -> dict[str, list[int]]:
    action = config["action"]
    benchmark = config["benchmark"]
    action_horizon = action["horizon"]
    action_dimension = action["dimension"]
    timestep_dimension = action["timestep_embedding_dimension"]
    state_dimension = benchmark["state_dimension"]
    return {
        "action_projector.horizon_embedding": [action_horizon, hidden_size],
        "action_projector.action_type_embedding": [hidden_size],
        "action_projector.action_projection.weight": [hidden_size, action_dimension],
        "action_projector.action_projection.bias": [hidden_size],
        "action_projector.timestep_mlp.net.0.weight": [hidden_size, timestep_dimension],
        "action_projector.timestep_mlp.net.0.bias": [hidden_size],
        "action_projector.timestep_mlp.net.2.weight": [hidden_size, hidden_size],
        "action_projector.timestep_mlp.net.2.bias": [hidden_size],
        "action_projector.state_mlp.net.0.weight": [hidden_size, state_dimension],
        "action_projector.state_mlp.net.0.bias": [hidden_size],
        "action_projector.state_mlp.net.2.weight": [hidden_size, hidden_size],
        "action_projector.state_mlp.net.2.bias": [hidden_size],
        "velocity_head.projection.weight": [action_dimension, hidden_size],
        "velocity_head.projection.bias": [action_dimension],
    }


def _validate_calvin_optimizer_parameter_inventory(
    value: Any,
    *,
    run: RunAudit,
    rank: int,
) -> str:
    context = f"{run.role} rank {rank} optimizer parameter inventory"
    require(
        isinstance(value, Mapping) and set(value) == {"groups", "schema"},
        f"{context} schema differs",
    )
    assert isinstance(value, Mapping)
    require(value["schema"] == OPTIMIZER_PARAMETER_SCHEMA, f"{context} version differs")
    groups = value["groups"]
    require(isinstance(groups, list) and len(groups) == 2, f"{context} groups differ")
    require(
        all(isinstance(group, Mapping) and set(group) == {"name", "parameters"} for group in groups),
        f"{context} group schema differs",
    )
    lora_group, interface_group = groups
    require(
        lora_group["name"] == "lora" and interface_group["name"] == "interface",
        f"{context} group ordering differs",
    )
    all_parameters: list[Mapping[str, Any]] = []
    for group in groups:
        parameters = group["parameters"]
        require(isinstance(parameters, list) and parameters, f"{context} contains an empty group")
        require(
            all(
                isinstance(parameter, Mapping) and set(parameter) == {"dtype", "name", "requires_grad", "shape"}
                for parameter in parameters
            ),
            f"{context} entry schema differs",
        )
        all_parameters.extend(parameters)
    names = [parameter["name"] for parameter in all_parameters]
    require(
        all(isinstance(name, str) and name for name in names) and len(names) == len(set(names)),
        f"{context} names are invalid",
    )
    require(
        all(
            parameter["dtype"] == "torch.float32"
            and parameter["requires_grad"] is True
            and isinstance(parameter["shape"], list)
            and all(type(dimension) is int and dimension > 0 for dimension in parameter["shape"])
            for parameter in all_parameters
        ),
        f"{context} dtype, trainability, or shapes differ",
    )

    lora_parameters = lora_group["parameters"]
    expected_lora_shapes = _expected_lora_parameter_shapes(run.config["lora"]["rank"])
    lora_names = tuple(parameter["name"] for parameter in lora_parameters)
    require(
        lora_names == tuple(expected_lora_shapes),
        f"{context} LoRA ordering or complete pinned topology differs",
    )
    require(
        all(parameter["shape"] == expected_lora_shapes[parameter["name"]] for parameter in lora_parameters),
        f"{context} LoRA shapes differ from the pinned model topology",
    )

    interface_parameters = interface_group["parameters"]
    interface_names = tuple(parameter["name"] for parameter in interface_parameters)
    require(interface_names == _EXPECTED_INTERFACE_PARAMETER_NAMES, f"{context} interface ordering differs")
    expected_shapes = _expected_interface_parameter_shapes(run.config, EXPECTED_MODEL_HIDDEN_SIZE)
    require(
        all(parameter["shape"] == expected_shapes[parameter["name"]] for parameter in interface_parameters),
        f"{context} interface shapes differ",
    )
    try:
        return optimizer_parameter_inventory_sha256(value)
    except (TypeError, ValueError) as exc:
        raise QualificationError(f"{context} is invalid: {exc}") from exc


def _validate_optimizer_scheduler_state(
    payload: Mapping[str, Any],
    *,
    checkpoint: CheckpointAudit,
    run: RunAudit,
    rank: int,
) -> None:
    optimizer = payload["optimizer"]
    scheduler = payload["scheduler"]
    inventory = payload["optimizer_parameter_inventory"]
    inventory_sha256 = _validate_calvin_optimizer_parameter_inventory(inventory, run=run, rank=rank)
    run_contract = payload["run_contract"]
    require(
        isinstance(run_contract, Mapping) and run_contract.get("optimizer_parameter_schema_sha256") == inventory_sha256,
        f"{run.role} rank {rank} optimizer inventory hash differs from its run contract",
    )
    try:
        validate_optimizer_state_dict(
            optimizer,
            inventory,
            expected_update=checkpoint.update,
        )
    except (TypeError, ValueError) as exc:
        raise QualificationError(
            f"{run.role} rank {rank} optimizer schema/inventory binding is invalid: {exc}"
        ) from exc
    require(
        isinstance(optimizer, Mapping) and set(optimizer) == {"param_groups", "state"},
        f"{run.role} rank {rank} optimizer schema differs",
    )
    groups = optimizer["param_groups"]
    states = optimizer["state"]
    require(isinstance(groups, list) and len(groups) == 2, f"{run.role} rank {rank} optimizer groups differ")
    expected_group_fields = {
        "amsgrad",
        "betas",
        "capturable",
        "decoupled_weight_decay",
        "differentiable",
        "eps",
        "foreach",
        "fused",
        "initial_lr",
        "lr",
        "maximize",
        "name",
        "params",
        "weight_decay",
    }
    optimization = run.config["optimization"]
    bases = [optimization["lora_learning_rate"], optimization["interface_learning_rate"]]
    names = ["lora", "interface"]
    expected_scale = _expected_learning_rate_scale(checkpoint.update, run.config)
    parameter_ids: list[int] = []
    for index, group in enumerate(groups):
        require(
            isinstance(group, dict) and set(group) == expected_group_fields,
            f"{run.role} rank {rank} optimizer group {index} schema differs",
        )
        expected_values = {
            "amsgrad": False,
            "betas": (optimization["adam_beta1"], optimization["adam_beta2"]),
            "capturable": False,
            "decoupled_weight_decay": True,
            "differentiable": False,
            "eps": optimization["adam_epsilon"],
            "foreach": None,
            "fused": None,
            "initial_lr": bases[index],
            "maximize": False,
            "name": names[index],
            "weight_decay": optimization["weight_decay"],
        }
        for field, expected in expected_values.items():
            require(
                type(group[field]) is type(expected) and group[field] == expected,
                f"{run.role} rank {rank} optimizer group {index} {field} drifted",
            )
        expected_lr = bases[index] * expected_scale
        require(
            type(group["lr"]) is float and math.isclose(group["lr"], expected_lr, rel_tol=1e-12, abs_tol=0.0),
            f"{run.role} rank {rank} optimizer group {index} learning rate drifted",
        )
        params = group["params"]
        require(
            isinstance(params, list)
            and params
            and all(type(parameter) is int and parameter >= 0 for parameter in params),
            f"{run.role} rank {rank} optimizer group {index} parameter inventory is invalid",
        )
        parameter_ids.extend(params)
    require(
        len(parameter_ids) == len(set(parameter_ids)) and sorted(parameter_ids) == list(range(len(parameter_ids))),
        f"{run.role} rank {rank} optimizer positional parameter inventory differs",
    )
    require(
        isinstance(states, Mapping) and set(states) == set(parameter_ids),
        f"{run.role} rank {rank} optimizer state coverage differs",
    )
    for parameter, state_value in states.items():
        require(
            isinstance(state_value, Mapping) and set(state_value) == {"exp_avg", "exp_avg_sq", "step"},
            f"{run.role} rank {rank} optimizer parameter {parameter} state schema differs",
        )
        step = state_value["step"]
        exp_avg = state_value["exp_avg"]
        exp_avg_sq = state_value["exp_avg_sq"]
        require(
            isinstance(step, torch.Tensor)
            and step.device.type == "cpu"
            and step.dtype == torch.float32
            and step.shape == torch.Size([])
            and step.item() == checkpoint.update,
            f"{run.role} rank {rank} optimizer parameter {parameter} step drifted",
        )
        require(
            isinstance(exp_avg, torch.Tensor)
            and isinstance(exp_avg_sq, torch.Tensor)
            and exp_avg.device.type == exp_avg_sq.device.type == "cpu"
            and exp_avg.dtype == exp_avg_sq.dtype == torch.float32
            and exp_avg.layout == exp_avg_sq.layout == torch.strided
            and exp_avg.numel() > 0
            and exp_avg.shape == exp_avg_sq.shape,
            f"{run.role} rank {rank} optimizer parameter {parameter} moments drifted",
        )
    expected_scheduler_fields = {
        "_get_lr_called_within_step",
        "_is_initial",
        "_last_lr",
        "_step_count",
        "base_lrs",
        "last_epoch",
        "lr_lambdas",
    }
    require(
        isinstance(scheduler, Mapping) and set(scheduler) == expected_scheduler_fields,
        f"{run.role} rank {rank} scheduler schema differs",
    )
    require(scheduler["base_lrs"] == bases, f"{run.role} rank {rank} scheduler base learning rates drifted")
    require(scheduler["last_epoch"] == checkpoint.update, f"{run.role} rank {rank} scheduler epoch drifted")
    require(scheduler["_step_count"] == checkpoint.update + 1, f"{run.role} rank {rank} scheduler step count drifted")
    require(scheduler["_is_initial"] is False, f"{run.role} rank {rank} scheduler initial flag drifted")
    require(
        scheduler["_get_lr_called_within_step"] is False,
        f"{run.role} rank {rank} scheduler get-lr flag drifted",
    )
    require(scheduler["lr_lambdas"] == [None, None], f"{run.role} rank {rank} scheduler lambda schema drifted")
    require(
        scheduler["_last_lr"] == [group["lr"] for group in groups],
        f"{run.role} rank {rank} scheduler/optimizer learning rates disagree",
    )


def _validate_rank_run_contract(
    contract: Any,
    *,
    checkpoint: CheckpointAudit,
    run: RunAudit,
    rank: int,
) -> dict[str, str]:
    require(
        isinstance(contract, dict)
        and set(contract) == EXPECTED_RUN_CONTRACT_FIELDS
        and all(isinstance(key, str) and isinstance(value, str) for key, value in contract.items()),
        f"{run.role} rank {rank} run contract inventory is invalid",
    )
    assert isinstance(contract, dict)
    identity = run.config["calvin_identity"]
    revisions = identity["calvin_source_revisions"]
    geometry = run.config["execution_geometry"]
    expected = {
        "action_adapter": identity["action_adapter"],
        "archive_bytes": str(identity["archive_bytes"]),
        "archive_sha256": identity["archive_sha256"],
        "calvin_env_revision": revisions["calvin_env"],
        "calvin_revision": revisions["calvin"],
        "calvin_source_revisions_sha256": identity["calvin_source_revisions_sha256"],
        "calvin_tacto_revision": revisions["tacto"],
        "camera_shapes_sha256": identity["camera_shapes_sha256"],
        "central_directory_sha256": identity["central_directory_sha256"],
        "config_sha256": run.config_sha256,
        "dataset_manifest_file_sha256": identity["dataset_manifest_file_sha256"],
        "dataset_manifest_schema": identity["dataset_manifest_schema"],
        "dataset_manifest_sha256": identity["dataset_manifest_sha256"],
        "execution_environment_sha256": canonical_config_sha256(run.config["execution_environment"]),
        "expert_batch_isolation": geometry["expert_batch_isolation"],
        "experts_implementation": geometry["experts_implementation"],
        "fixed_physical_prefix_width": str(geometry["fixed_physical_prefix_width"]),
        "member_index_bytes": str(identity["member_index"]["bytes"]),
        "member_index_path": identity["member_index"]["path"],
        "member_index_schema": identity["member_index"]["schema"],
        "member_index_sha256": identity["member_index"]["sha256"],
        "member_inventory_sha256": identity["member_inventory_sha256"],
        "metadata_sha256": identity["metadata_sha256"],
        "model_revision": run.config["model"]["revision"],
        "model_tree_sha256": run.config["artifact_trees"]["model_tree_sha256"],
        "normalization_sha256": identity["normalization_sha256"],
        "physical_batch_size": str(run.config["optimization"]["physical_batch_size"]),
        "policy_contract_sha256": canonical_config_sha256(_policy_contract(run.config)),
        "prefix_geometry_content_sha256": geometry["prefix_geometry_content_sha256"],
        "protocol": CALVIN_PROTOCOL,
        "reader_schema": identity["reader_schema"],
        "run_uuid": run.run_uuid,
        "source_tree_sha256": run.config["source_tree_sha256"],
        "split_sha256": identity["split_sha256"],
        "state_adapter": identity["state_adapter"],
        "storage_identity_sha256": identity["storage_identity_sha256"],
        "storage_mode": identity["storage_mode"],
        "train_episode_sha256": identity["split"]["train_episode_sha256"],
        "training_instruction_inventory_sha256": run.config["training_instruction_inventory_sha256"],
        "validation_episode_sha256": identity["split"]["validation_episode_sha256"],
    }
    optimizer_schema = _require_sha256(
        contract["optimizer_parameter_schema_sha256"],
        f"{run.role} rank {rank} optimizer parameter schema SHA-256",
    )
    expected["optimizer_parameter_schema_sha256"] = optimizer_schema
    _assert_equal(contract, dict(sorted(expected.items())), context=f"{run.role} rank {rank} full run contract")
    for key, expected_value in contract.items():
        observed = checkpoint.manifest.get(key)
        if key in INTEGER_RUN_CONTRACT_FIELDS:
            matches = type(observed) is int and str(observed) == expected_value
        else:
            matches = observed == expected_value
        require(matches, f"{run.role} rank {rank} run contract/manifest drift at {key}")
    return contract


def _load_rank_state(checkpoint: CheckpointAudit, *, rank: int, run: RunAudit) -> dict[str, Any]:
    name = f"training_rank_{rank:03d}"
    path = checkpoint.artifacts[name]
    expected_hash = checkpoint.manifest["artifacts"][name]["sha256"]
    raw, _ = _stable_regular_file_bytes(
        path,
        context=f"{run.role} update {checkpoint.update} rank {rank} state",
        require_single_link=True,
    )
    require(_sha256_bytes(raw) == expected_hash, f"{run.role} rank {rank} state hash drifted before load")
    try:
        payload = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
    except Exception as exc:
        raise QualificationError(f"cannot load authenticated {run.role} rank {rank} state on CPU: {exc}") from exc
    required_fields = {
        "schema",
        "rank",
        "world_size",
        "trainer_state",
        "optimizer",
        "optimizer_parameter_inventory",
        "scheduler",
        "rng",
        "run_contract",
    }
    require(
        isinstance(payload, dict) and set(payload) == required_fields, f"{run.role} rank {rank} state schema drifted"
    )
    require(payload["schema"] == TRAINING_RANK_STATE_SCHEMA, f"{run.role} rank {rank} state version drifted")
    require(type(payload["rank"]) is int and payload["rank"] == rank, f"{run.role} rank {rank} identity drifted")
    require(
        type(payload["world_size"]) is int and payload["world_size"] == EXPECTED_WORLD_SIZE,
        f"{run.role} rank {rank} topology drifted",
    )
    TrainerState.from_dict(payload["trainer_state"])
    _assert_equal(
        payload["trainer_state"],
        checkpoint.manifest["trainer_state"],
        context=f"{run.role} rank {rank} trainer-state/manifest",
    )
    require(isinstance(payload["optimizer"], Mapping), f"{run.role} rank {rank} optimizer state is invalid")
    require(isinstance(payload["scheduler"], Mapping), f"{run.role} rank {rank} scheduler state is invalid")
    _validate_optimizer_scheduler_state(payload, checkpoint=checkpoint, run=run, rank=rank)
    rng = payload["rng"]
    require(
        isinstance(rng, Mapping) and set(rng) == {"python", "numpy", "torch_cpu", "torch_cuda", "cuda_device_index"},
        f"{run.role} rank {rank} RNG inventory drifted",
    )
    require(
        type(rng["cuda_device_index"]) is int and rng["cuda_device_index"] == rank,
        f"{run.role} rank {rank} CUDA RNG identity drifted",
    )
    numpy_rng = rng["numpy"]
    require(
        isinstance(numpy_rng, Mapping)
        and set(numpy_rng) == {"bit_generator", "keys", "position", "has_gauss", "cached_gaussian"},
        f"{run.role} rank {rank} NumPy RNG inventory drifted",
    )
    require(
        isinstance(numpy_rng["keys"], torch.Tensor) and numpy_rng["keys"].dtype == torch.uint32,
        f"{run.role} rank {rank} NumPy RNG key dtype drifted",
    )
    require(
        isinstance(rng["torch_cpu"], torch.Tensor)
        and rng["torch_cpu"].dtype == torch.uint8
        and isinstance(rng["torch_cuda"], torch.Tensor)
        and rng["torch_cuda"].dtype == torch.uint8,
        f"{run.role} rank {rank} Torch RNG state drifted",
    )
    _validate_rank_run_contract(payload["run_contract"], checkpoint=checkpoint, run=run, rank=rank)
    return payload


def _compare_rank_states(left: RunAudit, right: RunAudit, *, checkpoint_index: int) -> dict[str, Any]:
    records: dict[str, Any] = {}
    require(checkpoint_index in {0, 1}, "rank-state checkpoint index must select update one or two")
    left_checkpoint = left.checkpoints[checkpoint_index]
    right_checkpoint = right.checkpoints[checkpoint_index]
    require(
        left_checkpoint.update == right_checkpoint.update == checkpoint_index + 1,
        "rank-state checkpoint update identity drifted",
    )
    update = left_checkpoint.update
    for rank in range(EXPECTED_WORLD_SIZE):
        left_payload = _load_rank_state(left_checkpoint, rank=rank, run=left)
        right_payload = _load_rank_state(right_checkpoint, rank=rank, run=right)
        left_contract = {key: value for key, value in left_payload["run_contract"].items() if key != "run_uuid"}
        right_contract = {key: value for key, value in right_payload["run_contract"].items() if key != "run_uuid"}
        left_normalized = {**left_payload, "run_contract": left_contract}
        right_normalized = {**right_payload, "run_contract": right_contract}
        _assert_equal(
            left_normalized,
            right_normalized,
            context=f"logical update {update} rank {rank} training state",
        )
        tensors, tensor_bytes = _tensor_stats(left_normalized)
        section_sha256 = {
            "optimizer": _logical_digest(left_payload["optimizer"]),
            "optimizer_parameter_inventory": _logical_digest(left_payload["optimizer_parameter_inventory"]),
            "rng": _logical_digest(left_payload["rng"]),
            "run_contract_without_run_uuid": _logical_digest(left_contract),
            "scheduler": _logical_digest(left_payload["scheduler"]),
            "trainer_state": _logical_digest(left_payload["trainer_state"]),
        }
        require(
            section_sha256
            == {
                "optimizer": _logical_digest(right_payload["optimizer"]),
                "optimizer_parameter_inventory": _logical_digest(right_payload["optimizer_parameter_inventory"]),
                "rng": _logical_digest(right_payload["rng"]),
                "run_contract_without_run_uuid": _logical_digest(right_contract),
                "scheduler": _logical_digest(right_payload["scheduler"]),
                "trainer_state": _logical_digest(right_payload["trainer_state"]),
            },
            f"logical update {update} rank {rank} section digests differ",
        )
        records[f"rank_{rank}"] = {
            "differences": 0,
            "section_sha256": section_sha256,
            "tensor_bytes": tensor_bytes,
            "tensors": tensors,
        }
        del left_payload, right_payload, left_normalized, right_normalized
    return records


def _run_report(run: RunAudit) -> dict[str, Any]:
    return {
        "checkpoints": [
            {
                "artifact_inventory_sha256": canonical_config_sha256(checkpoint.manifest["artifacts"]),
                "manifest_bytes": checkpoint.manifest_bytes,
                "manifest_sha256": checkpoint.manifest_sha256,
                "parent_manifest_sha256": checkpoint.manifest["parent_manifest_sha256"],
                "relative_path": checkpoint.path.relative_to(run.root).as_posix(),
                "update": checkpoint.update,
            }
            for checkpoint in run.checkpoints
        ],
        "config_sha256": run.config_sha256,
        "journal": {"bytes": run.journal_bytes, "sha256": run.journal_sha256},
        "metrics": {"bytes": run.metrics_bytes, "sha256": run.metrics_sha256},
        "resolved_config_raw_sha256": run.resolved_config_raw_sha256,
        "run_root": str(run.root),
        "run_uuid": run.run_uuid,
    }


def _reauthenticate_run_snapshot(run: RunAudit) -> None:
    """Reject input mutation between the initial audit and report finalization."""

    journal_path = _require_real_file(run.root / "run_journal.json", f"{run.role} run journal")
    metrics_path = _require_real_file(run.root / "metrics.jsonl", f"{run.role} metrics history")
    resolved_path = _require_real_file(run.resolved_config_path, f"{run.role} resolved config")
    journal_raw, journal_identity = _stable_regular_file_bytes(
        journal_path,
        context=f"{run.role} run journal reauthentication",
    )
    metrics_raw, metrics_identity = _stable_regular_file_bytes(
        metrics_path,
        context=f"{run.role} metrics reauthentication",
    )
    resolved_raw, resolved_identity = _stable_regular_file_bytes(
        resolved_path,
        context=f"{run.role} config reauthentication",
    )
    require(
        _stable_identity(journal_identity) == run.journal_identity,
        f"{run.role} journal file identity changed during qualification",
    )
    require(
        _stable_identity(metrics_identity) == run.metrics_identity,
        f"{run.role} metrics file identity changed during qualification",
    )
    require(
        _stable_identity(resolved_identity) == run.resolved_config_identity,
        f"{run.role} resolved config file identity changed during qualification",
    )
    require(_sha256_bytes(journal_raw) == run.journal_sha256, f"{run.role} journal changed during qualification")
    require(_sha256_bytes(metrics_raw) == run.metrics_sha256, f"{run.role} metrics changed during qualification")
    require(
        _sha256_bytes(resolved_raw) == run.resolved_config_raw_sha256,
        f"{run.role} resolved config changed during qualification",
    )
    _checkpoint_inventory(run.root)
    for checkpoint in run.checkpoints:
        manifest_path = _require_real_file(
            checkpoint.path / "manifest.json",
            f"{run.role} update {checkpoint.update} manifest",
        )
        manifest_raw, manifest_identity = _stable_regular_file_bytes(
            manifest_path,
            context=f"{run.role} update {checkpoint.update} manifest reauthentication",
        )
        require(
            _stable_identity(manifest_identity) == checkpoint.manifest_identity,
            f"{run.role} update {checkpoint.update} manifest file identity changed during qualification",
        )
        require(
            _sha256_bytes(manifest_raw) == checkpoint.manifest_sha256,
            f"{run.role} update {checkpoint.update} manifest changed during qualification",
        )
        for name, path in checkpoint.artifacts.items():
            entry = checkpoint.manifest["artifacts"][name]
            observed_path = _contained_real_file(
                checkpoint.path,
                entry["path"],
                f"{run.role} update {checkpoint.update} artifact {name}",
            )
            require(observed_path == path, f"{run.role} artifact {name} path changed during qualification")
            artifact_raw, artifact_stat = _stable_regular_file_bytes(
                path,
                context=f"{run.role} update {checkpoint.update} artifact {name} reauthentication",
                require_single_link=True,
            )
            require(
                _stable_identity(artifact_stat) == checkpoint.artifact_identities[name],
                f"{run.role} artifact {name} file identity changed during qualification",
            )
            require(artifact_stat.st_size == entry["bytes"], f"{run.role} artifact {name} size changed")
            require(
                _sha256_bytes(artifact_raw) == entry["sha256"],
                f"{run.role} artifact {name} changed during qualification",
            )


def finalized_report(value: dict[str, Any]) -> dict[str, Any]:
    require("report_sha256" not in value, "unfinalized report already has report_sha256")
    require(value.get("schema") == REPORT_SCHEMA, "qualification report schema is not canonical")
    require(value.get("kind") == QUALIFICATION_KIND, "qualification report kind is not canonical")
    require(value.get("status") == "passed", "only a passing qualification can be finalized")
    checks = value.get("pass_criteria")
    require(
        isinstance(checks, dict) and checks and all(type(result) is bool and result for result in checks.values()),
        "qualification report contains a failed or invalid pass criterion",
    )
    report = dict(value)
    report["report_sha256"] = _sha256_bytes(canonical_json_bytes(value))
    return report


def require_report_publication_guard(value: Mapping[str, Any], *, context: str) -> None:
    """Bind one publisher phase to the report's authenticated source/runtime."""

    execution = value.get("execution")
    require(
        isinstance(execution, dict)
        and execution.get("launcher_exact_comparator_bytes_bootstrap") is True
        and execution.get("bootstrap_mode") == _LAUNCHER_BOOTSTRAP_MODE,
        "qualification report was not produced by the closed exact-byte launcher bootstrap",
    )
    reported_source_identity = value.get("source_identity")
    require(isinstance(reported_source_identity, dict), "qualification report source identity is invalid")
    current_source_identity = require_qualification_source_unchanged(
        _IMPORTED_QUALIFICATION_SOURCE_IDENTITY,
        _bootstrap_source_identity(),
        context=context,
    )
    require_qualification_source_unchanged(
        reported_source_identity,
        current_source_identity,
        context=f"report/{context}",
    )
    reported_runtime_identity = value.get("runtime_identity")
    require(isinstance(reported_runtime_identity, dict), "qualification report runtime identity is invalid")
    require_runtime_unchanged(
        reported_runtime_identity,
        qualification_runtime_identity(),
        context=context,
    )


_FAILED_PUBLICATION_BYTES = b"DUO-VLA QUALIFICATION REPORT PUBLICATION FAILED\n"


def _require_parent_directory_bound(parent: Path, directory_fd: int, expected_identity: tuple[int, int]) -> None:
    descriptor_identity = os.fstat(directory_fd)
    path_identity = os.stat(parent, follow_symlinks=False)
    require(
        stat.S_ISDIR(descriptor_identity.st_mode)
        and stat.S_ISDIR(path_identity.st_mode)
        and (descriptor_identity.st_dev, descriptor_identity.st_ino) == expected_identity
        and (path_identity.st_dev, path_identity.st_ino) == expected_identity,
        "qualification output parent directory identity changed during publication",
    )


def _invalidate_failed_publication(descriptor: int, expected_identity: tuple[int, int]) -> None:
    """Make only the publisher-owned inode non-report data; never unlink a raced name."""

    observed = os.fstat(descriptor)
    require(
        stat.S_ISREG(observed.st_mode) and (observed.st_dev, observed.st_ino) == expected_identity,
        "qualification report descriptor identity changed during failure handling",
    )
    os.ftruncate(descriptor, 0)
    view = memoryview(_FAILED_PUBLICATION_BYTES)
    offset = 0
    while view:
        written = os.pwrite(descriptor, view, offset)
        require(written > 0, "zero-byte write while invalidating failed qualification report")
        offset += written
        view = view[written:]
    os.fsync(descriptor)


def write_canonical_json_exclusive(path: Path, value: dict[str, Any]) -> Path:
    require_report_publication_guard(value, context="before qualification report publication")
    published_bytes = canonical_json_bytes(value)
    recorded = value.get("report_sha256")
    unsigned = {key: item for key, item in value.items() if key != "report_sha256"}
    require(_valid_sha256(recorded), "qualification report self-hash is invalid")
    require(_sha256_bytes(canonical_json_bytes(unsigned)) == recorded, "qualification report self-hash is invalid")
    require(path.suffix == ".json" and path.name not in {"", ".", ".."}, "qualification output must be a JSON filename")
    parent = _require_real_directory(path.parent, "qualification output parent")
    target = parent / path.name
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"qualification report already exists: {target}")
    parent_before = os.stat(parent, follow_symlinks=False)
    directory_fd = os.open(parent, os.O_RDONLY | os.O_NONBLOCK | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC)
    parent_identity = (parent_before.st_dev, parent_before.st_ino)
    publication_verified = False
    target_link_created = False
    created_identity: tuple[int, int] | None = None
    target_descriptor: int | None = None
    try:
        _require_parent_directory_bound(parent, directory_fd, parent_identity)
        try:
            target_descriptor = os.open(
                ".",
                os.O_RDWR | os.O_NONBLOCK | os.O_TMPFILE | os.O_CLOEXEC,
                0o600,
                dir_fd=directory_fd,
            )
        except OSError as exc:
            raise QualificationError(
                f"qualification output filesystem does not support secure O_TMPFILE publication: {parent}: {exc}"
            ) from exc
        target_stat = os.fstat(target_descriptor)
        require(
            stat.S_ISREG(target_stat.st_mode) and target_stat.st_nlink == 0,
            "qualification report unnamed inode is invalid",
        )
        created_identity = (target_stat.st_dev, target_stat.st_ino)
        view = memoryview(published_bytes)
        while view:
            written = os.write(target_descriptor, view)
            require(written > 0, "zero-byte write while publishing qualification report")
            view = view[written:]
        os.fsync(target_descriptor)
        _read_pinned_regular_descriptor(
            target_descriptor,
            context="qualification report unnamed descriptor before publication",
            expected_identity=created_identity,
            expected_nlink=0,
            expected_bytes=published_bytes,
        )
        descriptor_link = Path(f"/proc/self/fd/{target_descriptor}")
        proc_identity = os.stat(descriptor_link, follow_symlinks=True)
        require(
            stat.S_ISREG(proc_identity.st_mode)
            and (proc_identity.st_dev, proc_identity.st_ino) == created_identity
            and proc_identity.st_nlink == 0,
            "qualification report descriptor link identity differs",
        )
        require_report_publication_guard(
            value,
            context="after durable qualification report staging and before atomic commit",
        )
        _require_parent_directory_bound(parent, directory_fd, parent_identity)
        _read_pinned_regular_descriptor(
            target_descriptor,
            context="qualification report unnamed descriptor immediately before atomic commit",
            expected_identity=created_identity,
            expected_nlink=0,
            expected_bytes=published_bytes,
        )
        os.link(
            descriptor_link,
            target.name,
            dst_dir_fd=directory_fd,
            follow_symlinks=True,
        )
        target_link_created = True
        _verify_published_entry(
            directory_fd,
            target.name,
            target_descriptor,
            expected_identity=created_identity,
            expected_nlink=1,
            expected_bytes=published_bytes,
            context="qualification report target after atomic publication",
        )
        os.fsync(directory_fd)
        _require_parent_directory_bound(parent, directory_fd, parent_identity)
        _read_pinned_regular_descriptor(
            target_descriptor,
            context="qualification report target descriptor after atomic commit",
            expected_identity=created_identity,
            expected_nlink=1,
            expected_bytes=published_bytes,
        )
        _verify_published_entry(
            directory_fd,
            target.name,
            target_descriptor,
            expected_identity=created_identity,
            expected_nlink=1,
            expected_bytes=published_bytes,
            context="qualification report target immediately after atomic commit",
        )
        _require_parent_directory_bound(parent, directory_fd, parent_identity)
        publication_verified = True
    finally:
        try:
            if (
                not publication_verified
                and target_link_created
                and target_descriptor is not None
                and created_identity is not None
            ):
                _invalidate_failed_publication(target_descriptor, created_identity)
                os.fsync(directory_fd)
        finally:
            try:
                if target_descriptor is not None:
                    os.close(target_descriptor)
            finally:
                os.close(directory_fd)
    return target


def compare_training_runs(
    interrupted_resumed_run: str | Path,
    uninterrupted_run: str | Path,
    *,
    interrupted_resumed_journal_sha256: str,
    uninterrupted_journal_sha256: str,
    interrupted_resumed_run_uuid: str,
    uninterrupted_run_uuid: str,
    expected_config_sha256: str,
) -> dict[str, Any]:
    require(_LAUNCHER_BOOTSTRAP_AUTHENTICATED, "use the closed exact-byte qualification launcher")
    source_identity_start = require_qualification_source_unchanged(
        _IMPORTED_QUALIFICATION_SOURCE_IDENTITY,
        _bootstrap_source_identity(),
        context="between pre-production-import snapshot and comparison start",
    )
    runtime_identity_start = require_canonical_cpu_runtime(
        qualification_runtime_identity(),
        context="comparison start",
    )
    left = _audit_run(
        "interrupted_then_resumed",
        interrupted_resumed_run,
        expected_journal_sha256=interrupted_resumed_journal_sha256,
        expected_run_uuid=interrupted_resumed_run_uuid,
        expected_config_sha256=expected_config_sha256,
        source_identity=source_identity_start,
    )
    right = _audit_run(
        "uninterrupted",
        uninterrupted_run,
        expected_journal_sha256=uninterrupted_journal_sha256,
        expected_run_uuid=uninterrupted_run_uuid,
        expected_config_sha256=expected_config_sha256,
        source_identity=source_identity_start,
    )
    require(left.root != right.root, "qualification inputs must be distinct run roots")
    require(left.run_uuid != right.run_uuid, "qualification inputs must have independently-created run UUIDs")
    require(left.config_sha256 == right.config_sha256, "run semantic config hashes differ")

    config_bytes, config_raw_sha256 = _files_are_exact(
        left.resolved_config_path,
        right.resolved_config_path,
        context="root resolved config",
    )
    require(
        config_raw_sha256 == left.resolved_config_raw_sha256 == right.resolved_config_raw_sha256,
        "root resolved config changed after authentication",
    )
    exact_artifacts: dict[str, Any] = {"root_resolved_config": {"bytes": config_bytes, "sha256": config_raw_sha256}}
    for artifact_name in EXACT_TIP_ARTIFACTS:
        artifact_bytes, artifact_sha256 = _files_are_exact(
            left.checkpoints[-1].artifacts[artifact_name],
            right.checkpoints[-1].artifacts[artifact_name],
            context=f"update-2 {artifact_name}",
        )
        require(
            artifact_sha256
            == left.checkpoints[-1].manifest["artifacts"][artifact_name]["sha256"]
            == right.checkpoints[-1].manifest["artifacts"][artifact_name]["sha256"],
            f"update-2 {artifact_name} changed after authentication",
        )
        exact_artifacts[artifact_name] = {"bytes": artifact_bytes, "sha256": artifact_sha256}

    manifest_comparison: dict[str, Any] = {}
    for update, (left_checkpoint, right_checkpoint) in enumerate(
        zip(left.checkpoints, right.checkpoints, strict=True),
        start=1,
    ):
        left_normalized = _normalized_manifest(left_checkpoint.manifest)
        right_normalized = _normalized_manifest(right_checkpoint.manifest)
        _assert_equal(left_normalized, right_normalized, context=f"normalized update {update} manifest")
        digest = canonical_config_sha256(left_normalized)
        require(
            digest == canonical_config_sha256(right_normalized), f"normalized update {update} manifest hash differs"
        )
        manifest_comparison[f"update_{update}"] = {"differences": 0, "normalized_sha256": digest}

    left_metrics = _normalized_metrics(left.metrics)
    right_metrics = _normalized_metrics(right.metrics)
    _assert_equal(left_metrics, right_metrics, context="training metrics excluding update_seconds")
    metrics_digest = canonical_config_sha256({"updates": left_metrics})
    require(
        metrics_digest == canonical_config_sha256({"updates": right_metrics}),
        "normalized metrics hash differs",
    )
    rank_records = {
        f"update_{checkpoint_index + 1}": _compare_rank_states(
            left,
            right,
            checkpoint_index=checkpoint_index,
        )
        for checkpoint_index in range(EXPECTED_UPDATE)
    }
    _reauthenticate_run_snapshot(left)
    _reauthenticate_run_snapshot(right)
    source_identity = require_qualification_source_unchanged(
        source_identity_start,
        _bootstrap_source_identity(),
        context="during training reproducibility comparison",
    )
    runtime_identity = require_runtime_unchanged(
        runtime_identity_start,
        qualification_runtime_identity(),
        context="during training reproducibility comparison",
    )
    payload = {
        "comparison": {
            "exact_update_2_artifacts": exact_artifacts,
            "logical_rank_state": {
                "ignored_fields": list(RANK_STATE_IGNORED_FIELDS),
                "updates": rank_records,
            },
            "manifests": {
                "normalized_fields": list(MANIFEST_NORMALIZED_FIELDS),
                **manifest_comparison,
            },
            "metrics": {
                "differences": 0,
                "ignored_fields": list(METRIC_IGNORED_FIELDS),
                "normalized_sha256": metrics_digest,
                "updates": EXPECTED_UPDATE,
            },
        },
        "inputs": {
            "interrupted_then_resumed": _run_report(left),
            "uninterrupted": _run_report(right),
        },
        "kind": QUALIFICATION_KIND,
        "pass_criteria": {
            "all_checkpoint_artifact_hashes_authenticated": True,
            "checkpoint_lineages_authenticated": True,
            "canonical_30000_update_config_with_declared_checkpoint_every_update_smoke_override": True,
            "canonical_python_311_isolated_no_site_cpu_runtime": True,
            "closed_launcher_process_local_capability_and_exact_comparator_bytes_authenticated": True,
            "cpu_only": True,
            "exact_canonical_resolved_config_inventory_and_dynamic_bindings_validated": True,
            "full_production_run_contract_and_optimizer_scheduler_schemas_validated": True,
            "independent_run_identities_authenticated": True,
            "injectively_framed_nofollow_production_source_tree_authenticated": True,
            "input_snapshots_reauthenticated_before_publication": True,
            "journals_externally_content_addressed": True,
            "live_production_source_tree_matches_both_run_contracts": True,
            "logical_update_1_and_2_optimizer_scheduler_rng_state_exact": True,
            "manifests_exact_after_declared_normalization": True,
            "metrics_exact_excluding_only_update_seconds": True,
            "optimizer_inventory_hash_group_names_shapes_and_state_coverage_validated": True,
            "qualification_source_unchanged_from_pre_import_through_final_precommit_guard": True,
            "qualification_source_and_runtime_rechecked_by_exclusive_publisher": True,
            "update_2_trainable_and_config_artifact_bytes_exact": True,
        },
        "protocol": CALVIN_PROTOCOL,
        "schema": REPORT_SCHEMA,
        "execution": {
            "bootstrap_mode": _QUALIFICATION_BOOTSTRAP_IDENTITY["mode"],
            "launcher_exact_comparator_bytes_bootstrap": _LAUNCHER_BOOTSTRAP_AUTHENTICATED,
            "process_history_authentication": "externally_asserted_not_cryptographically_proven",
            "verified_source_phases": [
                "pre_production_import",
                "main_start",
                "comparison_start",
                "comparison_end",
                "pre_publisher_invocation",
                "publisher_pre_link",
                "publisher_post_durable_staging_pre_atomic_commit",
            ],
        },
        "scope": {
            "benchmark_environment_accessed": False,
            "dataset_accessed": False,
            "gpu_accessed": False,
            "model_cache_accessed": False,
            "official_benchmark_claim": False,
            "qualification_only": True,
        },
        "runtime_identity": runtime_identity,
        "source_identity": source_identity,
        "status": "passed",
    }
    return finalized_report(payload)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interrupted-resumed-run", required=True, type=Path)
    parser.add_argument("--interrupted-resumed-journal-sha256", required=True)
    parser.add_argument("--interrupted-resumed-run-uuid", required=True)
    parser.add_argument("--uninterrupted-run", required=True, type=Path)
    parser.add_argument("--uninterrupted-journal-sha256", required=True)
    parser.add_argument("--uninterrupted-run-uuid", required=True)
    parser.add_argument("--expected-config-sha256", required=True)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    require(_LAUNCHER_BOOTSTRAP_AUTHENTICATED, "use the closed qualification launcher")
    source_identity_start = require_qualification_source_unchanged(
        _IMPORTED_QUALIFICATION_SOURCE_IDENTITY,
        _bootstrap_source_identity(),
        context="between pre-production-import snapshot and main startup",
    )
    runtime_identity_start = require_canonical_cpu_runtime(
        qualification_runtime_identity(),
        context="main startup",
    )
    args = _parse_args()
    output_parent = _require_real_directory(args.output.parent, "qualification output parent")
    output = output_parent / args.output.name
    require(not output.exists() and not output.is_symlink(), f"qualification report already exists: {output}")
    run_roots = (
        _require_real_directory(args.interrupted_resumed_run, "interrupted/resumed run root"),
        _require_real_directory(args.uninterrupted_run, "uninterrupted run root"),
    )
    require(
        all(not output.is_relative_to(root) for root in run_roots),
        "qualification report must be published outside both immutable run roots",
    )
    report = compare_training_runs(
        run_roots[0],
        run_roots[1],
        interrupted_resumed_journal_sha256=args.interrupted_resumed_journal_sha256,
        uninterrupted_journal_sha256=args.uninterrupted_journal_sha256,
        interrupted_resumed_run_uuid=args.interrupted_resumed_run_uuid,
        uninterrupted_run_uuid=args.uninterrupted_run_uuid,
        expected_config_sha256=args.expected_config_sha256,
    )
    require_qualification_source_unchanged(
        source_identity_start,
        _bootstrap_source_identity(),
        context="immediately before publisher invocation",
    )
    require_runtime_unchanged(
        runtime_identity_start,
        qualification_runtime_identity(),
        context="immediately before publisher invocation",
    )
    published = write_canonical_json_exclusive(output, report)
    print(
        json.dumps(
            {"output": str(published), "report_sha256": report["report_sha256"], "status": "passed"},
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
