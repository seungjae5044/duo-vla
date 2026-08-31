#!/usr/bin/env python3
# ruff: noqa: I001, UP006, UP017, UP035, UP045
"""Fail-closed official CALVIN ABC->D long-horizon evaluator.

This file intentionally runs inside the pinned Python 3.8 simulator
environment.  It imports no project package: its only local dependencies are
the standalone ``calvin_bridge.py`` and ``preflight.py`` modules next to this
file.  Official CALVIN modules are imported lazily so unit tests can exercise the rollout state
machine without importing PyBullet, Hydra, or the legacy model stack.

``--mode infrastructure`` is outcome-blind by construction: it regenerates
and authenticates the official sequences, authenticates the validation-D
configuration, constructs and resets the environment once, and validates the
observation schema.  It neither constructs a policy client nor a task oracle.

``--mode official-score`` is the only mode that can call ``predict`` or the
task oracle.  It additionally requires a pre-registration JSON document with
this exact top-level shape::

    {
      "schema": "duovla-calvin-official-preregistration-v8",
      "aggregation_python_version": "3.8.20",
      "aggregator_sha256": "sha256 of aggregate_calvin_official.py",
      "benchmark_protocol": "duovla-calvin-abc-to-d-v1",
      "direct_nfe": 1,
      "evaluation_seed": 0,
      "execution_horizons": [1, 4],
      "final_checkpoint_update": 30000,
      "flow_nfes": [1, 5, 10],
      "inference_seed_domain": "duo-vla-calvin-inference-seed-v1",
      "official_output_roots": {
        "schema": "duovla-calvin-official-output-roots-v1",
        "runs": {"path": "/canonical/runs", "device": 1, "inode": 2},
        "claims": {"path": "/canonical/claims", "device": 1, "inode": 3}
      },
      "policy_warmup_calls": 2,
      "sequence_count": 1000,
      "sequence_sha256": "90191d...fd6446",
      "sequences": [...the complete canonical official sequence list...],
      "subtasks_per_sequence": 5,
      "training_seeds": [0, 1, 2],
      "cells": [{
        "cell_id": "seed-0-flow-nfe-10-k-4",
        "checkpoint": {"sha256": "..."},
        "output_claim": {
          "schema": "duovla-calvin-official-output-claim-v1",
          "claim_id_sha256": "...",
          "claim_path": "/canonical/claims/seed-0-flow-nfe-10-k-4.json",
          "output_dir": "/canonical/runs/seed-0-flow-nfe-10-k-4"
        },
        "policy": {
          "identity_sha256": "...",
          "inference_seed_behavior": "episode_identity_gaussian_noise",
          "nfe": 10,
          "objective": "rectified_flow",
          "sampler": "euler_uniform",
          "train_seed": 0
        },
        "serving_runtime_sha256": "sha256 of the frozen serving runtime contract",
        "execution_geometry": {
          "experts_implementation": "grouped_mm",
          "expert_batch_isolation": "sample_isolated_grouped_mm_v1",
          "physical_batch_size": 8,
          "fixed_physical_prefix_width": "frozen positive integer",
          "prefix_geometry_content_sha256": "externally frozen semantic SHA-256"
        },
        "execution_horizon": 1
      }, ...23 more canonical cells...],
      "runtime_attestation_sha256": "sha256 of the frozen runtime/data attestation",
      "final_freeze_token_sha256": "sha256 of the explicit CLI token"
    }

The token itself is never written to an evaluation artifact.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import hmac
import json
import math
import os
import platform
import random
import secrets
import stat
import sys
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Set, Tuple

_SCRIPT_DIR = Path(__file__).resolve().parent
_EVALUATOR_SOURCE_NAMES = ("evaluate_calvin.py", "calvin_bridge.py", "preflight.py")


def _source_file_identity(path: Path) -> Dict[str, Any]:
    flags = os.O_RDONLY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(str(path), flags)
    except OSError as exc:
        raise RuntimeError(f"cannot open evaluator source: {path}") from exc
    digest = hashlib.sha256()
    size = 0
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise RuntimeError(f"evaluator source is not a regular file: {path}")
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
            raise RuntimeError(f"evaluator source changed while being hashed: {path}")
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    return {"bytes": size, "path": str(path.resolve()), "sha256": digest.hexdigest()}


def _capture_evaluator_source_identities(script_dir: Path = _SCRIPT_DIR) -> Dict[str, Dict[str, Any]]:
    root = script_dir.resolve()
    return {name: _source_file_identity(root / name) for name in _EVALUATOR_SOURCE_NAMES}


def _require_evaluator_sources_unchanged(expected: Mapping[str, Any]) -> None:
    if set(expected) != set(_EVALUATOR_SOURCE_NAMES):
        raise RuntimeError("evaluator source snapshot names changed")
    observed = {}  # type: Dict[str, Dict[str, Any]]
    for name in _EVALUATOR_SOURCE_NAMES:
        identity = expected[name]
        if not isinstance(identity, Mapping) or set(identity) != {"bytes", "path", "sha256"}:
            raise RuntimeError(f"evaluator source identity {name} is invalid")
        path = identity["path"]
        if not isinstance(path, str):
            raise RuntimeError(f"evaluator source identity {name} path is invalid")
        observed[name] = _source_file_identity(Path(path))
    if observed != expected:
        raise RuntimeError("evaluator sources changed after import-time snapshot")


def _require_imported_local_module(module: Any, source_name: str, expected: Mapping[str, Any]) -> None:
    """Bind an imported local module to the exact pre-import source snapshot."""

    if source_name not in _EVALUATOR_SOURCE_NAMES or source_name == "evaluate_calvin.py":
        raise RuntimeError(f"invalid evaluator dependency name: {source_name}")
    identity = expected.get(source_name)
    if not isinstance(identity, Mapping) or set(identity) != {"bytes", "path", "sha256"}:
        raise RuntimeError(f"evaluator dependency identity {source_name} is invalid")
    module_file = getattr(module, "__file__", None)
    module_spec = getattr(module, "__spec__", None)
    spec_origin = getattr(module_spec, "origin", None)
    if not isinstance(module_file, str) or Path(module_file).resolve() != Path(identity["path"]):
        raise RuntimeError(f"imported evaluator dependency {source_name} has an unexpected __file__")
    if not isinstance(spec_origin, str) or Path(spec_origin).resolve() != Path(identity["path"]):
        raise RuntimeError(f"imported evaluator dependency {source_name} has an unexpected spec origin")
    if _source_file_identity(Path(module_file)) != identity:
        raise RuntimeError(f"imported evaluator dependency {source_name} differs from the pre-import snapshot")


_IMPORT_EVALUATOR_SOURCE_IDENTITIES = _capture_evaluator_source_identities()

import numpy as np  # noqa: E402

with contextlib.suppress(ValueError):
    sys.path.remove(str(_SCRIPT_DIR))
sys.path.insert(0, str(_SCRIPT_DIR))

import calvin_bridge as _calvin_bridge  # noqa: E402

_require_imported_local_module(
    _calvin_bridge,
    "calvin_bridge.py",
    _IMPORT_EVALUATOR_SOURCE_IDENTITIES,
)

from calvin_bridge import (  # noqa: E402
    ACTION_DIM,
    ACTION_HORIZON,
    GRIPPER_IMAGE_SHAPE,
    INFERENCE_SEED_DOMAIN,
    MODEL_REVISION,
    NUM_SEQUENCES,
    PROTOCOL,
    SCHEMA,
    SEQUENCE_SHA256,
    STATE_DIM,
    STATIC_IMAGE_SHAPE,
    SUBTASKS_PER_SEQUENCE,
    SUPPORTED_EXECUTION_HORIZONS,
    PolicyClient,
)

import preflight as _preflight  # noqa: E402

_require_imported_local_module(
    _preflight,
    "preflight.py",
    _IMPORT_EVALUATOR_SOURCE_IDENTITIES,
)

from preflight import (  # noqa: E402
    ARCHIVE_BYTES,
    ARCHIVE_READER_SCHEMA,
    ARCHIVE_SHA256,
    CENTRAL_DIRECTORY_SHA256,
    DATASET_MANIFEST_SCHEMA,
    MEMBER_INDEX_NAME,
    MEMBER_INDEX_SCHEMA,
    PYTHON_VERSION,
    TASK_ORACLE_SHA256,
    TRAINING_METADATA_FILES,
    VALIDATION_ANNOTATIONS_SHA256,
    build_official_attestation,
)

_require_evaluator_sources_unchanged(_IMPORT_EVALUATOR_SOURCE_IDENTITIES)

EVALUATION_SEED = 0
MAX_ACTIONS_PER_SUBTASK = 360
CONTROL_FREQUENCY_HZ = 30
VALIDATION_SCENE = "calvin_scene_D"
PREREGISTRATION_SCHEMA = "duovla-calvin-official-preregistration-v8"
RUN_SCHEMA = "duovla-calvin-official-run-v5"
EPISODE_SCHEMA = "duovla-calvin-official-sequence-v1"
SUMMARY_SCHEMA = "duovla-calvin-official-summary-v2"
OUTPUT_ROOTS_SCHEMA = "duovla-calvin-official-output-roots-v1"
OUTPUT_CLAIM_SCHEMA = "duovla-calvin-official-output-claim-v1"
CLAIM_RECORD_SCHEMA = "duovla-calvin-official-claim-record-v1"
COMPLETION_SCHEMA = "duovla-calvin-official-completion-v1"
FINAL_CHECKPOINT_UPDATE = 30_000
DEFAULT_POLICY_WARMUP_CALLS = 2
MIN_POLICY_WARMUP_CALLS = 1
POLICY_WARMUP_REPLAN_BASE = MAX_ACTIONS_PER_SUBTASK
OFFICIAL_TRAIN_SEEDS = (0, 1, 2)
OFFICIAL_FLOW_NFES = (1, 5, 10)
OFFICIAL_DIRECT_NFE = 1

_PREREGISTRATION_FIELDS = {
    "aggregation_python_version",
    "aggregator_sha256",
    "benchmark_protocol",
    "cells",
    "direct_nfe",
    "evaluation_seed",
    "execution_horizons",
    "final_checkpoint_update",
    "final_freeze_token_sha256",
    "flow_nfes",
    "inference_seed_domain",
    "policy_warmup_calls",
    "official_output_roots",
    "schema",
    "sequence_count",
    "sequence_sha256",
    "sequences",
    "subtasks_per_sequence",
    "training_seeds",
    "runtime_attestation_sha256",
}
_CELL_FIELDS = {
    "cell_id",
    "checkpoint",
    "execution_geometry",
    "execution_horizon",
    "policy",
    "output_claim",
    "serving_runtime_sha256",
}
_CELL_CREATOR_FIELDS = _CELL_FIELDS - {"output_claim"}
_CHECKPOINT_FIELDS = {"sha256"}
_POLICY_FIELDS = {
    "identity_sha256",
    "inference_seed_behavior",
    "nfe",
    "objective",
    "sampler",
    "train_seed",
}
_POLICY_CREATOR_FIELDS = _POLICY_FIELDS - {"identity_sha256"}
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
    "episodes_jsonl_sha256",
    "preregistration_sha256",
    "run_json_sha256",
    "schema",
    "sequence_records",
    "summary_json_sha256",
}
_HEALTH_FIELDS = {
    "action_dim",
    "action_horizon",
    "calvin_identity",
    "checkpoint_manifest_sha256",
    "execution_horizons",
    "execution_geometry",
    "gripper_image_shape",
    "mode",
    "model_revision",
    "nfe",
    "normalization_content_sha256",
    "normalization_metadata_sha256",
    "objective",
    "operation",
    "policy_contract_sha256",
    "protocol",
    "request_id",
    "sampler",
    "schema",
    "sequence_sha256",
    "serving_runtime_sha256",
    "state_dim",
    "static_image_shape",
    "status",
    "train_seed",
}
_EXECUTION_GEOMETRY_FIELDS = {
    "expert_batch_isolation",
    "experts_implementation",
    "fixed_physical_prefix_width",
    "physical_batch_size",
    "prefix_geometry_content_sha256",
}
_CALVIN_IDENTITY_FIELDS = {
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
_CALVIN_MEMBER_INDEX_FIELDS = {"bytes", "path", "schema", "sha256"}


def require(condition: bool, message: str) -> None:
    """Raise one consistent exception for a failed evaluation invariant."""

    if not condition:
        raise RuntimeError(message)


def _require_attestation_sources_match_import(attestation: Mapping[str, Any]) -> None:
    try:
        attested_sources = attestation["runtime"]["sources"]
    except (KeyError, TypeError) as exc:
        raise RuntimeError("runtime/data attestation has no evaluator source identities") from exc
    require(
        attested_sources == _IMPORT_EVALUATOR_SOURCE_IDENTITIES,
        "runtime/data attestation sources differ from evaluator import-time snapshot",
    )
    _require_evaluator_sources_unchanged(_IMPORT_EVALUATOR_SOURCE_IDENTITIES)


def _is_integer(value: Any, minimum: int = 0, maximum: Optional[int] = None) -> bool:
    if type(value) is not int or value < minimum:
        return False
    return maximum is None or value < maximum


def _is_finite_number(value: Any) -> bool:
    if type(value) not in (int, float):
        return False
    return math.isfinite(value)


def _require_exact_keys(value: Mapping[str, Any], expected: Set[str], name: str) -> None:
    observed = set(value.keys())
    require(
        observed == expected,
        f"{name} fields differ: missing={sorted(expected - observed)}, extra={sorted(observed - expected)}",
    )


def _require_sha256(value: Any, name: str) -> None:
    require(
        isinstance(value, str) and len(value) == 64 and all(character in "0123456789abcdef" for character in value),
        f"{name} must be 64 lowercase hexadecimal characters",
    )


def _validate_execution_geometry(value: Any, name: str) -> Dict[str, Any]:
    require(isinstance(value, Mapping), f"{name} must be an object")
    _require_exact_keys(value, _EXECUTION_GEOMETRY_FIELDS, name)
    require(value["experts_implementation"] == "grouped_mm", f"{name} expert backend mismatch")
    require(
        value["expert_batch_isolation"] == "sample_isolated_grouped_mm_v1",
        f"{name} expert isolation mismatch",
    )
    require(
        type(value["physical_batch_size"]) is int and value["physical_batch_size"] == 8,
        f"{name} physical batch size must equal eight",
    )
    require(
        type(value["fixed_physical_prefix_width"]) is int
        and 0 < value["fixed_physical_prefix_width"] <= 1024 - ACTION_HORIZON,
        f"{name} fixed physical prefix width is invalid",
    )
    _require_sha256(value["prefix_geometry_content_sha256"], f"{name} prefix geometry SHA-256")
    return dict(value)


def _validate_calvin_identity(value: Any, name: str = "policy health CALVIN identity") -> Dict[str, Any]:
    require(isinstance(value, Mapping), f"{name} must be an object")
    _require_exact_keys(value, _CALVIN_IDENTITY_FIELDS, name)
    require(value["name"] == "task_ABC_D" and value["split"] == "training", f"{name} dataset/split mismatch")
    require(
        type(value["archive_bytes"]) is int and value["archive_bytes"] == ARCHIVE_BYTES,
        f"{name} archive bytes mismatch",
    )
    require(value["archive_sha256"] == ARCHIVE_SHA256, f"{name} archive SHA-256 mismatch")
    require(
        value["central_directory_sha256"] == CENTRAL_DIRECTORY_SHA256,
        f"{name} central-directory SHA-256 mismatch",
    )
    require(value["dataset_manifest_schema"] == DATASET_MANIFEST_SCHEMA, f"{name} manifest schema mismatch")
    require(value["reader_schema"] == ARCHIVE_READER_SCHEMA, f"{name} reader schema mismatch")
    require(value["storage_mode"] == "archive-direct", f"{name} storage mode mismatch")
    for field in (
        "dataset_manifest_file_sha256",
        "dataset_manifest_sha256",
        "member_inventory_sha256",
        "metadata_sha256",
        "storage_identity_sha256",
    ):
        _require_sha256(value[field], f"{name} {field}")
    require(value["metadata_files"] == list(TRAINING_METADATA_FILES), f"{name} metadata file list mismatch")
    member_index = value["member_index"]
    require(isinstance(member_index, Mapping), f"{name} member index must be an object")
    _require_exact_keys(member_index, _CALVIN_MEMBER_INDEX_FIELDS, f"{name} member index")
    require(
        type(member_index["bytes"]) is int and member_index["bytes"] > 0,
        f"{name} member-index byte length is invalid",
    )
    require(member_index["path"] == MEMBER_INDEX_NAME, f"{name} member-index path mismatch")
    require(member_index["schema"] == MEMBER_INDEX_SCHEMA, f"{name} member-index schema mismatch")
    _require_sha256(member_index["sha256"], f"{name} member-index SHA-256")
    return {
        **dict(value),
        "member_index": dict(member_index),
        "metadata_files": list(value["metadata_files"]),
    }


def canonical_json_bytes(value: Any) -> bytes:
    """Return the finite, sorted JSON representation used for identities."""

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
    """Hash a finite canonical JSON value."""

    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _real_directory_identity(path: Path, *, name: str, require_empty: bool = False) -> Dict[str, Any]:
    """Bind one canonical, non-symlink directory to its live device/inode."""

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
    flags = os.O_RDONLY | os.O_NOFOLLOW
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
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
            try:
                entries = os.listdir(descriptor)
            except (OSError, TypeError) as exc:
                raise RuntimeError(f"cannot inspect {name}") from exc
            require(not entries, f"{name} must be empty when pre-registration is created")
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
) -> Dict[str, Any]:
    """Capture the two dedicated roots used by all 24 official attempts."""

    runs = _real_directory_identity(Path(runs_root), name="official run root", require_empty=require_empty)
    claims = _real_directory_identity(Path(claims_root), name="official claim root", require_empty=require_empty)
    require(runs["path"] != claims["path"], "official run and claim roots must be distinct")
    common = os.path.commonpath((runs["path"], claims["path"]))
    require(
        common not in (runs["path"], claims["path"]),
        "official run and claim roots must not contain one another",
    )
    require(
        (runs["device"], runs["inode"]) != (claims["device"], claims["inode"]),
        "official run and claim roots must be different directories",
    )
    return {"claims": claims, "runs": runs, "schema": OUTPUT_ROOTS_SCHEMA}


def validate_official_output_roots(value: Any, *, require_live: bool = False) -> Dict[str, Any]:
    """Validate recorded root identities and optionally match their live inodes."""

    require(isinstance(value, Mapping), "official output roots must be an object")
    _require_exact_keys(value, _OUTPUT_ROOTS_FIELDS, "official output roots")
    require(value["schema"] == OUTPUT_ROOTS_SCHEMA, "official output roots schema mismatch")
    checked = {}  # type: Dict[str, Dict[str, Any]]
    for key in ("runs", "claims"):
        identity = value[key]
        require(isinstance(identity, Mapping), f"official {key} root identity must be an object")
        _require_exact_keys(identity, _OUTPUT_ROOT_IDENTITY_FIELDS, f"official {key} root identity")
        path = identity["path"]
        require(isinstance(path, str) and bool(path), f"official {key} root path is invalid")
        candidate = Path(path)
        require(candidate.is_absolute() and str(candidate) == path, f"official {key} root path is not canonical")
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
) -> Dict[str, Any]:
    """Derive, rather than accept, the sole run/claim paths for one cell."""

    require(isinstance(cell_id, str) and bool(cell_id), "output claim cell ID is invalid")
    checked_roots = validate_official_output_roots(roots)
    _require_sha256(final_freeze_token_sha256, "output claim freeze-token SHA-256")
    output_dir = str(Path(checked_roots["runs"]["path"]) / cell_id)
    claim_path = str(Path(checked_roots["claims"]["path"]) / f"{cell_id}.json")
    identity_payload = {
        "cell_id": cell_id,
        "claim_path": claim_path,
        "final_freeze_token_sha256": final_freeze_token_sha256,
        "output_dir": output_dir,
        "protocol": PROTOCOL,
    }
    return {
        "claim_id_sha256": canonical_sha256(identity_payload),
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
) -> Dict[str, Any]:
    require(isinstance(value, Mapping), "cell output claim must be an object")
    _require_exact_keys(value, _OUTPUT_CLAIM_FIELDS, "cell output claim")
    require(value["schema"] == OUTPUT_CLAIM_SCHEMA, "cell output claim schema mismatch")
    expected = derive_output_claim(cell_id, roots, final_freeze_token_sha256)
    require(canonical_json_bytes(value) == canonical_json_bytes(expected), "cell output claim is not canonical")
    return dict(value)


def official_cell_id(train_seed: int, objective: str, nfe: int, execution_horizon: int) -> str:
    """Return the only valid identifier for one canonical comparison cell."""

    require(objective in ("rectified_flow", "direct_regression"), "unsupported official objective")
    label = "flow" if objective == "rectified_flow" else "direct"
    return f"seed-{train_seed}-{label}-nfe-{nfe}-k-{execution_horizon}"


def official_factor_matrix() -> Set[Tuple[int, str, int, int]]:
    """Return the exact three-seed, two-method, NFE, and K matrix (24 cells)."""

    return {
        (seed, objective, nfe, execution_horizon)
        for seed in OFFICIAL_TRAIN_SEEDS
        for objective, nfes in (
            ("rectified_flow", OFFICIAL_FLOW_NFES),
            ("direct_regression", (OFFICIAL_DIRECT_NFE,)),
        )
        for nfe in nfes
        for execution_horizon in SUPPORTED_EXECUTION_HORIZONS
    }


def selected_policy_contract(objective: str, nfe: int) -> Dict[str, Any]:
    """Reconstruct the complete policy contract hashed by the policy server."""

    common = {
        "action_dim": ACTION_DIM,
        "action_horizon": ACTION_HORIZON,
        "clip_final_normalized_actions": True,
        "clip_intermediate_actions": False,
        "nfe": nfe,
        "objective": objective,
        "schema": "duo-vla-policy-contract-v1",
    }
    if objective == "rectified_flow":
        require(type(nfe) is int and nfe in OFFICIAL_FLOW_NFES, "flow cell NFE must be one of {1, 5, 10}")
        return {
            **common,
            "inference_seed_behavior": "episode_identity_gaussian_noise",
            "sampler": "euler_uniform",
            "training_input": "linear_noise_to_clean",
            "training_target": "velocity_clean_minus_noise",
            "training_timestep": "uniform_per_chunk",
        }
    require(objective == "direct_regression", "unsupported official objective")
    require(type(nfe) is int and nfe == OFFICIAL_DIRECT_NFE, "direct cell NFE must equal one")
    return {
        **common,
        "inference_seed_behavior": "episode_identity_echo_only",
        "sampler": "single_forward",
        "training_input": "zero_action_canvas",
        "training_target": "clean_action",
        "training_timestep": "fixed_one",
    }


def canonical_sequence_sha256(sequences: Sequence[Any]) -> str:
    """Hash the complete official sequence list exactly like CALVIN preflight."""

    return hashlib.sha256(canonical_json_bytes(sequences)).hexdigest()


def verify_sequence_contract(
    sequences: Sequence[Any],
    expected_count: int = NUM_SEQUENCES,
    expected_sha256: str = SEQUENCE_SHA256,
) -> List[Tuple[Dict[str, Any], Tuple[str, ...]]]:
    """Validate, authenticate, and detach a generated sequence collection."""

    require(isinstance(sequences, (list, tuple)), "official sequences must be a list or tuple")
    require(len(sequences) == expected_count, f"official sequence count must be exactly {expected_count}")
    payload = canonical_json_bytes(sequences)
    digest = hashlib.sha256(payload).hexdigest()
    require(
        hmac.compare_digest(digest, expected_sha256),
        f"official sequence digest mismatch: expected {expected_sha256}, found {digest}",
    )

    # A JSON round trip detaches the result from get_sequences' lru_cache and
    # normalizes NumPy string scalars without changing the authenticated bytes.
    normalized = json.loads(payload.decode("ascii"))
    checked = []  # type: List[Tuple[Dict[str, Any], Tuple[str, ...]]]
    for sequence_idx, item in enumerate(normalized):
        require(
            isinstance(item, list) and len(item) == 2,
            f"official sequence {sequence_idx} must contain initial state and task list",
        )
        initial_state, tasks = item
        require(isinstance(initial_state, dict), "official initial state must be an object")
        require(
            isinstance(tasks, list) and len(tasks) == SUBTASKS_PER_SEQUENCE,
            f"official sequence {sequence_idx} must contain exactly {SUBTASKS_PER_SEQUENCE} subtasks",
        )
        require(
            all(isinstance(task, str) and bool(task) for task in tasks),
            "official sequence tasks must be non-empty strings",
        )
        checked.append((dict(initial_state), tuple(tasks)))
    return checked


def regenerate_official_sequences(evaluation_seed: int) -> List[Tuple[Dict[str, Any], Tuple[str, ...]]]:
    """Regenerate and authenticate all 1,000 official sequences."""

    require(evaluation_seed == EVALUATION_SEED, "official CALVIN evaluation_seed must equal 0")
    from calvin_agent.evaluation.multistep_sequences import get_sequences

    # A scoring process must regenerate rather than reuse a sequence list that
    # another caller may have obtained from the official lru_cache.
    cache_clear = getattr(get_sequences, "cache_clear", None)
    if cache_clear is not None:
        cache_clear()
    generated = get_sequences(NUM_SEQUENCES)
    return verify_sequence_contract(generated)


def calvin_state(robot_obs: Any) -> np.ndarray:
    """Select the official robot_no_joints state into an owned float32[8]."""

    values = np.asarray(robot_obs)
    require(values.shape == (15,), "CALVIN robot_obs must have shape (15,)")
    require(bool(np.isfinite(values).all()), "CALVIN robot_obs must contain only finite values")
    with np.errstate(over="ignore", invalid="ignore"):
        state = np.concatenate((values[:7], values[14:15])).astype(np.float32, copy=True)
    require(state.shape == (STATE_DIM,), "constructed CALVIN policy state must have shape (8,)")
    require(bool(np.isfinite(state).all()), "CALVIN policy state cannot be represented as finite float32")
    require(state[7] in (-1.0, 1.0), "CALVIN previous gripper state must be exactly {-1, +1}")
    return np.ascontiguousarray(state).copy()


def calvin_env_action(action: Any) -> np.ndarray:
    """Return an owned clipped rel_action; -1 closes and +1 opens CALVIN's gripper."""

    values = np.asarray(action)
    require(values.shape == (ACTION_DIM,), "CALVIN policy action must have shape (7,)")
    require(bool(np.isfinite(values).all()), "CALVIN policy action must contain only finite values")
    with np.errstate(over="ignore", invalid="ignore"):
        output = np.array(values, dtype=np.float32, copy=True, order="C")
    require(bool(np.isfinite(output).all()), "CALVIN policy action cannot be represented as finite float32")
    require(output[6] in (-1.0, 1.0), "CALVIN gripper action must be -1 (close) or +1 (open)")
    output[:6] = np.clip(output[:6], -1.0, 1.0)
    return output


def _policy_observation(observation: Mapping[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    require(isinstance(observation, Mapping), "CALVIN observation must be an object")
    rgb_obs = observation.get("rgb_obs")
    require(isinstance(rgb_obs, Mapping), "CALVIN observation must contain rgb_obs")
    require("scene_obs" in observation, "CALVIN simulator observation must contain reset-only scene_obs")
    static = np.asarray(rgb_obs.get("rgb_static"))
    gripper = np.asarray(rgb_obs.get("rgb_gripper"))
    require(
        static.shape == STATIC_IMAGE_SHAPE and static.dtype == np.uint8,
        f"rgb_static must be uint8 with shape {STATIC_IMAGE_SHAPE}",
    )
    require(
        gripper.shape == GRIPPER_IMAGE_SHAPE and gripper.dtype == np.uint8,
        f"rgb_gripper must be uint8 with shape {GRIPPER_IMAGE_SHAPE}",
    )
    state = calvin_state(observation.get("robot_obs"))
    # These are the only three model inputs.  In particular, scene_obs is not
    # returned and therefore cannot be included in PolicyClient.predict.
    return static.copy(order="C"), gripper.copy(order="C"), state


def _validate_action_chunk(actions: Any) -> np.ndarray:
    require(isinstance(actions, np.ndarray), "policy action chunk must be a numpy array")
    require(actions.shape == (ACTION_HORIZON, ACTION_DIM), "policy action chunk must have shape (8, 7)")
    require(actions.dtype == np.float32, "policy action chunk must have dtype float32")
    require(bool(np.isfinite(actions).all()), "policy action chunk contains non-finite values")
    require(
        bool(np.logical_or(actions[:, 6] == -1.0, actions[:, 6] == 1.0).all()),
        "policy action chunk gripper values must be exactly {-1, +1}",
    )
    return actions


def _validate_prediction_metadata(
    response: Any,
    *,
    evaluation_seed: int,
    train_seed: int,
    sequence_idx: int,
    subtask_idx: int,
    subtask_name: str,
    replan_idx: int,
    execution_horizon: int,
) -> float:
    require(isinstance(response, Mapping), "policy prediction metadata must be an object")
    expected = {
        "evaluation_seed": evaluation_seed,
        "execution_horizon": execution_horizon,
        "replan_idx": replan_idx,
        "sequence_idx": sequence_idx,
        "sequence_sha256": SEQUENCE_SHA256,
        "subtask_idx": subtask_idx,
        "subtask_name": subtask_name,
        "train_seed": train_seed,
    }
    for name, expected_value in expected.items():
        observed = response.get(name)
        if type(expected_value) is int:
            matches = type(observed) is int and observed == expected_value
        else:
            matches = observed == expected_value
        require(matches, f"policy prediction {name} drifted")
    policy_seconds = response.get("policy_seconds")
    require(
        _is_finite_number(policy_seconds) and policy_seconds >= 0,
        "policy prediction has invalid policy_seconds",
    )
    return float(policy_seconds)


def _percentile(values: Sequence[float], percentage: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * percentage / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def first_language_phrase(language_annotations: Any, subtask_name: str) -> str:
    """Select exactly the first fixed phrase from new_playtable_validation."""

    try:
        phrases = language_annotations[subtask_name]
    except (KeyError, TypeError) as exc:
        raise RuntimeError(f"validation language has no task {subtask_name!r}") from exc
    require(not isinstance(phrases, str), "validation language entry must be a phrase sequence")
    try:
        phrase_count = len(phrases)
    except TypeError as exc:
        raise RuntimeError("validation language entry must be a phrase sequence") from exc
    require(phrase_count > 0, "validation language entry must contain at least one phrase")
    phrase = phrases[0]
    require(isinstance(phrase, str) and bool(phrase), "first validation language phrase must be non-empty text")
    return phrase


_POLICY_WARMUP_FIELDS = {
    "actions_sha256",
    "discarded",
    "evaluation_seed",
    "execution_horizon",
    "gripper_rgb_sha256",
    "inference_seed",
    "instruction_sha256",
    "nfe",
    "objective",
    "policy_latency_seconds",
    "replan_idx",
    "sampler",
    "sequence_idx",
    "sequence_sha256",
    "server_latency_seconds",
    "state_sha256",
    "static_rgb_sha256",
    "subtask_idx",
    "subtask_name",
    "train_seed",
    "warmup_index",
}


def synthetic_policy_warmup_inputs() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return deterministic, simulator-independent inputs for policy warm-up."""

    static_values = np.arange(math.prod(STATIC_IMAGE_SHAPE), dtype=np.uint32).reshape(STATIC_IMAGE_SHAPE)
    gripper_values = np.arange(math.prod(GRIPPER_IMAGE_SHAPE), dtype=np.uint32).reshape(GRIPPER_IMAGE_SHAPE)
    rgb_static = (static_values % 251).astype(np.uint8)
    rgb_gripper = ((gripper_values * 7 + 3) % 251).astype(np.uint8)
    state = np.zeros(STATE_DIM, dtype=np.float32)
    state[-1] = -1.0
    return rgb_static, rgb_gripper, state


def _tensor_bytes_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes(order="C")).hexdigest()


def validate_policy_warmup(
    value: Any,
    *,
    count: int,
    sequences: Sequence[Any],
    language_annotations: Any,
    execution_horizon: int,
    policy: Mapping[str, Any],
) -> Dict[str, Any]:
    """Authenticate the discarded synthetic calls recorded before scoring."""

    require(type(count) is int and count >= MIN_POLICY_WARMUP_CALLS, "policy warm-up count is below the minimum")
    require(isinstance(value, Mapping), "policy warm-up record must be an object")
    _require_exact_keys(value, {"count", "included_in_episode_latency", "reports"}, "policy warm-up record")
    require(type(value["count"]) is int and value["count"] == count, "policy warm-up count drifted")
    require(value["included_in_episode_latency"] is False, "policy warm-up calls entered episode latency")
    reports = value["reports"]
    require(isinstance(reports, list) and len(reports) == count, "policy warm-up report count drifted")
    require(bool(sequences), "policy warm-up requires the registered sequence inventory")
    first_sequence = sequences[0]
    require(isinstance(first_sequence, (list, tuple)) and len(first_sequence) == 2, "warm-up sequence is invalid")
    first_tasks = first_sequence[1]
    require(isinstance(first_tasks, (list, tuple)) and bool(first_tasks), "warm-up task inventory is invalid")
    subtask_name = first_tasks[0]
    require(isinstance(subtask_name, str) and bool(subtask_name), "warm-up task name is invalid")
    instruction = first_language_phrase(language_annotations, subtask_name)
    rgb_static, rgb_gripper, state = synthetic_policy_warmup_inputs()
    expected_hashes = {
        "gripper_rgb_sha256": _tensor_bytes_sha256(rgb_gripper),
        "instruction_sha256": hashlib.sha256(instruction.encode("utf-8")).hexdigest(),
        "state_sha256": _tensor_bytes_sha256(state),
        "static_rgb_sha256": _tensor_bytes_sha256(rgb_static),
    }
    train_seed = policy.get("train_seed")
    require(type(train_seed) is int and train_seed in OFFICIAL_TRAIN_SEEDS, "warm-up policy seed is invalid")
    checked = []  # type: List[Dict[str, Any]]
    for warmup_index, report in enumerate(reports):
        require(isinstance(report, Mapping), "policy warm-up report must be an object")
        _require_exact_keys(report, _POLICY_WARMUP_FIELDS, "policy warm-up report")
        warmup_replan_idx = POLICY_WARMUP_REPLAN_BASE + warmup_index
        expected_identity = {
            "discarded": True,
            "evaluation_seed": EVALUATION_SEED,
            "execution_horizon": execution_horizon,
            "nfe": policy.get("nfe"),
            "objective": policy.get("objective"),
            "replan_idx": warmup_replan_idx,
            "sampler": policy.get("sampler"),
            "sequence_idx": 0,
            "sequence_sha256": SEQUENCE_SHA256,
            "subtask_idx": 0,
            "subtask_name": subtask_name,
            "train_seed": train_seed,
            "warmup_index": warmup_index,
        }
        for name, expected in expected_identity.items():
            observed = report[name]
            matches = type(observed) is type(expected) and observed == expected
            require(matches, f"policy warm-up {name} drifted")
        expected_seed = _calvin_bridge.calvin_replan_seed(
            EVALUATION_SEED,
            SEQUENCE_SHA256,
            0,
            0,
            subtask_name,
            warmup_replan_idx,
        )
        require(
            type(report["inference_seed"]) is int and report["inference_seed"] == expected_seed,
            "policy warm-up inference seed drifted",
        )
        for name, expected in expected_hashes.items():
            require(report[name] == expected, f"policy warm-up {name} drifted")
        _require_sha256(report["actions_sha256"], "policy warm-up actions_sha256")
        require(
            _is_finite_number(report["policy_latency_seconds"]) and report["policy_latency_seconds"] >= 0,
            "policy warm-up latency is invalid",
        )
        require(
            _is_finite_number(report["server_latency_seconds"]) and report["server_latency_seconds"] >= 0,
            "policy warm-up server latency is invalid",
        )
        require(
            float(report["policy_latency_seconds"]) >= float(report["server_latency_seconds"]),
            "policy warm-up server latency exceeds the enclosing client latency",
        )
        checked.append(dict(report))
    return {
        "count": count,
        "included_in_episode_latency": False,
        "reports": checked,
    }


def run_policy_warmups(
    client: Any,
    health: Mapping[str, Any],
    *,
    count: int,
    sequences: Sequence[Any],
    language_annotations: Any,
    execution_horizon: int,
    attempt_callback: Optional[Callable[[int, Mapping[str, Any]], None]] = None,
    report_callback: Optional[Callable[[Sequence[Mapping[str, Any]]], None]] = None,
) -> Dict[str, Any]:
    """Run and record canonical synthetic calls before any D observation exists."""

    require(type(count) is int and count >= MIN_POLICY_WARMUP_CALLS, "official scoring requires a policy warm-up")
    first_sequence = sequences[0]
    subtask_name = first_sequence[1][0]
    instruction = first_language_phrase(language_annotations, subtask_name)
    rgb_static, rgb_gripper, state = synthetic_policy_warmup_inputs()
    reports = []  # type: List[Dict[str, Any]]
    for warmup_index in range(count):
        warmup_replan_idx = POLICY_WARMUP_REPLAN_BASE + warmup_index
        inference_seed = _calvin_bridge.calvin_replan_seed(
            EVALUATION_SEED,
            SEQUENCE_SHA256,
            0,
            0,
            subtask_name,
            warmup_replan_idx,
        )
        intent = {
            "evaluation_seed": EVALUATION_SEED,
            "execution_horizon": execution_horizon,
            "gripper_rgb_sha256": _tensor_bytes_sha256(rgb_gripper),
            "inference_seed": inference_seed,
            "instruction_sha256": hashlib.sha256(instruction.encode("utf-8")).hexdigest(),
            "replan_idx": warmup_replan_idx,
            "sequence_idx": 0,
            "sequence_sha256": SEQUENCE_SHA256,
            "state_sha256": _tensor_bytes_sha256(state),
            "static_rgb_sha256": _tensor_bytes_sha256(rgb_static),
            "subtask_idx": 0,
            "subtask_name": subtask_name,
            "train_seed": health["train_seed"],
            "warmup_index": warmup_index,
        }
        if attempt_callback is not None:
            attempt_callback(warmup_index, intent)
        request_started = time.perf_counter()
        actions, response = client.predict(
            evaluation_seed=EVALUATION_SEED,
            train_seed=health["train_seed"],
            sequence_sha256=SEQUENCE_SHA256,
            sequence_idx=0,
            subtask_idx=0,
            subtask_name=subtask_name,
            replan_idx=warmup_replan_idx,
            execution_horizon=execution_horizon,
            instruction=instruction,
            rgb_static=rgb_static,
            rgb_gripper=rgb_gripper,
            state=state,
        )
        policy_latency = time.perf_counter() - request_started
        chunk = _validate_action_chunk(actions)
        server_latency = _validate_prediction_metadata(
            response,
            evaluation_seed=EVALUATION_SEED,
            train_seed=health["train_seed"],
            sequence_idx=0,
            subtask_idx=0,
            subtask_name=subtask_name,
            replan_idx=warmup_replan_idx,
            execution_horizon=execution_horizon,
        )
        reports.append(
            {
                "actions_sha256": _tensor_bytes_sha256(chunk),
                "discarded": True,
                "evaluation_seed": EVALUATION_SEED,
                "execution_horizon": execution_horizon,
                "gripper_rgb_sha256": _tensor_bytes_sha256(rgb_gripper),
                "inference_seed": inference_seed,
                "instruction_sha256": hashlib.sha256(instruction.encode("utf-8")).hexdigest(),
                "nfe": health["nfe"],
                "objective": health["objective"],
                "policy_latency_seconds": policy_latency,
                "replan_idx": warmup_replan_idx,
                "sampler": health["sampler"],
                "sequence_idx": 0,
                "sequence_sha256": SEQUENCE_SHA256,
                "server_latency_seconds": server_latency,
                "state_sha256": _tensor_bytes_sha256(state),
                "static_rgb_sha256": _tensor_bytes_sha256(rgb_static),
                "subtask_idx": 0,
                "subtask_name": subtask_name,
                "train_seed": health["train_seed"],
                "warmup_index": warmup_index,
            }
        )
        if report_callback is not None:
            report_callback(tuple(reports))
    value = {"count": count, "included_in_episode_latency": False, "reports": reports}
    return validate_policy_warmup(
        value,
        count=count,
        sequences=sequences,
        language_annotations=language_annotations,
        execution_horizon=execution_horizon,
        policy=health,
    )


def rollout_subtask(
    environment: Any,
    client: Any,
    task_oracle: Any,
    observation: Mapping[str, Any],
    *,
    train_seed: int,
    evaluation_seed: int,
    sequence_idx: int,
    subtask_idx: int,
    subtask_name: str,
    instruction: str,
    execution_horizon: int,
    max_actions: int = MAX_ACTIONS_PER_SUBTASK,
) -> Tuple[Mapping[str, Any], Dict[str, Any]]:
    """Run one subtask while keeping the surrounding sequence environment live."""

    require(evaluation_seed == EVALUATION_SEED, "official CALVIN evaluation_seed must equal 0")
    require(
        type(execution_horizon) is int and execution_horizon in SUPPORTED_EXECUTION_HORIZONS,
        "CALVIN execution horizon K must be one of {1, 4}",
    )
    require(
        _is_integer(max_actions, 1) and max_actions <= MAX_ACTIONS_PER_SUBTASK,
        "CALVIN subtask action budget must be in [1, 360]",
    )
    require(_is_integer(train_seed, 0, 2**63), "train_seed must be an integer in [0, 2^63)")
    require(_is_integer(sequence_idx, 0, NUM_SEQUENCES), "sequence_idx is outside the official range")
    require(_is_integer(subtask_idx, 0, SUBTASKS_PER_SEQUENCE), "subtask_idx is outside the official range")
    require(isinstance(subtask_name, str) and bool(subtask_name), "subtask_name must be non-empty text")
    require(isinstance(instruction, str) and bool(instruction), "instruction must be non-empty text")

    # Queue and replan identity are deliberately local to one subtask.  A new
    # call to this function is the evaluator-side equivalent of model.reset():
    # the first request carries the new subtask identity and replan_idx == 0.
    action_queue = deque()
    replan_idx = 0
    environment_actions = 0
    clipped_channels = 0
    discarded_actions = 0
    policy_latencies = []  # type: List[float]
    server_latencies = []  # type: List[float]
    success = False
    started = time.perf_counter()
    start_info = environment.get_info()
    current_observation = observation

    while environment_actions < max_actions and not success:
        if not action_queue:
            rgb_static, rgb_gripper, state = _policy_observation(current_observation)
            request_started = time.perf_counter()
            actions, response = client.predict(
                evaluation_seed=evaluation_seed,
                train_seed=train_seed,
                sequence_sha256=SEQUENCE_SHA256,
                sequence_idx=sequence_idx,
                subtask_idx=subtask_idx,
                subtask_name=subtask_name,
                replan_idx=replan_idx,
                execution_horizon=execution_horizon,
                instruction=instruction,
                rgb_static=rgb_static,
                rgb_gripper=rgb_gripper,
                state=state,
            )
            policy_latencies.append(time.perf_counter() - request_started)
            chunk = _validate_action_chunk(actions)
            server_latencies.append(
                _validate_prediction_metadata(
                    response,
                    evaluation_seed=evaluation_seed,
                    train_seed=train_seed,
                    sequence_idx=sequence_idx,
                    subtask_idx=subtask_idx,
                    subtask_name=subtask_name,
                    replan_idx=replan_idx,
                    execution_horizon=execution_horizon,
                )
            )
            # Never queue row views: the official environment scales the array
            # passed to step() in place.
            action_queue.extend(
                np.array(row, dtype=np.float32, copy=True, order="C") for row in chunk[:execution_horizon]
            )
            replan_idx += 1

        queued_action = action_queue.popleft()
        clipped_channels += int(np.count_nonzero(np.abs(queued_action[:6]) > 1.0))
        env_action = calvin_env_action(queued_action)
        current_observation, _reward, _done, current_info = environment.step(env_action)
        environment_actions += 1

        # The official transition oracle is authoritative.  It is queried
        # after every action, including the first and the 360th.
        solved_tasks = task_oracle.get_task_info_for_set(start_info, current_info, {subtask_name})
        success = bool(solved_tasks)
        if success:
            discarded_actions = len(action_queue)
            action_queue.clear()

    return current_observation, {
        "action_clip_fraction": clipped_channels / (environment_actions * 6) if environment_actions else 0.0,
        "action_clipped_channels": clipped_channels,
        "action_continuous_channels": environment_actions * 6,
        "discarded_queued_actions_on_success": discarded_actions,
        "elapsed_seconds": time.perf_counter() - started,
        "environment_actions": environment_actions,
        "execution_horizon": execution_horizon,
        "instruction": instruction,
        "max_environment_actions": max_actions,
        "policy_calls": replan_idx,
        "policy_identity_first_replan_idx": 0,
        "policy_latency_p50_seconds": _percentile(policy_latencies, 50.0),
        "policy_latency_p95_seconds": _percentile(policy_latencies, 95.0),
        "policy_latency_seconds": policy_latencies,
        "server_latency_p50_seconds": _percentile(server_latencies, 50.0),
        "server_latency_p95_seconds": _percentile(server_latencies, 95.0),
        "server_latency_seconds": server_latencies,
        "steps_to_success": environment_actions if success else None,
        "subtask_idx": subtask_idx,
        "subtask_name": subtask_name,
        "success": success,
    }


def evaluate_sequence(
    environment: Any,
    client: Any,
    task_oracle: Any,
    initial_state: Mapping[str, Any],
    subtask_names: Sequence[str],
    language_annotations: Any,
    state_converter: Callable[[Mapping[str, Any]], Tuple[Any, Any]],
    *,
    train_seed: int,
    evaluation_seed: int,
    sequence_idx: int,
    execution_horizon: int,
) -> Dict[str, Any]:
    """Evaluate one official five-task sequence with exactly one env reset."""

    require(evaluation_seed == EVALUATION_SEED, "official CALVIN evaluation_seed must equal 0")
    require(
        len(subtask_names) == SUBTASKS_PER_SEQUENCE,
        "each official CALVIN sequence must contain exactly five subtasks",
    )
    robot_obs, scene_obs = state_converter(initial_state)
    robot_values = np.asarray(robot_obs)
    scene_values = np.asarray(scene_obs)
    require(robot_values.shape == (15,) and bool(np.isfinite(robot_values).all()), "invalid reset robot_obs")
    require(scene_values.shape == (24,) and bool(np.isfinite(scene_values).all()), "invalid reset scene_obs")
    # This is the only environment reset in the sequence.  Subsequent
    # subtasks consume the observation produced by the preceding subtask.
    observation = environment.reset(robot_obs=robot_values.copy(), scene_obs=scene_values.copy())
    _policy_observation(observation)

    started = time.perf_counter()
    subtask_records = []  # type: List[Dict[str, Any]]
    successful_subtasks = 0
    for subtask_idx, subtask_name in enumerate(subtask_names):
        instruction = first_language_phrase(language_annotations, subtask_name)
        observation, subtask_record = rollout_subtask(
            environment,
            client,
            task_oracle,
            observation,
            train_seed=train_seed,
            evaluation_seed=evaluation_seed,
            sequence_idx=sequence_idx,
            subtask_idx=subtask_idx,
            subtask_name=subtask_name,
            instruction=instruction,
            execution_horizon=execution_horizon,
        )
        subtask_records.append(subtask_record)
        if not subtask_record["success"]:
            break
        successful_subtasks += 1

    return {
        "elapsed_seconds": time.perf_counter() - started,
        "evaluation_seed": evaluation_seed,
        "execution_horizon": execution_horizon,
        "schema": EPISODE_SCHEMA,
        "sequence_idx": sequence_idx,
        "sequence_sha256": SEQUENCE_SHA256,
        "sequence_success": successful_subtasks == SUBTASKS_PER_SEQUENCE,
        "subtasks": subtask_records,
        "successful_subtasks": successful_subtasks,
    }


def evaluate_official_sequences(
    environment: Any,
    client: Any,
    task_oracle: Any,
    sequences: Sequence[Any],
    language_annotations: Any,
    state_converter: Callable[[Mapping[str, Any]], Tuple[Any, Any]],
    *,
    train_seed: int,
    evaluation_seed: int,
    execution_horizon: int,
    episode_callback: Optional[Callable[[Mapping[str, Any]], None]] = None,
) -> List[Dict[str, Any]]:
    """Authenticate first, then run the indivisible 1,000-sequence score."""

    require(evaluation_seed == EVALUATION_SEED, "official CALVIN evaluation_seed must equal 0")
    # This validation is intentionally inside the function that can predict;
    # no caller can accidentally bypass the digest gate by pre-validating a
    # differently shaped sequence collection.
    checked_sequences = verify_sequence_contract(sequences)
    results = []  # type: List[Dict[str, Any]]
    for sequence_idx, (initial_state, subtask_names) in enumerate(checked_sequences):
        record = evaluate_sequence(
            environment,
            client,
            task_oracle,
            initial_state,
            subtask_names,
            language_annotations,
            state_converter,
            train_seed=train_seed,
            evaluation_seed=evaluation_seed,
            sequence_idx=sequence_idx,
            execution_horizon=execution_horizon,
        )
        results.append(record)
        if episode_callback is not None:
            episode_callback(record)
    return results


def summarize_sequences(sequence_records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Compute official scores and authenticated episode-only performance metrics."""

    require(bool(sequence_records), "cannot summarize zero CALVIN sequences")
    lengths = []  # type: List[int]
    task_totals = {}  # type: Dict[str, int]
    task_successes = {}  # type: Dict[str, int]
    policy_latencies = []  # type: List[float]
    server_latencies = []  # type: List[float]
    attempted_subtasks = 0
    environment_actions = 0
    policy_calls = 0
    rollout_elapsed_seconds = 0.0
    for expected_idx, record in enumerate(sequence_records):
        require(record.get("sequence_idx") == expected_idx, "CALVIN sequence records are not contiguous and ordered")
        sequence_elapsed = record.get("elapsed_seconds")
        require(
            _is_finite_number(sequence_elapsed) and sequence_elapsed > 0,
            "CALVIN sequence elapsed time must be positive and finite",
        )
        rollout_elapsed_seconds += float(sequence_elapsed)
        length = record.get("successful_subtasks")
        require(
            _is_integer(length, 0, SUBTASKS_PER_SEQUENCE + 1),
            "successful_subtasks must be an integer in [0, 5]",
        )
        lengths.append(length)
        subtasks = record.get("subtasks")
        require(isinstance(subtasks, list), "CALVIN sequence record subtasks must be a list")
        for subtask in subtasks:
            require(isinstance(subtask, Mapping), "CALVIN subtask metric must be an object")
            name = subtask.get("subtask_name")
            require(isinstance(name, str) and bool(name), "invalid subtask metric name")
            task_totals[name] = task_totals.get(name, 0) + 1
            task_successes[name] = task_successes.get(name, 0) + int(bool(subtask.get("success")))
            subtask_policy_calls = subtask.get("policy_calls")
            subtask_environment_actions = subtask.get("environment_actions")
            require(
                type(subtask_policy_calls) is int and subtask_policy_calls > 0,
                "CALVIN subtask policy_calls must be positive",
            )
            require(
                type(subtask_environment_actions) is int and subtask_environment_actions > 0,
                "CALVIN subtask environment_actions must be positive",
            )
            subtask_policy_latencies = subtask.get("policy_latency_seconds")
            subtask_server_latencies = subtask.get("server_latency_seconds")
            require(
                isinstance(subtask_policy_latencies, list)
                and len(subtask_policy_latencies) == subtask_policy_calls
                and all(_is_finite_number(value) and value >= 0 for value in subtask_policy_latencies),
                "CALVIN subtask policy latency samples are invalid",
            )
            require(
                isinstance(subtask_server_latencies, list)
                and len(subtask_server_latencies) == subtask_policy_calls
                and all(_is_finite_number(value) and value >= 0 for value in subtask_server_latencies),
                "CALVIN subtask server latency samples are invalid",
            )
            require(
                all(
                    float(subtask_policy_latencies[index]) >= float(subtask_server_latencies[index])
                    for index in range(subtask_policy_calls)
                ),
                "CALVIN server latency exceeds its enclosing client latency",
            )
            attempted_subtasks += 1
            environment_actions += subtask_environment_actions
            policy_calls += subtask_policy_calls
            policy_latencies.extend(float(value) for value in subtask_policy_latencies)
            server_latencies.extend(float(value) for value in subtask_server_latencies)

    count = len(lengths)
    require(policy_calls > 0 and len(policy_latencies) == policy_calls, "CALVIN policy call inventory is empty")
    require(len(server_latencies) == policy_calls, "CALVIN server latency inventory is incomplete")
    policy_latency_total = sum(policy_latencies)
    server_latency_total = sum(server_latencies)
    require(policy_latency_total > 0, "CALVIN aggregate policy latency must be positive")
    require(server_latency_total > 0, "CALVIN aggregate server latency must be positive")
    require(rollout_elapsed_seconds > 0, "CALVIN aggregate rollout elapsed time must be positive")
    summary = {
        "AvgLen": sum(lengths) / count,
        "attempted_subtasks": attempted_subtasks,
        "environment_actions": environment_actions,
        "policy_calls": policy_calls,
        "policy_latency_p50_seconds": _percentile(policy_latencies, 50.0),
        "policy_latency_p95_seconds": _percentile(policy_latencies, 95.0),
        "policy_latency_total_seconds": policy_latency_total,
        "policy_throughput_calls_per_second": policy_calls / policy_latency_total,
        "rollout_elapsed_seconds": rollout_elapsed_seconds,
        "rollout_environment_actions_per_second": environment_actions / rollout_elapsed_seconds,
        "schema": SUMMARY_SCHEMA,
        "sequence_count": count,
        "sequence_sha256": SEQUENCE_SHA256,
        "server_latency_p50_seconds": _percentile(server_latencies, 50.0),
        "server_latency_p95_seconds": _percentile(server_latencies, 95.0),
        "server_latency_total_seconds": server_latency_total,
        "server_throughput_calls_per_second": policy_calls / server_latency_total,
        "task_metrics": [
            {
                "attempted": task_totals[name],
                "conditional_success_rate": task_successes[name] / task_totals[name],
                "subtask_name": name,
                "successes": task_successes[name],
            }
            for name in sorted(task_totals)
        ],
    }  # type: Dict[str, Any]
    for depth in range(1, SUBTASKS_PER_SEQUENCE + 1):
        summary[f"SR{depth}"] = sum(length >= depth for length in lengths) / count
    require(
        abs(summary["AvgLen"] - sum(summary[f"SR{depth}"] for depth in range(1, 6))) < 1e-12,
        "AvgLen/SR identity failed",
    )
    return summary


def validate_policy_health(
    health: Any,
    *,
    execution_horizon: int,
    expected_cell: Optional[Mapping[str, Any]] = None,
    expected_calvin_identity: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Fail closed on health and bind the live server to the frozen cell."""

    require(isinstance(health, Mapping), "policy health must be an object")
    _require_exact_keys(health, _HEALTH_FIELDS, "policy health")
    require(health["schema"] == SCHEMA, "policy health schema mismatch")
    require(health["status"] == "ok" and health["operation"] == "health", "policy health envelope mismatch")
    require(
        isinstance(health["request_id"], str) and bool(health["request_id"]),
        "policy health request_id is invalid",
    )
    require(health["protocol"] == PROTOCOL, "policy protocol mismatch")
    require(health["sequence_sha256"] == SEQUENCE_SHA256, "policy sequence digest mismatch")
    require(
        type(health["action_horizon"]) is int and health["action_horizon"] == ACTION_HORIZON,
        "policy action horizon mismatch",
    )
    require(
        type(health["action_dim"]) is int and health["action_dim"] == ACTION_DIM,
        "policy action dimension mismatch",
    )
    require(
        type(health["state_dim"]) is int and health["state_dim"] == STATE_DIM,
        "policy state dimension mismatch",
    )
    require(
        isinstance(health["static_image_shape"], list)
        and all(type(item) is int for item in health["static_image_shape"])
        and health["static_image_shape"] == list(STATIC_IMAGE_SHAPE),
        "policy static image shape mismatch",
    )
    require(
        isinstance(health["gripper_image_shape"], list)
        and all(type(item) is int for item in health["gripper_image_shape"])
        and health["gripper_image_shape"] == list(GRIPPER_IMAGE_SHAPE),
        "policy gripper image shape mismatch",
    )
    require(
        isinstance(health["execution_horizons"], list)
        and all(type(item) is int for item in health["execution_horizons"])
        and health["execution_horizons"] == list(SUPPORTED_EXECUTION_HORIZONS),
        "policy execution horizon set mismatch",
    )
    require(
        type(execution_horizon) is int and execution_horizon in health["execution_horizons"],
        "selected execution horizon is not supported by policy health",
    )
    train_seed = health["train_seed"]
    require(
        type(train_seed) is int and train_seed in OFFICIAL_TRAIN_SEEDS,
        "policy health train_seed must be one of {0, 1, 2}",
    )
    require(health["mode"] == "real", "official-score refuses fake policy health")
    live_calvin_identity = _validate_calvin_identity(health["calvin_identity"])
    require(
        health["normalization_metadata_sha256"] == live_calvin_identity["metadata_sha256"],
        "policy normalization/CALVIN metadata identities differ",
    )
    if expected_calvin_identity is not None:
        attested_calvin_identity = _validate_calvin_identity(
            expected_calvin_identity,
            "runtime-attested CALVIN identity",
        )
        require(
            canonical_json_bytes(live_calvin_identity) == canonical_json_bytes(attested_calvin_identity),
            "live policy CALVIN identity differs from runtime/data attestation",
        )
    objective = health["objective"]
    sampler = health["sampler"]
    nfe = health["nfe"]
    require(objective in ("rectified_flow", "direct_regression"), "policy objective is unsupported")
    if objective == "rectified_flow":
        require(sampler == "euler_uniform", "rectified-flow policy sampler mismatch")
        require(type(nfe) is int and nfe in (1, 5, 10), "rectified-flow policy NFE is unsupported")
    else:
        require(sampler == "single_forward", "direct-regression policy sampler mismatch")
        require(type(nfe) is int and nfe == 1, "direct-regression policy NFE must equal one")
    for name in (
        "checkpoint_manifest_sha256",
        "normalization_content_sha256",
        "normalization_metadata_sha256",
        "policy_contract_sha256",
        "serving_runtime_sha256",
    ):
        _require_sha256(health[name], f"policy health {name}")
    require(health["model_revision"] == MODEL_REVISION, "policy health model revision mismatch")
    live_execution_geometry = _validate_execution_geometry(
        health["execution_geometry"],
        "policy health execution geometry",
    )

    if expected_cell is not None:
        require(isinstance(expected_cell, Mapping), "pre-registered policy cell must be an object")
        registered_cell = _validate_registered_cell(dict(expected_cell))
        checkpoint = registered_cell["checkpoint"]
        policy = registered_cell["policy"]
        require(isinstance(checkpoint, Mapping), "pre-registered checkpoint identity must be an object")
        require(isinstance(policy, Mapping), "pre-registered policy identity must be an object")
        _require_exact_keys(checkpoint, _CHECKPOINT_FIELDS, "pre-registered checkpoint identity")
        _require_exact_keys(policy, _POLICY_FIELDS, "pre-registered policy identity")
        _require_sha256(checkpoint["sha256"], "pre-registered checkpoint sha256")
        _require_sha256(policy["identity_sha256"], "pre-registered policy identity_sha256")
        _require_sha256(expected_cell["serving_runtime_sha256"], "pre-registered serving_runtime_sha256")
        registered_execution_geometry = _validate_execution_geometry(
            registered_cell["execution_geometry"],
            "pre-registered execution geometry",
        )
        require(
            health["checkpoint_manifest_sha256"] == checkpoint["sha256"],
            "live checkpoint identity differs from pre-registration",
        )
        require(
            health["policy_contract_sha256"] == policy["identity_sha256"],
            "live policy identity differs from pre-registration",
        )
        require(
            hmac.compare_digest(health["serving_runtime_sha256"], registered_cell["serving_runtime_sha256"]),
            "live serving runtime identity differs from pre-registration",
        )
        require(
            canonical_json_bytes(live_execution_geometry) == canonical_json_bytes(registered_execution_geometry),
            "live execution geometry differs from pre-registration",
        )
        require(
            train_seed == policy["train_seed"],
            "live policy train_seed differs from pre-registration",
        )
        require(
            execution_horizon == registered_cell["execution_horizon"],
            "live execution horizon differs from pre-registration",
        )
        require(
            (objective, nfe, sampler) == (policy["objective"], policy["nfe"], policy["sampler"]),
            "live serving policy factors differ from pre-registration",
        )
    return dict(health)


def _unique_object(pairs: List[Tuple[str, Any]]) -> Dict[str, Any]:
    result = {}  # type: Dict[str, Any]
    for name, value in pairs:
        if name in result:
            raise ValueError(f"duplicate JSON field {name!r}")
        result[name] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant {value}")


def _read_strict_json(path: Path, *, expected_sha256: Optional[str] = None) -> Tuple[Dict[str, Any], str]:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise RuntimeError(f"cannot read pre-registration manifest: {path}") from exc
    digest = hashlib.sha256(raw).hexdigest()
    if expected_sha256 is not None:
        _require_sha256(expected_sha256, "externally supplied pre-registration SHA-256")
        require(
            hmac.compare_digest(digest, expected_sha256),
            "pre-registration file differs from externally supplied --preregistration-sha256",
        )
    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, ValueError) as exc:
        raise RuntimeError("pre-registration manifest is not strict finite UTF-8 JSON") from exc
    require(isinstance(value, dict), "pre-registration manifest root must be an object")
    return value, digest


def _validate_registered_cell(
    candidate: Any,
    *,
    roots: Optional[Mapping[str, Any]] = None,
    final_freeze_token_sha256: Optional[str] = None,
) -> Dict[str, Any]:
    """Validate one cell against the canonical official factor contract."""

    require(isinstance(candidate, dict), "pre-registration cell must be an object")
    _require_exact_keys(candidate, _CELL_FIELDS, "pre-registration cell")
    checkpoint = candidate["checkpoint"]
    policy = candidate["policy"]
    require(isinstance(checkpoint, dict), "cell checkpoint identity must be an object")
    require(isinstance(policy, dict), "cell policy identity must be an object")
    _require_exact_keys(checkpoint, _CHECKPOINT_FIELDS, "checkpoint identity")
    _require_exact_keys(policy, _POLICY_FIELDS, "policy identity")
    _require_sha256(checkpoint["sha256"], "checkpoint sha256")
    _require_sha256(policy["identity_sha256"], "policy identity_sha256")
    _require_sha256(candidate["serving_runtime_sha256"], "serving_runtime_sha256")
    _validate_execution_geometry(candidate["execution_geometry"], "execution geometry")

    train_seed = policy["train_seed"]
    objective = policy["objective"]
    nfe = policy["nfe"]
    execution_horizon = candidate["execution_horizon"]
    require(
        type(train_seed) is int and train_seed in OFFICIAL_TRAIN_SEEDS,
        "policy train_seed must be one of {0, 1, 2}",
    )
    require(
        type(execution_horizon) is int and execution_horizon in SUPPORTED_EXECUTION_HORIZONS,
        "cell execution_horizon must be one of {1, 4}",
    )
    contract = selected_policy_contract(objective, nfe)
    require(policy["sampler"] == contract["sampler"], "cell policy sampler mismatch")
    require(
        policy["inference_seed_behavior"] == contract["inference_seed_behavior"],
        "cell inference seed behavior mismatch",
    )
    require(
        hmac.compare_digest(policy["identity_sha256"], canonical_sha256(contract)),
        "cell selected policy identity SHA-256 mismatch",
    )
    require(
        candidate["cell_id"] == official_cell_id(train_seed, objective, nfe, execution_horizon),
        "pre-registration cell_id is not canonical",
    )
    if roots is None or final_freeze_token_sha256 is None:
        require(
            roots is None and final_freeze_token_sha256 is None,
            "output claim validation context is incomplete",
        )
        output_claim = candidate["output_claim"]
        require(isinstance(output_claim, Mapping), "cell output claim must be an object")
        _require_exact_keys(output_claim, _OUTPUT_CLAIM_FIELDS, "cell output claim")
        require(output_claim["schema"] == OUTPUT_CLAIM_SCHEMA, "cell output claim schema mismatch")
        _require_sha256(output_claim["claim_id_sha256"], "cell output claim ID")
        for name in ("claim_path", "output_dir"):
            require(
                isinstance(output_claim[name], str) and Path(output_claim[name]).is_absolute(),
                f"cell output claim {name} is invalid",
            )
    else:
        validate_output_claim(
            candidate["output_claim"],
            cell_id=candidate["cell_id"],
            roots=roots,
            final_freeze_token_sha256=final_freeze_token_sha256,
        )
    return dict(candidate)


def validate_preregistration_manifest(
    manifest: Any,
    generated_sequences: Sequence[Any],
    *,
    runtime_attestation_sha256: str,
) -> List[Dict[str, Any]]:
    """Validate the complete sealed 24-cell official evaluation contract."""

    require(isinstance(manifest, dict), "pre-registration manifest must be an object")
    _require_exact_keys(manifest, _PREREGISTRATION_FIELDS, "pre-registration manifest")
    require(manifest["schema"] == PREREGISTRATION_SCHEMA, "pre-registration schema mismatch")
    roots = validate_official_output_roots(manifest["official_output_roots"])
    require(
        manifest["aggregation_python_version"] == PYTHON_VERSION,
        f"pre-registration aggregation runtime must be Python {PYTHON_VERSION}",
    )
    _require_sha256(manifest["aggregator_sha256"], "pre-registration aggregator_sha256")
    require(manifest["benchmark_protocol"] == PROTOCOL, "pre-registration benchmark protocol mismatch")
    require(
        isinstance(manifest["training_seeds"], list)
        and all(type(value) is int for value in manifest["training_seeds"])
        and manifest["training_seeds"] == list(OFFICIAL_TRAIN_SEEDS),
        "pre-registration training seeds changed",
    )
    require(
        type(manifest["direct_nfe"]) is int and manifest["direct_nfe"] == OFFICIAL_DIRECT_NFE,
        "pre-registration direct NFE changed",
    )
    require(
        isinstance(manifest["execution_horizons"], list)
        and all(type(value) is int for value in manifest["execution_horizons"])
        and manifest["execution_horizons"] == list(SUPPORTED_EXECUTION_HORIZONS),
        "pre-registration execution horizons changed",
    )
    require(
        isinstance(manifest["flow_nfes"], list)
        and all(type(value) is int for value in manifest["flow_nfes"])
        and manifest["flow_nfes"] == list(OFFICIAL_FLOW_NFES),
        "pre-registration flow NFEs changed",
    )
    require(
        type(manifest["final_checkpoint_update"]) is int
        and manifest["final_checkpoint_update"] == FINAL_CHECKPOINT_UPDATE,
        "pre-registration final checkpoint update changed",
    )
    require(
        manifest["inference_seed_domain"] == INFERENCE_SEED_DOMAIN,
        "pre-registration inference seed domain changed",
    )
    require(
        type(manifest["policy_warmup_calls"]) is int and manifest["policy_warmup_calls"] >= MIN_POLICY_WARMUP_CALLS,
        f"pre-registration policy warm-up count must be at least {MIN_POLICY_WARMUP_CALLS}",
    )
    require(
        type(manifest["subtasks_per_sequence"]) is int and manifest["subtasks_per_sequence"] == SUBTASKS_PER_SEQUENCE,
        "pre-registration subtasks-per-sequence changed",
    )
    _require_sha256(manifest["runtime_attestation_sha256"], "pre-registration runtime_attestation_sha256")
    _require_sha256(runtime_attestation_sha256, "current runtime attestation SHA-256")
    require(
        hmac.compare_digest(manifest["runtime_attestation_sha256"], runtime_attestation_sha256),
        "current runtime/data attestation differs from pre-registration",
    )
    require(
        type(manifest["evaluation_seed"]) is int and manifest["evaluation_seed"] == EVALUATION_SEED,
        "pre-registration evaluation_seed must equal integer 0",
    )
    require(
        type(manifest["sequence_count"]) is int and manifest["sequence_count"] == NUM_SEQUENCES,
        f"pre-registration must contain exactly {NUM_SEQUENCES} sequences",
    )
    _require_sha256(manifest["sequence_sha256"], "pre-registration sequence_sha256")
    require(manifest["sequence_sha256"] == SEQUENCE_SHA256, "pre-registration sequence digest mismatch")
    registered_sequences = manifest["sequences"]
    require(
        isinstance(registered_sequences, list) and len(registered_sequences) == NUM_SEQUENCES,
        f"pre-registration sequences must contain exactly {NUM_SEQUENCES} entries",
    )
    require(
        hmac.compare_digest(canonical_sequence_sha256(registered_sequences), SEQUENCE_SHA256),
        "pre-registration sequence payload digest mismatch",
    )
    require(
        hmac.compare_digest(canonical_json_bytes(registered_sequences), canonical_json_bytes(generated_sequences)),
        "pre-registered sequences differ from the freshly generated official sequences",
    )
    _require_sha256(manifest["final_freeze_token_sha256"], "final_freeze_token_sha256")

    cells = manifest["cells"]
    require(isinstance(cells, list), "pre-registration cells must be a list")
    checked_cells = [
        _validate_registered_cell(
            candidate,
            roots=roots,
            final_freeze_token_sha256=manifest["final_freeze_token_sha256"],
        )
        for candidate in cells
    ]
    cell_ids = [candidate["cell_id"] for candidate in checked_cells]
    require(len(cell_ids) == len(set(cell_ids)), "pre-registration cell_id values must be unique")
    factors = {
        (
            candidate["policy"]["train_seed"],
            candidate["policy"]["objective"],
            candidate["policy"]["nfe"],
            candidate["execution_horizon"],
        )
        for candidate in checked_cells
    }
    require(factors == official_factor_matrix(), "pre-registration does not contain the exact 24-cell factor matrix")
    require(len(checked_cells) == len(factors) == 24, "pre-registration must contain exactly 24 policy cells")
    require(
        len({candidate["output_claim"]["output_dir"] for candidate in checked_cells}) == 24,
        "pre-registration output directories must be unique",
    )
    require(
        len({candidate["output_claim"]["claim_path"] for candidate in checked_cells}) == 24,
        "pre-registration claim paths must be unique",
    )
    require(
        len({candidate["output_claim"]["claim_id_sha256"] for candidate in checked_cells}) == 24,
        "pre-registration claim identities must be unique",
    )

    require(
        len({canonical_json_bytes(candidate["execution_geometry"]) for candidate in checked_cells}) == 1,
        "all official cells must share one execution geometry",
    )
    require(
        len({candidate["serving_runtime_sha256"] for candidate in checked_cells}) == 1,
        "all official cells must share one evaluation serving runtime",
    )
    for seed in OFFICIAL_TRAIN_SEEDS:
        for objective in ("rectified_flow", "direct_regression"):
            group = [
                candidate
                for candidate in checked_cells
                if candidate["policy"]["train_seed"] == seed and candidate["policy"]["objective"] == objective
            ]
            require(
                len({candidate["checkpoint"]["sha256"] for candidate in group}) == 1,
                "NFE/K cells must share one final checkpoint per seed/objective",
            )
        flow_checkpoint = next(
            candidate["checkpoint"]["sha256"]
            for candidate in checked_cells
            if candidate["policy"]["train_seed"] == seed and candidate["policy"]["objective"] == "rectified_flow"
        )
        direct_checkpoint = next(
            candidate["checkpoint"]["sha256"]
            for candidate in checked_cells
            if candidate["policy"]["train_seed"] == seed and candidate["policy"]["objective"] == "direct_regression"
        )
        require(flow_checkpoint != direct_checkpoint, "flow and direct cells cannot share a checkpoint manifest")
    require(
        len({candidate["checkpoint"]["sha256"] for candidate in checked_cells}) == 6,
        "official matrix must contain one distinct final checkpoint per seed/objective",
    )
    return checked_cells


def load_preregistration(
    path: Path,
    generated_sequences: Sequence[Any],
    *,
    cell_id: str,
    execution_horizon: int,
    final_freeze_token: str,
    preregistration_sha256: str,
    runtime_attestation_sha256: str,
) -> Tuple[Dict[str, Any], Dict[str, Any], str]:
    """Authenticate the externally frozen file, runtime, sequences, cell, and token."""

    manifest, manifest_sha256 = _read_strict_json(path, expected_sha256=preregistration_sha256)
    checked_cells = validate_preregistration_manifest(
        manifest,
        generated_sequences,
        runtime_attestation_sha256=runtime_attestation_sha256,
    )
    require(
        isinstance(final_freeze_token, str) and bool(final_freeze_token.strip()),
        "official-score requires an explicit non-empty --final-freeze-token string",
    )
    token_sha256 = hashlib.sha256(final_freeze_token.encode("utf-8")).hexdigest()
    require(
        hmac.compare_digest(token_sha256, manifest["final_freeze_token_sha256"]),
        "final freeze token does not match the pre-registration manifest",
    )

    matching = [candidate for candidate in checked_cells if candidate["cell_id"] == cell_id]
    require(len(matching) == 1, "selected --cell-id is absent from the pre-registration manifest")
    selected_cell = matching[0]
    require(
        selected_cell["execution_horizon"] == execution_horizon,
        "CLI execution horizon differs from the pre-registered cell",
    )
    return dict(manifest), selected_cell, manifest_sha256


def build_claim_record(
    output_claim: Mapping[str, Any],
    *,
    cell_id: str,
    preregistration_sha256: str,
    final_freeze_token_sha256: str,
    created_utc: str,
) -> Dict[str, Any]:
    """Build the immutable external record that consumes one cell attempt."""

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
) -> Dict[str, Any]:
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


def _read_stable_regular_bytes(path: Path, *, name: str) -> Tuple[bytes, Dict[str, Any]]:
    """Read one non-symlink regular file once and prove descriptor stability."""

    flags = os.O_RDONLY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(str(path), flags)
    except OSError as exc:
        raise RuntimeError(f"cannot open authenticated {name}: {path}") from exc
    chunks = []  # type: List[bytes]
    size = 0
    digest = hashlib.sha256()
    try:
        before = os.fstat(descriptor)
        require(stat.S_ISREG(before.st_mode), f"authenticated {name} is not a regular file")
        with os.fdopen(descriptor, "rb") as source:
            descriptor = -1
            while True:
                block = source.read(1024 * 1024)
                if not block:
                    break
                chunks.append(block)
                size += len(block)
                digest.update(block)
            after = os.fstat(source.fileno())
        stable_fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
        require(
            all(getattr(before, field) == getattr(after, field) for field in stable_fields) and size == after.st_size,
            f"authenticated {name} changed while being read",
        )
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    return b"".join(chunks), {"bytes": size, "path": str(path.resolve()), "sha256": digest.hexdigest()}


def _read_published_target_identity(path: Path, *, name: str) -> Dict[str, Any]:
    """Read one publication target without following links and bind its inode."""

    require(hasattr(os, "O_NOFOLLOW"), "publication verification requires O_NOFOLLOW")
    flags = os.O_RDONLY | os.O_NOFOLLOW
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
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
        stable_fields = ("st_dev", "st_ino", "st_nlink", "st_size", "st_mtime_ns", "st_ctime_ns")
        require(
            all(getattr(before, field) == getattr(after, field) for field in stable_fields) and size == after.st_size,
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


def _publication_identity(value: Mapping[str, Any]) -> Dict[str, Any]:
    """Drop the expected-to-change link count from a target identity."""

    return {
        "bytes": value["bytes"],
        "device": value["device"],
        "inode": value["inode"],
        "sha256": value["sha256"],
    }


def _verify_published_target(
    path: Path,
    expected: Mapping[str, Any],
    *,
    links: int,
    name: str,
) -> None:
    """Require a path to still name the exact created inode and content."""

    require(
        set(expected) == {"bytes", "device", "inode", "sha256"},
        f"published {name} identity fields are invalid",
    )
    observed = _read_published_target_identity(path, name=name)
    require(
        _publication_identity(observed) == expected and observed["links"] == links,
        f"published {name} target identity, content, or link count changed",
    )


def _unlink_created_path(path: Path, expected: Optional[Mapping[str, Any]]) -> bool:
    """Unlink a path only when it still names the inode created by this process."""

    if expected is None:
        return False
    try:
        current = os.stat(str(path), follow_symlinks=False)
    except OSError:
        return False
    if not stat.S_ISREG(current.st_mode):
        return False
    if current.st_dev != expected.get("device") or current.st_ino != expected.get("inode"):
        return False
    try:
        path.unlink()
    except OSError:
        return False
    return True


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(str(path), os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def publish_bytes_and_sha256_exclusive(
    path: Path,
    payload: bytes,
    *,
    commit_guard: Optional[Callable[[], None]] = None,
) -> str:
    """Durably publish an immutable payload/sidecar pair without overwriting either.

    The sidecar is linked first and the payload is the commit marker.  A caller
    that observes the payload after the final directory fsync therefore also
    observes its complete sidecar.  Any in-process failure removes only links
    created by this call; pre-existing collision targets are never changed.
    """

    require(isinstance(payload, bytes), "exclusive publication payload must be bytes")
    parent = path.parent
    parent.mkdir(parents=True, exist_ok=True)
    companion = path.with_suffix(path.suffix + ".sha256")
    require(companion != path, "exclusive publication companion path collides with payload path")
    digest = hashlib.sha256(payload).hexdigest()
    companion_payload = f"{digest}  {path.name}\n".encode("ascii")
    nonce = f"{os.getpid()}-{time.time_ns()}-{secrets.token_hex(8)}"
    temporary_payload = parent / f".{path.name}.tmp-{nonce}"
    temporary_companion = parent / f".{companion.name}.tmp-{nonce}"
    created_payload = False
    created_companion = False
    temporary_payload_identity = None  # type: Optional[Dict[str, Any]]
    temporary_companion_identity = None  # type: Optional[Dict[str, Any]]

    def _write_temporary(candidate: Path, content: bytes, name: str) -> Dict[str, Any]:
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        require(hasattr(os, "O_NOFOLLOW"), "exclusive publication requires O_NOFOLLOW")
        flags |= os.O_NOFOLLOW
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

    def _verify_temporaries(*, links: int) -> None:
        require(
            temporary_payload_identity is not None and temporary_companion_identity is not None,
            "publication temporary identities are missing",
        )
        _verify_published_target(
            temporary_payload,
            temporary_payload_identity,
            links=links,
            name="payload temporary",
        )
        _verify_published_target(
            temporary_companion,
            temporary_companion_identity,
            links=links,
            name="SHA-256 temporary",
        )

    def _verify_final_targets(*, links: int, payload_exists: bool) -> None:
        require(temporary_companion_identity is not None, "publication companion identity is missing")
        _verify_published_target(
            companion,
            temporary_companion_identity,
            links=links,
            name="SHA-256 companion",
        )
        if payload_exists:
            require(temporary_payload_identity is not None, "publication payload identity is missing")
            _verify_published_target(
                path,
                temporary_payload_identity,
                links=links,
                name="payload",
            )

    try:
        temporary_payload_identity = _write_temporary(temporary_payload, payload, "payload")
        temporary_companion_identity = _write_temporary(temporary_companion, companion_payload, "SHA-256 companion")
        _verify_temporaries(links=1)
        if commit_guard is not None:
            commit_guard()
        _verify_temporaries(links=1)
        os.link(str(temporary_companion), str(companion), follow_symlinks=False)
        created_companion = True
        _verify_published_target(
            temporary_payload,
            temporary_payload_identity,
            links=1,
            name="payload temporary",
        )
        _verify_published_target(
            temporary_companion,
            temporary_companion_identity,
            links=2,
            name="SHA-256 temporary",
        )
        _verify_final_targets(links=2, payload_exists=False)
        if commit_guard is not None:
            commit_guard()
        _verify_published_target(
            temporary_payload,
            temporary_payload_identity,
            links=1,
            name="payload temporary",
        )
        _verify_published_target(
            temporary_companion,
            temporary_companion_identity,
            links=2,
            name="SHA-256 temporary",
        )
        _verify_final_targets(links=2, payload_exists=False)
        os.link(str(temporary_payload), str(path), follow_symlinks=False)
        created_payload = True
        _fsync_directory(parent)
        _verify_temporaries(links=2)
        _verify_final_targets(links=2, payload_exists=True)
        if commit_guard is not None:
            commit_guard()
        _verify_temporaries(links=2)
        _verify_final_targets(links=2, payload_exists=True)
        require(
            _unlink_created_path(temporary_payload, temporary_payload_identity),
            "payload temporary target changed before cleanup",
        )
        require(
            _unlink_created_path(temporary_companion, temporary_companion_identity),
            "SHA-256 temporary target changed before cleanup",
        )
        _fsync_directory(parent)
        _verify_final_targets(links=1, payload_exists=True)
        return digest
    except BaseException:
        if created_payload:
            _unlink_created_path(path, temporary_payload_identity)
        if created_companion:
            _unlink_created_path(companion, temporary_companion_identity)
        _unlink_created_path(temporary_payload, temporary_payload_identity)
        _unlink_created_path(temporary_companion, temporary_companion_identity)
        with contextlib.suppress(OSError):
            _fsync_directory(parent)
        raise
    finally:
        _unlink_created_path(temporary_payload, temporary_payload_identity)
        _unlink_created_path(temporary_companion, temporary_companion_identity)


def _capture_exclusive_pair(
    path: Path,
    *,
    expected_sha256: str,
    name: str,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Authenticate one already-published payload/sidecar pair and its inodes."""

    _require_sha256(expected_sha256, f"{name} SHA-256")
    payload_observed = _read_published_target_identity(path, name=name)
    require(
        payload_observed["links"] == 1 and payload_observed["sha256"] == expected_sha256,
        f"published {name} digest or link count is invalid",
    )
    companion = path.with_suffix(path.suffix + ".sha256")
    companion_observed = _read_published_target_identity(companion, name=f"{name} SHA-256 companion")
    expected_companion = f"{expected_sha256}  {path.name}\n".encode("ascii")
    require(
        companion_observed["links"] == 1
        and companion_observed["bytes"] == len(expected_companion)
        and companion_observed["sha256"] == hashlib.sha256(expected_companion).hexdigest(),
        f"published {name} SHA-256 companion is invalid",
    )
    return _publication_identity(payload_observed), _publication_identity(companion_observed)


def write_json_atomic(path: Path, value: Mapping[str, Any]) -> Dict[str, Any]:
    """Atomically replace JSON and return its verified target inode/content identity."""

    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}-{time.time_ns()}")
    payload = (json.dumps(value, allow_nan=False, ensure_ascii=True, indent=2, sort_keys=True) + "\n").encode("ascii")
    temporary_identity = None  # type: Optional[Dict[str, Any]]
    try:
        require(hasattr(os, "O_NOFOLLOW"), "atomic JSON publication requires O_NOFOLLOW")
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW
        descriptor = os.open(str(temporary), flags, 0o600)
        with os.fdopen(descriptor, "wb") as sink:
            sink.write(payload)
            sink.flush()
            os.fsync(sink.fileno())
        observed = _read_published_target_identity(temporary, name="atomic JSON temporary")
        require(
            observed["bytes"] == len(payload)
            and observed["sha256"] == hashlib.sha256(payload).hexdigest()
            and observed["links"] == 1,
            "atomic JSON temporary identity is invalid",
        )
        temporary_identity = _publication_identity(observed)
        os.replace(str(temporary), str(path))
        _fsync_directory(path.parent)
        _verify_published_target(path, temporary_identity, links=1, name="atomic JSON")
        return temporary_identity
    finally:
        _unlink_created_path(temporary, temporary_identity)


class EvaluationJournal:
    """Durable, interruption-auditable owner of one official score directory."""

    def __init__(
        self,
        output_dir: Path,
        run_manifest: Mapping[str, Any],
        *,
        commit_guard: Optional[Callable[[], None]] = None,
    ) -> None:
        self.output_dir = Path(output_dir)
        require(self.output_dir.is_absolute(), "evaluation journal output directory must be absolute")
        self.run_path = self.output_dir / "run.json"
        self.episodes_path = self.output_dir / "episodes.jsonl"
        self.summary_path = self.output_dir / "summary.json"
        self.completion_path = self.output_dir / "completion.json"
        self.run_manifest = dict(run_manifest)
        self._episode_sink = None  # type: Optional[Any]
        self._episode_digest = hashlib.sha256()
        self._episode_bytes = 0
        self._episode_count = 0
        self._episodes_identity = None  # type: Optional[Dict[str, Any]]
        self._claim_targets = ()  # type: Sequence[Tuple[Path, Mapping[str, Any], str]]
        self._commit_guard = commit_guard

    def start(self) -> None:
        claimed = False
        try:
            self.output_dir.mkdir(parents=False, exist_ok=False, mode=0o700)
            claimed = True
            self.run_manifest["status"] = "running"
            write_json_atomic(self.run_path, self.run_manifest)
            require(hasattr(os, "O_NOFOLLOW"), "evaluation journal requires O_NOFOLLOW")
            descriptor = os.open(
                str(self.episodes_path),
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                0o600,
            )
            self._episode_sink = os.fdopen(descriptor, "w", encoding="utf-8")
            self._episode_sink.flush()
            os.fsync(self._episode_sink.fileno())
            opened = os.fstat(self._episode_sink.fileno())
            require(
                stat.S_ISREG(opened.st_mode) and opened.st_nlink == 1 and opened.st_size == 0,
                "evaluation journal episodes target is invalid",
            )
            self._episodes_identity = {
                "bytes": 0,
                "device": opened.st_dev,
                "inode": opened.st_ino,
                "sha256": self._episode_digest.hexdigest(),
            }
            _verify_published_target(
                self.episodes_path,
                self._episodes_identity,
                links=1,
                name="episodes JSONL",
            )
            _fsync_directory(self.output_dir)
        except BaseException as exc:
            if claimed:
                try:
                    self.fail(exc)
                except BaseException as fail_exc:
                    raise RuntimeError(
                        f"evaluation journal start failed and could not record failure: "
                        f"{type(fail_exc).__name__}: {fail_exc}"
                    ) from exc
            raise

    def bind_claim(self, claim_path: Path, claim_json_sha256: str) -> None:
        """Bind the immutable external attempt claim into the durable run journal."""

        require(self.run_manifest.get("status") == "running", "evaluation journal is not running")
        registered = self.run_manifest.get("output_claim")
        if registered is not None:
            registered_path = registered.get("claim_path") if isinstance(registered, Mapping) else None
            require(
                isinstance(registered_path, str) and Path(registered_path) == claim_path,
                "journal claim path differs from the pre-registered output claim",
            )
        payload_identity, companion_identity = _capture_exclusive_pair(
            claim_path,
            expected_sha256=claim_json_sha256,
            name="official claim JSON",
        )
        self._claim_targets = (
            (claim_path, payload_identity, "official claim JSON"),
            (
                claim_path.with_suffix(claim_path.suffix + ".sha256"),
                companion_identity,
                "official claim SHA-256 companion",
            ),
        )
        self._guard_targets(())
        self.run_manifest["claim_json_sha256"] = claim_json_sha256
        self.update_running({})

    def append_episode(self, record: Mapping[str, Any]) -> None:
        require(self._episode_sink is not None, "evaluation journal is not running")
        require(record.get("sequence_idx") == self._episode_count, "episode append order is not contiguous")
        line = json.dumps(record, allow_nan=False, ensure_ascii=True, separators=(",", ":"), sort_keys=True) + "\n"
        self._episode_sink.write(line)
        self._episode_sink.flush()
        os.fsync(self._episode_sink.fileno())
        encoded = line.encode("ascii")
        self._episode_digest.update(encoded)
        self._episode_bytes += len(encoded)
        self._episode_count += 1

    def update_running(self, values: Mapping[str, Any]) -> None:
        """Durably checkpoint pre-episode setup without relinquishing ownership."""

        require(self.run_manifest.get("status") == "running", "evaluation journal is not running")
        require(self._episode_sink is not None, "evaluation journal episode stream is not open")
        self.run_manifest.update(dict(values))
        self._guard_targets(())
        write_json_atomic(self.run_path, self.run_manifest)
        self._guard_targets(())

    def _close_episodes(self) -> None:
        if self._episode_sink is not None:
            sink = self._episode_sink
            try:
                sink.flush()
                os.fsync(sink.fileno())
                closed = os.fstat(sink.fileno())
                require(
                    self._episodes_identity is not None
                    and stat.S_ISREG(closed.st_mode)
                    and closed.st_dev == self._episodes_identity["device"]
                    and closed.st_ino == self._episodes_identity["inode"]
                    and closed.st_nlink == 1
                    and closed.st_size == self._episode_bytes,
                    "evaluation journal episodes descriptor identity changed",
                )
                self._episodes_identity = {
                    "bytes": self._episode_bytes,
                    "device": closed.st_dev,
                    "inode": closed.st_ino,
                    "sha256": self._episode_digest.hexdigest(),
                }
            finally:
                sink.close()
                self._episode_sink = None
        require(self._episodes_identity is not None, "evaluation journal episodes identity is missing")
        _verify_published_target(
            self.episodes_path,
            self._episodes_identity,
            links=1,
            name="episodes JSONL",
        )

    def _guard_targets(self, targets: Sequence[Tuple[Path, Mapping[str, Any], str]]) -> None:
        guarded = tuple(self._claim_targets) + tuple(targets)
        for path, identity, name in guarded:
            _verify_published_target(path, identity, links=1, name=name)
        if self._commit_guard is not None:
            self._commit_guard()
        for path, identity, name in guarded:
            _verify_published_target(path, identity, links=1, name=name)

    def complete(self, summary: Mapping[str, Any]) -> None:
        try:
            if self.run_manifest.get("mode") == "official-score":
                require(
                    "claim_json_sha256" in self.run_manifest and "output_claim" in self.run_manifest,
                    "official score cannot complete without its immutable external claim",
                )
            self._close_episodes()
            require(self._episode_count == NUM_SEQUENCES, "official score did not write exactly 1000 sequence records")
            require(self._episodes_identity is not None, "evaluation journal episodes identity is missing")
            episode_target = (self.episodes_path, self._episodes_identity, "episodes JSONL")
            self._guard_targets((episode_target,))
            summary_identity = write_json_atomic(self.summary_path, summary)
            summary_target = (self.summary_path, summary_identity, "summary JSON")
            self._guard_targets((episode_target, summary_target))
            self.run_manifest.update(
                {
                    "episodes_jsonl_sha256": self._episode_digest.hexdigest(),
                    "finished_utc": datetime.now(timezone.utc).isoformat(),
                    "sequence_records": self._episode_count,
                    "status": "complete",
                    "summary_json_sha256": summary_identity["sha256"],
                }
            )
            run_identity = write_json_atomic(self.run_path, self.run_manifest)
            run_target = (self.run_path, run_identity, "complete run JSON")
            self._guard_targets((episode_target, summary_target, run_target))
            if "claim_json_sha256" in self.run_manifest:
                completion = {
                    "cell_id": self.run_manifest["cell"]["cell_id"],
                    "claim_json_sha256": self.run_manifest["claim_json_sha256"],
                    "episodes_jsonl_sha256": self.run_manifest["episodes_jsonl_sha256"],
                    "preregistration_sha256": self.run_manifest["preregistration_sha256"],
                    "run_json_sha256": run_identity["sha256"],
                    "schema": COMPLETION_SCHEMA,
                    "sequence_records": self._episode_count,
                    "summary_json_sha256": self.run_manifest["summary_json_sha256"],
                }
                completion_payload = (
                    json.dumps(completion, allow_nan=False, ensure_ascii=True, indent=2, sort_keys=True) + "\n"
                ).encode("ascii")

                def completion_guard() -> None:
                    self._guard_targets((episode_target, summary_target, run_target))

                completion_sha256 = publish_bytes_and_sha256_exclusive(
                    self.completion_path,
                    completion_payload,
                    commit_guard=completion_guard,
                )
                completion_identity, companion_identity = _capture_exclusive_pair(
                    self.completion_path,
                    expected_sha256=completion_sha256,
                    name="official completion JSON",
                )
                self._guard_targets(
                    (
                        episode_target,
                        summary_target,
                        run_target,
                        (self.completion_path, completion_identity, "official completion JSON"),
                        (
                            self.completion_path.with_suffix(self.completion_path.suffix + ".sha256"),
                            companion_identity,
                            "official completion SHA-256 companion",
                        ),
                    )
                )
        except BaseException as exc:
            self.fail(exc)
            raise

    def fail(self, error: BaseException) -> None:
        close_error = None  # type: Optional[BaseException]
        try:
            self._close_episodes()
        except BaseException as exc:
            close_error = exc
        message = str(error)
        if close_error is not None:
            message = f"{message}; journal target integrity failure: {type(close_error).__name__}: {close_error}"
        self.run_manifest.update(
            {
                "error": {"message": message, "type": type(error).__name__},
                "failed_utc": datetime.now(timezone.utc).isoformat(),
                "partial_episodes_jsonl_sha256": self._episode_digest.hexdigest(),
                "sequence_records": self._episode_count,
                "status": "failed",
            }
        )
        write_json_atomic(self.run_path, self.run_manifest)


def _official_conf_dir() -> Path:
    import calvin_agent

    module_path = Path(calvin_agent.__file__).resolve()
    conf_dir = module_path.parent.parent / "conf"
    require(conf_dir.is_dir(), "cannot locate official calvin_agent configuration")
    return conf_dir


def _verify_attested_file(
    path: Path,
    expected_identity: Mapping[str, Any],
    *,
    name: str,
    pinned_sha256: Optional[str] = None,
) -> Tuple[bytes, Dict[str, Any]]:
    """Read and authenticate the exact bytes that the caller will parse."""

    require(isinstance(expected_identity, Mapping), f"{name} attestation identity must be an object")
    expected_path = expected_identity.get("path")
    expected_bytes = expected_identity.get("bytes")
    expected_sha256 = expected_identity.get("sha256")
    _require_sha256(expected_sha256, f"{name} attested SHA-256")
    require(
        isinstance(expected_path, str) and Path(expected_path).resolve() == path.resolve(),
        f"{name} attested path mismatch",
    )
    require(type(expected_bytes) is int and expected_bytes >= 0, f"{name} attested byte length is invalid")
    raw, identity = _read_stable_regular_bytes(path, name=name)
    require(identity["path"] == expected_path, f"{name} resolved path changed while being authenticated")
    require(identity["bytes"] == expected_bytes, f"{name} byte length differs from attestation")
    require(hmac.compare_digest(identity["sha256"], expected_sha256), f"{name} raw bytes differ from attestation")
    if pinned_sha256 is not None:
        _require_sha256(pinned_sha256, f"{name} pinned SHA-256")
        require(
            hmac.compare_digest(identity["sha256"], pinned_sha256),
            f"{name} raw bytes differ from pinned checkout",
        )
    return raw, identity


def load_validation_annotations(expected_identity: Mapping[str, Any]) -> Tuple[Any, Dict[str, Any]]:
    path = _official_conf_dir() / "annotations" / "new_playtable_validation.yaml"
    raw, identity = _verify_attested_file(
        path,
        expected_identity,
        name="validation annotation YAML",
        pinned_sha256=VALIDATION_ANNOTATIONS_SHA256,
    )
    from omegaconf import OmegaConf

    try:
        annotations = OmegaConf.create(raw.decode("utf-8"))
    except UnicodeDecodeError as exc:
        raise RuntimeError("validation annotation YAML is not UTF-8") from exc
    return annotations, identity


def load_task_oracle(expected_identity: Mapping[str, Any]) -> Tuple[Any, Dict[str, Any]]:
    path = _official_conf_dir() / "callbacks" / "rollout" / "tasks" / "new_playtable_tasks.yaml"
    raw, identity = _verify_attested_file(
        path,
        expected_identity,
        name="task oracle YAML",
        pinned_sha256=TASK_ORACLE_SHA256,
    )
    import hydra
    from omegaconf import OmegaConf

    try:
        task_config = OmegaConf.create(raw.decode("utf-8"))
    except UnicodeDecodeError as exc:
        raise RuntimeError("task oracle YAML is not UTF-8") from exc
    return hydra.utils.instantiate(task_config), identity


def construct_validation_environment(
    dataset_root: Path,
    expected_config_identity: Mapping[str, Any],
) -> Tuple[Any, Dict[str, Any]]:
    """Authenticate validation-D merged config and instantiate only its two RGB cameras."""

    dataset_path = dataset_root.resolve()
    validation_path = dataset_path / "validation"
    merged_config_path = validation_path / ".hydra" / "merged_config.yaml"
    require(validation_path.is_dir(), "CALVIN dataset validation directory is missing")
    raw, config_identity = _verify_attested_file(
        merged_config_path,
        expected_config_identity,
        name="CALVIN validation merged config",
    )
    import hydra
    from omegaconf import OmegaConf

    try:
        config = OmegaConf.create(raw.decode("utf-8"))
    except UnicodeDecodeError as exc:
        raise RuntimeError("CALVIN validation merged config is not UTF-8") from exc
    require(OmegaConf.select(config, "scene.name") == VALIDATION_SCENE, "CALVIN validation config is not scene D")
    require(
        OmegaConf.select(config, "env._target_") == "calvin_env.envs.play_table_env.PlayTableSimEnv",
        "CALVIN validation environment target changed",
    )
    require(
        OmegaConf.select(config, "env.control_freq") == CONTROL_FREQUENCY_HZ,
        "CALVIN validation control frequency must be 30 Hz",
    )
    require(OmegaConf.select(config, "env.use_egl") is True, "CALVIN validation config must use EGL")
    require(
        OmegaConf.select(config, "cameras.static.width") == STATIC_IMAGE_SHAPE[1]
        and OmegaConf.select(config, "cameras.static.height") == STATIC_IMAGE_SHAPE[0],
        "CALVIN validation static camera shape changed",
    )
    require(
        OmegaConf.select(config, "cameras.gripper.width") == GRIPPER_IMAGE_SHAPE[1]
        and OmegaConf.select(config, "cameras.gripper.height") == GRIPPER_IMAGE_SHAPE[0],
        "CALVIN validation gripper camera shape changed",
    )
    selected_cameras = {"static", "gripper"}
    configured_cameras = set(config.cameras.keys())
    require(selected_cameras <= configured_cameras, "CALVIN validation config is missing a required RGB camera")
    for camera_name in configured_cameras - selected_cameras:
        del config.cameras[camera_name]
    require(set(config.cameras.keys()) == selected_cameras, "CALVIN validation camera filtering failed")
    if not hydra.core.global_hydra.GlobalHydra.instance().is_initialized():
        hydra.initialize(".")
    environment = hydra.utils.instantiate(
        config.env,
        show_gui=False,
        use_vr=False,
        use_scene_info=True,
    )
    require(environment.control_freq == CONTROL_FREQUENCY_HZ, "constructed CALVIN environment is not 30 Hz")
    return environment, {
        "control_frequency_hz": CONTROL_FREQUENCY_HZ,
        "merged_config_path": str(merged_config_path),
        "merged_config_sha256": config_identity["sha256"],
        "scene": VALIDATION_SCENE,
        "validation_path": str(validation_path),
    }


def _close_environment(environment: Any) -> None:
    environment.close()
    # The pinned upstream destructor otherwise performs a noisy second close.
    if hasattr(environment, "cid"):
        environment.cid = -1
    if hasattr(environment, "p"):
        environment.p = None


def run_infrastructure_mode(
    dataset_root: Path,
    evaluation_seed: int,
    source_root: Path,
) -> Dict[str, Any]:
    """Run outcome-blind validation-D infrastructure checks only."""

    _require_evaluator_sources_unchanged(_IMPORT_EVALUATOR_SOURCE_IDENTITIES)
    require(evaluation_seed == EVALUATION_SEED, "official CALVIN evaluation_seed must equal 0")
    attestation = build_official_attestation(source_root, dataset_root, script_dir=_SCRIPT_DIR)
    _require_attestation_sources_match_import(attestation)
    attestation_sha256 = attestation["attestation_sha256"]
    _require_sha256(attestation_sha256, "runtime/data attestation SHA-256")
    runtime_identity = attestation["runtime"]
    dataset_identity = attestation["dataset"]
    official_yaml = runtime_identity["official_yaml"]
    sequences = regenerate_official_sequences(evaluation_seed)
    annotations, annotation_identity = load_validation_annotations(official_yaml["validation_annotations"])
    unique_tasks = sorted({task for _state, tasks in sequences for task in tasks})
    phrases = {task: first_language_phrase(annotations, task) for task in unique_tasks}

    from calvin_agent.evaluation.utils import get_env_state_for_initial_condition

    environment, environment_identity = construct_validation_environment(
        dataset_root,
        dataset_identity["validation_critical_files"]["validation/.hydra/merged_config.yaml"],
    )
    try:
        robot_obs, scene_obs = get_env_state_for_initial_condition(sequences[0][0])
        observation = environment.reset(
            robot_obs=np.asarray(robot_obs).copy(),
            scene_obs=np.asarray(scene_obs).copy(),
        )
        rgb_static, rgb_gripper, state = _policy_observation(observation)
        # Do not call environment.step(), PolicyClient, predict(), get_info(),
        # or task_oracle in this mode.
        report = {
            "annotation": annotation_identity,
            "attestation": attestation,
            "attestation_sha256": attestation_sha256,
            "environment": environment_identity,
            "evaluation_seed": evaluation_seed,
            "first_phrases_sha256": hashlib.sha256(canonical_json_bytes(phrases)).hexdigest(),
            "mode": "infrastructure",
            "observation": {
                "gripper_rgb_dtype": str(rgb_gripper.dtype),
                "gripper_rgb_shape": list(rgb_gripper.shape),
                "state_dtype": str(state.dtype),
                "state_shape": list(state.shape),
                "static_rgb_dtype": str(rgb_static.dtype),
                "static_rgb_shape": list(rgb_static.shape),
            },
            "oracle_outcomes_inspected": False,
            "policy_predict_calls": 0,
            "sequence_count": len(sequences),
            "sequence_sha256": canonical_sequence_sha256(sequences),
            "status": "ok",
            "unique_language_tasks": len(unique_tasks),
        }
        _require_evaluator_sources_unchanged(_IMPORT_EVALUATOR_SOURCE_IDENTITIES)
        return report
    finally:
        _close_environment(environment)


def run_official_score_mode(args: argparse.Namespace) -> Dict[str, Any]:
    """Execute one frozen pre-registered 1,000-sequence scoring cell."""

    _require_evaluator_sources_unchanged(_IMPORT_EVALUATOR_SOURCE_IDENTITIES)
    require(args.evaluation_seed == EVALUATION_SEED, "official CALVIN evaluation_seed must equal 0")
    # This gate performs every possible raw-byte, Git, package, module-origin,
    # platform, and validation-data check before YAML parsing, Hydra, the task
    # oracle, environment construction, or a policy connection can occur.
    attestation = build_official_attestation(
        args.source_root.resolve(),
        args.dataset_root,
        script_dir=_SCRIPT_DIR,
    )
    _require_attestation_sources_match_import(attestation)
    attestation_sha256 = attestation["attestation_sha256"]
    _require_sha256(attestation_sha256, "runtime/data attestation SHA-256")
    sequences = regenerate_official_sequences(args.evaluation_seed)
    manifest, selected_cell, preregistration_sha256 = load_preregistration(
        args.preregistration_manifest.resolve(),
        sequences,
        cell_id=args.cell_id,
        execution_horizon=args.execution_horizon,
        final_freeze_token=args.final_freeze_token,
        preregistration_sha256=args.preregistration_sha256,
        runtime_attestation_sha256=attestation_sha256,
    )
    roots = validate_official_output_roots(manifest["official_output_roots"], require_live=True)
    output_claim = selected_cell["output_claim"]
    expected_output_dir = Path(output_claim["output_dir"])
    expected_claim_path = Path(output_claim["claim_path"])
    require(
        args.output_dir.is_absolute() and args.output_dir == expected_output_dir,
        "official-score --output-dir differs from the pre-registered canonical output directory",
    )
    require(
        args.policy_warmup_calls == manifest["policy_warmup_calls"],
        "official-score policy warm-up count differs from the pre-registration",
    )
    runtime_identity = attestation["runtime"]
    dataset_identity = attestation["dataset"]
    official_yaml = runtime_identity["official_yaml"]
    created_utc = datetime.now(timezone.utc).isoformat()
    run_manifest = {
        "attestation": attestation,
        "attestation_sha256": attestation_sha256,
        "cell": selected_cell,
        "created_utc": created_utc,
        "evaluation_seed": args.evaluation_seed,
        "execution_horizon": args.execution_horizon,
        "final_freeze_token_sha256": hashlib.sha256(args.final_freeze_token.encode("utf-8")).hexdigest(),
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
        "schema": RUN_SCHEMA,
        "sequence_count": NUM_SEQUENCES,
        "sequence_sha256": SEQUENCE_SHA256,
    }

    def output_commit_guard() -> None:
        _require_evaluator_sources_unchanged(_IMPORT_EVALUATOR_SOURCE_IDENTITIES)
        validate_official_output_roots(roots, require_live=True)

    journal = EvaluationJournal(
        expected_output_dir,
        run_manifest,
        commit_guard=output_commit_guard,
    )
    _require_evaluator_sources_unchanged(_IMPORT_EVALUATOR_SOURCE_IDENTITIES)
    # Claim and journal the output before the first policy connection or call.
    # A collision, concurrent invocation, or later setup failure is therefore
    # incapable of producing an unaudited warm-up attempt for this cell.
    journal.start()
    environment = None
    try:
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
        claim_json_sha256 = publish_bytes_and_sha256_exclusive(
            expected_claim_path,
            claim_payload,
            commit_guard=output_commit_guard,
        )
        journal.bind_claim(expected_claim_path, claim_json_sha256)

        annotations, annotation_identity = load_validation_annotations(official_yaml["validation_annotations"])
        # Authenticate every instruction before constructing anything capable of
        # policy inference.
        for _initial_state, tasks in sequences:
            for task in tasks:
                first_language_phrase(annotations, task)
        journal.update_running({"annotation": annotation_identity})

        with PolicyClient(args.socket, timeout_seconds=args.policy_timeout_seconds) as client:
            health = validate_policy_health(
                client.health(),
                execution_horizon=args.execution_horizon,
                expected_cell=selected_cell,
                expected_calvin_identity=dataset_identity["calvin_identity"],
            )
            journal.update_running({"policy_health": health})
            partial_warmup_reports = []  # type: List[Dict[str, Any]]

            def record_warmup_attempt(warmup_index: int, intent: Mapping[str, Any]) -> None:
                journal.update_running(
                    {
                        "policy_warmup": {
                            "attempted_count": warmup_index + 1,
                            "completed_count": len(partial_warmup_reports),
                            "current_request": dict(intent),
                            "expected_count": args.policy_warmup_calls,
                            "included_in_episode_latency": False,
                            "reports": list(partial_warmup_reports),
                            "status": "running",
                        }
                    }
                )

            def record_partial_warmup(reports: Sequence[Mapping[str, Any]]) -> None:
                partial_warmup_reports[:] = [dict(report) for report in reports]
                journal.update_running(
                    {
                        "policy_warmup": {
                            "attempted_count": len(reports),
                            "completed_count": len(reports),
                            "current_request": None,
                            "expected_count": args.policy_warmup_calls,
                            "included_in_episode_latency": False,
                            "reports": [dict(report) for report in reports],
                            "status": "running",
                        }
                    }
                )

            policy_warmup = run_policy_warmups(
                client,
                health,
                count=args.policy_warmup_calls,
                sequences=sequences,
                language_annotations=annotations,
                execution_horizon=args.execution_horizon,
                attempt_callback=record_warmup_attempt,
                report_callback=record_partial_warmup,
            )
            journal.update_running({"policy_warmup": policy_warmup})
            # Only discarded synthetic policy calls are allowed before this point;
            # no validation-D observation or task-oracle outcome exists yet.
            from calvin_agent.evaluation.utils import get_env_state_for_initial_condition

            task_oracle, oracle_identity = load_task_oracle(official_yaml["task_oracle"])
            environment, environment_identity = construct_validation_environment(
                args.dataset_root,
                dataset_identity["validation_critical_files"]["validation/.hydra/merged_config.yaml"],
            )
            journal.update_running({"environment": environment_identity, "oracle": oracle_identity})
            records = evaluate_official_sequences(
                environment,
                client,
                task_oracle,
                sequences,
                annotations,
                get_env_state_for_initial_condition,
                train_seed=health["train_seed"],
                evaluation_seed=args.evaluation_seed,
                execution_horizon=args.execution_horizon,
                episode_callback=journal.append_episode,
            )
            summary = summarize_sequences(records)
            require(summary["sequence_count"] == NUM_SEQUENCES, "summary is not the complete official score")
            _require_evaluator_sources_unchanged(_IMPORT_EVALUATOR_SOURCE_IDENTITIES)
            environment_to_close = environment
            environment = None
            _close_environment(environment_to_close)
            journal.complete(summary)
    except BaseException as exc:
        failure = exc
        if environment is not None:
            environment_to_close = environment
            environment = None
            try:
                _close_environment(environment_to_close)
            except BaseException as close_exc:
                failure = RuntimeError(
                    f"{type(exc).__name__}: {exc}; environment close failed: {type(close_exc).__name__}: {close_exc}"
                )
        if journal.run_manifest.get("status") != "failed":
            journal.fail(failure)
        raise
    return summary


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    cache_root = Path(os.environ.get("DUO_VLA_CACHE_ROOT", "/root/.cache/duo-vla"))
    parser.add_argument("--mode", choices=("infrastructure", "official-score"), required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path(os.environ.get("CALVIN_SOURCE_ROOT", str(cache_root / "simulators/calvin"))),
    )
    parser.add_argument("--evaluation-seed", type=int, default=EVALUATION_SEED)
    parser.add_argument("--execution-horizon", type=int, choices=SUPPORTED_EXECUTION_HORIZONS)
    parser.add_argument(
        "--socket",
        type=Path,
    )
    parser.add_argument("--policy-timeout-seconds", type=float)
    parser.add_argument(
        "--policy-warmup-calls",
        type=int,
        help=(
            "discarded synthetic policy calls before scoring; required in official-score mode "
            f"(recommended {DEFAULT_POLICY_WARMUP_CALLS}, minimum {MIN_POLICY_WARMUP_CALLS})"
        ),
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--preregistration-manifest", type=Path)
    parser.add_argument(
        "--preregistration-sha256",
        help="externally stored raw SHA-256 of the frozen pre-registration JSON",
    )
    parser.add_argument("--cell-id")
    parser.add_argument(
        "--final-freeze-token",
        help="explicit string whose SHA-256 must match the frozen pre-registration; never recorded verbatim",
    )
    return parser.parse_args(argv)


def validate_mode_arguments(args: argparse.Namespace) -> None:
    require(args.evaluation_seed == EVALUATION_SEED, "official CALVIN evaluation_seed must equal 0")
    if args.mode == "infrastructure":
        require(args.execution_horizon is None, "infrastructure mode must not select a policy execution horizon")
        require(args.output_dir is None, "infrastructure mode does not create scoring artifacts")
        require(args.preregistration_manifest is None, "infrastructure mode must not receive a pre-registration")
        require(
            args.preregistration_sha256 is None,
            "infrastructure mode must not receive a pre-registration SHA-256",
        )
        require(args.cell_id is None, "infrastructure mode must not select a scoring cell")
        require(args.final_freeze_token is None, "infrastructure mode must not receive a final freeze token")
        require(args.policy_warmup_calls is None, "infrastructure mode must not receive policy warm-up calls")
        require(args.socket is None, "infrastructure mode must not receive a policy socket")
        require(args.policy_timeout_seconds is None, "infrastructure mode must not receive a policy timeout")
        return
    require(args.execution_horizon in SUPPORTED_EXECUTION_HORIZONS, "official-score requires K in {1, 4}")
    require(isinstance(args.socket, Path), "official-score requires an explicit --socket")
    if args.policy_timeout_seconds is None:
        args.policy_timeout_seconds = 300.0
    require(
        _is_finite_number(args.policy_timeout_seconds) and args.policy_timeout_seconds > 0,
        "policy timeout must be positive and finite",
    )
    require(args.output_dir is not None, "official-score requires --output-dir")
    require(args.preregistration_manifest is not None, "official-score requires --preregistration-manifest")
    _require_sha256(args.preregistration_sha256, "official-score --preregistration-sha256")
    require(isinstance(args.cell_id, str) and bool(args.cell_id), "official-score requires --cell-id")
    require(
        isinstance(args.final_freeze_token, str) and bool(args.final_freeze_token.strip()),
        "official-score requires an explicit --final-freeze-token string",
    )
    require(
        type(args.policy_warmup_calls) is int and args.policy_warmup_calls >= MIN_POLICY_WARMUP_CALLS,
        f"official-score policy warm-up calls must be at least {MIN_POLICY_WARMUP_CALLS}",
    )


def main(argv: Optional[Sequence[str]] = None) -> None:
    _require_evaluator_sources_unchanged(_IMPORT_EVALUATOR_SOURCE_IDENTITIES)
    _preflight.require_canonical_evaluator_runtime()
    args = parse_args(argv)
    validate_mode_arguments(args)
    require(
        platform.python_version() == PYTHON_VERSION,
        f"CALVIN evaluator requires Python {PYTHON_VERSION}, found {platform.python_version()}",
    )
    # Reserve the original stdout exclusively for the single JSON report, then
    # leave fd 1 directed to stderr. Native CALVIN/PyBullet libraries can write
    # to C stdout during mode execution and process teardown; restoring fd 1
    # after the report would let those diagnostics corrupt the JSON stream.
    sys.stdout.flush()
    report_descriptor = os.dup(sys.stdout.fileno())
    try:
        os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    except BaseException:
        os.close(report_descriptor)
        raise
    try:
        random.seed(EVALUATION_SEED)
        np.random.seed(EVALUATION_SEED)
        if args.mode == "infrastructure":
            report = run_infrastructure_mode(args.dataset_root, args.evaluation_seed, args.source_root)
        else:
            report = run_official_score_mode(args)
        _require_evaluator_sources_unchanged(_IMPORT_EVALUATOR_SOURCE_IDENTITIES)
        payload = (json.dumps(report, allow_nan=False, ensure_ascii=True, indent=2, sort_keys=True) + "\n").encode(
            "ascii"
        )
        while payload:
            written = os.write(report_descriptor, payload)
            require(written > 0, "CALVIN evaluator report write made no progress")
            payload = payload[written:]
    finally:
        os.close(report_descriptor)


if __name__ == "__main__":
    main()
