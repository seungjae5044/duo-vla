#!/usr/bin/env python3
"""Validate and exclusively publish the 40-task LIBERO expert-replay gate.

This program intentionally does not generate simulator evidence.  It accepts a
content-addressed evidence bundle produced by the pinned LIBERO runtime, checks
the complete fail-closed contract, and publishes a deterministic qualification
report.  The raw evidence manifest and every file it names must already exist.
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import importlib.metadata
import importlib.util
import json
import math
import os
import stat
import sys
from collections.abc import Mapping, Sequence
from itertools import pairwise
from pathlib import Path, PurePosixPath
from typing import Any


def _activate_project_source_root() -> Path:
    source_path = Path(__file__).resolve().parents[1] / "src"
    source_identity = os.lstat(source_path)
    if not stat.S_ISDIR(source_identity.st_mode):
        raise RuntimeError("project src import root must be a real directory")
    source_root = source_path.resolve()
    entries = list(os.scandir(source_root))
    if len(entries) != 1 or entries[0].name != "duo_vla" or not entries[0].is_dir(follow_symlinks=False):
        raise RuntimeError("project src import root must contain only the real duo_vla package directory")

    def visit(directory: Path, *, in_cache: bool) -> None:
        for entry in os.scandir(directory):
            mode = entry.stat(follow_symlinks=False).st_mode
            path = directory / entry.name
            if stat.S_ISDIR(mode):
                visit(path, in_cache=in_cache or entry.name == "__pycache__")
            elif stat.S_ISREG(mode):
                allowed = entry.name.endswith(".pyc") if in_cache else entry.name.endswith(".py")
                if not allowed:
                    raise RuntimeError(f"project package import tree contains a forbidden file: {path}")
            else:
                raise RuntimeError(f"project package import tree contains a linked or special entry: {path}")

    visit(source_root / "duo_vla", in_cache=False)
    source_text = str(source_root)
    sys.path[:] = [source_text] + [
        entry for entry in sys.path if str(Path(entry or os.getcwd()).resolve()) != source_text
    ]
    return source_root


def _validate_project_module_origins(source_root: Path, required_modules: set[str]) -> dict[str, str]:
    package_root = (source_root / "duo_vla").resolve(strict=True)
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


_PROJECT_SOURCE_ROOT = _activate_project_source_root()

from duo_vla.hf_snapshot import verify_huggingface_snapshot  # noqa: E402
from duo_vla.libero_replay_evidence import (  # noqa: E402
    ORIGINAL_HDF5_CONTENT_SHA256,
    ORIGINAL_HDF5_FILE_COUNT,
    ORIGINAL_HDF5_REPOSITORY_ID,
    ORIGINAL_HDF5_REVISION,
    ORIGINAL_HDF5_TOTAL_BYTES,
    PARQUET_BINDING_SCHEMA,
    PARQUET_TASK_SCHEMA,
    SIMULATOR_STAGE_SCHEMA,
    SIMULATOR_TASK_SCHEMA,
    ReplayEvidenceError,
    load_original_hdf5_inventory,
    load_source_parquet_alignment,
    observation_alignment_metrics,
    task_slug,
    trajectory_sha256,
    validate_observation_alignment_metrics,
    validate_original_hdf5_inventory,
)
from duo_vla.runtime_integrity import (  # noqa: E402
    content_address_eval_venv,
    content_address_train_venv,
    require_matching_eval_venv,
    require_matching_train_venv,
)

_REQUIRED_PROJECT_MODULES = {
    "duo_vla",
    "duo_vla.hf_snapshot",
    "duo_vla.libero_replay_evidence",
    "duo_vla.runtime_integrity",
}
_validate_project_module_origins(_PROJECT_SOURCE_ROOT, _REQUIRED_PROJECT_MODULES)

PROTOCOL = "duovla-libero-v1"
EVIDENCE_SCHEMA = "duo-vla-libero-expert-replay-evidence-v1"
QUALIFICATION_SCHEMA = "duo-vla-libero-expert-replay-qualification-v1"
QUALIFICATION_KIND = "libero-40-task-regenerated-expert-replay"
SIMULATOR_ATTESTATION_SCHEMA = "duo-vla-libero-simulator-attestation-v3"
VALIDATOR_RUNTIME_SCHEMA = "duo-vla-libero-expert-replay-validator-runtime-v1"
SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
DATASET_REVISION = "86958911c0f959db2bbbdb107eb3e17c5f9c798e"
DATASET_TREE_METADATA_SHA256 = "d9c14b4aff28bcc56f341b171c6a5a3b10510d4bd0378662891c5156d245add8"
DATASET_TREE_FILE_COUNT = 383
DATASET_TREE_TOTAL_BYTES = 34_926_157_548
DATASET_CONTENT_INVENTORY_SHA256 = "63fd7a951ebb397a33c43cad4a7c48c7c6911bd8d1481ff99b07da5f7890782c"
DATASET_SNAPSHOT_FILES_VERIFIED = 382
DATASET_SNAPSHOT_TOTAL_BYTES = 34_926_155_087
NORMALIZATION_CONTENT_SHA256 = "a972b5d95a8aaa8ae7582bafcbc071261979cb46c2a3515b4da7a7cf0156ac73"
NORMALIZATION_RAW_SHA256 = "8a0428184e4db8463f3986e9a8c4f912e1027815669f4b4880c4bd3e8d3bcf29"
TASK_INVENTORY_SHA256 = "d00c211a09f34003089ba5a4dbbbb0e11af2543f4bba9cb1901a04a2a25e0117"
ENVIRONMENT_SEED = 0

GATE_NAMES = (
    "schema_revision_counts",
    "exact_gripper_set",
    "controller_impulse_directions",
    "normalization_round_trip",
    "episode_boundary_chunk_fixture",
    "pre_action_observation_alignment",
    "camera_transform_parity",
    "deterministic_fixed_state_reset",
    "pre_dispatch_integrity_controls",
)
PRE_DISPATCH_MUTATIONS = ("zero_action", "mismatched_language", "swapped_cameras", "inverted_gripper")
IMPULSE_DIRECTIONS = ("+x", "-x", "+y", "-y", "+z", "-z", "+rx", "-rx", "+ry", "-ry", "+rz", "-rz")

_EVIDENCE_FIELDS = {"demonstrations", "gates", "inputs", "raw_evidence", "schema"}
_INPUT_FIELDS = {
    "config_file_sha256",
    "dataset_content_inventory_sha256",
    "dataset_revision",
    "dataset_snapshot_files_verified",
    "dataset_snapshot_total_bytes",
    "dataset_tree_file_count",
    "dataset_tree_metadata_sha256",
    "dataset_tree_total_bytes",
    "normalization_content_sha256",
    "normalization_raw_sha256",
    "original_hdf5_file_count",
    "original_hdf5_inventory_content_sha256",
    "original_hdf5_inventory_raw_sha256",
    "original_hdf5_repository_id",
    "original_hdf5_revision",
    "original_hdf5_total_bytes",
    "project_source_tree_sha256",
    "simulator_attestation_raw_sha256",
    "simulator_attestation_sha256",
    "simulator_runtime_sha256",
    "source_files_sha256",
    "task_inventory_sha256",
    "train_venv_identity",
}
_RAW_EVIDENCE_FIELDS = {"bytes", "id", "path", "sha256"}
_DEMONSTRATION_FIELDS = {
    "action_sequence_sha256",
    "demonstration_id",
    "initial_state_sha256",
    "instruction",
    "observation_sequence_sha256",
    "raw_evidence_ids",
    "regenerated",
    "source_episode_index",
    "step_count",
    "success",
    "suite",
    "task_id",
    "task_name",
    "trajectory_sha256",
}
_GATE_FIELDS = {"evidence_ids", "name", "passed", "result", "result_sha256"}
_TASK_IDENTITY_FIELDS = {"instruction", "suite", "task_id", "task_name"}
_REPORT_FIELDS = {
    "content_sha256",
    "evidence",
    "evidence_manifest_raw_sha256",
    "inputs",
    "kind",
    "pass_criteria",
    "replay_summary",
    "schema",
    "status",
    "task_inventory",
    "validator_runtime_identity",
    "validator_source_identity",
}
_PASS_CRITERIA = {
    "all_required_gates_passed": True,
    "all_raw_evidence_hashes_verified": True,
    "canonical_task_coverage_40_of_40": True,
    "every_included_demonstration_regenerated": True,
    "every_included_demonstration_successful": True,
    "exclusive_output_directory_owned": True,
    "inputs_and_validator_source_rechecked_before_commit": True,
    "live_training_snapshot_rechecked_before_commit": True,
    "train_venv_rechecked_before_commit": True,
    "two_stage_semantics_cross_checked": True,
    "validator_runtime_rechecked_before_commit": True,
}
_VALIDATOR_RUNTIME_FIELDS = {
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
_SIMULATOR_STAGE_FIELDS = {
    "collector",
    "inputs",
    "schema",
    "gates",
    "source_scan",
    "status",
    "task_records",
}
_SIMULATOR_TASK_FIELDS = {
    "attempts",
    "collector_source_sha256",
    "pre_dispatch_integrity_controls",
    "reset_determinism",
    "schema",
    "selected",
    "source_file",
    "source_scan",
    "task",
}
_PARQUET_STAGE_FIELDS = {
    "binder",
    "gates",
    "inputs",
    "process",
    "schema",
    "status",
    "task_records",
    "train_venv",
}
_SELECTED_REPLAY_FIELDS = {
    "action_sequence_sha256",
    "alignment_probe",
    "initial_state_sha256",
    "inverted_gripper_action_sequence_sha256",
    "observation_alignment_features",
    "observation_sequence_sha256",
    "source_episode_index",
    "step_count",
    "success",
    "swapped_observation_sequence_sha256",
    "trajectory_sha256",
    "zero_action_sequence_sha256",
}
_REPLAY_ATTEMPT_FIELDS = {
    "action_sequence_sha256",
    "source_episode_index",
    "step_count",
    "success",
    "training_linked",
}
_RESET_EVIDENCE_FIELDS = {
    "environment_seed",
    "first_observation_sha256",
    "first_simulator_state_sha256",
    "probe_environment_count",
    "seed_calls_per_environment",
    "second_observation_sha256",
    "second_simulator_state_sha256",
    "settle_steps",
}
_BINDER_REQUIRED_ENVIRONMENT = {
    "HF_HUB_OFFLINE": "1",
    "LANG": "C.UTF-8",
    "LC_ALL": "C.UTF-8",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "OMP_DYNAMIC": "FALSE",
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "PYTHONHASHSEED": "0",
    "PYTHONNOUSERSITE": "1",
    "PYTHONSAFEPATH": "1",
    "PYTHONDONTWRITEBYTECODE": "1",
    "RAYON_NUM_THREADS": "1",
    "TOKENIZERS_PARALLELISM": "false",
    "TRANSFORMERS_OFFLINE": "1",
    "TZ": "UTC",
    "VECLIB_MAXIMUM_THREADS": "1",
}


class QualificationError(RuntimeError):
    """Raised before publication when a qualification invariant fails."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise QualificationError(message)


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
        raise QualificationError(f"value is not finite canonical JSON: {exc}") from exc


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _require_exact_keys(value: Mapping[str, Any], expected: set[str], name: str) -> None:
    observed = set(value)
    require(
        observed == expected,
        f"{name} fields differ: missing={sorted(expected - observed)}, extra={sorted(observed - expected)}",
    )


def _require_sha256(value: Any, name: str) -> str:
    require(
        isinstance(value, str) and len(value) == 64 and all(character in "0123456789abcdef" for character in value),
        f"{name} must be 64 lowercase hexadecimal characters",
    )
    return value


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, value in pairs:
        if name in result:
            raise ValueError(f"duplicate JSON field {name!r}")
        result[name] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant {value}")


def _require_finite_json_numbers(value: Any, *, name: str) -> None:
    if isinstance(value, float):
        require(math.isfinite(value), f"{name} contains a non-finite number")
    elif isinstance(value, Mapping):
        for key, item in value.items():
            _require_finite_json_numbers(item, name=f"{name}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _require_finite_json_numbers(item, name=f"{name}[{index}]")


def _read_descriptor_json(descriptor: int, *, name: str) -> tuple[dict[str, Any], bytes]:
    chunks: list[bytes] = []
    before = os.fstat(descriptor)
    require(stat.S_ISREG(before.st_mode), f"{name} is not a regular file")
    with os.fdopen(os.dup(descriptor), "rb") as source:
        while block := source.read(1024 * 1024):
            chunks.append(block)
        after = os.fstat(source.fileno())
    stable = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns", "st_nlink")
    require(all(getattr(before, field) == getattr(after, field) for field in stable), f"{name} changed while read")
    raw = b"".join(chunks)
    require(len(raw) == after.st_size, f"{name} byte count changed while read")
    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_unique_json_object,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, ValueError) as exc:
        raise QualificationError(f"{name} is not strict finite UTF-8 JSON") from exc
    require(isinstance(value, dict), f"{name} root must be an object")
    _require_finite_json_numbers(value, name=name)
    return value, raw


def read_stable_json(path: Path, *, name: str, expected_sha256: str | None = None) -> tuple[dict[str, Any], str]:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(str(path), flags)
    except OSError as exc:
        raise QualificationError(f"cannot open {name}: {path}") from exc
    try:
        value, raw = _read_descriptor_json(descriptor, name=name)
    finally:
        os.close(descriptor)
    digest = hashlib.sha256(raw).hexdigest()
    if expected_sha256 is not None:
        _require_sha256(expected_sha256, f"expected {name} SHA-256")
        require(hmac.compare_digest(digest, expected_sha256), f"{name} differs from its externally recorded SHA-256")
    return value, digest


def stable_file_sha256(path: Path) -> str:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(str(path), flags)
    digest = hashlib.sha256()
    size = 0
    try:
        before = os.fstat(descriptor)
        require(stat.S_ISREG(before.st_mode), f"source is not a regular file: {path}")
        with os.fdopen(descriptor, "rb") as source:
            descriptor = -1
            while block := source.read(1024 * 1024):
                size += len(block)
                digest.update(block)
            after = os.fstat(source.fileno())
        stable = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns", "st_nlink")
        require(
            all(getattr(before, field) == getattr(after, field) for field in stable) and size == after.st_size,
            f"source changed while being hashed: {path}",
        )
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    return digest.hexdigest()


def _load_authenticated_preflight(project_root: Path, attestation: Mapping[str, Any]) -> Any:
    project_sources = attestation.get("project_sources")
    require(isinstance(project_sources, Mapping), "simulator attestation project-source identity is invalid")
    path = project_root / "scripts/preflight_libero_env.py"
    require(
        stable_file_sha256(path) == project_sources.get("preflight"),
        "simulator preflight source differs from the attestation",
    )
    require(
        stable_file_sha256(Path(__file__)) == project_sources.get("replay_qualification")
        and stable_file_sha256(project_root / "src/duo_vla/libero_replay_evidence.py")
        == project_sources.get("replay_contract"),
        "qualification source differs from the simulator attestation",
    )
    spec = importlib.util.spec_from_file_location("_duo_vla_authenticated_qualification_preflight", path)
    require(spec is not None and spec.loader is not None, "cannot construct authenticated simulator preflight loader")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validator_runtime_identity(
    project_root: Path,
    cache_root: Path,
    attestation: Mapping[str, Any],
) -> dict[str, Any]:
    """Recompute the exact eval interpreter/import runtime bound by attestation v3."""

    source_root = _activate_project_source_root()
    require(source_root == (project_root / "src").resolve(), "qualifier checkout source root differs")
    _validate_project_module_origins(source_root, _REQUIRED_PROJECT_MODULES)
    preflight = _load_authenticated_preflight(project_root, attestation)
    try:
        process = preflight.validate_process_environment(project_root, cache_root)
        require(process == attestation.get("process"), "validator process differs from the simulator attestation")
        observed_eval_venv = content_address_eval_venv(cache_root / "venvs/libero-eval")
        require_matching_eval_venv(attestation.get("eval_venv_identity"), observed_eval_venv)
        assets = attestation.get("assets")
        require(isinstance(assets, Mapping) and isinstance(assets.get("path"), str), "attested assets path is invalid")
        assets_path = Path(assets["path"])
        site_packages_identity = preflight.verify_site_packages_inventory(
            cache_root / "venvs/libero-eval",
            assets_path,
        )
        module_origins = preflight.module_origin_identity(cache_root / "venvs/libero-eval")
        distribution_records = preflight.verify_distribution_records(cache_root / "venvs/libero-eval")
        installed_distributions = preflight.installed_distribution_identity()
        packages = {name: importlib.metadata.version(name) for name in preflight.EXPECTED_PACKAGES}
        project_sources = preflight.source_file_identities(project_root)
    except (OSError, RuntimeError, ValueError) as exc:
        raise QualificationError(f"cannot authenticate validator runtime: {exc}") from exc
    expected_values = {
        "distribution_records": distribution_records,
        "eval_venv_identity": observed_eval_venv,
        "installed_distributions": installed_distributions,
        "module_origins": module_origins,
        "packages": packages,
        "process": process,
        "project_sources": project_sources,
        "site_packages": site_packages_identity,
    }
    for name, observed in expected_values.items():
        require(observed == attestation.get(name), f"validator runtime {name} differs from the simulator attestation")
    _validate_project_module_origins(source_root, _REQUIRED_PROJECT_MODULES)
    return {
        **expected_values,
        "schema": VALIDATOR_RUNTIME_SCHEMA,
        "simulator_runtime_sha256": simulator_runtime_sha256(attestation),
    }


def training_source_tree_sha256(root: Path) -> str:
    """Match the exact source identity used by the LIBERO trainer and server."""

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
        digest.update(path.relative_to(root).as_posix().encode("ascii"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def source_identity(project_root: Path, *, single_gpu: bool = False) -> dict[str, Any]:
    binder_launcher = (
        "run_bind_libero_expert_replay_single_gpu.sh" if single_gpu else "run_bind_libero_expert_replay.sh"
    )
    qualification_launcher = (
        "run_qualify_libero_expert_replay_single_gpu.sh" if single_gpu else "run_qualify_libero_expert_replay.sh"
    )
    training_launcher = "run_libero_train_single_gpu.sh" if single_gpu else "run_libero_train.sh"
    named_paths = {
        "data_reader": project_root / "src/duo_vla/data/libero.py",
        "normalization": project_root / "src/duo_vla/data/libero_stats.py",
        "expert_replay_binder": project_root / "scripts/bind_libero_expert_replay.py",
        "expert_replay_binder_launcher": project_root / "scripts" / binder_launcher,
        "expert_replay_collector": project_root / "scripts/collect_libero_expert_replay.py",
        "expert_replay_collector_launcher": project_root / "scripts/run_collect_libero_expert_replay.sh",
        "expert_replay_contract": project_root / "src/duo_vla/libero_replay_evidence.py",
        "original_hdf5_inventory": project_root / "configs/libero_original_hdf5_inventory.json",
        "source_parquet_alignment": project_root / "configs/libero_source_parquet_alignment.json",
        "qualification": Path(__file__),
        "qualification_launcher": project_root / "scripts" / qualification_launcher,
        "simulator_preflight": project_root / "scripts/preflight_libero_env.py",
        "simulator_preflight_launcher": project_root / "scripts/run_libero_preflight.sh",
        "training_launcher": project_root / "scripts" / training_launcher,
        "training_program": project_root / "scripts/train_libero.py",
    }
    source_files = {name: stable_file_sha256(path) for name, path in named_paths.items()}
    configs = {
        "direct_regression": stable_file_sha256(
            project_root
            / "configs"
            / ("libero_direct_regression_single_gpu.toml" if single_gpu else "libero_direct_regression.toml")
        ),
        "rectified_flow": stable_file_sha256(
            project_root / "configs" / ("libero_single_gpu.toml" if single_gpu else "libero.toml")
        ),
    }
    return {
        "config_file_sha256": configs,
        "project_source_tree_sha256": training_source_tree_sha256(project_root),
        "source_files_sha256": source_files,
    }


def _validate_normalization(value: Mapping[str, Any]) -> None:
    require(value.get("schema") == "duo-vla-libero-normalization-v1", "normalization schema mismatch")
    recorded = _require_sha256(value.get("content_sha256"), "normalization content_sha256")
    unsigned = {name: item for name, item in value.items() if name != "content_sha256"}
    require(hmac.compare_digest(canonical_sha256(unsigned), recorded), "normalization semantic self-hash mismatch")
    require(recorded == NORMALIZATION_CONTENT_SHA256, "normalization content identity mismatch")
    dataset = value.get("dataset")
    counts = value.get("counts")
    action = value.get("action")
    require(isinstance(dataset, Mapping), "normalization dataset identity is invalid")
    require(dataset.get("id") == "HuggingFaceVLA/libero", "normalization dataset ID mismatch")
    require(dataset.get("revision") == DATASET_REVISION, "normalization dataset revision mismatch")
    require(isinstance(counts, Mapping), "normalization counts are invalid")
    require(
        {name: counts.get(name) for name in ("tasks", "total_episodes", "total_frames")}
        == {"tasks": 40, "total_episodes": 1693, "total_frames": 273465},
        "normalization dataset counts mismatch",
    )
    require(isinstance(action, Mapping), "normalization action contract is invalid")
    require(action.get("observed_gripper_values") == [-1.0, 1.0], "normalization gripper set mismatch")


def dataset_content_inventory_sha256(value: Mapping[str, Any]) -> str:
    files = value.get("files")
    require(isinstance(files, Mapping), "dataset tree file inventory is invalid")
    records: list[dict[str, Any]] = []
    for name in sorted(set(files) - {".gitattributes"}):
        entry = files[name]
        require(isinstance(entry, Mapping), f"dataset tree entry is invalid: {name}")
        if "lfs_sha256" in entry:
            algorithm = "sha256"
            size = entry.get("lfs_size")
            digest = entry.get("lfs_sha256")
        else:
            algorithm = "git-sha1"
            size = entry.get("size")
            digest = entry.get("blob_id")
        require(type(size) is int and size >= 0, f"dataset tree entry byte count is invalid: {name}")
        _require_sha256(digest, f"dataset tree entry digest {name}") if algorithm == "sha256" else require(
            isinstance(digest, str)
            and len(digest) == 40
            and all(character in "0123456789abcdef" for character in digest),
            f"dataset tree Git digest is invalid: {name}",
        )
        records.append({"algorithm": algorithm, "bytes": size, "digest": digest, "path": name})
    return canonical_sha256(records)


def _validate_dataset_tree(value: Mapping[str, Any]) -> None:
    _require_exact_keys(value, {"files", "format_version"}, "dataset tree metadata")
    require(value["format_version"] == 1, "dataset tree format version mismatch")
    files = value["files"]
    require(isinstance(files, Mapping), "dataset tree file inventory is invalid")
    require(len(files) == DATASET_TREE_FILE_COUNT, "dataset tree file count mismatch")
    sizes = [item.get("size") for item in files.values() if isinstance(item, Mapping)]
    require(
        len(sizes) == DATASET_TREE_FILE_COUNT
        and all(type(size) is int and size >= 0 for size in sizes)
        and sum(sizes) == DATASET_TREE_TOTAL_BYTES,
        "dataset tree total byte count mismatch",
    )
    require(
        dataset_content_inventory_sha256(value) == DATASET_CONTENT_INVENTORY_SHA256,
        "dataset content inventory identity mismatch",
    )


def _task_identities(attestation: Mapping[str, Any]) -> list[dict[str, Any]]:
    require(attestation.get("schema") == SIMULATOR_ATTESTATION_SCHEMA, "simulator attestation schema mismatch")
    require(attestation.get("status") == "ok", "simulator attestation did not pass")
    require(
        attestation.get("environment_constructed") is True, "simulator attestation did not construct an environment"
    )
    require(attestation.get("task_inventory_count") == 40, "simulator task inventory count mismatch")
    require(
        attestation.get("task_inventory_sha256") == TASK_INVENTORY_SHA256, "simulator task inventory identity mismatch"
    )
    inventory = attestation.get("task_inventory")
    require(isinstance(inventory, list) and len(inventory) == 40, "simulator task inventory is incomplete")
    require(
        canonical_sha256(inventory) == attestation["task_inventory_sha256"] == TASK_INVENTORY_SHA256,
        "simulator task inventory content does not match its declared identity",
    )
    tasks: list[dict[str, Any]] = []
    for index, item in enumerate(inventory):
        require(isinstance(item, Mapping), f"simulator task inventory entry {index} is invalid")
        task = {name: item.get(name) for name in _TASK_IDENTITY_FIELDS}
        require(
            isinstance(task["suite"], str)
            and type(task["task_id"]) is int
            and isinstance(task["task_name"], str)
            and bool(task["task_name"])
            and isinstance(task["instruction"], str)
            and bool(task["instruction"]),
            f"simulator task inventory entry {index} has invalid identity fields",
        )
        require(item.get("reset_count") == 50, f"simulator task {index} reset count mismatch")
        tasks.append(task)
    expected_pairs = [(suite, task_id) for suite in SUITES for task_id in range(10)]
    require(
        [(task["suite"], task["task_id"]) for task in tasks] == expected_pairs,
        "simulator task inventory is not in canonical 40-task order",
    )
    return tasks


def simulator_runtime_sha256(attestation: Mapping[str, Any]) -> str:
    names = (
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
    runtime = {name: attestation.get(name) for name in names}
    require(
        all(value is not None for value in runtime.values()), "simulator attestation runtime identity is incomplete"
    )
    return canonical_sha256(runtime)


def _validate_dataset_snapshot_report(value: Mapping[str, Any]) -> None:
    _require_exact_keys(
        value,
        {
            "content_inventory_sha256",
            "files_verified",
            "revision",
            "snapshot",
            "total_bytes",
            "tree_metadata_sha256",
        },
        "live dataset snapshot report",
    )
    require(value["revision"] == DATASET_REVISION, "live dataset snapshot revision mismatch")
    require(
        value["tree_metadata_sha256"] == DATASET_TREE_METADATA_SHA256
        and value["content_inventory_sha256"] == DATASET_CONTENT_INVENTORY_SHA256,
        "live dataset snapshot content identity mismatch",
    )
    require(
        value["files_verified"] == DATASET_SNAPSHOT_FILES_VERIFIED
        and value["total_bytes"] == DATASET_SNAPSHOT_TOTAL_BYTES,
        "live dataset snapshot verified counts mismatch",
    )
    require(
        isinstance(value["snapshot"], str) and bool(value["snapshot"]),
        "live dataset snapshot path is invalid",
    )


def _validate_train_venv_identity(value: Any) -> dict[str, Any]:
    require(isinstance(value, dict), "train-venv identity must be an object")
    try:
        return require_matching_train_venv(value, dict(value))
    except RuntimeError as exc:
        raise QualificationError(str(exc)) from exc


def _train_python_version(train_venv: Path) -> str:
    config_path = train_venv / "pyvenv.cfg"
    try:
        lines = config_path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise QualificationError(f"cannot read train-venv Python version: {config_path}") from exc
    values = [line.split("=", 1)[1].strip() for line in lines if line.startswith("version_info =")]
    require(len(values) == 1 and bool(values[0]), "train-venv pyvenv.cfg version_info differs")
    return values[0]


def _expected_binder_process(
    project_root: Path,
    train_venv: Path,
    train_venv_identity: Mapping[str, Any],
) -> dict[str, Any]:
    cache_root = train_venv.parent.parent
    base_runtime = train_venv_identity["base_python_runtime"]
    base_prefix = Path(base_runtime["base_prefix"])
    base_exec_prefix = Path(base_runtime["venv_python_link_target"]).parent.parent
    version_info = _train_python_version(train_venv)
    major, minor, *_rest = version_info.split(".")
    version = f"python{major}.{minor}"
    compact_version = f"python{major}{minor}"
    sys_path = [
        str((project_root / "src").resolve()),
        str(base_prefix / "lib" / f"{compact_version}.zip"),
        str(base_prefix / "lib" / version),
        str(base_exec_prefix / "lib" / version / "lib-dynload"),
        str(train_venv / "lib" / version / "site-packages"),
    ]
    environment = {
        **_BINDER_REQUIRED_ENVIRONMENT,
        "DUO_VLA_CACHE_ROOT": str(cache_root),
        "DUO_VLA_PROJECT_ROOT": str(project_root),
        "DUO_VLA_TRAIN_VENV": str(train_venv),
        "HF_HOME": "/root/.cache/huggingface",
        "HOME": "/root",
        "PATH": f"{train_venv / 'bin'}:/usr/bin:/bin",
    }
    return {
        "environment": environment,
        "environment_sha256": canonical_sha256(environment),
        "python_base_exec_prefix": str(base_exec_prefix),
        "python_base_prefix": str(base_prefix),
        "python_executable": str(train_venv / "bin/python"),
        "python_flags": {"dont_write_bytecode": True, "no_user_site": True, "safe_path": True},
        "python_invocation_flags": ["-P", "-B", "-X", "pycache_prefix=/dev/null"],
        "python_prefix": str(train_venv),
        "python_pycache_prefix": "/dev/null",
        "python_version": version_info,
        "sys_path": sys_path,
    }


def build_expected_inputs(
    project_root: Path,
    *,
    simulator_attestation: Mapping[str, Any],
    simulator_attestation_raw_sha256: str,
    dataset_tree: Mapping[str, Any],
    dataset_tree_raw_sha256: str,
    normalization: Mapping[str, Any],
    normalization_raw_sha256: str,
    dataset_snapshot: Mapping[str, Any],
    train_venv_identity: Mapping[str, Any],
    original_hdf5_inventory: Mapping[str, Any] | None = None,
    original_hdf5_inventory_raw_sha256: str | None = None,
) -> dict[str, Any]:
    _task_identities(simulator_attestation)
    _validate_dataset_tree(dataset_tree)
    _validate_normalization(normalization)
    require(dataset_tree_raw_sha256 == DATASET_TREE_METADATA_SHA256, "dataset tree metadata raw identity mismatch")
    require(normalization_raw_sha256 == NORMALIZATION_RAW_SHA256, "normalization raw identity mismatch")
    _validate_dataset_snapshot_report(dataset_snapshot)
    validated_train_venv = _validate_train_venv_identity(dict(train_venv_identity))
    if original_hdf5_inventory is None:
        inventory_path = project_root / "configs/libero_original_hdf5_inventory.json"
        try:
            original_hdf5_inventory, _records, observed_inventory_raw_sha256 = load_original_hdf5_inventory(
                inventory_path
            )
        except ReplayEvidenceError as exc:
            raise QualificationError(str(exc)) from exc
        if original_hdf5_inventory_raw_sha256 is None:
            original_hdf5_inventory_raw_sha256 = observed_inventory_raw_sha256
    else:
        try:
            validate_original_hdf5_inventory(original_hdf5_inventory)
        except ReplayEvidenceError as exc:
            raise QualificationError(str(exc)) from exc
    _require_sha256(original_hdf5_inventory_raw_sha256, "original HDF5 inventory raw SHA-256")
    require(
        original_hdf5_inventory.get("content_sha256") == ORIGINAL_HDF5_CONTENT_SHA256,
        "original HDF5 inventory content identity mismatch",
    )
    train_venv_root = Path(validated_train_venv["root"])
    identity = source_identity(project_root, single_gpu=train_venv_root.name == "train-single-gpu")
    return {
        "config_file_sha256": identity["config_file_sha256"],
        "dataset_content_inventory_sha256": DATASET_CONTENT_INVENTORY_SHA256,
        "dataset_revision": DATASET_REVISION,
        "dataset_snapshot_files_verified": DATASET_SNAPSHOT_FILES_VERIFIED,
        "dataset_snapshot_total_bytes": DATASET_SNAPSHOT_TOTAL_BYTES,
        "dataset_tree_file_count": DATASET_TREE_FILE_COUNT,
        "dataset_tree_metadata_sha256": dataset_tree_raw_sha256,
        "dataset_tree_total_bytes": DATASET_TREE_TOTAL_BYTES,
        "normalization_content_sha256": NORMALIZATION_CONTENT_SHA256,
        "normalization_raw_sha256": normalization_raw_sha256,
        "original_hdf5_file_count": ORIGINAL_HDF5_FILE_COUNT,
        "original_hdf5_inventory_content_sha256": ORIGINAL_HDF5_CONTENT_SHA256,
        "original_hdf5_inventory_raw_sha256": original_hdf5_inventory_raw_sha256,
        "original_hdf5_repository_id": ORIGINAL_HDF5_REPOSITORY_ID,
        "original_hdf5_revision": ORIGINAL_HDF5_REVISION,
        "original_hdf5_total_bytes": ORIGINAL_HDF5_TOTAL_BYTES,
        "project_source_tree_sha256": identity["project_source_tree_sha256"],
        "simulator_attestation_raw_sha256": simulator_attestation_raw_sha256,
        "simulator_attestation_sha256": canonical_sha256(simulator_attestation),
        "simulator_runtime_sha256": simulator_runtime_sha256(simulator_attestation),
        "source_files_sha256": identity["source_files_sha256"],
        "task_inventory_sha256": TASK_INVENTORY_SHA256,
        "train_venv_identity": json.loads(canonical_json_bytes(validated_train_venv).decode("ascii")),
    }


def _result_integer(value: Mapping[str, Any], name: str, *, minimum: int = 0) -> int:
    result = value.get(name)
    require(type(result) is int and result >= minimum, f"gate result {name} must be an integer >= {minimum}")
    return result


def canonical_gate_results() -> dict[str, dict[str, Any]]:
    """Return the literal result contracts an evidence collector must fill."""

    return {
        "schema_revision_counts": {
            "dataset_revision": DATASET_REVISION,
            "declared_file_pointer_mismatches": 1690,
            "episodes": 1693,
            "frames": 273465,
            "physical_data_files": 377,
            "tasks": 40,
        },
        "exact_gripper_set": {
            "observed_values": [-1.0, 1.0],
            "unexpected_value_count": 0,
            "zero_value_count": 0,
        },
        "controller_impulse_directions": {
            "action_dimension": 7,
            "checked_directions": list(IMPULSE_DIRECTIONS),
            "controller": "OSC_POSE",
            "direction_match_count": len(IMPULSE_DIRECTIONS),
            "gripper_close": 1.0,
            "gripper_open": -1.0,
        },
        "normalization_round_trip": {
            "continuous_action_dimensions": 6,
            "gripper_sign_exact": True,
            "max_abs_error": 0.0,
            "samples": 1,
            "state_dimensions": 8,
        },
        "episode_boundary_chunk_fixture": {
            "checked_terminal_anchors": 1,
            "cross_boundary_count": 0,
            "horizon": 8,
            "padding_mask_exact": True,
            "padding_value": "zeros",
        },
        "pre_action_observation_alignment": {
            "action_index": "t",
            "checked_tasks": 40,
            "checked_transitions": 40,
            "exact_match_count": 40,
            "observation_index": "t",
            "post_action_pairing_rejected": True,
        },
        "camera_transform_parity": {
            "checked_tasks": 40,
            "dataset_transform": "already_rotated_no_additional_transform",
            "dtype": "uint8",
            "image_shape": [256, 256, 3],
            "pixel_parity_all": True,
            "positive_stride_contiguous": True,
            "processor_order": ["agentview", "eye_in_hand"],
            "rollout_transform": "rotate_180_once",
        },
        "deterministic_fixed_state_reset": {
            "all_serialized_state_repeats_equal": True,
            "checked_tasks": 40,
            "environment_seed": ENVIRONMENT_SEED,
            "official_fixed_states_used": False,
            "resets_per_task": 2,
        },
        "pre_dispatch_integrity_controls": {
            "checked_tasks": 40,
            "mutations": {name: {"mutation_detected": True} for name in PRE_DISPATCH_MUTATIONS},
        },
    }


def _validate_gate_result(name: str, result: Mapping[str, Any]) -> None:
    if name == "schema_revision_counts":
        _require_exact_keys(
            result,
            {
                "dataset_revision",
                "declared_file_pointer_mismatches",
                "episodes",
                "frames",
                "physical_data_files",
                "tasks",
            },
            name,
        )
        require(
            result
            == {
                "dataset_revision": DATASET_REVISION,
                "declared_file_pointer_mismatches": 1690,
                "episodes": 1693,
                "frames": 273465,
                "physical_data_files": 377,
                "tasks": 40,
            },
            "schema/revision/count gate result mismatch",
        )
    elif name == "exact_gripper_set":
        _require_exact_keys(result, {"observed_values", "unexpected_value_count", "zero_value_count"}, name)
        require(
            result == {"observed_values": [-1.0, 1.0], "unexpected_value_count": 0, "zero_value_count": 0},
            "exact gripper gate result mismatch",
        )
    elif name == "controller_impulse_directions":
        _require_exact_keys(
            result,
            {
                "action_dimension",
                "checked_directions",
                "controller",
                "direction_match_count",
                "gripper_close",
                "gripper_open",
            },
            name,
        )
        require(result.get("controller") == "OSC_POSE", "controller impulse gate must use OSC_POSE")
        require(result.get("action_dimension") == 7, "controller impulse action dimension mismatch")
        require(
            result.get("checked_directions") == list(IMPULSE_DIRECTIONS),
            "controller impulse direction inventory mismatch",
        )
        require(
            result.get("direction_match_count") == len(IMPULSE_DIRECTIONS),
            "controller impulse directions did not all match",
        )
        require(
            result.get("gripper_open") == -1.0 and result.get("gripper_close") == 1.0,
            "controller gripper direction mismatch",
        )
    elif name == "normalization_round_trip":
        _require_exact_keys(
            result,
            {"continuous_action_dimensions", "gripper_sign_exact", "max_abs_error", "samples", "state_dimensions"},
            name,
        )
        samples = _result_integer(result, "samples", minimum=1)
        error = result.get("max_abs_error")
        require(
            samples > 0
            and isinstance(error, (int, float))
            and not isinstance(error, bool)
            and math.isfinite(float(error)),
            "normalization error is invalid",
        )
        require(0.0 <= float(error) <= 1e-6, "normalization round-trip error exceeds tolerance")
        require(
            result.get("continuous_action_dimensions") == 6
            and result.get("state_dimensions") == 8
            and result.get("gripper_sign_exact") is True,
            "normalization round-trip contract mismatch",
        )
    elif name == "episode_boundary_chunk_fixture":
        _require_exact_keys(
            result,
            {"checked_terminal_anchors", "cross_boundary_count", "horizon", "padding_mask_exact", "padding_value"},
            name,
        )
        require(
            result.get("horizon") == 8
            and _result_integer(result, "checked_terminal_anchors", minimum=1) > 0
            and result.get("cross_boundary_count") == 0
            and result.get("padding_mask_exact") is True
            and result.get("padding_value") == "zeros",
            "episode-boundary chunk gate result mismatch",
        )
    elif name == "pre_action_observation_alignment":
        _require_exact_keys(
            result,
            {
                "action_index",
                "checked_tasks",
                "checked_transitions",
                "exact_match_count",
                "observation_index",
                "post_action_pairing_rejected",
            },
            name,
        )
        transitions = _result_integer(result, "checked_transitions", minimum=1)
        require(
            result.get("action_index") == "t"
            and result.get("observation_index") == "t"
            and result.get("checked_tasks") == 40
            and transitions >= result["checked_tasks"]
            and result.get("exact_match_count") == transitions
            and result.get("post_action_pairing_rejected") is True,
            "pre-action observation alignment gate result mismatch",
        )
    elif name == "camera_transform_parity":
        _require_exact_keys(
            result,
            {
                "checked_tasks",
                "dataset_transform",
                "dtype",
                "image_shape",
                "pixel_parity_all",
                "positive_stride_contiguous",
                "processor_order",
                "rollout_transform",
            },
            name,
        )
        require(
            result
            == {
                "checked_tasks": 40,
                "dataset_transform": "already_rotated_no_additional_transform",
                "dtype": "uint8",
                "image_shape": [256, 256, 3],
                "pixel_parity_all": True,
                "positive_stride_contiguous": True,
                "processor_order": ["agentview", "eye_in_hand"],
                "rollout_transform": "rotate_180_once",
            },
            "camera transform parity gate result mismatch",
        )
    elif name == "deterministic_fixed_state_reset":
        _require_exact_keys(
            result,
            {
                "all_serialized_state_repeats_equal",
                "checked_tasks",
                "environment_seed",
                "official_fixed_states_used",
                "resets_per_task",
            },
            name,
        )
        require(
            result.get("all_serialized_state_repeats_equal") is True
            and result.get("checked_tasks") == 40
            and result.get("environment_seed") == ENVIRONMENT_SEED
            and result.get("official_fixed_states_used") is False
            and _result_integer(result, "resets_per_task", minimum=2) >= 2,
            "deterministic reset gate result mismatch",
        )
    elif name == "pre_dispatch_integrity_controls":
        _require_exact_keys(result, {"checked_tasks", "mutations"}, name)
        require(result.get("checked_tasks") == 40, "pre-dispatch controls did not cover all 40 tasks")
        mutations = result.get("mutations")
        require(isinstance(mutations, Mapping), "pre-dispatch mutation results are invalid")
        _require_exact_keys(mutations, set(PRE_DISPATCH_MUTATIONS), "pre-dispatch mutations")
        for mutation_name in PRE_DISPATCH_MUTATIONS:
            mutation = mutations[mutation_name]
            require(isinstance(mutation, Mapping), f"pre-dispatch mutation {mutation_name} is invalid")
            _require_exact_keys(mutation, {"mutation_detected"}, f"pre-dispatch mutation {mutation_name}")
            require(
                mutation == {"mutation_detected": True},
                f"pre-dispatch mutation {mutation_name} was not detected",
            )
    else:  # pragma: no cover - guarded by exact name inventory
        raise QualificationError(f"unknown qualification gate {name!r}")


def _open_relative_regular(root_descriptor: int, relative: str) -> int:
    path = PurePosixPath(relative)
    require(
        not path.is_absolute() and path.parts and all(part not in {"", ".", ".."} for part in path.parts),
        f"unsafe raw evidence path: {relative!r}",
    )
    require("\\" not in relative, f"raw evidence path must use POSIX separators: {relative!r}")
    directory = os.dup(root_descriptor)
    try:
        for part in path.parts[:-1]:
            child = os.open(
                part,
                os.O_RDONLY
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_DIRECTORY", 0)
                | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=directory,
            )
            os.close(directory)
            directory = child
        return os.open(
            path.parts[-1],
            os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=directory,
        )
    except OSError as exc:
        raise QualificationError(f"cannot open raw evidence without following links: {relative}") from exc
    finally:
        os.close(directory)


def _verify_raw_evidence(root: Path, records: Sequence[Mapping[str, Any]]) -> None:
    root_descriptor = os.open(
        str(root),
        os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        for record in records:
            descriptor = _open_relative_regular(root_descriptor, record["path"])
            digest = hashlib.sha256()
            size = 0
            try:
                before = os.fstat(descriptor)
                require(stat.S_ISREG(before.st_mode), f"raw evidence is not a regular file: {record['path']}")
                with os.fdopen(descriptor, "rb") as source:
                    descriptor = -1
                    while block := source.read(1024 * 1024):
                        size += len(block)
                        digest.update(block)
                    after = os.fstat(source.fileno())
                stable = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns", "st_nlink")
                require(
                    all(getattr(before, field) == getattr(after, field) for field in stable),
                    f"raw evidence changed while read: {record['path']}",
                )
            finally:
                if descriptor >= 0:
                    os.close(descriptor)
            require(size == record["bytes"] == after.st_size, f"raw evidence byte count mismatch: {record['path']}")
            require(
                hmac.compare_digest(digest.hexdigest(), record["sha256"]),
                f"raw evidence hash mismatch: {record['path']}",
            )
    finally:
        os.close(root_descriptor)


def _read_raw_evidence_json(
    root_descriptor: int,
    record: Mapping[str, Any],
    *,
    name: str,
) -> dict[str, Any]:
    descriptor = _open_relative_regular(root_descriptor, record["path"])
    try:
        value, raw = _read_descriptor_json(descriptor, name=name)
    finally:
        os.close(descriptor)
    require(len(raw) == record["bytes"], f"{name} byte count differs")
    require(hashlib.sha256(raw).hexdigest() == record["sha256"], f"{name} SHA-256 differs")
    return value


def _validate_generated_bundle_semantics(
    root: Path,
    raw_records: Sequence[Mapping[str, Any]],
    demonstrations: Sequence[Mapping[str, Any]],
    *,
    evidence_document: Mapping[str, Any],
    expected_inputs: Mapping[str, Any],
    task_inventory: Sequence[Mapping[str, Any]],
) -> None:
    """Cross-check the two generated stages instead of trusting result booleans."""

    record_by_id = {record["id"]: record for record in raw_records}
    alignment, training_source_indices, alignment_raw_sha256 = load_source_parquet_alignment(
        _PROJECT_SOURCE_ROOT.parent / "configs/libero_source_parquet_alignment.json"
    )
    evidence_gate_results = {gate["name"]: gate["result"] for gate in evidence_document["gates"]}
    expected_ids = {"parquet-binding", "simulator-stage", "simulator-stage-commit"}
    for suite in SUITES:
        for task_id in range(10):
            slug = task_slug(suite, task_id)
            expected_ids.update({f"parquet-{slug}", f"simulator-{slug}"})
    require(set(record_by_id) == expected_ids, "generated replay raw-evidence ID inventory differs")
    require(
        record_by_id["parquet-binding"]["path"] == "parquet-binding.json"
        and record_by_id["simulator-stage"]["path"] == "simulator-stage.json"
        and record_by_id["simulator-stage-commit"]["path"] == "simulator-stage.commit.json",
        "generated replay stage paths differ",
    )
    expected_files = {record["path"] for record in raw_records} | {"evidence.json", "evidence.json.sha256"}
    entries = list(root.rglob("*"))
    observed_files: set[str] = set()
    observed_directories: set[str] = set()
    for path in entries:
        relative = path.relative_to(root).as_posix()
        identity = path.lstat()
        if stat.S_ISREG(identity.st_mode):
            require(identity.st_nlink == 1, f"generated replay bundle file has multiple hard links: {relative}")
            observed_files.add(relative)
        elif stat.S_ISDIR(identity.st_mode):
            observed_directories.add(relative)
        else:
            raise QualificationError(f"generated replay bundle contains a special or linked entry: {relative}")
    require(observed_files == expected_files, "generated replay bundle file inventory differs")
    require(observed_directories == {"binding", "raw"}, "generated replay bundle directory inventory differs")
    manifest_now, manifest_raw_sha256 = read_stable_json(root / "evidence.json", name="expert replay evidence manifest")
    require(manifest_now == evidence_document, "expert replay evidence manifest changed during validation")
    try:
        companion = (root / "evidence.json.sha256").read_text(encoding="ascii").strip().split()
    except (OSError, UnicodeError) as exc:
        raise QualificationError("cannot read expert replay evidence SHA-256 companion") from exc
    require(
        companion == [manifest_raw_sha256, "evidence.json"],
        "expert replay evidence SHA-256 companion differs",
    )

    inventory_path = Path(__file__).resolve().parents[1] / "configs/libero_original_hdf5_inventory.json"
    try:
        inventory, inventory_records, inventory_raw_sha256 = load_original_hdf5_inventory(inventory_path)
    except ReplayEvidenceError as exc:
        raise QualificationError(str(exc)) from exc
    require(
        inventory["content_sha256"] == expected_inputs["original_hdf5_inventory_content_sha256"]
        and inventory_raw_sha256 == expected_inputs["original_hdf5_inventory_raw_sha256"],
        "generated replay original-HDF5 inventory identity differs",
    )
    inventory_by_task = {(record["suite"], record["task_id"]): record for record in inventory_records}
    tasks = {(task["suite"], task["task_id"]): task for task in task_inventory}
    demos = {(demo["suite"], demo["task_id"]): demo for demo in demonstrations}
    require(len(demos) == 40, "generated replay must contain exactly one demonstration per task")

    root_descriptor = os.open(
        str(root),
        os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        simulator_stage = _read_raw_evidence_json(
            root_descriptor, record_by_id["simulator-stage"], name="simulator stage evidence"
        )
        parquet_stage = _read_raw_evidence_json(
            root_descriptor, record_by_id["parquet-binding"], name="parquet binding evidence"
        )
        simulator_commit = _read_raw_evidence_json(
            root_descriptor,
            record_by_id["simulator-stage-commit"],
            name="simulator stage commit evidence",
        )
        _require_exact_keys(simulator_stage, _SIMULATOR_STAGE_FIELDS, "simulator stage evidence")
        _require_exact_keys(parquet_stage, _PARQUET_STAGE_FIELDS, "parquet binding evidence")
        require(
            simulator_stage.get("schema") == SIMULATOR_STAGE_SCHEMA and simulator_stage.get("status") == "complete",
            "simulator stage evidence is incomplete",
        )
        require(
            parquet_stage.get("schema") == PARQUET_BINDING_SCHEMA and parquet_stage.get("status") == "complete",
            "parquet binding evidence is incomplete",
        )
        require(
            simulator_commit
            == {
                "schema": "duo-vla-libero-expert-replay-simulator-stage-commit-v1",
                "simulator_stage_sha256": record_by_id["simulator-stage"]["sha256"],
                "task_record_root_sha256": canonical_sha256(simulator_stage["task_records"]),
            },
            "simulator stage commit differs",
        )
        expected_simulator_inputs = {
            "original_hdf5_inventory_content_sha256": expected_inputs["original_hdf5_inventory_content_sha256"],
            "original_hdf5_inventory_raw_sha256": expected_inputs["original_hdf5_inventory_raw_sha256"],
            "simulator_attestation_raw_sha256": expected_inputs["simulator_attestation_raw_sha256"],
            "simulator_attestation_sha256": expected_inputs["simulator_attestation_sha256"],
            "simulator_runtime_sha256": expected_inputs["simulator_runtime_sha256"],
            "task_inventory_sha256": expected_inputs["task_inventory_sha256"],
        }
        require(
            simulator_stage.get("inputs") == expected_simulator_inputs,
            "simulator stage source/data/runtime inputs differ",
        )
        simulator_collector = simulator_stage.get("collector")
        require(isinstance(simulator_collector, Mapping), "simulator collector identity is invalid")
        _require_exact_keys(
            simulator_collector,
            {
                "path",
                "sha256",
                "shared_contract_sha256",
                "source_parquet_alignment_content_sha256",
                "source_parquet_alignment_raw_sha256",
            },
            "simulator collector identity",
        )
        require(
            simulator_collector.get("path") == "scripts/collect_libero_expert_replay.py"
            and simulator_collector.get("sha256")
            == expected_inputs["source_files_sha256"].get("expert_replay_collector")
            and simulator_collector.get("shared_contract_sha256")
            == expected_inputs["source_files_sha256"].get("expert_replay_contract"),
            "simulator collector source identity differs",
        )
        require(
            simulator_collector.get("source_parquet_alignment_content_sha256") == alignment["content_sha256"]
            and simulator_collector.get("source_parquet_alignment_raw_sha256") == alignment_raw_sha256,
            "simulator collector source/parquet alignment identity differs",
        )
        simulator_checks = simulator_stage.get("gates")
        require(
            isinstance(simulator_checks, Mapping)
            and set(simulator_checks) == {"controller_impulse_directions", "exact_gripper_set"},
            "simulator-stage check inventory differs",
        )
        require(
            simulator_checks["exact_gripper_set"]
            == {"observed_values": [-1.0, 1.0], "unexpected_value_count": 0, "zero_value_count": 0},
            "simulator-stage gripper scan differs",
        )
        controller_check = simulator_checks["controller_impulse_directions"]
        require(isinstance(controller_check, Mapping), "simulator-stage controller check is invalid")
        controller_summary = {name: item for name, item in controller_check.items() if name != "raw"}
        _validate_gate_result("controller_impulse_directions", controller_summary)
        require(
            controller_summary == evidence_gate_results["controller_impulse_directions"]
            and simulator_checks["exact_gripper_set"] == evidence_gate_results["exact_gripper_set"],
            "simulator-stage checks differ from published gate results",
        )
        controller_raw = controller_check.get("raw")
        require(
            isinstance(controller_raw, Mapping)
            and set(controller_raw) == {"gripper_apertures", "responses"}
            and isinstance(controller_raw["responses"], list)
            and len(controller_raw["responses"]) == len(IMPULSE_DIRECTIONS),
            "simulator-stage raw controller evidence differs",
        )
        apertures = controller_raw["gripper_apertures"]
        responses = controller_raw["responses"]
        require(
            isinstance(apertures, Mapping)
            and set(apertures) == {"close", "open"}
            and all(isinstance(value, (int, float)) and not isinstance(value, bool) for value in apertures.values())
            and all(math.isfinite(float(value)) for value in apertures.values())
            and apertures["open"] > apertures["close"],
            "simulator-stage raw gripper response differs",
        )
        require(
            [item.get("direction") for item in responses] == list(IMPULSE_DIRECTIONS)
            and all(
                isinstance(item, Mapping)
                and set(item) == {"direction", "leakage", "primary_delta"}
                and isinstance(item["leakage"], (int, float))
                and not isinstance(item["leakage"], bool)
                and math.isfinite(float(item["leakage"]))
                and float(item["leakage"]) >= 0.0
                and isinstance(item["primary_delta"], (int, float))
                and not isinstance(item["primary_delta"], bool)
                and math.isfinite(float(item["primary_delta"]))
                and (1.0 if item["direction"].startswith("+") else -1.0) * float(item["primary_delta"]) > 0.0
                and float(item["leakage"]) <= abs(float(item["primary_delta"])) * 0.01 + 1e-8
                for item in responses
            ),
            "simulator-stage raw controller impulse responses differ",
        )
        expected_parquet_inputs = {
            "dataset_content_inventory_sha256": expected_inputs["dataset_content_inventory_sha256"],
            "dataset_tree_sha256": expected_inputs["dataset_tree_metadata_sha256"],
            "original_hdf5_inventory_content_sha256": expected_inputs["original_hdf5_inventory_content_sha256"],
            "original_hdf5_inventory_raw_sha256": expected_inputs["original_hdf5_inventory_raw_sha256"],
            "simulator_stage_sha256": record_by_id["simulator-stage"]["sha256"],
            "simulator_stage_commit_sha256": record_by_id["simulator-stage-commit"]["sha256"],
            "dataset_snapshot_files_verified": expected_inputs["dataset_snapshot_files_verified"],
            "dataset_snapshot_total_bytes": expected_inputs["dataset_snapshot_total_bytes"],
            "train_venv_root_sha256": expected_inputs["train_venv_identity"]["root_sha256"],
        }
        require(parquet_stage.get("inputs") == expected_parquet_inputs, "parquet binding inputs differ")
        parquet_gates = parquet_stage.get("gates")
        require(
            isinstance(parquet_gates, Mapping)
            and set(parquet_gates)
            == {"episode_boundary_chunk_fixture", "normalization_round_trip", "training_gripper_set"},
            "parquet binding gate inventory differs",
        )
        require(
            parquet_gates["episode_boundary_chunk_fixture"] == evidence_gate_results["episode_boundary_chunk_fixture"]
            and parquet_gates["normalization_round_trip"] == evidence_gate_results["normalization_round_trip"]
            and parquet_gates["training_gripper_set"]
            == {
                **evidence_gate_results["exact_gripper_set"],
                "validated_actions": 273465,
                "validated_episodes": 1693,
            },
            "parquet binding gates differ from published gate results",
        )
        parquet_binder = parquet_stage.get("binder")
        require(isinstance(parquet_binder, Mapping), "parquet binder identity is invalid")
        _require_exact_keys(
            parquet_binder,
            {"path", "sha256", "shared_contract_sha256"},
            "parquet binder identity",
        )
        require(
            parquet_binder.get("path") == "scripts/bind_libero_expert_replay.py"
            and parquet_binder.get("sha256") == expected_inputs["source_files_sha256"].get("expert_replay_binder")
            and parquet_binder.get("shared_contract_sha256")
            == expected_inputs["source_files_sha256"].get("expert_replay_contract"),
            "parquet binder source identity differs",
        )
        train_venv = parquet_stage.get("train_venv")
        require(train_venv == expected_inputs["train_venv_identity"], "parquet binder train-venv identity differs")
        require(
            parquet_stage.get("process")
            == _expected_binder_process(
                Path(__file__).resolve().parents[1],
                Path(train_venv["root"]),
                train_venv,
            ),
            "parquet binder process identity differs",
        )

        simulator_records = simulator_stage.get("task_records")
        parquet_records = parquet_stage.get("task_records")
        stage_source_scan = simulator_stage.get("source_scan")
        require(
            isinstance(simulator_records, list)
            and isinstance(parquet_records, list)
            and isinstance(stage_source_scan, list)
            and len(simulator_records) == len(parquet_records) == 40,
            "generated replay stage task inventories are incomplete",
        )
        require(len(stage_source_scan) == 40, "simulator-stage source scan is incomplete")
        expected_order = [(suite, task_id) for suite in SUITES for task_id in range(10)]
        parquet_episode_indices: list[int] = []
        parquet_task_indices: list[int] = []
        parquet_spans: list[tuple[int, int]] = []
        for index, identity in enumerate(expected_order):
            suite, task_id = identity
            slug = task_slug(suite, task_id)
            simulator_raw = record_by_id[f"simulator-{slug}"]
            parquet_raw = record_by_id[f"parquet-{slug}"]
            simulator_record = simulator_records[index]
            parquet_record = parquet_records[index]
            require(
                simulator_record
                == {
                    "bytes": simulator_raw["bytes"],
                    "path": simulator_raw["path"],
                    "sha256": simulator_raw["sha256"],
                    "suite": suite,
                    "task_id": task_id,
                },
                f"simulator stage task record differs: {suite}:{task_id}",
            )
            require(
                parquet_record
                == {
                    "bytes": parquet_raw["bytes"],
                    "path": parquet_raw["path"],
                    "sha256": parquet_raw["sha256"],
                    "suite": suite,
                    "task_id": task_id,
                },
                f"parquet stage task record differs: {suite}:{task_id}",
            )
            simulator_task = _read_raw_evidence_json(
                root_descriptor, simulator_raw, name=f"simulator task evidence {suite}:{task_id}"
            )
            parquet_task = _read_raw_evidence_json(
                root_descriptor, parquet_raw, name=f"parquet task evidence {suite}:{task_id}"
            )
            _require_exact_keys(simulator_task, _SIMULATOR_TASK_FIELDS, "simulator task evidence")
            require(simulator_task.get("schema") == SIMULATOR_TASK_SCHEMA, "simulator task schema differs")
            require(
                simulator_task.get("collector_source_sha256") == simulator_collector["sha256"],
                f"simulator task collector source identity differs: {suite}:{task_id}",
            )
            require(parquet_task.get("schema") == PARQUET_TASK_SCHEMA, "parquet task schema differs")
            task = tasks[identity]
            require(
                inventory_by_task[identity]["task_name"] == task["task_name"],
                f"original HDF5 task name differs from simulator inventory: {suite}:{task_id}",
            )
            require(
                simulator_task.get("task")
                == {
                    "instruction": task["instruction"],
                    "suite": suite,
                    "task_id": task_id,
                    "task_name": task["task_name"],
                },
                f"simulator task metadata differs: {suite}:{task_id}",
            )
            require(
                simulator_task.get("source_file") == inventory_by_task[identity],
                f"simulator source HDF5 identity differs: {suite}:{task_id}",
            )
            attempts = simulator_task.get("attempts")
            require(isinstance(attempts, list) and bool(attempts), "simulator replay attempts are empty")
            require(
                all(isinstance(item, Mapping) and set(item) == _REPLAY_ATTEMPT_FIELDS for item in attempts),
                f"simulator replay attempt fields differ: {suite}:{task_id}",
            )
            require(
                all(not (item.get("success") is True and item.get("training_linked") is True) for item in attempts[:-1])
                and attempts[-1].get("success") is True
                and attempts[-1].get("training_linked") is True
                and [item.get("source_episode_index") for item in attempts] == list(range(len(attempts))),
                f"simulator did not use canonical first training-linked success selection: {suite}:{task_id}",
            )
            require(
                all(
                    item["training_linked"]
                    == (item["source_episode_index"] in training_source_indices[(suite, task_id)])
                    for item in attempts
                ),
                f"simulator training-link flags differ from the pinned alignment: {suite}:{task_id}",
            )
            selected = simulator_task.get("selected")
            require(
                isinstance(selected, Mapping) and set(selected) == _SELECTED_REPLAY_FIELDS,
                "simulator selected replay is invalid",
            )
            require(
                selected.get("success") is True
                and selected.get("source_episode_index") == attempts[-1].get("source_episode_index")
                and selected.get("action_sequence_sha256") == attempts[-1].get("action_sequence_sha256")
                and selected.get("step_count") == attempts[-1].get("step_count"),
                f"simulator selected replay differs from first training-linked success: {suite}:{task_id}",
            )
            for digest_name in (
                "action_sequence_sha256",
                "initial_state_sha256",
                "inverted_gripper_action_sequence_sha256",
                "observation_sequence_sha256",
                "swapped_observation_sequence_sha256",
                "trajectory_sha256",
                "zero_action_sequence_sha256",
            ):
                _require_sha256(selected[digest_name], f"selected replay {digest_name}")
            simulator_alignment = observation_alignment_metrics(
                selected["observation_alignment_features"],
                selected["observation_alignment_features"].get("frames", []),
            )
            require(
                simulator_alignment["passed"] is True and simulator_alignment["frames"] == selected["step_count"],
                f"simulator observation-alignment feature structure differs: {suite}:{task_id}",
            )
            scan = simulator_task.get("source_scan")
            require(
                isinstance(scan, Mapping)
                and set(scan)
                == {
                    "demonstrations",
                    "gripper_counts",
                    "raw_transition_count",
                    "retained_transition_count",
                },
                "simulator source scan is invalid",
            )
            require(
                stage_source_scan[index]
                == {
                    "gripper_counts": scan.get("gripper_counts"),
                    "path": inventory_by_task[identity]["path"],
                    "raw_transition_count": scan.get("raw_transition_count"),
                    "retained_transition_count": scan.get("retained_transition_count"),
                    "suite": suite,
                    "task_id": task_id,
                },
                f"simulator-stage source scan differs: {suite}:{task_id}",
            )
            scan_demos = scan.get("demonstrations")
            source_episode_index = selected["source_episode_index"]
            require(
                isinstance(scan_demos, list)
                and bool(scan_demos)
                and all(
                    isinstance(item, Mapping)
                    and set(item)
                    == {
                        "action_sequence_sha256",
                        "initial_state_sha256",
                        "raw_transition_count",
                        "retained_transition_count",
                        "source_action_dtype",
                        "source_episode_index",
                        "source_state_sequence_sha256",
                    }
                    for item in scan_demos
                )
                and source_episode_index < len(scan_demos)
                and scan_demos[source_episode_index].get("action_sequence_sha256") == selected["action_sequence_sha256"]
                and scan_demos[source_episode_index].get("initial_state_sha256") == selected["initial_state_sha256"],
                f"selected replay does not match the complete source scan: {suite}:{task_id}",
            )
            require(
                [item["source_episode_index"] for item in scan_demos] == list(range(len(scan_demos)))
                and scan["raw_transition_count"] == sum(item["raw_transition_count"] for item in scan_demos)
                and scan["retained_transition_count"] == sum(item["retained_transition_count"] for item in scan_demos)
                and bool(scan["gripper_counts"])
                and set(scan["gripper_counts"]) <= {"-1", "1"}
                and all(type(count) is int and count > 0 for count in scan["gripper_counts"].values())
                and all(item["source_action_dtype"] in {"<f4", "<f8"} for item in scan_demos),
                f"simulator source scan counts/order/dtype differ: {suite}:{task_id}",
            )
            for scan_demo in scan_demos:
                require(
                    type(scan_demo["raw_transition_count"]) is int
                    and type(scan_demo["retained_transition_count"]) is int
                    and 0 < scan_demo["retained_transition_count"] <= scan_demo["raw_transition_count"],
                    f"simulator source demo counts differ: {suite}:{task_id}",
                )
                _require_sha256(scan_demo["action_sequence_sha256"], "source action sequence")
                _require_sha256(scan_demo["initial_state_sha256"], "source initial state")
                _require_sha256(scan_demo["source_state_sequence_sha256"], "source state sequence")
            require(
                all(
                    attempt["source_episode_index"] == scan_demos[attempt_index]["source_episode_index"]
                    and attempt["action_sequence_sha256"] == scan_demos[attempt_index]["action_sequence_sha256"]
                    and attempt["step_count"] == scan_demos[attempt_index]["retained_transition_count"]
                    for attempt_index, attempt in enumerate(attempts)
                ),
                f"simulator attempts do not match the source scan: {suite}:{task_id}",
            )
            reset = simulator_task.get("reset_determinism")
            require(
                isinstance(reset, Mapping)
                and set(reset) == _RESET_EVIDENCE_FIELDS
                and reset.get("environment_seed") == ENVIRONMENT_SEED
                and reset.get("probe_environment_count") == 2
                and reset.get("seed_calls_per_environment") == 1
                and reset.get("settle_steps") == 10
                and reset.get("first_observation_sha256") == reset.get("second_observation_sha256")
                and reset.get("first_simulator_state_sha256") == reset.get("second_simulator_state_sha256"),
                f"simulator reset evidence differs: {suite}:{task_id}",
            )
            for digest_name in (
                "first_observation_sha256",
                "first_simulator_state_sha256",
                "second_observation_sha256",
                "second_simulator_state_sha256",
            ):
                _require_sha256(reset[digest_name], f"simulator reset {digest_name}")
            alignment = selected.get("alignment_probe")
            require(
                isinstance(alignment, Mapping)
                and set(alignment)
                == {
                    "post_action_observation_sha256",
                    "pre_action_observation_sha256",
                    "retained_transition_index",
                    "source_transition_index",
                }
                and alignment.get("pre_action_observation_sha256") != alignment.get("post_action_observation_sha256"),
                f"pre/post-action alignment probe did not distinguish observations: {suite}:{task_id}",
            )
            require(
                type(alignment["retained_transition_index"]) is int
                and 0 <= alignment["retained_transition_index"] < selected["step_count"]
                and type(alignment["source_transition_index"]) is int
                and 0
                <= alignment["source_transition_index"]
                < scan_demos[source_episode_index]["raw_transition_count"],
                f"pre/post-action alignment indices are out of range: {suite}:{task_id}",
            )
            _require_sha256(alignment["pre_action_observation_sha256"], "pre-action observation probe")
            _require_sha256(alignment["post_action_observation_sha256"], "post-action observation probe")
            require(
                selected["trajectory_sha256"]
                == trajectory_sha256(
                    suite=suite,
                    task_id=task_id,
                    source_episode_index=selected["source_episode_index"],
                    initial_state_digest=selected["initial_state_sha256"],
                    action_digest=selected["action_sequence_sha256"],
                    observation_digest=selected["observation_sequence_sha256"],
                ),
                f"selected replay trajectory digest differs: {suite}:{task_id}",
            )
            require(
                selected["inverted_gripper_action_sequence_sha256"] != selected["action_sequence_sha256"]
                and selected["zero_action_sequence_sha256"] != selected["action_sequence_sha256"]
                and selected["swapped_observation_sequence_sha256"] != selected["observation_sequence_sha256"],
                f"pre-dispatch mutation digest was not distinguished: {suite}:{task_id}",
            )
            controls = simulator_task.get("pre_dispatch_integrity_controls")
            require(
                isinstance(controls, Mapping)
                and set(controls) == set(PRE_DISPATCH_MUTATIONS)
                and all(item == {"mutation_detected": True} for item in controls.values()),
                f"pre-dispatch integrity-control evidence differs: {suite}:{task_id}",
            )
            require(
                _require_sha256(
                    parquet_task.get("dataset_observation_sequence_sha256"),
                    "parquet dataset observation sequence",
                )
                and validate_observation_alignment_metrics(parquet_task.get("observation_alignment_metrics"))["passed"]
                is True,
                f"parquet observation-alignment evidence differs: {suite}:{task_id}",
            )
            require(
                parquet_task
                == {
                    "action_sequence_sha256": selected["action_sequence_sha256"],
                    "dataset_observation_sequence_sha256": parquet_task.get("dataset_observation_sequence_sha256"),
                    "dataset_episode_index": parquet_task.get("dataset_episode_index"),
                    "dataset_global_start": parquet_task.get("dataset_global_start"),
                    "dataset_global_stop": parquet_task.get("dataset_global_stop"),
                    "dataset_task_index": parquet_task.get("dataset_task_index"),
                    "instruction": task["instruction"],
                    "observation_alignment_metrics": parquet_task.get("observation_alignment_metrics"),
                    "observation_sequence_sha256": selected["observation_sequence_sha256"],
                    "schema": PARQUET_TASK_SCHEMA,
                    "source_episode_index": selected["source_episode_index"],
                    "step_count": selected["step_count"],
                    "suite": suite,
                    "task_id": task_id,
                    "task_name": task["task_name"],
                    "trajectory_sha256": selected["trajectory_sha256"],
                },
                f"parquet binding does not match simulator replay: {suite}:{task_id}",
            )
            require(
                type(parquet_task["dataset_episode_index"]) is int
                and 0 <= parquet_task["dataset_episode_index"] < 1693
                and type(parquet_task["dataset_global_start"]) is int
                and type(parquet_task["dataset_global_stop"]) is int
                and 0 <= parquet_task["dataset_global_start"] < parquet_task["dataset_global_stop"] <= 273465
                and type(parquet_task["dataset_task_index"]) is int
                and 0 <= parquet_task["dataset_task_index"] < 40
                and parquet_task["dataset_global_stop"] - parquet_task["dataset_global_start"]
                == selected["step_count"],
                f"parquet binding episode span differs: {suite}:{task_id}",
            )
            parquet_episode_indices.append(parquet_task["dataset_episode_index"])
            parquet_task_indices.append(parquet_task["dataset_task_index"])
            parquet_spans.append((parquet_task["dataset_global_start"], parquet_task["dataset_global_stop"]))
            demo = demos[identity]
            require(
                demo["demonstration_id"] == f"{slug}-source-{selected['source_episode_index']:04d}"
                and demo["action_sequence_sha256"] == selected["action_sequence_sha256"]
                and demo["initial_state_sha256"] == selected["initial_state_sha256"]
                and demo["observation_sequence_sha256"] == selected["observation_sequence_sha256"]
                and demo["trajectory_sha256"] == selected["trajectory_sha256"]
                and demo["source_episode_index"] == selected["source_episode_index"]
                and demo["step_count"] == selected["step_count"]
                and demo["raw_evidence_ids"] == sorted([f"parquet-{slug}", f"simulator-{slug}"]),
                f"qualification demonstration does not match generated stages: {suite}:{task_id}",
            )
        require(
            len(parquet_episode_indices) == len(set(parquet_episode_indices)) == 40,
            "parquet bindings reuse a dataset episode",
        )
        require(
            {label for item in stage_source_scan for label in item["gripper_counts"]} == {"-1", "1"},
            "simulator source scans do not contain the exact global gripper set",
        )
        require(set(parquet_task_indices) == set(range(40)), "parquet bindings do not cover all dataset task indices")
        ordered_spans = sorted(parquet_spans)
        require(
            all(left[1] <= right[0] for left, right in pairwise(ordered_spans)),
            "parquet binding episode spans overlap",
        )
    finally:
        os.close(root_descriptor)


def _canonical_raw_evidence_paths() -> dict[str, str]:
    paths = {
        "parquet-binding": "parquet-binding.json",
        "simulator-stage": "simulator-stage.json",
        "simulator-stage-commit": "simulator-stage.commit.json",
    }
    for suite in SUITES:
        for task_id in range(10):
            slug = task_slug(suite, task_id)
            paths[f"parquet-{slug}"] = f"binding/{slug}.json"
            paths[f"simulator-{slug}"] = f"raw/{slug}.json"
    return paths


def _canonical_gate_evidence_ids() -> dict[str, list[str]]:
    simulator_ids = [f"simulator-{task_slug(suite, task_id)}" for suite in SUITES for task_id in range(10)]
    task_ids = [
        identifier
        for suite in SUITES
        for task_id in range(10)
        for identifier in (
            f"parquet-{task_slug(suite, task_id)}",
            f"simulator-{task_slug(suite, task_id)}",
        )
    ]
    commit = ["simulator-stage-commit"]
    return {
        "schema_revision_counts": ["parquet-binding"],
        "exact_gripper_set": sorted(["parquet-binding", "simulator-stage", *commit]),
        "controller_impulse_directions": sorted(["simulator-stage", *commit]),
        "normalization_round_trip": ["parquet-binding"],
        "episode_boundary_chunk_fixture": ["parquet-binding"],
        "pre_action_observation_alignment": sorted([*task_ids, *commit]),
        "camera_transform_parity": sorted([*task_ids, *commit]),
        "deterministic_fixed_state_reset": sorted([*simulator_ids, *commit]),
        "pre_dispatch_integrity_controls": sorted([*simulator_ids, *commit]),
    }


def validate_evidence_document(
    evidence: Any,
    *,
    expected_inputs: Mapping[str, Any],
    task_inventory: Sequence[Mapping[str, Any]],
    raw_evidence_root: Path | None,
) -> dict[str, Any]:
    require(isinstance(evidence, Mapping), "expert replay evidence root must be an object")
    _require_exact_keys(evidence, _EVIDENCE_FIELDS, "expert replay evidence")
    require(evidence["schema"] == EVIDENCE_SCHEMA, "expert replay evidence schema mismatch")
    inputs = evidence["inputs"]
    require(isinstance(inputs, Mapping), "expert replay evidence inputs are invalid")
    _require_exact_keys(inputs, _INPUT_FIELDS, "expert replay evidence inputs")
    require(dict(inputs) == dict(expected_inputs), "expert replay source/config/data/runtime inputs differ")

    raw_evidence = evidence["raw_evidence"]
    require(isinstance(raw_evidence, list) and bool(raw_evidence), "expert replay raw evidence inventory is empty")
    raw_ids: list[str] = []
    raw_paths: list[str] = []
    for index, item in enumerate(raw_evidence):
        require(isinstance(item, Mapping), f"raw evidence record {index} is invalid")
        _require_exact_keys(item, _RAW_EVIDENCE_FIELDS, f"raw evidence record {index}")
        identifier = item["id"]
        path = item["path"]
        require(
            isinstance(identifier, str)
            and bool(identifier)
            and identifier == identifier.strip()
            and all(character.isalnum() or character in "._-" for character in identifier),
            f"raw evidence record {index} has an invalid ID",
        )
        require(isinstance(path, str) and bool(path), f"raw evidence record {index} path is invalid")
        require(type(item["bytes"]) is int and item["bytes"] > 0, f"raw evidence record {index} byte count is invalid")
        _require_sha256(item["sha256"], f"raw evidence record {index} SHA-256")
        raw_ids.append(identifier)
        raw_paths.append(path)
    require(raw_ids == sorted(raw_ids), "raw evidence records must be sorted by ID")
    require(len(raw_ids) == len(set(raw_ids)), "raw evidence IDs must be unique")
    require(len(raw_paths) == len(set(raw_paths)), "raw evidence paths must be unique")
    canonical_raw_paths = _canonical_raw_evidence_paths()
    require(
        len(raw_ids) == 83
        and raw_ids == sorted(canonical_raw_paths)
        and {item["id"]: item["path"] for item in raw_evidence} == canonical_raw_paths,
        "raw evidence must contain the 83 canonical IDs and paths",
    )
    if raw_evidence_root is not None:
        _verify_raw_evidence(raw_evidence_root, raw_evidence)

    gates = evidence["gates"]
    require(isinstance(gates, list) and len(gates) == len(GATE_NAMES), "qualification gate inventory is incomplete")
    referenced_ids: set[str] = set()
    for gate_index, expected_name in enumerate(GATE_NAMES):
        gate = gates[gate_index]
        require(isinstance(gate, Mapping), f"qualification gate {expected_name} is invalid")
        _require_exact_keys(gate, _GATE_FIELDS, f"qualification gate {expected_name}")
        require(gate["name"] == expected_name, "qualification gates are not in canonical order")
        require(gate["passed"] is True, f"qualification gate did not pass: {expected_name}")
        evidence_ids = gate["evidence_ids"]
        require(
            isinstance(evidence_ids, list)
            and bool(evidence_ids)
            and evidence_ids == sorted(evidence_ids)
            and len(evidence_ids) == len(set(evidence_ids)),
            f"qualification gate {expected_name} evidence IDs are invalid",
        )
        require(
            set(evidence_ids) <= set(raw_ids), f"qualification gate {expected_name} references missing raw evidence"
        )
        require(
            evidence_ids == _canonical_gate_evidence_ids()[expected_name],
            f"qualification gate {expected_name} raw-evidence mapping differs",
        )
        referenced_ids.update(evidence_ids)
        result = gate["result"]
        require(isinstance(result, Mapping), f"qualification gate {expected_name} result is invalid")
        _require_sha256(gate["result_sha256"], f"qualification gate {expected_name} result_sha256")
        require(
            hmac.compare_digest(gate["result_sha256"], canonical_sha256(result)),
            f"qualification gate {expected_name} result self-hash mismatch",
        )
        _validate_gate_result(expected_name, result)

    expected_tasks = [dict(task) for task in task_inventory]
    require(len(expected_tasks) == 40, "canonical task inventory must contain 40 tasks")
    task_lookup = {(task["suite"], task["task_id"]): task for task in expected_tasks}
    require(len(task_lookup) == 40, "canonical task inventory contains duplicate task identities")
    demonstrations = evidence["demonstrations"]
    require(
        isinstance(demonstrations, list) and len(demonstrations) == 40,
        "expert replay evidence must contain exactly 40 demonstrations",
    )
    observed_demo_ids: list[str] = []
    trajectories: list[str] = []
    tasks_with_success: set[tuple[str, int]] = set()
    for index, demonstration in enumerate(demonstrations):
        require(isinstance(demonstration, Mapping), f"expert demonstration {index} is invalid")
        _require_exact_keys(demonstration, _DEMONSTRATION_FIELDS, f"expert demonstration {index}")
        suite = demonstration["suite"]
        task_id = demonstration["task_id"]
        key = (suite, task_id)
        require(key in task_lookup, f"expert demonstration {index} is not a canonical LIBERO task")
        expected_task = task_lookup[key]
        require(
            demonstration["task_name"] == expected_task["task_name"]
            and demonstration["instruction"] == expected_task["instruction"],
            f"expert demonstration {index} task metadata differs from the simulator attestation",
        )
        require(demonstration["regenerated"] is True, f"expert demonstration {index} is not regenerated")
        require(demonstration["success"] is True, f"expert demonstration {index} did not succeed")
        require(
            type(demonstration["source_episode_index"]) is int and demonstration["source_episode_index"] >= 0,
            f"expert demonstration {index} source episode index is invalid",
        )
        require(
            type(demonstration["step_count"]) is int and demonstration["step_count"] > 0,
            f"expert demonstration {index} step count is invalid",
        )
        identifier = demonstration["demonstration_id"]
        require(isinstance(identifier, str) and bool(identifier), f"expert demonstration {index} ID is invalid")
        observed_demo_ids.append(identifier)
        for name in (
            "action_sequence_sha256",
            "initial_state_sha256",
            "observation_sequence_sha256",
            "trajectory_sha256",
        ):
            _require_sha256(demonstration[name], f"expert demonstration {index} {name}")
        trajectories.append(demonstration["trajectory_sha256"])
        evidence_ids = demonstration["raw_evidence_ids"]
        require(
            isinstance(evidence_ids, list)
            and bool(evidence_ids)
            and evidence_ids == sorted(evidence_ids)
            and len(evidence_ids) == len(set(evidence_ids)),
            f"expert demonstration {index} evidence IDs are invalid",
        )
        require(set(evidence_ids) <= set(raw_ids), f"expert demonstration {index} references missing raw evidence")
        slug = task_slug(suite, task_id)
        require(
            demonstration["demonstration_id"] == f"{slug}-source-{demonstration['source_episode_index']:04d}"
            and evidence_ids == sorted([f"parquet-{slug}", f"simulator-{slug}"]),
            f"expert demonstration {index} canonical ID/raw references differ",
        )
        referenced_ids.update(evidence_ids)
        tasks_with_success.add(key)
    require(observed_demo_ids == sorted(observed_demo_ids), "expert demonstrations must be sorted by demonstration ID")
    require(len(observed_demo_ids) == len(set(observed_demo_ids)), "expert demonstration IDs must be unique")
    require(len(trajectories) == len(set(trajectories)), "expert demonstrations must have distinct trajectory hashes")
    require(tasks_with_success == set(task_lookup), "expert replay does not cover all canonical 40 tasks")
    require(referenced_ids == set(raw_ids), "raw evidence inventory contains unreferenced files")
    if raw_evidence_root is not None:
        _validate_generated_bundle_semantics(
            raw_evidence_root,
            raw_evidence,
            demonstrations,
            evidence_document=evidence,
            expected_inputs=expected_inputs,
            task_inventory=expected_tasks,
        )
    return {
        "demonstration_count": len(demonstrations),
        "successful_demonstration_count": len(demonstrations),
        "successful_task_count": len(tasks_with_success),
        "task_count": len(task_lookup),
    }


def _validate_validator_runtime_identity(
    value: Any,
    *,
    simulator_attestation: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    require(isinstance(value, Mapping), "qualification validator runtime identity is invalid")
    _require_exact_keys(value, _VALIDATOR_RUNTIME_FIELDS, "qualification validator runtime identity")
    require(value["schema"] == VALIDATOR_RUNTIME_SCHEMA, "qualification validator runtime schema differs")
    _require_sha256(value["simulator_runtime_sha256"], "qualification validator simulator runtime SHA-256")
    eval_venv = value["eval_venv_identity"]
    require(isinstance(eval_venv, dict), "qualification validator eval-venv identity is invalid")
    try:
        require_matching_eval_venv(eval_venv, dict(eval_venv))
    except RuntimeError as exc:
        raise QualificationError(str(exc)) from exc

    process = value["process"]
    require(isinstance(process, Mapping), "qualification validator process identity is invalid")
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
        "qualification validator process identity",
    )
    environment = process["environment"]
    require(isinstance(environment, Mapping), "qualification validator environment identity is invalid")
    require(
        "PYTHONPATH" not in environment
        and environment.get("PYTHONSAFEPATH") == "1"
        and environment.get("PYTHONDONTWRITEBYTECODE") == "1",
        "qualification validator environment does not enforce safe imports",
    )
    require(
        process["python_flags"] == {"dont_write_bytecode": True, "no_user_site": True, "safe_path": True}
        and process["python_invocation_flags"] == ["-P", "-B", "-X", "pycache_prefix=/dev/null"]
        and process["python_pycache_prefix"] == "/dev/null",
        "qualification validator Python startup flags differ",
    )
    sys_path = process["sys_path"]
    require(
        isinstance(sys_path, list)
        and len(sys_path) == 5
        and len(set(sys_path)) == 5
        and all(isinstance(path, str) and Path(path).is_absolute() for path in sys_path),
        "qualification validator import search path is invalid",
    )
    require(
        all(
            isinstance(value[name], Mapping)
            for name in _VALIDATOR_RUNTIME_FIELDS - {"schema", "simulator_runtime_sha256"}
        ),
        "qualification validator runtime contains invalid nested identities",
    )
    project_sources = value["project_sources"]
    _require_exact_keys(
        project_sources,
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
        "qualification validator project sources",
    )
    for name, digest in project_sources.items():
        _require_sha256(digest, f"qualification validator project source {name}")
    _require_exact_keys(
        value["site_packages"],
        {"declared_files", "startup_files", "symlinks", "unregistered_files"},
        "qualification validator site-packages identity",
    )
    _require_exact_keys(
        value["installed_distributions"],
        {"count", "inventory_sha256", "packages"},
        "qualification validator installed-distribution identity",
    )
    if simulator_attestation is not None:
        expected = {
            name: simulator_attestation.get(name)
            for name in _VALIDATOR_RUNTIME_FIELDS - {"schema", "simulator_runtime_sha256"}
        }
        require(
            all(value[name] == expected[name] for name in expected)
            and value["simulator_runtime_sha256"] == simulator_runtime_sha256(simulator_attestation),
            "qualification validator runtime differs from the simulator attestation",
        )
    return dict(value)


def build_report(
    evidence: Mapping[str, Any],
    *,
    evidence_manifest_raw_sha256: str,
    expected_inputs: Mapping[str, Any],
    task_inventory: Sequence[Mapping[str, Any]],
    replay_summary: Mapping[str, Any],
    validator_runtime: Mapping[str, Any],
    validator_source: Mapping[str, Any],
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "evidence": json.loads(canonical_json_bytes(evidence).decode("ascii")),
        "evidence_manifest_raw_sha256": evidence_manifest_raw_sha256,
        "inputs": json.loads(canonical_json_bytes(expected_inputs).decode("ascii")),
        "kind": QUALIFICATION_KIND,
        "pass_criteria": dict(_PASS_CRITERIA),
        "replay_summary": dict(replay_summary),
        "schema": QUALIFICATION_SCHEMA,
        "status": "passed",
        "task_inventory": [dict(task) for task in task_inventory],
        "validator_runtime_identity": json.loads(canonical_json_bytes(validator_runtime).decode("ascii")),
        "validator_source_identity": json.loads(canonical_json_bytes(validator_source).decode("ascii")),
    }
    report["content_sha256"] = canonical_sha256(report)
    return report


def validate_qualification_report_document(
    report: Any,
    *,
    project_root: Path | None = None,
    simulator_attestation: Mapping[str, Any] | None = None,
    simulator_attestation_raw_sha256: str | None = None,
) -> dict[str, Any]:
    require(isinstance(report, Mapping), "expert replay qualification report root must be an object")
    _require_exact_keys(report, _REPORT_FIELDS, "expert replay qualification report")
    require(report["schema"] == QUALIFICATION_SCHEMA, "expert replay qualification report schema mismatch")
    require(report["kind"] == QUALIFICATION_KIND, "expert replay qualification report kind mismatch")
    require(report["status"] == "passed", "expert replay qualification report did not pass")
    _require_sha256(report["content_sha256"], "expert replay qualification content_sha256")
    unsigned = {name: item for name, item in report.items() if name != "content_sha256"}
    require(
        hmac.compare_digest(report["content_sha256"], canonical_sha256(unsigned)),
        "expert replay qualification content self-hash mismatch",
    )
    require(report["pass_criteria"] == _PASS_CRITERIA, "expert replay qualification pass criteria are incomplete")
    _validate_validator_runtime_identity(
        report["validator_runtime_identity"],
        simulator_attestation=simulator_attestation,
    )
    _require_sha256(report["evidence_manifest_raw_sha256"], "expert replay evidence manifest raw SHA-256")
    inputs = report["inputs"]
    require(isinstance(inputs, Mapping), "expert replay qualification inputs are invalid")
    _require_exact_keys(inputs, _INPUT_FIELDS, "expert replay qualification inputs")
    for name in (
        "dataset_tree_metadata_sha256",
        "dataset_content_inventory_sha256",
        "normalization_content_sha256",
        "normalization_raw_sha256",
        "original_hdf5_inventory_content_sha256",
        "original_hdf5_inventory_raw_sha256",
        "project_source_tree_sha256",
        "simulator_attestation_raw_sha256",
        "simulator_attestation_sha256",
        "simulator_runtime_sha256",
        "task_inventory_sha256",
    ):
        _require_sha256(inputs[name], f"qualification input {name}")
    require(inputs["dataset_revision"] == DATASET_REVISION, "qualification dataset revision mismatch")
    require(
        inputs["dataset_content_inventory_sha256"] == DATASET_CONTENT_INVENTORY_SHA256,
        "qualification dataset content inventory mismatch",
    )
    require(
        inputs["dataset_tree_metadata_sha256"] == DATASET_TREE_METADATA_SHA256, "qualification dataset tree mismatch"
    )
    require(inputs["dataset_tree_file_count"] == DATASET_TREE_FILE_COUNT, "qualification dataset file count mismatch")
    require(inputs["dataset_tree_total_bytes"] == DATASET_TREE_TOTAL_BYTES, "qualification dataset byte count mismatch")
    require(
        inputs["dataset_snapshot_files_verified"] == DATASET_SNAPSHOT_FILES_VERIFIED
        and inputs["dataset_snapshot_total_bytes"] == DATASET_SNAPSHOT_TOTAL_BYTES,
        "qualification live dataset snapshot counts mismatch",
    )
    require(
        inputs["normalization_content_sha256"] == NORMALIZATION_CONTENT_SHA256,
        "qualification normalization content mismatch",
    )
    require(
        inputs["normalization_raw_sha256"] == NORMALIZATION_RAW_SHA256,
        "qualification normalization raw identity mismatch",
    )
    require(
        inputs["original_hdf5_repository_id"] == ORIGINAL_HDF5_REPOSITORY_ID
        and inputs["original_hdf5_revision"] == ORIGINAL_HDF5_REVISION,
        "qualification original HDF5 repository identity mismatch",
    )
    require(
        inputs["original_hdf5_inventory_content_sha256"] == ORIGINAL_HDF5_CONTENT_SHA256,
        "qualification original HDF5 inventory content mismatch",
    )
    require(
        inputs["original_hdf5_file_count"] == ORIGINAL_HDF5_FILE_COUNT
        and inputs["original_hdf5_total_bytes"] == ORIGINAL_HDF5_TOTAL_BYTES,
        "qualification original HDF5 inventory counts mismatch",
    )
    require(inputs["task_inventory_sha256"] == TASK_INVENTORY_SHA256, "qualification task inventory mismatch")
    _validate_train_venv_identity(inputs["train_venv_identity"])
    tasks = report["task_inventory"]
    require(isinstance(tasks, list) and len(tasks) == 40, "qualification report task inventory is incomplete")
    for index, task in enumerate(tasks):
        require(isinstance(task, Mapping), f"qualification report task {index} is invalid")
        _require_exact_keys(task, _TASK_IDENTITY_FIELDS, f"qualification report task {index}")
    require(
        [(task["suite"], task["task_id"]) for task in tasks]
        == [(suite, task_id) for suite in SUITES for task_id in range(10)],
        "qualification report task inventory is not canonical",
    )
    summary = validate_evidence_document(
        report["evidence"],
        expected_inputs=inputs,
        task_inventory=tasks,
        raw_evidence_root=None,
    )
    require(report["replay_summary"] == summary, "qualification replay summary mismatch")
    validator_source = report["validator_source_identity"]
    require(isinstance(validator_source, Mapping), "qualification validator source identity is invalid")
    require(
        validator_source
        == {
            "config_file_sha256": inputs["config_file_sha256"],
            "project_source_tree_sha256": inputs["project_source_tree_sha256"],
            "source_files_sha256": inputs["source_files_sha256"],
        },
        "qualification validator and input source identities differ",
    )
    if project_root is not None:
        train_venv_root = Path(inputs["train_venv_identity"]["root"])
        require(
            validator_source == source_identity(project_root, single_gpu=train_venv_root.name == "train-single-gpu"),
            "qualification source/config identity is no longer current",
        )
    if simulator_attestation is not None:
        tasks_now = _task_identities(simulator_attestation)
        require(tasks == tasks_now, "qualification task inventory differs from the current simulator attestation")
        require(
            inputs["simulator_attestation_sha256"] == canonical_sha256(simulator_attestation),
            "qualification simulator attestation semantic identity mismatch",
        )
        require(
            inputs["simulator_runtime_sha256"] == simulator_runtime_sha256(simulator_attestation),
            "qualification simulator runtime identity mismatch",
        )
    if simulator_attestation_raw_sha256 is not None:
        _require_sha256(simulator_attestation_raw_sha256, "current simulator attestation raw SHA-256")
        require(
            inputs["simulator_attestation_raw_sha256"] == simulator_attestation_raw_sha256,
            "qualification simulator attestation raw identity mismatch",
        )
    return dict(report)


def load_qualification_report(
    path: Path,
    *,
    expected_raw_sha256: str,
    project_root: Path | None = None,
    simulator_attestation: Mapping[str, Any] | None = None,
    simulator_attestation_raw_sha256: str | None = None,
) -> tuple[dict[str, Any], str]:
    report, digest = read_stable_json(
        path, name="expert replay qualification report", expected_sha256=expected_raw_sha256
    )
    companion = path.with_suffix(path.suffix + ".sha256")
    try:
        companion_fields = companion.read_text(encoding="ascii").strip().split()
    except (OSError, UnicodeError) as exc:
        raise QualificationError(f"cannot read expert replay qualification SHA-256 companion: {companion}") from exc
    require(
        companion_fields == [digest, path.name],
        "expert replay qualification SHA-256 companion mismatch",
    )
    return (
        validate_qualification_report_document(
            report,
            project_root=project_root,
            simulator_attestation=simulator_attestation,
            simulator_attestation_raw_sha256=simulator_attestation_raw_sha256,
        ),
        digest,
    )


def qualification_identity(report: Mapping[str, Any], *, report_raw_sha256: str) -> dict[str, Any]:
    """Reduce a validated report to the exact identity sealed by pre-registration."""

    validated = validate_qualification_report_document(report)
    _require_sha256(report_raw_sha256, "expert replay qualification report raw SHA-256")
    evidence = validated["evidence"]
    inputs = validated["inputs"]
    return {
        "config_file_sha256": inputs["config_file_sha256"],
        "content_sha256": validated["content_sha256"],
        "dataset_content_inventory_sha256": inputs["dataset_content_inventory_sha256"],
        "dataset_snapshot_files_verified": inputs["dataset_snapshot_files_verified"],
        "dataset_snapshot_total_bytes": inputs["dataset_snapshot_total_bytes"],
        "dataset_tree_metadata_sha256": inputs["dataset_tree_metadata_sha256"],
        "demonstration_count": validated["replay_summary"]["demonstration_count"],
        "evidence_manifest_raw_sha256": validated["evidence_manifest_raw_sha256"],
        "gates_sha256": canonical_sha256(evidence["gates"]),
        "kind": validated["kind"],
        "normalization_content_sha256": inputs["normalization_content_sha256"],
        "normalization_raw_sha256": inputs["normalization_raw_sha256"],
        "original_hdf5_file_count": inputs["original_hdf5_file_count"],
        "original_hdf5_inventory_content_sha256": inputs["original_hdf5_inventory_content_sha256"],
        "original_hdf5_inventory_raw_sha256": inputs["original_hdf5_inventory_raw_sha256"],
        "original_hdf5_repository_id": inputs["original_hdf5_repository_id"],
        "original_hdf5_revision": inputs["original_hdf5_revision"],
        "original_hdf5_total_bytes": inputs["original_hdf5_total_bytes"],
        "project_source_tree_sha256": inputs["project_source_tree_sha256"],
        "raw_evidence_root_sha256": canonical_sha256(evidence["raw_evidence"]),
        "report_sha256": report_raw_sha256,
        "schema": validated["schema"],
        "simulator_attestation_raw_sha256": inputs["simulator_attestation_raw_sha256"],
        "simulator_attestation_sha256": inputs["simulator_attestation_sha256"],
        "simulator_runtime_sha256": inputs["simulator_runtime_sha256"],
        "status": validated["status"],
        "successful_demonstration_count": validated["replay_summary"]["successful_demonstration_count"],
        "successful_task_count": validated["replay_summary"]["successful_task_count"],
        "task_count": validated["replay_summary"]["task_count"],
        "task_inventory_sha256": inputs["task_inventory_sha256"],
        "train_venv_identity": json.loads(canonical_json_bytes(inputs["train_venv_identity"]).decode("ascii")),
        "validator_runtime_identity": json.loads(
            canonical_json_bytes(validated["validator_runtime_identity"]).decode("ascii")
        ),
    }


def _fsync_directory_descriptor(descriptor: int) -> None:
    os.fsync(descriptor)


def _write_exclusive_at(directory: int, name: str, payload: bytes) -> None:
    descriptor = os.open(
        name,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
        0o600,
        dir_fd=directory,
    )
    with os.fdopen(descriptor, "wb") as sink:
        sink.write(payload)
        sink.flush()
        os.fsync(sink.fileno())


def publish_report_exclusive(output_dir: Path, report: Mapping[str, Any]) -> tuple[Path, str]:
    """Own a new directory and publish the report last as its commit marker."""

    parent = output_dir.parent.resolve(strict=True)
    require(parent.is_dir(), f"qualification output parent is not a directory: {parent}")
    target = parent / output_dir.name
    payload = (json.dumps(report, allow_nan=False, ensure_ascii=True, indent=2, sort_keys=True) + "\n").encode("ascii")
    serialized = json.loads(
        payload.decode("ascii"),
        object_pairs_hook=_unique_json_object,
        parse_constant=_reject_json_constant,
    )
    _require_finite_json_numbers(serialized, name="serialized qualification report")
    validate_qualification_report_document(serialized)
    digest = hashlib.sha256(payload).hexdigest()
    companion_payload = f"{digest}  qualification.json\n".encode("ascii")
    parent_descriptor = os.open(
        str(parent),
        os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        try:
            os.mkdir(output_dir.name, 0o700, dir_fd=parent_descriptor)
        except FileExistsError as exc:
            raise FileExistsError(f"qualification output directory already exists: {target}") from exc
        os.fsync(parent_descriptor)
        directory = os.open(
            output_dir.name,
            os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=parent_descriptor,
        )
        try:
            before = os.fstat(directory)
            require(stat.S_ISDIR(before.st_mode), "qualification output target is not a directory")
            _write_exclusive_at(directory, "qualification.json.sha256", companion_payload)
            _fsync_directory_descriptor(directory)
            after_sidecar = os.fstat(directory)
            require(
                (before.st_dev, before.st_ino) == (after_sidecar.st_dev, after_sidecar.st_ino),
                "qualification output directory identity changed before commit",
            )
            _write_exclusive_at(directory, "qualification.json", payload)
            _fsync_directory_descriptor(directory)
        finally:
            os.close(directory)
    finally:
        os.close(parent_descriptor)
    return target / "qualification.json", digest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-manifest", type=Path, required=True)
    parser.add_argument("--evidence-manifest-sha256", required=True)
    parser.add_argument("--simulator-attestation", type=Path, required=True)
    parser.add_argument("--simulator-attestation-sha256", required=True, help="externally recorded raw file digest")
    parser.add_argument("--dataset-tree-metadata", type=Path, required=True)
    parser.add_argument("--dataset-tree-metadata-sha256", required=True)
    parser.add_argument("--snapshot-root", type=Path, required=True)
    parser.add_argument("--normalization-artifact", type=Path, required=True)
    parser.add_argument("--normalization-artifact-sha256", required=True)
    parser.add_argument("--original-hdf5-inventory", type=Path, required=True)
    parser.add_argument("--original-hdf5-inventory-sha256", required=True)
    parser.add_argument(
        "--train-venv-profile",
        choices=("legacy-tp2", "single-gpu-tp1"),
        default="legacy-tp2",
    )
    parser.add_argument("--output-dir", type=Path, required=True, help="must not already exist")
    return parser.parse_args(argv)


def run_qualification(args: argparse.Namespace, *, project_root: Path | None = None) -> tuple[Path, str]:
    root = Path(__file__).resolve().parents[1] if project_root is None else project_root.resolve()
    cache_root = Path(os.environ.get("DUO_VLA_CACHE_ROOT", "/root/.cache/duo-vla")).resolve()
    train_venv_profile = getattr(args, "train_venv_profile", "legacy-tp2")
    require(train_venv_profile in {"legacy-tp2", "single-gpu-tp1"}, "unknown train-venv profile")
    single_gpu = train_venv_profile == "single-gpu-tp1"
    train_venv = cache_root / "venvs" / ("train-single-gpu" if single_gpu else "train")
    attestation, attestation_raw_sha256 = read_stable_json(
        args.simulator_attestation.resolve(),
        name="simulator attestation",
        expected_sha256=args.simulator_attestation_sha256,
    )
    require(attestation.get("schema") == SIMULATOR_ATTESTATION_SCHEMA, "simulator attestation schema differs")
    validator_runtime_start = validator_runtime_identity(root, cache_root, attestation)
    try:
        dataset_snapshot_start = verify_huggingface_snapshot(
            args.snapshot_root.resolve(),
            expected_revision=DATASET_REVISION,
        )
        train_venv_start = content_address_train_venv(train_venv)
    except (OSError, RuntimeError, ValueError) as exc:
        raise QualificationError(f"cannot authenticate live training data/runtime: {exc}") from exc
    dataset_tree, dataset_tree_raw_sha256 = read_stable_json(
        args.dataset_tree_metadata.resolve(),
        name="dataset tree metadata",
        expected_sha256=args.dataset_tree_metadata_sha256,
    )
    normalization, normalization_raw_sha256 = read_stable_json(
        args.normalization_artifact.resolve(),
        name="normalization artifact",
        expected_sha256=args.normalization_artifact_sha256,
    )
    original_hdf5_inventory, original_hdf5_inventory_raw_sha256 = read_stable_json(
        args.original_hdf5_inventory.resolve(),
        name="original HDF5 inventory",
        expected_sha256=args.original_hdf5_inventory_sha256,
    )
    expected_inputs = build_expected_inputs(
        root,
        simulator_attestation=attestation,
        simulator_attestation_raw_sha256=attestation_raw_sha256,
        dataset_tree=dataset_tree,
        dataset_tree_raw_sha256=dataset_tree_raw_sha256,
        normalization=normalization,
        normalization_raw_sha256=normalization_raw_sha256,
        dataset_snapshot=dataset_snapshot_start,
        train_venv_identity=train_venv_start,
        original_hdf5_inventory=original_hdf5_inventory,
        original_hdf5_inventory_raw_sha256=original_hdf5_inventory_raw_sha256,
    )
    start_source = source_identity(root, single_gpu=single_gpu)
    require(
        start_source
        == {
            "config_file_sha256": expected_inputs["config_file_sha256"],
            "project_source_tree_sha256": expected_inputs["project_source_tree_sha256"],
            "source_files_sha256": expected_inputs["source_files_sha256"],
        },
        "qualification source/config differs from the evidence inputs",
    )
    tasks = _task_identities(attestation)
    evidence_path = args.evidence_manifest.resolve()
    evidence, evidence_raw_sha256 = read_stable_json(
        evidence_path,
        name="expert replay evidence manifest",
        expected_sha256=args.evidence_manifest_sha256,
    )
    summary = validate_evidence_document(
        evidence,
        expected_inputs=expected_inputs,
        task_inventory=tasks,
        raw_evidence_root=evidence_path.parent,
    )
    try:
        dataset_snapshot_end = verify_huggingface_snapshot(
            args.snapshot_root.resolve(),
            expected_revision=DATASET_REVISION,
        )
        train_venv_end = content_address_train_venv(train_venv)
        require_matching_train_venv(train_venv_start, train_venv_end)
    except (OSError, RuntimeError, ValueError) as exc:
        raise QualificationError(f"live training data/runtime changed during qualification: {exc}") from exc
    require(
        dataset_snapshot_end == dataset_snapshot_start,
        "live training snapshot changed during qualification",
    )
    final_inputs = (
        (args.simulator_attestation, "simulator attestation", args.simulator_attestation_sha256, attestation),
        (args.dataset_tree_metadata, "dataset tree metadata", args.dataset_tree_metadata_sha256, dataset_tree),
        (args.normalization_artifact, "normalization artifact", args.normalization_artifact_sha256, normalization),
        (
            args.original_hdf5_inventory,
            "original HDF5 inventory",
            args.original_hdf5_inventory_sha256,
            original_hdf5_inventory,
        ),
        (args.evidence_manifest, "expert replay evidence manifest", args.evidence_manifest_sha256, evidence),
    )
    for input_path, input_name, input_sha256, expected_value in final_inputs:
        observed_value, _observed_sha256 = read_stable_json(
            input_path.resolve(),
            name=input_name,
            expected_sha256=input_sha256,
        )
        require(observed_value == expected_value, f"{input_name} changed during qualification")
    final_summary = validate_evidence_document(
        evidence,
        expected_inputs=expected_inputs,
        task_inventory=tasks,
        raw_evidence_root=evidence_path.parent,
    )
    require(final_summary == summary, "expert replay evidence summary changed during qualification")
    require(
        source_identity(root, single_gpu=single_gpu) == start_source,
        "qualification source/config changed before report construction",
    )
    report = build_report(
        evidence,
        evidence_manifest_raw_sha256=evidence_raw_sha256,
        expected_inputs=expected_inputs,
        task_inventory=tasks,
        replay_summary=summary,
        validator_runtime=validator_runtime_start,
        validator_source=start_source,
    )
    validate_qualification_report_document(
        report,
        project_root=root,
        simulator_attestation=attestation,
        simulator_attestation_raw_sha256=attestation_raw_sha256,
    )
    require(
        source_identity(root, single_gpu=single_gpu) == start_source,
        "qualification source/config changed before publication",
    )
    validator_runtime_end = validator_runtime_identity(root, cache_root, attestation)
    require(validator_runtime_end == validator_runtime_start, "validator runtime changed before publication")
    return publish_report_exclusive(args.output_dir.resolve(), report)


def main(argv: Sequence[str] | None = None) -> None:
    path, digest = run_qualification(parse_args(argv))
    print(json.dumps({"qualification_report": str(path), "sha256": digest}, sort_keys=True))


if __name__ == "__main__":
    main()
