#!/usr/bin/env python3
"""Persistent TP=2 Duo-VLA policy server for the isolated CALVIN evaluator."""

from __future__ import annotations

import argparse
import base64
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import site
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from calvin_bridge import (
    ACTION_DIM,
    ACTION_HORIZON,
    MODEL_REVISION,
    PROTOCOL,
    STATE_DIM,
    make_error_response,
    make_health_response,
    make_predict_response,
    make_success_response,
    serve_unix_policy,
)

MODEL_ID = "google/diffusiongemma-26B-A4B-it"
ARCHIVE_SHA256 = "c2036c67eb4c06966af1d1e1665bdb572c69e1404f5e77ffd46b384ff2b79f74"
TRAIN_LOCK_SHA256 = "0b1fb188747ee99224078b3c40975ca7e6f8e082e22d2860f9b50ee679a67c46"
EXPECTED_TRAIN_PYTHON = "3.11.15"
CALVIN_STATE_ADAPTER = "robot_obs[0:7]+robot_obs[14:15]"
CALVIN_ACTION_ADAPTER = "identity_official_scaled_rel_actions"
CALVIN_CAMERA_SHAPES = {"rgb_static": [200, 200, 3], "rgb_gripper": [84, 84, 3]}
EXPERTS_IMPLEMENTATION = "grouped_mm"
EXPERT_BATCH_ISOLATION = "sample_isolated_grouped_mm_v1"
PHYSICAL_BATCH_SIZE = 8
_EXECUTION_GEOMETRY_FIELDS = {
    "expert_batch_isolation",
    "experts_implementation",
    "fixed_physical_prefix_width",
    "physical_batch_size",
    "prefix_geometry_content_sha256",
}
PINNED_CALVIN_SOURCE_REVISIONS = {
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
    "torch": "2.13.0+cu126",
    "torchvision": "0.28.0+cu126",
    "transformers": "5.15.0",
}
REQUIRED_TRAIN_ENVIRONMENT = {
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
REQUIRED_SERVING_ENVIRONMENT = {
    **REQUIRED_TRAIN_ENVIRONMENT,
    "BLIS_NUM_THREADS": "1",
    "LANG": "C.UTF-8",
    "LC_ALL": "C.UTF-8",
    "PATH": "/usr/bin:/bin",
    "PYTHONHASHSEED": "0",
    "PYTHONSAFEPATH": "1",
    "PYTHONDONTWRITEBYTECODE": "1",
    "RAYON_NUM_THREADS": "1",
    "TORCH_NCCL_ASYNC_ERROR_HANDLING": "1",
    "TZ": "UTC",
    "VECLIB_MAXIMUM_THREADS": "1",
}
SERVING_RUNTIME_SCHEMA = "duo-vla-calvin-serving-runtime-v3"
_RELEVANT_ENVIRONMENT_PREFIXES = (
    "BASH_",
    "BLIS_",
    "CUBLAS_",
    "CUDA_",
    "CUDNN_",
    "CDPATH",
    "DUO_VLA_",
    "ENV",
    "GCONV_PATH",
    "GLIBC_",
    "GOMP_",
    "GLOBIGNORE",
    "HF_",
    "KMP_",
    "LANG",
    "LC_",
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
    "PYTHON",
    "PYTORCH_",
    "RAYON_",
    "TOKENIZERS_",
    "TORCH_NCCL_",
    "TRANSFORMERS_",
    "TZ",
    "VECLIB_",
)
_DISTRIBUTION_IMPORT_NAMES = {
    "accelerate": "accelerate",
    "huggingface-hub": "huggingface_hub",
    "numpy": "numpy",
    "peft": "peft",
    "pillow": "PIL",
    "pyarrow": "pyarrow",
    "safetensors": "safetensors",
    "tokenizers": "tokenizers",
    "torch": "torch",
    "torchvision": "torchvision",
    "transformers": "transformers",
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
_CALVIN_NORMALIZATION_FIELDS = frozenset(
    {"action", "algorithm", "content_sha256", "counts", "dataset", "schema", "split", "state"}
)
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
_CALVIN_PERMANENT_STORAGE_FIELDS = frozenset(
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
        "reader_schema",
        "storage_identity_sha256",
        "storage_mode",
    }
)
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
_HEALTH_WIRE_IDENTITY_FIELDS = frozenset(
    {
        "calvin_identity",
        "checkpoint_manifest_sha256",
        "execution_geometry",
        "mode",
        "model_revision",
        "nfe",
        "normalization_content_sha256",
        "normalization_metadata_sha256",
        "objective",
        "policy_contract_sha256",
        "sampler",
        "serving_runtime_sha256",
        "train_seed",
    }
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, allow_nan=False, separators=(",", ":"), sort_keys=True).encode()
    return hashlib.sha256(payload).hexdigest()


def _valid_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _normalization_permanent_storage_identity(dataset: Any) -> dict[str, Any]:
    """Validate and copy the exact archive-direct identity emitted by normalization v4."""

    from duo_vla.data.calvin_archive import (
        CALVIN_ARCHIVE_READER_SCHEMA,
        CALVIN_INDEX_NAME,
        CALVIN_MEMBER_INDEX_SCHEMA,
        OFFICIAL_CENTRAL_DIRECTORY_SHA256,
    )
    from duo_vla.data.calvin_stats import (
        CALVIN_ABC_D_ARCHIVE_BYTES,
        CALVIN_ABC_D_ARCHIVE_SHA256,
        CALVIN_CRITICAL_TRAIN_METADATA,
        CALVIN_DATASET_MANIFEST_SCHEMA,
        CALVIN_STORAGE_MODE_ARCHIVE_DIRECT,
    )

    require(isinstance(dataset, dict), "CALVIN normalization dataset identity is missing")
    assert isinstance(dataset, dict)
    require(
        set(dataset) == _CALVIN_NORMALIZATION_DATASET_FIELDS,
        "CALVIN normalization dataset identity field inventory differs",
    )
    member_index = dataset.get("member_index")
    require(
        isinstance(member_index, dict) and set(member_index) == _CALVIN_MEMBER_INDEX_FIELDS,
        "CALVIN normalization member-index identity field inventory differs",
    )
    assert isinstance(member_index, dict)
    expected = {
        "archive_bytes": CALVIN_ABC_D_ARCHIVE_BYTES,
        "archive_sha256": CALVIN_ABC_D_ARCHIVE_SHA256,
        "central_directory_sha256": OFFICIAL_CENTRAL_DIRECTORY_SHA256,
        "dataset_manifest_schema": CALVIN_DATASET_MANIFEST_SCHEMA,
        "member_index.path": CALVIN_INDEX_NAME,
        "member_index.schema": CALVIN_MEMBER_INDEX_SCHEMA,
        "metadata_files": list(CALVIN_CRITICAL_TRAIN_METADATA),
        "name": "task_ABC_D",
        "reader_schema": CALVIN_ARCHIVE_READER_SCHEMA,
        "split": "training",
        "storage_mode": CALVIN_STORAGE_MODE_ARCHIVE_DIRECT,
    }
    observed = {
        "archive_bytes": dataset.get("archive_bytes"),
        "archive_sha256": dataset.get("archive_sha256"),
        "central_directory_sha256": dataset.get("central_directory_sha256"),
        "dataset_manifest_schema": dataset.get("dataset_manifest_schema"),
        "member_index.path": member_index.get("path"),
        "member_index.schema": member_index.get("schema"),
        "metadata_files": dataset.get("metadata_files"),
        "name": dataset.get("name"),
        "reader_schema": dataset.get("reader_schema"),
        "split": dataset.get("split"),
        "storage_mode": dataset.get("storage_mode"),
    }
    mismatches = sorted(name for name, expected_value in expected.items() if observed[name] != expected_value)
    require(not mismatches, f"CALVIN normalization archive-direct identity mismatch: {mismatches}")
    require(
        type(member_index.get("bytes")) is int and member_index["bytes"] > 0,
        "CALVIN normalization member-index byte count is invalid",
    )
    for name, value in {
        "central_directory_sha256": dataset.get("central_directory_sha256"),
        "dataset_manifest_file_sha256": dataset.get("dataset_manifest_file_sha256"),
        "dataset_manifest_sha256": dataset.get("dataset_manifest_sha256"),
        "member_index.sha256": member_index.get("sha256"),
        "member_inventory_sha256": dataset.get("member_inventory_sha256"),
        "metadata_sha256": dataset.get("metadata_sha256"),
        "storage_identity_sha256": dataset.get("storage_identity_sha256"),
    }.items():
        require(_valid_sha256(value), f"CALVIN normalization {name} is invalid")
    identity = {
        "archive_bytes": dataset["archive_bytes"],
        "archive_sha256": dataset["archive_sha256"],
        "central_directory_sha256": dataset["central_directory_sha256"],
        "dataset_manifest_file_sha256": dataset["dataset_manifest_file_sha256"],
        "dataset_manifest_schema": dataset["dataset_manifest_schema"],
        "dataset_manifest_sha256": dataset["dataset_manifest_sha256"],
        "member_index": {
            "bytes": member_index["bytes"],
            "path": member_index["path"],
            "schema": member_index["schema"],
            "sha256": member_index["sha256"],
        },
        "member_inventory_sha256": dataset["member_inventory_sha256"],
        "metadata_files": list(dataset["metadata_files"]),
        "metadata_sha256": dataset["metadata_sha256"],
        "reader_schema": dataset["reader_schema"],
        "storage_identity_sha256": dataset["storage_identity_sha256"],
        "storage_mode": dataset["storage_mode"],
    }
    require(set(identity) == _CALVIN_PERMANENT_STORAGE_FIELDS, "CALVIN permanent storage identity drifted")
    return identity


def _validate_checkpoint_permanent_storage_identity(
    manifest: dict[str, Any],
    expected: dict[str, Any],
) -> None:
    """Cross-check the typed checkpoint fields against one normalization identity."""

    member_index = expected.get("member_index")
    require(
        set(expected) == _CALVIN_PERMANENT_STORAGE_FIELDS
        and isinstance(member_index, dict)
        and set(member_index) == _CALVIN_MEMBER_INDEX_FIELDS,
        "expected CALVIN permanent storage identity is invalid",
    )
    assert isinstance(member_index, dict)
    observed = {
        "archive_bytes": manifest.get("archive_bytes"),
        "archive_sha256": manifest.get("archive_sha256"),
        "central_directory_sha256": manifest.get("central_directory_sha256"),
        "dataset_manifest_file_sha256": manifest.get("dataset_manifest_file_sha256"),
        "dataset_manifest_schema": manifest.get("dataset_manifest_schema"),
        "dataset_manifest_sha256": manifest.get("dataset_manifest_sha256"),
        "member_index": {
            "bytes": manifest.get("member_index_bytes"),
            "path": manifest.get("member_index_path"),
            "schema": manifest.get("member_index_schema"),
            "sha256": manifest.get("member_index_sha256"),
        },
        # The canonical checkpoint carries this exact list inside
        # ``calvin_identity``; its flat run contract authenticates the digest.
        "metadata_files": expected["metadata_files"],
        "member_inventory_sha256": manifest.get("member_inventory_sha256"),
        "metadata_sha256": manifest.get("metadata_sha256"),
        "reader_schema": manifest.get("reader_schema"),
        "storage_identity_sha256": manifest.get("storage_identity_sha256"),
        "storage_mode": manifest.get("storage_mode"),
    }
    mismatches = sorted(name for name in expected if observed[name] != expected[name])
    require(not mismatches, f"checkpoint permanent CALVIN storage identity mismatch: {mismatches}")


def _validate_checkpoint_calvin_identity(value: Any, expected_storage: dict[str, Any]) -> dict[str, Any]:
    require(isinstance(value, dict), "checkpoint CALVIN identity is missing")
    assert isinstance(value, dict)
    require(set(value) == _CALVIN_IDENTITY_FIELDS, "checkpoint CALVIN identity field inventory differs")
    observed_storage = {name: value.get(name) for name in _CALVIN_PERMANENT_STORAGE_FIELDS}
    require(observed_storage == expected_storage, "checkpoint CALVIN identity storage fields mismatch")
    return value


def _execution_geometry_from_config(config: dict[str, Any]) -> dict[str, Any]:
    model = config.get("model")
    optimization = config.get("optimization")
    benchmark = config.get("benchmark")
    require(
        isinstance(model, dict) and isinstance(optimization, dict) and isinstance(benchmark, dict),
        "resolved config has no execution-geometry sections",
    )
    assert isinstance(model, dict) and isinstance(optimization, dict) and isinstance(benchmark, dict)
    geometry = {
        "expert_batch_isolation": model.get("expert_batch_isolation"),
        "experts_implementation": model.get("experts_implementation"),
        "fixed_physical_prefix_width": benchmark.get("fixed_physical_prefix_width"),
        "physical_batch_size": optimization.get("physical_batch_size"),
        "prefix_geometry_content_sha256": benchmark.get("prefix_geometry_content_sha256"),
    }
    require(geometry["experts_implementation"] == EXPERTS_IMPLEMENTATION, "expert backend contract mismatch")
    require(
        geometry["expert_batch_isolation"] == EXPERT_BATCH_ISOLATION,
        "expert batch-isolation contract mismatch",
    )
    require(
        type(geometry["physical_batch_size"]) is int and geometry["physical_batch_size"] == PHYSICAL_BATCH_SIZE,
        "physical batch contract must equal eight",
    )
    require(
        type(geometry["fixed_physical_prefix_width"]) is int
        and 0 < geometry["fixed_physical_prefix_width"] <= 1024 - ACTION_HORIZON,
        "fixed physical prefix width is invalid",
    )
    require(_valid_sha256(geometry["prefix_geometry_content_sha256"]), "prefix geometry SHA-256 is invalid")
    require(config.get("execution_geometry") == geometry, "resolved execution-geometry identity mismatch")
    return geometry


def _validate_execution_geometry(value: Any, *, allow_none: bool = False) -> dict[str, Any] | None:
    if value is None and allow_none:
        return None
    require(isinstance(value, dict), "execution geometry must be an object")
    assert isinstance(value, dict)
    require(set(value) == _EXECUTION_GEOMETRY_FIELDS, "execution geometry fields differ")
    # Reuse the config validator without permitting a second representation.
    envelope = {
        "model": {
            "expert_batch_isolation": value.get("expert_batch_isolation"),
            "experts_implementation": value.get("experts_implementation"),
        },
        "optimization": {"physical_batch_size": value.get("physical_batch_size")},
        "benchmark": {
            "fixed_physical_prefix_width": value.get("fixed_physical_prefix_width"),
            "prefix_geometry_content_sha256": value.get("prefix_geometry_content_sha256"),
        },
        "execution_geometry": value,
    }
    return _execution_geometry_from_config(envelope)


def _source_tree_sha256(root: Path) -> str:
    """Repeat the trainer's injectively framed source identity."""

    from duo_vla.calvin_source_identity import calvin_source_tree_sha256

    return calvin_source_tree_sha256(
        root,
        explicit_relative_paths=_CALVIN_SOURCE_EXPLICIT_RELATIVE_PATHS,
        magic=_CALVIN_SOURCE_TREE_HASH_MAGIC,
    )


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _validated_serving_process_environment(
    project_root: Path,
    *,
    require_tp_launch: bool = True,
) -> tuple[dict[str, Any], Path]:
    """Require the canonical wrapper environment before any scored CUDA work."""

    require(
        platform.python_version() == EXPECTED_TRAIN_PYTHON,
        f"policy server requires Python {EXPECTED_TRAIN_PYTHON}",
    )
    require(not site.ENABLE_USER_SITE and sys.flags.no_user_site == 1, "policy server requires the user site disabled")
    require(sys.flags.safe_path, "policy server requires Python safe-path mode")
    require(sys.flags.dont_write_bytecode, "policy server requires bytecode writes disabled")

    canonical_project = project_root.resolve()
    cache_value = os.environ.get("DUO_VLA_CACHE_ROOT")
    hf_value = os.environ.get("HF_HOME")
    require(isinstance(cache_value, str) and Path(cache_value).is_absolute(), "DUO_VLA_CACHE_ROOT must be absolute")
    require(isinstance(hf_value, str) and Path(hf_value).is_absolute(), "HF_HOME must be absolute")
    cache_root = Path(cache_value).resolve()
    hf_home = Path(hf_value).resolve()
    require(str(cache_root) == cache_value, "DUO_VLA_CACHE_ROOT must be canonical")
    require(str(hf_home) == hf_value, "HF_HOME must be canonical")
    expected_prefix = (cache_root / "venvs/train").resolve()
    require(
        Path(sys.prefix).resolve() == expected_prefix,
        f"policy server requires the pinned train venv: {expected_prefix}",
    )

    expected_paths = {
        "DUO_VLA_CACHE_ROOT": str(cache_root),
        "DUO_VLA_PROJECT_ROOT": str(canonical_project),
        "DUO_VLA_TRAIN_VENV": str(expected_prefix),
        "HF_HOME": str(hf_home),
        "PYTHONPATH": os.pathsep.join(
            (
                str((canonical_project / "src").resolve()),
                str((canonical_project / "scripts/calvin").resolve()),
            )
        ),
    }
    expected_environment = {**REQUIRED_SERVING_ENVIRONMENT, **expected_paths}
    observed_environment = {name: os.environ.get(name) for name in expected_environment}
    require(
        observed_environment == expected_environment,
        f"policy server environment differs from the canonical launcher: {observed_environment}",
    )
    relevant_names = {name for name in os.environ if name.startswith(_RELEVANT_ENVIRONMENT_PREFIXES)}
    unexpected = sorted(relevant_names - set(expected_environment))
    require(not unexpected, f"policy server environment contains unpinned overrides: {unexpected}")

    if require_tp_launch:
        require(os.environ.get("WORLD_SIZE") == "2", "policy server requires WORLD_SIZE=2")
        require(os.environ.get("LOCAL_WORLD_SIZE") == "2", "policy server requires LOCAL_WORLD_SIZE=2")
        rank = os.environ.get("RANK")
        local_rank = os.environ.get("LOCAL_RANK")
        require(rank in {"0", "1"} and local_rank == rank, "policy server requires one local rank for each TP rank")
    return (
        {
            "distributed": {
                "local_rank_equals_rank": require_tp_launch,
                "local_world_size": 2 if require_tp_launch else None,
                "world_size": 2 if require_tp_launch else None,
            },
            "environment": dict(sorted(expected_environment.items())),
        },
        expected_prefix,
    )


def _file_digest(path: Path) -> tuple[bytes, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            size += len(block)
            digest.update(block)
    return digest.digest(), size


def _verified_distribution_identity(name: str, expected_version: str, venv_root: Path) -> dict[str, Any]:
    """Verify every installed wheel file against RECORD and return a compact content identity."""

    distribution = importlib.metadata.distribution(name)
    require(distribution.version == expected_version, f"installed {name} version differs from the lock")
    files = distribution.files
    require(files is not None and bool(files), f"installed {name} distribution has no RECORD inventory")
    inventory: list[dict[str, Any]] = []
    unhashed: list[str] = []
    total_bytes = 0
    for entry in sorted(files, key=lambda value: str(value)):
        relative = str(entry).replace(os.sep, "/")
        unresolved = Path(distribution.locate_file(entry))
        require(unresolved.is_file(), f"installed {name} file is missing: {relative}")
        resolved = unresolved.resolve()
        require(_is_relative_to(resolved, venv_root), f"installed {name} file escapes the train venv: {resolved}")
        digest, size = _file_digest(resolved)
        total_bytes += size
        recorded_hash = entry.hash
        if recorded_hash is None:
            unhashed.append(relative)
        else:
            require(recorded_hash.mode == "sha256", f"installed {name} RECORD uses a non-SHA256 digest")
            encoded = base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")
            require(encoded == recorded_hash.value, f"installed {name} file hash differs from RECORD: {relative}")
            if entry.size is not None:
                require(size == entry.size, f"installed {name} file size differs from RECORD: {relative}")
        inventory.append({"bytes": size, "path": relative, "sha256": digest.hex()})
    require(
        len(unhashed) == 1 and unhashed[0].endswith(".dist-info/RECORD"),
        f"installed {name} has unexpected unhashed RECORD entries: {unhashed}",
    )
    return {
        "content_inventory_sha256": _canonical_sha256(inventory),
        "files_verified": len(inventory),
        "record_sha256": next(item["sha256"] for item in inventory if item["path"] == unhashed[0]),
        "total_bytes": total_bytes,
        "version": expected_version,
    }


def _serving_process_identity(project_root: Path, package_versions: dict[str, str]) -> dict[str, Any]:
    """Authenticate process paths, imports, and installed distribution bytes."""

    environment, venv_root = _validated_serving_process_environment(project_root)
    project_root = project_root.resolve()
    project_src = (project_root / "src").resolve()
    project_calvin = (project_root / "scripts/calvin").resolve()
    expected_prefix = venv_root.resolve()
    site_packages = expected_prefix / f"lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages"

    effective_sys_path: list[str] = []
    for entry in sys.path:
        require(bool(entry), "policy server sys.path contains the current working directory")
        resolved = Path(entry).resolve()
        require(
            resolved in {project_src, project_calvin}
            or _is_relative_to(resolved, Path(sys.base_prefix).resolve())
            or _is_relative_to(resolved, expected_prefix),
            f"policy server sys.path contains an untrusted entry: {resolved}",
        )
        effective_sys_path.append(str(resolved))
    require(
        effective_sys_path[:2] == [str(project_src), str(project_calvin)],
        "policy server project import roots are not first and canonical",
    )
    require(
        len(effective_sys_path) == len(set(effective_sys_path)),
        "policy server sys.path contains duplicate entries",
    )
    require(str(site_packages.resolve()) in effective_sys_path, "policy server train-venv site-packages is absent")

    expected_module_roots = {
        "calvin_bridge": project_calvin,
        "duo_vla": project_src,
        **{module: site_packages for module in _DISTRIBUTION_IMPORT_NAMES.values()},
    }
    module_origins: dict[str, str] = {}
    for module, expected_root in expected_module_roots.items():
        spec = importlib.util.find_spec(module)
        origin = None if spec is None else spec.origin
        require(
            isinstance(origin, str) and _is_relative_to(Path(origin).resolve(), expected_root.resolve()),
            f"policy server module {module!r} resolves outside {expected_root}: {origin}",
        )
        module_origins[module] = str(Path(origin).resolve())
    for module in ("sitecustomize", "usercustomize"):
        require(importlib.util.find_spec(module) is None, f"policy server forbids {module}.py injection")

    pth_files = {path.name: sha256_file(path) for path in sorted(site_packages.glob("*.pth")) if path.is_file()}
    distributions = {
        name: _verified_distribution_identity(name, package_versions[name], expected_prefix)
        for name in sorted(package_versions)
    }
    return {
        **environment,
        "distributions": distributions,
        "module_origins": module_origins,
        "python": {
            "base_prefix": str(Path(sys.base_prefix).resolve()),
            "executable": str(Path(sys.executable).resolve()),
            "prefix": str(expected_prefix),
            "version": platform.python_version(),
        },
        "site_packages_pth": pth_files,
        "sys_path": effective_sys_path,
    }


def load_pinned_source_revisions(project_root: Path) -> dict[str, str]:
    path = project_root / "scripts/calvin/revisions.env"
    require(path.is_file(), f"pinned CALVIN revision file is missing: {path}")
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line and not line.startswith("#"):
            key, separator, value = line.partition("=")
            require(bool(separator and key and value), f"invalid CALVIN revision entry: {line!r}")
            values[key] = value
    observed = {
        "calvin": values.get("CALVIN_REVISION"),
        "calvin_env": values.get("CALVIN_ENV_REVISION"),
        "tacto": values.get("CALVIN_TACTO_REVISION"),
    }
    require(observed == PINNED_CALVIN_SOURCE_REVISIONS, f"CALVIN source revision pin mismatch: {observed}")
    return dict(PINNED_CALVIN_SOURCE_REVISIONS)


def _verify_dataset_with_generation(
    dataset_root: Path,
    manifest_path: Path | None = None,
    *,
    verify_archive: bool = True,
    allow_legacy_v3: bool = False,
) -> tuple[dict[str, Any], Any | None]:
    """Authenticate production archive-direct storage and current A/B/C metadata."""

    root = dataset_root.resolve()
    require(root.is_dir() and root.name == "task_ABC_D", f"invalid CALVIN dataset root: {root}")
    from duo_vla.data.calvin_stats import (
        CALVIN_ABC_D_ARCHIVE_SHA256,
        CALVIN_CRITICAL_TRAIN_METADATA,
        _read_strict_json,
        _storage_identity_from_manifest,
        authenticate_calvin_dataset_generation,
        calvin_dataset_manifest_path,
        calvin_metadata_sha256,
        load_calvin_dataset_manifest,
    )

    require(CALVIN_ABC_D_ARCHIVE_SHA256 == ARCHIVE_SHA256, "server/stats archive pin mismatch")
    training_root = root / "training"
    selected_manifest = calvin_dataset_manifest_path(training_root).resolve()
    if manifest_path is not None:
        require(manifest_path.resolve() == selected_manifest, "--dataset-manifest differs from the canonical path")
    authenticated_generation = None
    if allow_legacy_v3:
        payload = load_calvin_dataset_manifest(
            training_root,
            verify_archive=verify_archive,
            allow_legacy_v3=True,
        )
        current_payload, manifest_file_sha256 = _read_strict_json(selected_manifest)
        require(current_payload == payload, "CALVIN dataset manifest changed after authentication")
        storage = _storage_identity_from_manifest(payload, manifest_file_sha256=manifest_file_sha256)
        metadata_sha256 = calvin_metadata_sha256(training_root)
    else:
        payload, authenticated_generation = authenticate_calvin_dataset_generation(training_root)
        storage = authenticated_generation.storage
        metadata_sha256 = authenticated_generation.metadata_sha256
    permanent_identity = {
        "archive_bytes": storage.archive_bytes,
        "archive_sha256": storage.archive_sha256,
        "central_directory_sha256": (
            dict(storage.central_directory)["sha256"] if storage.central_directory is not None else None
        ),
        "dataset_manifest_file_sha256": storage.manifest_file_sha256,
        "dataset_manifest_schema": storage.manifest_schema,
        "dataset_manifest_sha256": storage.manifest_content_sha256,
        "member_index": {
            "bytes": storage.member_index_bytes,
            "path": storage.member_index_path,
            "schema": storage.member_index_schema,
            "sha256": storage.member_index_sha256,
        },
        "member_inventory_sha256": storage.member_inventory_sha256,
        "metadata_files": list(CALVIN_CRITICAL_TRAIN_METADATA),
        "metadata_sha256": metadata_sha256,
        "reader_schema": storage.reader_schema,
        "storage_identity_sha256": storage.content_sha256,
        "storage_mode": storage.mode,
    }
    require(
        set(permanent_identity) == _CALVIN_PERMANENT_STORAGE_FIELDS,
        "CALVIN dataset permanent identity field inventory drifted",
    )
    calvin_identity = {"name": "task_ABC_D", "split": "training", **permanent_identity}
    if not allow_legacy_v3:
        require(
            _normalization_permanent_storage_identity(calvin_identity) == permanent_identity,
            "CALVIN live archive-direct identity is not canonical",
        )
    report = {
        **permanent_identity,
        "calvin_identity": calvin_identity,
        "dataset_manifest_content_sha256": storage.manifest_content_sha256,
        "dataset_manifest": str(selected_manifest),
        "dataset_root": str(root),
    }
    return report, authenticated_generation


def verify_dataset(
    dataset_root: Path,
    manifest_path: Path | None = None,
    *,
    verify_archive: bool = True,
    allow_legacy_v3: bool = False,
) -> dict[str, Any]:
    report, _authenticated_generation = _verify_dataset_with_generation(
        dataset_root,
        manifest_path,
        verify_archive=verify_archive,
        allow_legacy_v3=allow_legacy_v3,
    )
    return report


def model_snapshot_preflight() -> dict[str, Any]:
    from duo_vla.hf_snapshot import verify_huggingface_snapshot

    hf_home = Path(os.environ.get("HF_HOME", "/root/.cache/huggingface"))
    snapshot = hf_home / "hub/models--google--diffusiongemma-26B-A4B-it/snapshots" / MODEL_REVISION
    verified = verify_huggingface_snapshot(snapshot, expected_revision=MODEL_REVISION)
    index = json.loads((snapshot / "model.safetensors.index.json").read_text(encoding="utf-8"))
    shards = sorted(set(index.get("weight_map", {}).values()))
    require(len(shards) == 11, f"expected 11 DiffusionGemma weight shards, found {len(shards)}")
    return {
        "content_inventory_sha256": verified["content_inventory_sha256"],
        "files_verified": verified["files_verified"],
        "id": MODEL_ID,
        "revision": MODEL_REVISION,
        "shards": len(shards),
        "snapshot": str(snapshot),
        "total_snapshot_bytes": verified["total_bytes"],
        "tree_metadata_sha256": verified["tree_metadata_sha256"],
    }


def _validate_split(split: Any) -> None:
    require(isinstance(split, dict), "CALVIN normalization split identity is missing")
    required = {
        "algorithm",
        "seed",
        "train_episode_indices",
        "train_episode_sha256",
        "validation_episode_indices",
        "validation_episode_sha256",
        "validation_fraction",
    }
    require(set(split) == required, "CALVIN normalization split fields differ")
    for prefix in ("train", "validation"):
        indices = split[f"{prefix}_episode_indices"]
        require(
            isinstance(indices, list)
            and bool(indices)
            and all(type(value) is int and value >= 0 for value in indices)
            and indices == sorted(set(indices)),
            f"CALVIN {prefix} episode indices are invalid",
        )
        digest = hashlib.sha256(",".join(map(str, indices)).encode()).hexdigest()
        require(split[f"{prefix}_episode_sha256"] == digest, f"CALVIN {prefix} split hash mismatch")
    require(
        set(split["train_episode_indices"]).isdisjoint(split["validation_episode_indices"]),
        "CALVIN normalization train/validation split overlaps",
    )


def _validate_normalization_contract(stats: dict[str, Any]) -> dict[str, Any]:
    from duo_vla.data.calvin_stats import CALVIN_STATS_SCHEMA

    dataset = stats.get("dataset")
    state = stats.get("state")
    action = stats.get("action")
    algorithm = stats.get("algorithm")
    require(set(stats) == _CALVIN_NORMALIZATION_FIELDS, "normalization-v4 field inventory differs")
    require(stats.get("schema") == CALVIN_STATS_SCHEMA, "official serving requires normalization-v4")
    require(_valid_sha256(stats.get("content_sha256")), "normalization content SHA-256 is invalid")
    require(isinstance(stats.get("counts"), dict), "normalization count identity is missing")
    require(
        all(isinstance(value, dict) for value in (dataset, state, action, algorithm)),
        "normalization identity is incomplete",
    )
    assert (
        isinstance(dataset, dict)
        and isinstance(state, dict)
        and isinstance(action, dict)
        and isinstance(algorithm, dict)
    )
    permanent_identity = _normalization_permanent_storage_identity(dataset)
    require(state.get("dimension") == 8, "normalization state dimension mismatch")
    require(
        state.get("continuous_dimensions") == list(range(7)) and state.get("gripper_index") == 7,
        "state normalization channels mismatch",
    )
    require(state.get("observed_gripper_values") == [-1.0, 1.0], "state gripper normalization contract mismatch")
    q01 = np.asarray(state.get("q01"), dtype=np.float64)
    q99 = np.asarray(state.get("q99"), dtype=np.float64)
    require(q01.shape == (7,) and q99.shape == (7,), "state percentile vectors must contain seven values")
    require(
        bool(np.isfinite(q01).all() and np.isfinite(q99).all() and (q99 >= q01).all()),
        "state percentile values are invalid",
    )
    require(action.get("dimension") == 7, "normalization action dimension mismatch")
    require(
        action.get("continuous_dimensions") == list(range(6)) and action.get("gripper_index") == 6,
        "action channel contract mismatch",
    )
    require(action.get("observed_gripper_values") == [-1.0, 1.0], "action gripper contract mismatch")
    require(action.get("transform") == CALVIN_ACTION_ADAPTER, "CALVIN actions must remain official scaled rel_actions")
    minimum = np.asarray(action.get("continuous_min"), dtype=np.float64)
    maximum = np.asarray(action.get("continuous_max"), dtype=np.float64)
    require(minimum.shape == (6,) and maximum.shape == (6,), "action range identity must contain six values")
    require(bool(np.isfinite(minimum).all() and np.isfinite(maximum).all()), "action range identity is non-finite")
    require(
        bool((minimum >= -1.0 - 1e-6).all() and (maximum <= 1.0 + 1e-6).all()), "official scaled actions exceed [-1,1]"
    )
    require(algorithm.get("actions_re_normalized") is False, "CALVIN actions must not be percentile normalized")
    _validate_split(stats.get("split"))
    return permanent_identity


def _load_checkpoint_prefix_geometry(
    checkpoint: Path,
    manifest: dict[str, Any],
    config: dict[str, Any],
    *,
    model_snapshot_report: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Authenticate the copied contract against config, model, cameras, and manifest."""

    from duo_vla.prefix_geometry import (
        CameraGeometry,
        SnapshotTreeIdentity,
        load_prefix_geometry_contract,
    )

    geometry = _execution_geometry_from_config(config)
    artifacts = manifest.get("artifacts")
    require(isinstance(artifacts, dict), "checkpoint artifact table is missing")
    artifact = artifacts.get("prefix_geometry")
    require(
        isinstance(artifact, dict) and artifact.get("path") == "artifacts/prefix_geometry.json",
        "checkpoint has no canonical copied prefix-geometry artifact",
    )
    expected_snapshot = SnapshotTreeIdentity(
        repository_id=MODEL_ID,
        revision=model_snapshot_report.get("revision"),
        tree_metadata_sha256=model_snapshot_report.get("tree_metadata_sha256"),
        content_inventory_sha256=model_snapshot_report.get("content_inventory_sha256"),
        files_verified=model_snapshot_report.get("files_verified"),
        total_bytes=model_snapshot_report.get("total_snapshot_bytes"),
    )
    cameras = tuple(
        CameraGeometry(name=name, height=CALVIN_CAMERA_SHAPES[name][0], width=CALVIN_CAMERA_SHAPES[name][1])
        for name in ("rgb_static", "rgb_gripper")
    )
    payload = load_prefix_geometry_contract(
        checkpoint / artifact["path"],
        expected_content_sha256=geometry["prefix_geometry_content_sha256"],
        expected_model_identity=expected_snapshot,
        expected_processor_identity=expected_snapshot,
        expected_ordered_cameras=cameras,
        expected_fixed_physical_prefix_width=geometry["fixed_physical_prefix_width"],
    )
    require(manifest.get("execution_geometry") == geometry, "checkpoint execution geometry differs from config")
    require(
        manifest.get("experts_implementation") == geometry["experts_implementation"]
        and manifest.get("expert_batch_isolation") == geometry["expert_batch_isolation"]
        and manifest.get("physical_batch_size") == geometry["physical_batch_size"]
        and manifest.get("fixed_physical_prefix_width") == geometry["fixed_physical_prefix_width"]
        and manifest.get("prefix_geometry_content_sha256") == geometry["prefix_geometry_content_sha256"],
        "checkpoint execution-geometry fields differ from the resolved contract",
    )
    summary = manifest.get("prefix_geometry")
    require(
        isinstance(summary, dict)
        and summary
        == {
            "instruction_inventory_sha256": payload["instruction_inventory"]["sha256"],
            "maximum_valid_prefix_length": payload["geometry"]["maximum_valid_prefix_length"],
            "padding_side": payload["tokenization"]["padding_side"],
        },
        "checkpoint prefix-geometry summary mismatch",
    )
    return payload, geometry


def _validate_checkpoint_training_environment(
    manifest: dict[str, Any],
    config: dict[str, Any],
    *,
    project_root: Path,
    run_seed: int,
) -> dict[str, Any]:
    """Bind a checkpoint to the canonical Python 3.11 train runtime that produced it."""

    execution = manifest.get("execution_environment")
    require(isinstance(execution, dict), "checkpoint has no training execution environment")
    require(execution == config.get("execution_environment"), "checkpoint/config training environments differ")
    require(
        manifest.get("execution_environment_sha256") == _canonical_sha256(execution),
        "checkpoint training environment SHA-256 mismatch",
    )
    expected_outer_fields = {
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
    require(set(execution) == expected_outer_fields, "checkpoint training environment fields differ")
    runtime = execution.get("authenticated_runtime")
    require(isinstance(runtime, dict), "checkpoint has no authenticated training runtime")
    require(
        set(runtime) == {"environment", "lock_sha256", "module_origins", "packages", "python", "sys_path"},
        "checkpoint authenticated training-runtime fields differ",
    )
    require(runtime.get("lock_sha256") == TRAIN_LOCK_SHA256, "checkpoint training lock identity mismatch")
    require(runtime.get("packages") == EXPECTED_TRAIN_PACKAGES, "checkpoint training package identity mismatch")
    require(runtime.get("python") == EXPECTED_TRAIN_PYTHON, "checkpoint training Python identity mismatch")

    project_root = project_root.resolve()
    expected_pythonpath = str((project_root / "src").resolve())
    expected_environment = {**REQUIRED_TRAIN_ENVIRONMENT, "PYTHONPATH": expected_pythonpath}
    require(runtime.get("environment") == expected_environment, "checkpoint training process environment mismatch")
    cache_root = Path(os.environ.get("DUO_VLA_CACHE_ROOT", "/root/.cache/duo-vla")).resolve()
    train_prefix = (cache_root / "venvs/train").resolve()
    site_packages = train_prefix / f"lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages"
    expected_origins = {
        "duo_vla": str((project_root / "src/duo_vla/__init__.py").resolve()),
        **{
            module: str((site_packages / module / "__init__.py").resolve())
            for module in _DISTRIBUTION_IMPORT_NAMES.values()
        },
    }
    # Pillow and compiled/namespace package layouts are discovered rather than
    # guessed, but must still resolve below the exact pinned venv root.
    observed_origins = runtime.get("module_origins")
    require(
        isinstance(observed_origins, dict) and set(observed_origins) == set(expected_origins),
        "checkpoint training module-origin fields differ",
    )
    for module, origin in observed_origins.items():
        require(isinstance(origin, str), f"checkpoint training module origin is invalid: {module}")
        expected_root = project_root / "src" if module == "duo_vla" else site_packages
        require(
            _is_relative_to(Path(origin).resolve(), expected_root.resolve()),
            f"checkpoint training module {module!r} resolved outside {expected_root}",
        )

    recorded_sys_path = runtime.get("sys_path")
    safe_roots = (project_root / "scripts", project_root / "src", Path(sys.base_prefix), train_prefix)
    require(isinstance(recorded_sys_path, list) and bool(recorded_sys_path), "checkpoint training sys.path is invalid")
    for entry in recorded_sys_path:
        require(isinstance(entry, str) and bool(entry), "checkpoint training sys.path contains an empty entry")
        resolved = Path(entry).resolve()
        require(
            any(_is_relative_to(resolved, root.resolve()) for root in safe_roots),
            f"checkpoint training sys.path contains an untrusted entry: {resolved}",
        )

    expected_values = {
        "cublas_workspace_config": REQUIRED_TRAIN_ENVIRONMENT["CUBLAS_WORKSPACE_CONFIG"],
        "cuda_runtime": "12.6",
        "cudnn_benchmark": False,
        "cudnn_deterministic": True,
        "cudnn_tf32": False,
        "deterministic_algorithms": True,
        "deterministic_warn_only": False,
        "float32_matmul_precision": "highest",
        "matmul_tf32": False,
        "peft": EXPECTED_TRAIN_PACKAGES["peft"],
        "python": EXPECTED_TRAIN_PYTHON,
        "python_hash_seed": str(run_seed),
        "torch": EXPECTED_TRAIN_PACKAGES["torch"],
        "transformers": EXPECTED_TRAIN_PACKAGES["transformers"],
        "world_size": 2,
    }
    mismatches = {
        name: {"expected": expected, "observed": execution.get(name)}
        for name, expected in expected_values.items()
        if execution.get(name) != expected
    }
    require(not mismatches, f"checkpoint was not produced by the canonical training runtime: {mismatches}")
    gpu_capability = execution.get("gpu_capability")
    gpu_names = execution.get("gpu_names")
    require(
        isinstance(gpu_capability, list)
        and len(gpu_capability) == 2
        and all(
            isinstance(value, list) and len(value) == 2 and all(type(component) is int for component in value)
            for value in gpu_capability
        ),
        "checkpoint training GPU capability identity is invalid",
    )
    require(
        isinstance(gpu_names, list)
        and len(gpu_names) == 2
        and all(isinstance(value, str) and bool(value) for value in gpu_names),
        "checkpoint training GPU name identity is invalid",
    )
    require(
        type(execution.get("cudnn")) is int and execution["cudnn"] > 0,
        "checkpoint training cuDNN identity is invalid",
    )
    return execution


def _committed_checkpoint_record(checkpoint: Path, config_sha256: str) -> Any:
    """Require the selected checkpoint to be the run journal's authenticated tip."""

    from duo_vla.run_journal import load_run_journal, validate_resume_checkpoint

    require(checkpoint.parent.name == "checkpoints", "official checkpoint must live below a checkpoints directory")
    output_dir = checkpoint.parent.parent
    journal = load_run_journal(output_dir, expected_config_sha256=config_sha256)
    require(journal.latest_checkpoint is not None, "training run has no committed checkpoint")
    return validate_resume_checkpoint(output_dir, checkpoint)


def resolve_checkpoint(
    checkpoint_dir: Path,
    *,
    training_root: Path,
    project_root: Path,
    train_seed_override: int | None,
    model_snapshot_report: dict[str, Any],
    authenticated_generation: Any | None = None,
) -> tuple[dict[str, Any], Path, int, dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Resolve and cross-check every checkpoint, data, and source identity."""

    from duo_vla.backbones.loading import (
        validate_decoder_attention_lora_adapter_config,
        validate_decoder_attention_lora_weights,
    )
    from duo_vla.checkpointing import load_checkpoint_manifest
    from duo_vla.data.calvin_stats import load_calvin_state_normalizer
    from duo_vla.policy_contract import validate_manifest_policy_contract
    from duo_vla.run_config import canonical_config_sha256, load_resolved_toml, load_verified_resolved_config

    checkpoint = checkpoint_dir.resolve()
    manifest = load_checkpoint_manifest(checkpoint, verify_hashes=True)
    require(manifest.get("kind") == "resumable-calvin-abc-to-d-training", "checkpoint kind is not CALVIN ABC-to-D")
    require(manifest.get("model_id") == MODEL_ID, "checkpoint model identity mismatch")
    require(manifest.get("model_revision") == MODEL_REVISION, "checkpoint model revision mismatch")
    require(manifest.get("protocol") == PROTOCOL, "checkpoint CALVIN protocol mismatch")
    require(
        manifest.get("dataset") == "task_ABC_D" and manifest.get("dataset_split") == "training",
        "checkpoint dataset split mismatch",
    )
    artifacts = manifest.get("artifacts")
    require(isinstance(artifacts, dict), "checkpoint artifact table is missing")
    expected_artifact_paths = {
        "interface": "interface.safetensors",
        "lora_config": "lora/adapter_config.json",
        "lora_weights": "lora/adapter_model.safetensors",
    }
    mismatched_artifacts = [
        name
        for name, expected_path in expected_artifact_paths.items()
        if not isinstance(artifacts.get(name), dict) or artifacts[name].get("path") != expected_path
    ]
    require(
        not mismatched_artifacts,
        f"checkpoint artifact paths are not canonical: {mismatched_artifacts}",
    )

    resolved_artifact = artifacts.get("resolved_config")
    require(
        isinstance(resolved_artifact, dict) and isinstance(resolved_artifact.get("path"), str),
        "checkpoint has no resolved configuration",
    )
    resolved_path = checkpoint / resolved_artifact["path"]
    require(resolved_path.resolve().is_relative_to(checkpoint), "resolved configuration escapes checkpoint")
    config_sha256 = manifest.get("config_sha256")
    require(isinstance(config_sha256, str) and len(config_sha256) == 64, "checkpoint configuration hash is invalid")
    config, _ = load_verified_resolved_config(resolved_path, expected_sha256=config_sha256)
    contract = validate_manifest_policy_contract(manifest, config)
    contract_dict = contract.to_dict()
    contract_sha256 = canonical_config_sha256(contract_dict)
    require(manifest.get("policy_contract_sha256") == contract_sha256, "checkpoint policy contract hash mismatch")

    model = config.get("model")
    benchmark = config.get("benchmark")
    action_config = config.get("action")
    lora = config.get("lora")
    optimization = config.get("optimization")
    training = config.get("training")
    run = config.get("run")
    require(
        all(isinstance(value, dict) for value in (model, benchmark, action_config, lora, optimization, training, run)),
        "resolved config is incomplete",
    )
    assert isinstance(model, dict) and isinstance(benchmark, dict) and isinstance(action_config, dict)
    assert (
        isinstance(lora, dict)
        and isinstance(optimization, dict)
        and isinstance(training, dict)
        and isinstance(run, dict)
    )
    expected_config = {
        "protocol": (config.get("protocol"), PROTOCOL),
        "model.id": (model.get("id"), MODEL_ID),
        "model.revision": (model.get("revision"), MODEL_REVISION),
        "model.dtype": (model.get("dtype"), "bfloat16"),
        "model.tensor_parallel_size": (model.get("tensor_parallel_size"), 2),
        "model.attention_implementation": (model.get("attention_implementation"), "sdpa"),
        "model.experts_implementation": (model.get("experts_implementation"), EXPERTS_IMPLEMENTATION),
        "model.expert_batch_isolation": (model.get("expert_batch_isolation"), EXPERT_BATCH_ISOLATION),
        "benchmark.dataset": (benchmark.get("dataset"), "task_ABC_D"),
        "benchmark.train_split": (benchmark.get("train_split"), "training"),
        "benchmark.evaluation_split": (benchmark.get("evaluation_split"), "validation"),
        "benchmark.train_environments": (benchmark.get("train_environments"), ["A", "B", "C"]),
        "benchmark.evaluation_environment": (benchmark.get("evaluation_environment"), "D"),
        "benchmark.state_dimension": (benchmark.get("state_dimension"), STATE_DIM),
        "benchmark.camera_order": (benchmark.get("camera_order"), ["rgb_static", "rgb_gripper"]),
        "benchmark.execution_horizons": (benchmark.get("execution_horizons"), [1, 4]),
        "action.horizon": (action_config.get("horizon"), ACTION_HORIZON),
        "action.dimension": (action_config.get("dimension"), ACTION_DIM),
        "action.timestep_embedding_dimension": (action_config.get("timestep_embedding_dimension"), 256),
        "action.timestep_scale": (action_config.get("timestep_scale"), 1000.0),
        "action.timestep_max_period": (action_config.get("timestep_max_period"), 10000.0),
        "action.conditioning_mlp_activation": (action_config.get("conditioning_mlp_activation"), "silu"),
        "action.output_head_initialization_std": (action_config.get("output_head_initialization_std"), 1e-3),
        "lora.rank": (lora.get("rank"), 16),
        "lora.alpha": (lora.get("alpha"), 32),
        "lora.dropout": (lora.get("dropout"), 0.0),
        "lora.projections": (lora.get("projections"), ["q_proj", "k_proj", "v_proj", "o_proj"]),
        "lora.scope": (lora.get("scope"), "decoder_self_attention_only"),
        "optimization.global_batch_size": (optimization.get("global_batch_size"), 64),
        "optimization.microbatch_size": (optimization.get("microbatch_size"), PHYSICAL_BATCH_SIZE),
        "optimization.gradient_accumulation_steps": (optimization.get("gradient_accumulation_steps"), 8),
        "optimization.physical_batch_size": (optimization.get("physical_batch_size"), PHYSICAL_BATCH_SIZE),
        "optimization.total_updates": (optimization.get("total_updates"), 30_000),
        "training.seeds": (training.get("seeds"), [0, 1, 2]),
        "run.task": (run.get("task"), None),
    }
    mismatches = [name for name, (observed, expected) in expected_config.items() if observed != expected]
    require(not mismatches, f"resolved CALVIN configuration mismatches: {mismatches}")
    canonical_config_name = (
        "calvin_abc_to_d.toml" if contract.objective == "rectified_flow" else "calvin_abc_to_d_direct.toml"
    )
    canonical_recipe = load_resolved_toml(project_root / "configs" / canonical_config_name)
    recipe_sections = (
        "protocol",
        "model",
        "policy",
        "action",
        "lora",
        "optimization",
        "training",
        "sampling",
        "benchmark",
        "reproducibility",
    )
    recipe_mismatches = [name for name in recipe_sections if config.get(name) != canonical_recipe.get(name)]
    require(not recipe_mismatches, f"resolved config differs from the canonical comparison recipe: {recipe_mismatches}")
    validate_decoder_attention_lora_adapter_config(
        checkpoint / "lora/adapter_config.json",
        rank=int(lora["rank"]),
        alpha=int(lora["alpha"]),
        dropout=float(lora["dropout"]),
    )
    validate_decoder_attention_lora_weights(
        checkpoint / "lora/adapter_model.safetensors",
        rank=int(lora["rank"]),
    )
    artifact_trees = config.get("artifact_trees")
    require(isinstance(artifact_trees, dict), "resolved config has no authenticated model tree")
    model_tree_sha256 = artifact_trees.get("model_tree_sha256")
    require(
        isinstance(model_tree_sha256, str)
        and len(model_tree_sha256) == 64
        and manifest.get("model_tree_sha256") == model_tree_sha256,
        "checkpoint/config model tree identity mismatch",
    )
    require(
        model_snapshot_report.get("tree_metadata_sha256") == model_tree_sha256,
        "current model snapshot differs from the checkpoint before prefix authentication",
    )
    prefix_geometry, execution_geometry = _load_checkpoint_prefix_geometry(
        checkpoint,
        manifest,
        config,
        model_snapshot_report=model_snapshot_report,
    )
    training_instruction_inventory_sha256 = config.get("training_instruction_inventory_sha256")
    require(
        _valid_sha256(training_instruction_inventory_sha256)
        and manifest.get("training_instruction_inventory_sha256") == training_instruction_inventory_sha256,
        "checkpoint/config authenticated training-instruction inventory mismatch",
    )

    normalization_artifact = artifacts.get("normalization")
    require(
        isinstance(normalization_artifact, dict) and isinstance(normalization_artifact.get("path"), str),
        "checkpoint has no normalization artifact",
    )
    normalization_path = checkpoint / normalization_artifact["path"]
    require(normalization_path.resolve().is_relative_to(checkpoint), "normalization artifact escapes checkpoint")
    _normalizer, stats = load_calvin_state_normalizer(
        normalization_path,
        expected_archive_sha256=ARCHIVE_SHA256,
        training_root=training_root if authenticated_generation is not None else None,
        verify_archive=False,
        authenticated_generation=authenticated_generation,
    )
    permanent_storage_identity = _validate_normalization_contract(stats)
    normalization_dataset_identity = {
        "name": "task_ABC_D",
        "split": "training",
        **permanent_storage_identity,
    }
    normalization_sha256 = stats["content_sha256"]
    metadata_sha256 = permanent_storage_identity["metadata_sha256"]
    require(manifest.get("normalization_sha256") == normalization_sha256, "checkpoint normalization hash mismatch")
    _validate_checkpoint_permanent_storage_identity(manifest, permanent_storage_identity)

    calvin_identity = manifest.get("calvin_identity")
    config_identity = config.get("calvin_identity")
    require(
        isinstance(calvin_identity, dict) and calvin_identity == config_identity,
        "checkpoint/config CALVIN identity mismatch",
    )
    calvin_identity = _validate_checkpoint_calvin_identity(calvin_identity, permanent_storage_identity)
    require(
        calvin_identity.get("normalization_sha256") == normalization_sha256, "CALVIN identity normalization mismatch"
    )
    require(calvin_identity.get("protocol") == PROTOCOL, "CALVIN identity protocol mismatch")
    require(calvin_identity.get("split") == stats["split"] == manifest.get("split"), "CALVIN split identity mismatch")
    split_sha256 = canonical_config_sha256(stats["split"])
    require(
        calvin_identity.get("split_sha256") == split_sha256 == manifest.get("split_sha256"),
        "CALVIN split identity hash mismatch",
    )
    require(
        manifest.get("train_episode_sha256") == stats["split"]["train_episode_sha256"]
        and manifest.get("validation_episode_sha256") == stats["split"]["validation_episode_sha256"],
        "checkpoint split membership hashes mismatch",
    )
    require(
        calvin_identity.get("camera_shapes") == CALVIN_CAMERA_SHAPES == manifest.get("camera_shapes"),
        "CALVIN camera identity mismatch",
    )
    require(
        manifest.get("camera_shapes_sha256") == canonical_config_sha256(CALVIN_CAMERA_SHAPES),
        "CALVIN camera identity hash mismatch",
    )
    require(
        calvin_identity.get("state_adapter") == CALVIN_STATE_ADAPTER == manifest.get("state_adapter"),
        "CALVIN state adapter mismatch",
    )
    require(
        calvin_identity.get("action_adapter") == CALVIN_ACTION_ADAPTER == manifest.get("action_adapter"),
        "CALVIN action adapter mismatch",
    )

    source_revisions = load_pinned_source_revisions(project_root)
    source_revision_sha256 = canonical_config_sha256(source_revisions)
    require(manifest.get("calvin_source_revisions") == source_revisions, "checkpoint CALVIN source revisions mismatch")
    require(
        calvin_identity.get("calvin_source_revisions") == source_revisions, "CALVIN identity source revisions mismatch"
    )
    require(
        manifest.get("calvin_source_revisions_sha256") == source_revision_sha256, "CALVIN source revision hash mismatch"
    )
    require(
        manifest.get("calvin_revision") == source_revisions["calvin"]
        and manifest.get("calvin_env_revision") == source_revisions["calvin_env"]
        and manifest.get("calvin_tacto_revision") == source_revisions["tacto"],
        "checkpoint individual CALVIN source revisions mismatch",
    )
    current_source_sha256 = _source_tree_sha256(project_root)
    require(
        manifest.get("source_tree_sha256") == current_source_sha256 == config.get("source_tree_sha256"),
        "training source tree differs from checkpoint",
    )

    recorded_seed = manifest.get("run_seed")
    require(type(recorded_seed) is int and recorded_seed in (0, 1, 2), "checkpoint run seed is not a declared seed")
    require(run.get("seed") == recorded_seed, "checkpoint and resolved configuration run seeds differ")
    training_execution_environment = _validate_checkpoint_training_environment(
        manifest,
        config,
        project_root=project_root,
        run_seed=recorded_seed,
    )
    total_updates = optimization["total_updates"]
    expected_examples = total_updates * int(optimization.get("global_batch_size", -1))
    trainer_state = manifest.get("trainer_state")
    last_metrics = manifest.get("last_metrics")
    require(manifest.get("complete") is True, "official serving requires a complete final checkpoint")
    require(
        manifest.get("configured_total_updates") == total_updates,
        "checkpoint and resolved total-update contracts differ",
    )
    require(
        isinstance(trainer_state, dict)
        and trainer_state.get("schema") == "duo-vla-trainer-state-v1"
        and trainer_state.get("next_update") == total_updates
        and trainer_state.get("examples_seen") == expected_examples,
        "checkpoint trainer state is not the configured final update",
    )
    require(
        isinstance(last_metrics, dict)
        and last_metrics.get("update") == total_updates
        and last_metrics.get("examples_seen") == expected_examples,
        "checkpoint final metrics do not match the configured final update",
    )
    require(
        checkpoint.name == f"update-{total_updates:06d}",
        "final checkpoint directory name does not match the configured update",
    )
    committed_record = _committed_checkpoint_record(checkpoint, config_sha256)
    if train_seed_override is not None:
        require(train_seed_override == recorded_seed, "--train-seed differs from checkpoint run_seed")
    checkpoint_manifest_sha256 = sha256_file(checkpoint / "manifest.json")
    require(
        committed_record.update == total_updates,
        "run journal update differs from the official final checkpoint",
    )
    require(
        committed_record.manifest_sha256 == checkpoint_manifest_sha256,
        "run journal manifest SHA-256 differs from the official final checkpoint",
    )
    report = {
        **permanent_storage_identity,
        "calvin_identity": normalization_dataset_identity,
        "checkpoint_calvin_identity": calvin_identity,
        "kind": manifest["kind"],
        "manifest_sha256": checkpoint_manifest_sha256,
        "normalization_content_sha256": normalization_sha256,
        "path": str(checkpoint),
        "policy_contract": contract_dict,
        "policy_contract_sha256": contract_sha256,
        "model_tree_sha256": model_tree_sha256,
        "execution_geometry": execution_geometry,
        "prefix_geometry_instruction_inventory_sha256": prefix_geometry["instruction_inventory"]["sha256"],
        "training_instruction_inventory_sha256": training_instruction_inventory_sha256,
        "source_tree_sha256": current_source_sha256,
        "split_sha256": split_sha256,
        "train_seed": recorded_seed,
        "training_execution_environment": training_execution_environment,
        "training_execution_environment_sha256": manifest["execution_environment_sha256"],
    }
    identities: dict[str, Any] = {
        "calvin_identity": normalization_dataset_identity,
        "checkpoint_manifest_sha256": checkpoint_manifest_sha256,
        "execution_geometry": execution_geometry,
        "normalization_content_sha256": normalization_sha256,
        "normalization_metadata_sha256": metadata_sha256,
    }
    return manifest, normalization_path, recorded_seed, report, config, contract_dict, identities


def train_runtime_preflight(
    project_root: Path,
    checkpoint_dir: Path,
    dataset_root: Path,
    *,
    dataset_manifest: Path | None,
    train_seed_override: int | None,
) -> tuple[dict[str, Any], Path, int, dict[str, Any], dict[str, Any], dict[str, Any]]:
    require(
        platform.python_version() == EXPECTED_TRAIN_PYTHON,
        f"policy server requires Python {EXPECTED_TRAIN_PYTHON}, found {platform.python_version()}",
    )
    package_versions = {name: importlib.metadata.version(name) for name in EXPECTED_TRAIN_PACKAGES}
    require(package_versions == EXPECTED_TRAIN_PACKAGES, f"train package pin mismatch: {package_versions}")
    lock_path = project_root / "uv.lock"
    require(lock_path.is_file(), f"missing train lockfile: {lock_path}")
    require(sha256_file(lock_path) == TRAIN_LOCK_SHA256, "train lockfile SHA-256 mismatch")
    dataset_report, authenticated_generation = _verify_dataset_with_generation(dataset_root, dataset_manifest)
    require(authenticated_generation is not None, "official serving requires an authenticated v4 dataset generation")
    snapshot_report = model_snapshot_preflight()
    _, normalization_path, train_seed, checkpoint_report, config, contract, identities = resolve_checkpoint(
        checkpoint_dir,
        training_root=dataset_root.resolve() / "training",
        project_root=project_root,
        train_seed_override=train_seed_override,
        model_snapshot_report=snapshot_report,
        authenticated_generation=authenticated_generation,
    )
    require(
        dataset_report["calvin_identity"] == checkpoint_report["calvin_identity"] == identities["calvin_identity"],
        "dataset/checkpoint permanent CALVIN identity mismatch",
    )
    require(
        snapshot_report["tree_metadata_sha256"] == checkpoint_report["model_tree_sha256"],
        "installed model tree differs from the checkpoint contract",
    )
    report = {
        "checkpoint": checkpoint_report,
        "dataset": dataset_report,
        "lock_sha256": TRAIN_LOCK_SHA256,
        "model": snapshot_report,
        "execution_geometry": checkpoint_report["execution_geometry"],
        "packages": package_versions,
        "python": sys.version.split()[0],
        "source_revisions": load_pinned_source_revisions(project_root),
        "status": "ok",
    }
    return report, normalization_path, train_seed, config, contract, identities


def configure_and_identify_serving_runtime(torch: Any, preflight_report: dict[str, Any]) -> tuple[dict[str, Any], str]:
    """Enable deterministic inference and bind the exact software/hardware runtime used for scoring."""

    project_root = Path(__file__).resolve().parents[2]
    process_identity = _serving_process_identity(project_root, preflight_report["packages"])
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    require(torch.are_deterministic_algorithms_enabled(), "PyTorch deterministic algorithms did not enable")
    require(not torch.is_deterministic_algorithms_warn_only_enabled(), "deterministic algorithms are warn-only")
    require(not torch.backends.cudnn.benchmark, "cuDNN benchmarking must be disabled")
    require(torch.backends.cudnn.deterministic, "cuDNN deterministic mode must be enabled")
    require(not torch.backends.cudnn.allow_tf32, "cuDNN TF32 must be disabled")
    require(not torch.backends.cuda.matmul.allow_tf32, "CUDA matmul TF32 must be disabled")
    require(torch.cuda.device_count() == 2, "serving runtime must expose exactly two GPUs")
    try:
        completed = subprocess.run(
            [
                "/usr/bin/nvidia-smi",
                "--query-gpu=index,name,uuid,driver_version,compute_cap",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError("cannot attest the NVIDIA driver/GPU runtime") from exc
    gpu_rows = []
    for line in completed.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        require(len(fields) == 5 and all(fields), f"invalid nvidia-smi identity row: {line!r}")
        gpu_rows.append(
            {
                "compute_capability": fields[4],
                "driver_version": fields[3],
                "index": int(fields[0]),
                "name": fields[1],
                "uuid": fields[2],
            }
        )
    require([row["index"] for row in gpu_rows] == [0, 1], "nvidia-smi did not report the two visible GPUs in order")
    logical_gpu_rows = []
    for logical_index, physical in enumerate(gpu_rows):
        properties = torch.cuda.get_device_properties(logical_index)
        logical_uuid = str(getattr(properties, "uuid", ""))
        require(bool(logical_uuid), f"CUDA logical device {logical_index} has no UUID")
        physical_uuid = physical["uuid"].removeprefix("GPU-")
        require(
            logical_uuid.lower() == physical_uuid.lower(),
            f"CUDA logical device {logical_index} maps to an unpinned physical GPU",
        )
        capability = list(torch.cuda.get_device_capability(logical_index))
        require(
            properties.name == physical["name"],
            f"CUDA logical device {logical_index} name differs from nvidia-smi",
        )
        require(
            ".".join(map(str, capability)) == physical["compute_capability"],
            f"CUDA logical device {logical_index} capability differs from nvidia-smi",
        )
        logical_gpu_rows.append(
            {
                "compute_capability": capability,
                "logical_index": logical_index,
                "name": properties.name,
                "physical_index": physical["index"],
                "uuid": logical_uuid,
            }
        )
    checkpoint_report = preflight_report["checkpoint"]
    model_report = preflight_report["model"]
    training_environment = checkpoint_report["training_execution_environment"]
    training_origins = training_environment["authenticated_runtime"]["module_origins"]
    serving_origins = process_identity["module_origins"]
    require(
        all(serving_origins[module] == origin for module, origin in training_origins.items()),
        "serving package/code import origins differ from training",
    )
    require(torch.version.cuda == training_environment["cuda_runtime"], "serving CUDA runtime differs from training")
    require(torch.backends.cudnn.version() == training_environment["cudnn"], "serving cuDNN differs from training")
    require(
        [row["name"] for row in logical_gpu_rows] == training_environment["gpu_names"],
        "serving GPU names differ from training",
    )
    require(
        [row["compute_capability"] for row in logical_gpu_rows] == training_environment["gpu_capability"],
        "serving GPU capabilities differ from training",
    )
    payload = {
        "authenticated_software": {
            "bridge_sha256": sha256_file(Path(__file__).with_name("calvin_bridge.py")),
            "lock_sha256": preflight_report["lock_sha256"],
            "model_content_inventory_sha256": model_report["content_inventory_sha256"],
            "model_tree_sha256": model_report["tree_metadata_sha256"],
            "nvidia_smi_sha256": sha256_file(Path("/usr/bin/nvidia-smi")),
            "packages": preflight_report["packages"],
            "policy_launcher_sha256": sha256_file(Path(__file__).with_name("run_policy_server.sh")),
            "serve_policy_sha256": sha256_file(Path(__file__)),
            "source_revisions": preflight_report["source_revisions"],
            "source_tree_sha256": checkpoint_report["source_tree_sha256"],
            "train_launcher_sha256": sha256_file(project_root / "scripts/run_calvin_train.sh"),
            "training_execution_environment_sha256": checkpoint_report["training_execution_environment_sha256"],
        },
        "determinism": {
            "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cudnn_tf32": torch.backends.cudnn.allow_tf32,
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "deterministic_warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
            "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
            "python_hash_seed": os.environ["PYTHONHASHSEED"],
        },
        "execution_geometry": _validate_execution_geometry(preflight_report.get("execution_geometry")),
        "hardware": {
            "logical_cuda_devices": logical_gpu_rows,
            "nvidia_smi_devices": gpu_rows,
        },
        "platform": {
            "cuda_runtime": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "machine": platform.machine(),
            "nccl": list(torch.cuda.nccl.version()),
            "platform": platform.platform(),
            "python": sys.version.split()[0],
            "torch": torch.__version__,
        },
        "process": process_identity,
        "schema": SERVING_RUNTIME_SCHEMA,
        "sdpa_backends": {
            "cudnn": torch.backends.cuda.cudnn_sdp_enabled(),
            "flash": torch.backends.cuda.flash_sdp_enabled(),
            "math": torch.backends.cuda.math_sdp_enabled(),
            "memory_efficient": torch.backends.cuda.mem_efficient_sdp_enabled(),
        },
    }
    return payload, _canonical_sha256(payload)


def select_serving_policy_contract(
    checkpoint_policy_contract: dict[str, Any],
    *,
    flow_steps_override: int | None,
) -> dict[str, Any]:
    """Select rollout NFE while retaining all authenticated objective semantics."""

    selected = dict(checkpoint_policy_contract)
    if selected["objective"] == "rectified_flow":
        if flow_steps_override is not None:
            require(type(flow_steps_override) is int, "--flow-steps must be an integer")
            selected["nfe"] = flow_steps_override
        require(selected["nfe"] in {1, 5, 10}, "rectified-flow serving NFE must be one of {1, 5, 10}")
    elif selected["objective"] == "direct_regression":
        require(flow_steps_override is None, "--flow-steps is forbidden for direct regression")
        require(selected["nfe"] == 1, "direct-regression NFE must equal one")
    else:
        raise RuntimeError(f"unsupported checkpoint objective: {selected['objective']!r}")
    return selected


def _health_identity(
    *,
    mode: str,
    train_seed: int,
    policy_contract: dict[str, Any],
    artifact_identities: dict[str, Any] | None,
    serving_runtime_sha256: str | None = None,
) -> dict[str, Any]:
    if mode == "fake":
        result = {
            "calvin_identity": None,
            "checkpoint_manifest_sha256": None,
            "execution_geometry": None,
            "mode": "fake",
            "model_revision": None,
            "nfe": 0,
            "normalization_content_sha256": None,
            "normalization_metadata_sha256": None,
            "objective": "test_fake",
            "policy_contract_sha256": None,
            "sampler": "seeded_test_normal",
            "serving_runtime_sha256": None,
            "train_seed": train_seed,
        }
        require(set(result) == _HEALTH_WIRE_IDENTITY_FIELDS, "fake health identity field inventory drifted")
        return result
    assert artifact_identities is not None
    require(
        set(artifact_identities)
        == {
            "calvin_identity",
            "checkpoint_manifest_sha256",
            "execution_geometry",
            "normalization_content_sha256",
            "normalization_metadata_sha256",
        },
        "real health artifact identity field inventory differs",
    )
    require(
        isinstance(serving_runtime_sha256, str) and len(serving_runtime_sha256) == 64,
        "real health requires a serving runtime SHA-256",
    )
    calvin_identity = artifact_identities.get("calvin_identity")
    permanent_storage_identity = _normalization_permanent_storage_identity(calvin_identity)
    require(
        calvin_identity == {"name": "task_ABC_D", "split": "training", **permanent_storage_identity},
        "real health CALVIN identity differs from normalization-v4",
    )
    require(
        artifact_identities.get("normalization_metadata_sha256") == permanent_storage_identity["metadata_sha256"],
        "real health normalization metadata differs from CALVIN identity",
    )
    execution_geometry = _validate_execution_geometry(artifact_identities.get("execution_geometry"))
    result = {
        **artifact_identities,
        "execution_geometry": execution_geometry,
        "mode": "real",
        "model_revision": MODEL_REVISION,
        "nfe": policy_contract["nfe"],
        "objective": policy_contract["objective"],
        "policy_contract_sha256": _canonical_sha256(policy_contract),
        "sampler": policy_contract["sampler"],
        "serving_runtime_sha256": serving_runtime_sha256,
        "train_seed": train_seed,
    }
    require(set(result) == _HEALTH_WIRE_IDENTITY_FIELDS, "real health identity field inventory drifted")
    return result


def finalize_calvin_actions(raw_actions: Any) -> Any:
    """Apply only CALVIN's official final-space clip and gripper threshold."""

    require(
        len(raw_actions.shape) == 3
        and type(raw_actions.shape[0]) is int
        and raw_actions.shape[0] > 0
        and tuple(raw_actions.shape[1:]) == (ACTION_HORIZON, ACTION_DIM),
        "raw CALVIN action chunk shape mismatch",
    )
    require(raw_actions.is_floating_point(), "raw CALVIN action chunk must be floating point")
    clipped = raw_actions.clamp(-1.0, 1.0)
    # The pinned official CALVIN wrappers use a strict positive threshold.
    clipped[..., 6] = (clipped[..., 6] > 0).to(dtype=clipped.dtype).mul(2.0).sub(1.0)
    return clipped


def _replicate_singleton_tensor(value: Any, *, physical_batch_size: int = PHYSICAL_BATCH_SIZE) -> Any:
    require(len(value.shape) >= 1 and value.shape[0] == 1, "only singleton tensors may be physically replicated")
    require(
        type(physical_batch_size) is int and physical_batch_size == PHYSICAL_BATCH_SIZE,
        "serving physical batch must equal eight",
    )
    repeats = (physical_batch_size,) + (1,) * (len(value.shape) - 1)
    return value.repeat(*repeats)


def _require_bitwise_exact_replicas(value: Any) -> None:
    require(
        len(value.shape) == 3
        and value.shape[0] == PHYSICAL_BATCH_SIZE
        and tuple(value.shape[1:]) == (ACTION_HORIZON, ACTION_DIM),
        "policy output does not have the fixed physical batch shape",
    )
    reference = value[0:1].expand_as(value)
    require(bool(value.equal(reference)), "fixed-batch policy replicas are not bitwise identical")


def place_calvin_state_normalizer(normalizer: Any, *, device: Any, dtype: Any) -> Any:
    """Move only the seven percentile tensors while preserving binary channel 7."""

    from duo_vla.normalization import ActionNormalizer

    require(normalizer.action_dim == 8 and normalizer.resolved_gripper_index == 7, "invalid CALVIN state normalizer")
    return ActionNormalizer(
        continuous=normalizer.continuous.to(device=device, dtype=dtype),
        action_dim=8,
        gripper_index=7,
    )


class FakePolicy:
    """Deterministic IPC-only policy which cannot claim a model identity."""

    def predict(self, request: dict[str, Any]) -> tuple[np.ndarray, float]:
        started = time.perf_counter()
        generator = np.random.default_rng(request["inference_seed"])
        actions = generator.normal(0.0, 0.05, size=(ACTION_HORIZON, ACTION_DIM)).astype(np.float32)
        np.clip(actions[:, :6], -1.0, 1.0, out=actions[:, :6])
        actions[:, 6] = np.where(actions[:, 6] > 0, 1.0, -1.0)
        return np.ascontiguousarray(actions), time.perf_counter() - started


def _load_calvin_interface_checkpoint(
    checkpoint_dir: Path,
    action_projector: Any,
    velocity_head: Any,
) -> None:
    """Route official and development serving through the strict shared loader."""

    from duo_vla.checkpointing import load_interface_state_dict

    load_interface_state_dict(
        checkpoint_dir / "interface.safetensors",
        {"action_projector": action_projector, "velocity_head": velocity_head},
    )


class RealPolicy:
    """TP-sharded DiffusionGemma with the trained CALVIN action interface."""

    def __init__(
        self,
        checkpoint_dir: Path,
        normalization_path: Path,
        device: Any,
        *,
        resolved_config: dict[str, Any],
        policy_contract: dict[str, Any],
    ) -> None:
        import torch
        from PIL import Image
        from transformers import AutoProcessor

        from duo_vla.action_interface import ActionInputProjector, VelocityHead
        from duo_vla.backbones.diffusion_gemma import DiffusionGemmaActionDecoder, encode_diffusion_gemma_prefix
        from duo_vla.backbones.loading import DEFAULT_DIFFUSION_GEMMA_SPEC, load_diffusion_gemma_bf16_tp
        from duo_vla.backbones.sample_isolated_experts import (
            install_sample_isolated_grouped_mm_experts,
            verify_sample_isolated_grouped_mm_experts,
        )
        from duo_vla.checkpointing import load_lora_checkpoint
        from duo_vla.config import ActionInterfaceConfig
        from duo_vla.data.calvin_stats import load_calvin_state_normalizer
        from duo_vla.flow import euler_sample
        from duo_vla.modeling import DuoVLADenoiser
        from duo_vla.prefix_geometry import (
            CameraGeometry,
            apply_fixed_prefix_chat_template,
            load_prefix_geometry_contract,
        )

        self.torch = torch
        self.Image = Image
        self.encode_prefix = encode_diffusion_gemma_prefix
        self.euler_sample = euler_sample
        self.device = device
        self.policy_contract = policy_contract
        self.execution_geometry = _execution_geometry_from_config(resolved_config)
        cameras = tuple(
            CameraGeometry(name=name, height=CALVIN_CAMERA_SHAPES[name][0], width=CALVIN_CAMERA_SHAPES[name][1])
            for name in ("rgb_static", "rgb_gripper")
        )
        self.prefix_geometry = load_prefix_geometry_contract(
            checkpoint_dir / "artifacts/prefix_geometry.json",
            expected_content_sha256=self.execution_geometry["prefix_geometry_content_sha256"],
            expected_ordered_cameras=cameras,
            expected_fixed_physical_prefix_width=self.execution_geometry["fixed_physical_prefix_width"],
        )
        self.allowed_instructions = frozenset(
            record["instruction"] for record in self.prefix_geometry["instruction_inventory"]["records"]
        )
        self.apply_fixed_prefix_chat_template = apply_fixed_prefix_chat_template
        self.processor = AutoProcessor.from_pretrained(
            DEFAULT_DIFFUSION_GEMMA_SPEC.model_id,
            revision=DEFAULT_DIFFUSION_GEMMA_SPEC.revision,
            local_files_only=True,
        )
        require(
            getattr(getattr(self.processor, "tokenizer", None), "padding_side", None)
            == self.prefix_geometry["tokenization"]["padding_side"],
            "loaded processor padding side differs from the prefix contract",
        )
        self.model = load_diffusion_gemma_bf16_tp(local_files_only=True, tp_size=2)
        install_sample_isolated_grouped_mm_experts(
            self.model,
            physical_batch_size=PHYSICAL_BATCH_SIZE,
        )
        verify_sample_isolated_grouped_mm_experts(self.model, physical_batch_size=PHYSICAL_BATCH_SIZE)
        backend = DiffusionGemmaActionDecoder.from_block_diffusion_model(self.model)
        self.adapted, _ = load_lora_checkpoint(
            checkpoint_dir,
            self.model,
            is_trainable=False,
            validate_decoder_contract=True,
            expected_rank=int(resolved_config["lora"]["rank"]),
        )
        verify_sample_isolated_grouped_mm_experts(self.model, physical_batch_size=PHYSICAL_BATCH_SIZE)
        action = resolved_config["action"]
        benchmark = resolved_config["benchmark"]
        interface_config = ActionInterfaceConfig(
            hidden_size=2816,
            state_dim=int(benchmark["state_dimension"]),
            action_horizon=int(action["horizon"]),
            action_dim=int(action["dimension"]),
            timestep_embedding_dim=int(action["timestep_embedding_dimension"]),
            timestep_scale=float(action["timestep_scale"]),
            timestep_max_period=float(action["timestep_max_period"]),
            output_init_std=float(action["output_head_initialization_std"]),
        )
        projector = ActionInputProjector(interface_config).to(device)
        head = VelocityHead(interface_config.hidden_size, interface_config.action_dim).to(device)
        _load_calvin_interface_checkpoint(checkpoint_dir, projector, head)
        self.denoiser = DuoVLADenoiser(projector, backend, head).eval()
        self.adapted.eval()
        self.model.model.encoder.eval()
        state_normalizer, _ = load_calvin_state_normalizer(normalization_path)
        self.state_normalizer = place_calvin_state_normalizer(
            state_normalizer,
            device=device,
            dtype=torch.float32,
        )

    def _processor_inputs(self, request: dict[str, Any]) -> dict[str, Any]:
        observation = request["observation"]
        instruction = request["instruction"]
        require(instruction in self.allowed_instructions, "request instruction is absent from prefix geometry")
        conversations = [
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": self.Image.fromarray(observation["rgb_static"])},
                        {"type": "image", "image": self.Image.fromarray(observation["rgb_gripper"])},
                        {"type": "text", "text": instruction},
                    ],
                }
            ]
            for _ in range(PHYSICAL_BATCH_SIZE)
        ]
        values = self.apply_fixed_prefix_chat_template(
            self.processor,
            conversations,
            fixed_physical_prefix_width=self.execution_geometry["fixed_physical_prefix_width"],
            padding_side=self.prefix_geometry["tokenization"]["padding_side"],
            expected_batch_size=PHYSICAL_BATCH_SIZE,
            images_per_prefix=len(self.prefix_geometry["ordered_cameras"]),
        )
        expected_length = next(
            record["valid_prefix_length"]
            for record in self.prefix_geometry["instruction_inventory"]["records"]
            if record["instruction"] == instruction
        )
        observed_lengths = tuple(int(value) for value in values["attention_mask"].sum(dim=1).tolist())
        require(
            observed_lengths == (expected_length,) * PHYSICAL_BATCH_SIZE,
            "runtime valid-prefix length differs from the authenticated instruction record",
        )
        return dict(values.to(self.device))

    def predict(self, request: dict[str, Any]) -> tuple[np.ndarray, float]:
        torch = self.torch
        torch.cuda.synchronize(self.device)
        started = time.perf_counter()
        singleton_state = (
            torch.from_numpy(request["observation"]["state"])
            .to(
                device=self.device,
                dtype=torch.float32,
            )
            .unsqueeze(0)
        )
        state = _replicate_singleton_tensor(singleton_state)
        normalized_state = self.state_normalizer.normalize(state)
        valid = torch.ones((PHYSICAL_BATCH_SIZE, ACTION_HORIZON), device=self.device, dtype=torch.bool)
        objective = self.policy_contract["objective"]
        with torch.no_grad():
            # Match training exactly: the frozen prefix path runs outside
            # autocast, once per request, and its cache is reused by Euler.
            prefix_inputs = self._processor_inputs(request)
            prefix = self.encode_prefix(self.model, prefix_inputs)
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            if objective == "rectified_flow":
                generator = torch.Generator(device=self.device).manual_seed(request["inference_seed"])
                singleton_noise = torch.randn(
                    (1, ACTION_HORIZON, ACTION_DIM),
                    device=self.device,
                    dtype=normalized_state.dtype,
                    generator=generator,
                )
                noise = _replicate_singleton_tensor(singleton_noise)

                def velocity(actions: Any, timesteps: Any) -> Any:
                    return self.denoiser(
                        actions,
                        timesteps,
                        normalized_state,
                        prefix_cache=prefix.past_key_values,
                        prefix_attention_mask=prefix.attention_mask,
                        action_valid_mask=valid,
                    )

                raw_actions = self.euler_sample(
                    velocity,
                    initial_noise=noise,
                    num_steps=self.policy_contract["nfe"],
                )
            elif objective == "direct_regression":
                raw_actions = self.denoiser(
                    torch.zeros(
                        (PHYSICAL_BATCH_SIZE, ACTION_HORIZON, ACTION_DIM),
                        device=self.device,
                        dtype=normalized_state.dtype,
                    ),
                    torch.ones(PHYSICAL_BATCH_SIZE, device=self.device, dtype=normalized_state.dtype),
                    normalized_state,
                    prefix_cache=prefix.past_key_values,
                    prefix_attention_mask=prefix.attention_mask,
                    action_valid_mask=valid,
                )
            else:
                raise RuntimeError(f"unsupported loaded policy objective: {objective!r}")
        # CALVIN rel_actions are already in official scaled action space.  There
        # is deliberately no percentile inversion on any action channel.
        _require_bitwise_exact_replicas(raw_actions)
        clipped = finalize_calvin_actions(raw_actions)
        torch.cuda.synchronize(self.device)
        values = clipped[0].detach().float().cpu().numpy().astype(np.float32, copy=True)
        return np.ascontiguousarray(values), time.perf_counter() - started


def _dispatch_local(
    request: dict[str, Any],
    *,
    policy: FakePolicy | RealPolicy,
    health_identity: dict[str, Any],
) -> dict[str, Any]:
    operation = request["operation"]
    if operation == "health":
        require(set(health_identity) == _HEALTH_WIRE_IDENTITY_FIELDS, "policy health identity field inventory differs")
        return make_health_response(request, **health_identity)
    if operation == "shutdown":
        return make_success_response(request, stopped=True)
    if request["train_seed"] != health_identity["train_seed"]:
        raise ValueError("request train_seed differs from the loaded checkpoint")
    actions, seconds = policy.predict(request)
    return make_predict_response(request, actions, policy_seconds=seconds)


def _canonical_prediction(value: dict[str, Any]) -> str:
    comparable = {key: item for key, item in value.items() if key != "policy_seconds"}
    return json.dumps(comparable, allow_nan=False, separators=(",", ":"), sort_keys=True)


def run_fake_server(socket_path: Path, *, train_seed: int) -> None:
    policy = FakePolicy()
    health = _health_identity(mode="fake", train_seed=train_seed, policy_contract={}, artifact_identities=None)

    def dispatch(request: dict[str, Any]) -> dict[str, Any]:
        return _dispatch_local(request, policy=policy, health_identity=health)

    serve_unix_policy(
        socket_path,
        dispatch,
        ready=lambda: print(json.dumps({"mode": "fake", "socket": str(socket_path), "status": "ready"}), flush=True),
    )


def run_distributed_server(
    socket_path: Path,
    *,
    checkpoint_dir: Path,
    normalization_path: Path,
    train_seed: int,
    resolved_config: dict[str, Any],
    policy_contract: dict[str, Any],
    artifact_identities: dict[str, Any],
    preflight_report: dict[str, Any],
) -> None:
    import torch
    import torch.distributed as dist

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    serving_runtime, serving_runtime_sha256 = configure_and_identify_serving_runtime(torch, preflight_report)
    dist.init_process_group("nccl", device_id=device)
    require(dist.get_world_size() == 2, "real CALVIN policy serving requires TP world size 2")
    rank = dist.get_rank()
    try:
        runtime_hashes: list[str | None] = [None] * dist.get_world_size()
        dist.all_gather_object(runtime_hashes, serving_runtime_sha256)
        require(
            runtime_hashes == [serving_runtime_sha256] * dist.get_world_size(),
            f"TP ranks disagree on serving runtime identity: {runtime_hashes}",
        )
        policy = RealPolicy(
            checkpoint_dir,
            normalization_path,
            device,
            resolved_config=resolved_config,
            policy_contract=policy_contract,
        )
        health = _health_identity(
            mode="real",
            train_seed=train_seed,
            policy_contract=policy_contract,
            artifact_identities=artifact_identities,
            serving_runtime_sha256=serving_runtime_sha256,
        )
        dist.barrier()

        def execute(request: dict[str, Any]) -> dict[str, Any]:
            local_result: dict[str, Any] | None = None
            local_error: str | None = None
            try:
                local_result = _dispatch_local(request, policy=policy, health_identity=health)
            except Exception as exc:
                local_error = f"rank {rank}: {type(exc).__name__}: {exc}"
            errors: list[str | None] = [None] * dist.get_world_size()
            dist.all_gather_object(errors, local_error)
            failures = [error for error in errors if error is not None]
            if failures:
                return make_error_response(request, RuntimeError("; ".join(failures)))
            assert local_result is not None
            if request["operation"] == "predict":
                results: list[dict[str, Any] | None] = [None] * dist.get_world_size()
                dist.all_gather_object(results, local_result)
                concrete = [result for result in results if result is not None]
                require(len(concrete) == dist.get_world_size(), "a TP rank returned no prediction")
                reference = _canonical_prediction(concrete[0])
                if not all(_canonical_prediction(result) == reference for result in concrete[1:]):
                    return make_error_response(
                        request, RuntimeError("TP ranks produced different CALVIN action chunks")
                    )
                local_result["policy_seconds"] = max(float(result["policy_seconds"]) for result in concrete)
            return local_result

        if rank == 0:

            def dispatch(request: dict[str, Any]) -> dict[str, Any]:
                holder = [request]
                dist.broadcast_object_list(holder, src=0, device=device)
                return execute(request)

            serve_unix_policy(
                socket_path,
                dispatch,
                ready=lambda: print(
                    json.dumps(
                        {
                            **health,
                            "checkpoint": str(checkpoint_dir),
                            "socket": str(socket_path),
                            "serving_runtime": serving_runtime,
                            "status": "ready",
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                ),
            )
        else:
            while True:
                holder: list[dict[str, Any] | None] = [None]
                dist.broadcast_object_list(holder, src=0, device=device)
                request = holder[0]
                require(request is not None, "rank 0 broadcast an empty CALVIN policy request")
                execute(request)
                if request["operation"] == "shutdown":
                    break
    finally:
        dist.destroy_process_group()


def parse_args() -> argparse.Namespace:
    cache_root = Path(os.environ.get("DUO_VLA_CACHE_ROOT", "/root/.cache/duo-vla"))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", nargs="?", type=Path)
    parser.add_argument("--dataset-root", type=Path, default=cache_root / "data/calvin/task_ABC_D")
    parser.add_argument("--dataset-manifest", type=Path)
    parser.add_argument("--socket", type=Path, default=cache_root / "run/calvin-policy.sock")
    parser.add_argument("--train-seed", type=int)
    parser.add_argument(
        "--flow-steps",
        type=int,
        choices=(1, 5, 10),
        help="Override Euler NFE for rectified flow; forbidden for direct regression.",
    )
    parser.add_argument("--fake-policy", action="store_true", help="serve deterministic IPC-only action chunks")
    parser.add_argument("--preflight-only", action="store_true", help="authenticate artifacts without loading CUDA")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.fake_policy:
        require(args.checkpoint is None, "fake policy does not accept a checkpoint")
        require(type(args.train_seed) is int and 0 <= args.train_seed < 2**63, "fake policy requires --train-seed")
        require(args.flow_steps is None, "fake policy does not accept --flow-steps")
        require(not args.preflight_only, "--preflight-only is only for a real policy")
        run_fake_server(args.socket, train_seed=args.train_seed)
        return

    require(args.checkpoint is not None, "real policy requires a checkpoint directory")
    project_root = Path(__file__).resolve().parents[2]
    _validated_serving_process_environment(project_root, require_tp_launch=not args.preflight_only)
    report, normalization_path, train_seed, config, checkpoint_contract, identities = train_runtime_preflight(
        project_root,
        args.checkpoint,
        args.dataset_root,
        dataset_manifest=args.dataset_manifest,
        train_seed_override=args.train_seed,
    )
    serving_contract = select_serving_policy_contract(checkpoint_contract, flow_steps_override=args.flow_steps)
    report["serving_policy_contract"] = serving_contract
    report["serving_policy_contract_sha256"] = _canonical_sha256(serving_contract)
    if args.preflight_only:
        print(json.dumps(report, indent=2, sort_keys=True))
        return
    run_distributed_server(
        args.socket,
        checkpoint_dir=args.checkpoint.resolve(),
        normalization_path=normalization_path,
        train_seed=train_seed,
        resolved_config=config,
        policy_contract=serving_contract,
        artifact_identities=identities,
        preflight_report=report,
    )


if __name__ == "__main__":
    main()
