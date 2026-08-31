#!/usr/bin/env python3
"""Persistent Unix-socket Duo-VLA policy server for the isolated LIBERO evaluator."""

# ruff: noqa: E402 -- authenticate local import roots before importing project code.

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import site
import stat
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, ClassVar

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent


def _activate_project_source_root() -> Path:
    source_root = _SCRIPT_DIR.parent / "src"
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


def _load_local_module(name: str, path: Path) -> Any:
    expected = path.resolve(strict=True)
    existing = sys.modules.get(name)
    if existing is not None:
        module_file = getattr(existing, "__file__", None)
        origin = getattr(getattr(existing, "__spec__", None), "origin", None)
        if not isinstance(module_file, str) or not isinstance(origin, str):
            raise RuntimeError(f"existing local module {name} has no file origin")
        if Path(module_file).resolve() != expected or Path(origin).resolve() != expected:
            raise RuntimeError(f"existing local module {name} has an unexpected origin")
        return existing
    specification = importlib.util.spec_from_file_location(name, expected)
    if specification is None or specification.loader is None:
        raise RuntimeError(f"cannot construct import specification for {expected}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    try:
        specification.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


_libero_bridge = _load_local_module("libero_bridge", _SCRIPT_DIR / "libero_bridge.py")

from libero_bridge import (
    ACTION_DIM,
    ACTION_HORIZON,
    IMAGE_SHAPE,
    PROTOCOL,
    STATE_DIM,
    make_error_response,
    make_success_response,
    serve_unix_policy,
    validate_execution_geometry,
    validate_wire_policy_contract,
)

_bridge_file = getattr(_libero_bridge, "__file__", None)
_bridge_origin = getattr(getattr(_libero_bridge, "__spec__", None), "origin", None)
if (
    not isinstance(_bridge_file, str)
    or not isinstance(_bridge_origin, str)
    or Path(_bridge_file).resolve() != (_SCRIPT_DIR / "libero_bridge.py").resolve()
    or Path(_bridge_origin).resolve() != (_SCRIPT_DIR / "libero_bridge.py").resolve()
):
    raise RuntimeError("imported LIBERO bridge has an unexpected module origin")

from duo_vla.runtime_determinism import configure_strict_cuda_determinism, deterministic_torch_runtime
from duo_vla.runtime_integrity import (
    content_address_train_venv,
    require_matching_train_venv,
    static_environment_identity,
    validate_torchrun_rank_environment,
)

_validate_project_module_origins({"duo_vla", "duo_vla.runtime_determinism", "duo_vla.runtime_integrity"})

MODEL_ID = "google/diffusiongemma-26B-A4B-it"
MODEL_REVISION = "f7f5b7f5fa82ffc52addd066915886d497f5517b"
DATASET_ID = "HuggingFaceVLA/libero"
DATASET_REVISION = "86958911c0f959db2bbbdb107eb3e17c5f9c798e"
DATASET_TREE_SHA256 = "d9c14b4aff28bcc56f341b171c6a5a3b10510d4bd0378662891c5156d245add8"
DATASET_CONTENT_INVENTORY_SHA256 = "63fd7a951ebb397a33c43cad4a7c48c7c6911bd8d1481ff99b07da5f7890782c"
DATASET_FILES_VERIFIED = 382
DATASET_TOTAL_BYTES = 34_926_155_087
NORMALIZATION_SHA256 = "a972b5d95a8aaa8ae7582bafcbc071261979cb46c2a3515b4da7a7cf0156ac73"
TRAIN_LOCK_SHA256 = "0b1fb188747ee99224078b3c40975ca7e6f8e082e22d2860f9b50ee679a67c46"
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
PHYSICAL_BATCH_SIZE = 8
EXPERTS_IMPLEMENTATION = "grouped_mm"
EXPERT_BATCH_ISOLATION = "sample_isolated_grouped_mm_v1"
LIBERO_PREFIX_CAMERA_NAMES = ("agentview", "eye_in_hand")
REQUIRED_SERVING_ENVIRONMENT = {
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
    "PYTHONHASHSEED": "0",
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


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, allow_nan=False, separators=(",", ":"), sort_keys=True).encode()
    return hashlib.sha256(payload).hexdigest()


def _training_source_tree_sha256(root: Path) -> str:
    """Recompute the trainer's exact live source-tree identity."""

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
            root / "scripts/train_libero.py",
            root / "pyproject.toml",
            root / "uv.lock",
        )
        if path.is_file()
    )
    for path in sorted(paths):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _validate_serving_process_environment(project_root: Path) -> dict[str, Any]:
    """Require the canonical launcher environment before any CUDA initialization."""

    _activate_project_source_root()
    _validate_project_module_origins({"duo_vla", "duo_vla.runtime_determinism", "duo_vla.runtime_integrity"})
    project_root = project_root.resolve()
    cache_root = Path(os.environ.get("DUO_VLA_CACHE_ROOT", "/root/.cache/duo-vla")).resolve()
    train_venv = (cache_root / "venvs/train").resolve()
    expected = {
        **REQUIRED_SERVING_ENVIRONMENT,
        "DUO_VLA_CACHE_ROOT": str(cache_root),
        "DUO_VLA_PROJECT_ROOT": str(project_root),
        "DUO_VLA_TRAIN_VENV": str(train_venv),
        "HF_HOME": str(Path(os.environ.get("HF_HOME", "/root/.cache/huggingface")).resolve()),
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
    require(
        not present_forbidden and not present_algorithm_overrides,
        "LIBERO serving environment contains injection/algorithm overrides: "
        f"forbidden={present_forbidden}, algorithm_overrides={present_algorithm_overrides}",
    )
    observed = {name: os.environ.get(name) for name in expected}
    require(observed == expected, f"LIBERO serving environment differs from launcher: {observed}")
    require(Path(sys.prefix).resolve() == train_venv, f"LIBERO serving requires the pinned train venv: {train_venv}")
    require(sys.flags.safe_path == 1, "LIBERO serving requires Python safe-path mode")
    require(sys.flags.dont_write_bytecode == 1 and sys.dont_write_bytecode, "LIBERO serving requires -B")
    require(sys.flags.no_user_site == 1 and not site.ENABLE_USER_SITE, "LIBERO serving requires no user site")
    require(sys.pycache_prefix == "/dev/null", "LIBERO serving requires an impossible pycache lookup prefix")
    version = f"python{sys.version_info.major}.{sys.version_info.minor}"
    compact_version = f"python{sys.version_info.major}{sys.version_info.minor}"
    expected_sys_path = [
        str((project_root / "src").resolve()),
        str(Path(sys.base_prefix) / "lib" / f"{compact_version}.zip"),
        str(Path(sys.base_prefix) / "lib" / version),
        str(Path(sys.base_exec_prefix) / "lib" / version / "lib-dynload"),
        str(train_venv / "lib" / version / "site-packages"),
    ]
    require(sys.path == expected_sys_path, f"LIBERO serving import search path differs: {sys.path}")
    rank_environment = validate_torchrun_rank_environment(
        os.environ,
        required=any(name in os.environ for name in ("RANK", "LOCAL_RANK", "WORLD_SIZE")),
    )
    return {
        "environment": dict(sorted(observed.items())),
        "static_environment_sha256": static_environment_identity(observed)["sha256"],
        "torchrun": rank_environment,
    }


def _driver_and_binary_identity(torch: Any) -> dict[str, Any]:
    """Bind the latency runtime to its driver, device mapping, and loaded binaries."""

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
        raise RuntimeError("cannot attest the driver/device runtime") from exc
    rows: list[dict[str, Any]] = []
    for line in completed.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        require(len(fields) == 5 and all(fields), f"invalid driver/device identity row: {line!r}")
        rows.append(
            {
                "compute_capability": fields[4],
                "driver_version": fields[3],
                "index": int(fields[0]),
                "name": fields[1],
                "uuid": fields[2],
            }
        )
    require([row["index"] for row in rows] == [0, 1], "driver query did not report two devices in order")
    logical: list[dict[str, Any]] = []
    for index, physical in enumerate(rows):
        properties = torch.cuda.get_device_properties(index)
        uuid = str(getattr(properties, "uuid", ""))
        name = str(getattr(properties, "name", torch.cuda.get_device_name(index)))
        capability = list(torch.cuda.get_device_capability(index))
        require(uuid and uuid.lower() == physical["uuid"].removeprefix("GPU-").lower(), "device UUID mapping differs")
        require(name == physical["name"], "device name differs between CUDA and driver query")
        require(".".join(map(str, capability)) == physical["compute_capability"], "device capability differs")
        logical.append(
            {
                "compute_capability": capability,
                "logical_index": index,
                "name": name,
                "physical_index": physical["index"],
                "uuid": uuid,
            }
        )
    torch_binary = Path(torch._C.__file__).resolve()
    python_binary = Path(sys.executable).resolve()
    return {
        "binaries": {
            "nvidia_smi_sha256": sha256_file(Path("/usr/bin/nvidia-smi")),
            "python_executable": str(python_binary),
            "python_executable_sha256": sha256_file(python_binary),
            "torch_extension": str(torch_binary),
            "torch_extension_sha256": sha256_file(torch_binary),
        },
        "logical_cuda_devices": logical,
        "nvidia_smi_devices": rows,
    }


def configure_and_identify_serving_runtime(
    torch: Any,
    *,
    project_root: Path,
    checkpoint_report: dict[str, Any],
) -> tuple[dict[str, Any], str]:
    """Enable deterministic inference and return its content-addressed runtime state."""

    environment = _validate_serving_process_environment(project_root)
    configure_strict_cuda_determinism(torch)
    source_tree_sha256 = checkpoint_report.get("source_tree_sha256")
    require(
        isinstance(source_tree_sha256, str) and len(source_tree_sha256) == 64,
        "checkpoint report has no training source-tree identity",
    )
    live_source_tree_sha256 = _training_source_tree_sha256(project_root)
    require(
        live_source_tree_sha256 == source_tree_sha256,
        "live training source tree differs from the checkpoint",
    )
    training_environment_sha256 = checkpoint_report.get("training_execution_environment_sha256")
    require(
        isinstance(training_environment_sha256, str) and len(training_environment_sha256) == 64,
        "checkpoint report has no training execution-environment identity",
    )
    execution_geometry = validate_execution_geometry(checkpoint_report.get("execution_geometry"))
    train_venv = checkpoint_report.get("train_venv")
    require(isinstance(train_venv, dict), "checkpoint report has no authenticated train-venv identity")
    hardware = _driver_and_binary_identity(torch)
    payload = {
        "authenticated_software": {
            "bridge_sha256": sha256_file(project_root / "scripts/libero_bridge.py"),
            "checkpoint_source_tree_sha256": source_tree_sha256,
            "live_source_tree_sha256": live_source_tree_sha256,
            "packages": EXPECTED_TRAIN_PACKAGES,
            "policy_launcher_sha256": sha256_file(project_root / "scripts/run_libero_policy_server.sh"),
            "serve_policy_sha256": sha256_file(Path(__file__)),
            "train_launcher_sha256": sha256_file(project_root / "scripts/run_libero_train.sh"),
            "train_lock_sha256": TRAIN_LOCK_SHA256,
            "train_venv_content_inventory_sha256": train_venv["content_inventory_sha256"],
            "train_venv_root_sha256": train_venv["root_sha256"],
            "train_venv_tree_metadata_sha256": train_venv["tree_metadata_sha256"],
        },
        "determinism": deterministic_torch_runtime(torch),
        "environment": environment,
        "execution_geometry": execution_geometry,
        "gpu_capability": [row["compute_capability"] for row in hardware["logical_cuda_devices"]],
        "gpu_names": [row["name"] for row in hardware["logical_cuda_devices"]],
        "gpu_uuids": [row["uuid"] for row in hardware["logical_cuda_devices"]],
        "hardware": hardware,
        "platform": {
            "cuda_runtime": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "machine": platform.machine(),
            "nccl": list(torch.cuda.nccl.version()),
            "platform": platform.platform(),
            "python": sys.version.split()[0],
            "torch": torch.__version__,
        },
        "schema": "duovla-libero-serving-runtime-v2",
        "sdpa_backends": {
            "cudnn": torch.backends.cuda.cudnn_sdp_enabled(),
            "flash": torch.backends.cuda.flash_sdp_enabled(),
            "math": torch.backends.cuda.math_sdp_enabled(),
            "memory_efficient": torch.backends.cuda.mem_efficient_sdp_enabled(),
        },
        "training_execution_environment_sha256": training_environment_sha256,
    }
    require(len(payload["gpu_names"]) == 2, "LIBERO serving runtime must expose exactly two GPUs")
    require(all(payload["gpu_uuids"]), "LIBERO serving runtime could not identify both GPU UUIDs")
    return payload, _canonical_sha256(payload)


def latency_runtime_identity(serving_runtime: dict[str, Any]) -> tuple[dict[str, Any], str]:
    """Remove checkpoint-training provenance while retaining latency-relevant runtime identity."""

    expected = {
        "authenticated_software",
        "determinism",
        "environment",
        "execution_geometry",
        "gpu_capability",
        "gpu_names",
        "gpu_uuids",
        "hardware",
        "platform",
        "schema",
        "sdpa_backends",
        "training_execution_environment_sha256",
    }
    require(set(serving_runtime) == expected, "serving runtime fields changed before latency identity derivation")
    require(serving_runtime["schema"] == "duovla-libero-serving-runtime-v2", "serving runtime schema changed")
    identity = {
        name: value
        for name, value in serving_runtime.items()
        if name not in {"schema", "training_execution_environment_sha256"}
    }
    identity["schema"] = "duovla-libero-latency-runtime-v2"
    return identity, _canonical_sha256(identity)


def model_snapshot_preflight() -> dict[str, Any]:
    from duo_vla.hf_snapshot import verify_huggingface_snapshot

    hf_home = Path(os.environ.get("HF_HOME", "/root/.cache/huggingface"))
    snapshot = hf_home / "hub/models--google--diffusiongemma-26B-A4B-it/snapshots" / MODEL_REVISION
    tree_path = snapshot.parents[1] / "trees" / f"{MODEL_REVISION}.json"
    require(snapshot.is_dir(), f"pinned DiffusionGemma snapshot is missing: {snapshot}")
    require(tree_path.is_file(), f"pinned DiffusionGemma tree metadata is missing: {tree_path}")
    tree = json.loads(tree_path.read_text(encoding="utf-8"))
    require(tree.get("format_version") == 1 and isinstance(tree.get("files"), dict), "invalid model tree metadata")
    for name in ("config.json", "processor_config.json", "tokenizer.json", "model.safetensors.index.json"):
        require((snapshot / name).is_file(), f"pinned model snapshot is incomplete: missing {name}")
    index = json.loads((snapshot / "model.safetensors.index.json").read_text(encoding="utf-8"))
    shards = sorted(set(index.get("weight_map", {}).values()))
    require(len(shards) == 11, f"expected 11 DiffusionGemma weight shards, found {len(shards)}")
    total_bytes = 0
    for name in shards:
        path = snapshot / name
        entry = tree["files"].get(name)
        require(isinstance(entry, dict), f"model tree has no entry for {name}")
        expected_size = entry.get("lfs_size")
        expected_sha256 = entry.get("lfs_sha256")
        require(isinstance(expected_size, int) and expected_size > 0, f"model tree has no size for {name}")
        require(isinstance(expected_sha256, str) and len(expected_sha256) == 64, f"model tree has no hash for {name}")
        require(path.is_file() and path.stat().st_size == expected_size, f"model shard size mismatch: {name}")
        require(path.resolve().name == expected_sha256, f"model shard is not backed by its pinned content hash: {name}")
        total_bytes += expected_size
    verified = verify_huggingface_snapshot(snapshot, expected_revision=MODEL_REVISION)
    require(
        verified["tree_metadata_sha256"] == sha256_file(tree_path),
        "full model snapshot authentication disagrees with tree metadata",
    )
    require(verified["total_bytes"] >= total_bytes, "full model snapshot byte count is smaller than its weight shards")
    return {
        **verified,
        "id": MODEL_ID,
        "revision": MODEL_REVISION,
        "shards": len(shards),
        "snapshot": str(snapshot),
        "total_weight_bytes": total_bytes,
    }


def _execution_geometry_from_config(config: dict[str, Any]) -> dict[str, str | int]:
    model = config.get("model")
    optimization = config.get("optimization")
    benchmark = config.get("benchmark")
    require(isinstance(model, dict), "resolved config has no model table")
    require(isinstance(optimization, dict), "resolved config has no optimization table")
    require(isinstance(benchmark, dict), "resolved config has no benchmark table")
    content_sha256 = benchmark.get("prefix_geometry_content_sha256")
    require(
        isinstance(content_sha256, str)
        and len(content_sha256) == 64
        and all(character in "0123456789abcdef" for character in content_sha256),
        "resolved prefix geometry SHA-256 is invalid",
    )
    fixed_width = benchmark.get("fixed_physical_prefix_width")
    require(type(fixed_width) is int and fixed_width > 0, "resolved fixed physical prefix width is invalid")
    expected = {
        "experts_implementation": EXPERTS_IMPLEMENTATION,
        "expert_batch_isolation": EXPERT_BATCH_ISOLATION,
        "physical_batch_size": PHYSICAL_BATCH_SIZE,
        "fixed_physical_prefix_width": fixed_width,
        "prefix_geometry_content_sha256": content_sha256,
    }
    observed = {
        "experts_implementation": model.get("experts_implementation"),
        "expert_batch_isolation": model.get("expert_batch_isolation"),
        "physical_batch_size": optimization.get("physical_batch_size"),
        "fixed_physical_prefix_width": fixed_width,
        "prefix_geometry_content_sha256": content_sha256,
    }
    mismatches = {
        name: {"expected": value, "observed": observed[name]}
        for name, value in expected.items()
        if observed[name] != value
    }
    require(not mismatches, f"resolved fixed-batch execution geometry differs: {mismatches}")
    require(optimization.get("microbatch_size") == PHYSICAL_BATCH_SIZE, "resolved microbatch size must be 8")
    require(optimization.get("gradient_accumulation_steps") == 8, "resolved accumulation steps must be 8")
    require(optimization.get("global_batch_size") == 64, "resolved global batch size must be 64")
    require(
        benchmark.get("camera_order") == list(LIBERO_PREFIX_CAMERA_NAMES),
        "resolved LIBERO camera order differs from the prefix contract",
    )
    return expected


def _authenticated_training_environment(
    resolved_config: dict[str, Any],
    manifest: dict[str, Any],
    *,
    project_root: Path,
    train_seed: int,
) -> dict[str, Any]:
    """Require a checkpoint produced under the strict deterministic trainer contract."""

    environment = resolved_config.get("execution_environment")
    require(isinstance(environment, dict), "resolved config has no authenticated training execution environment")
    require(manifest.get("execution_environment") == environment, "checkpoint training environment differs from config")
    environment_sha256 = _canonical_sha256(environment)
    require(
        manifest.get("execution_environment_sha256") == environment_sha256,
        "checkpoint training environment SHA-256 mismatch",
    )
    expected_determinism = {
        "cublas_workspace_config": ":4096:8",
        "cudnn_benchmark": False,
        "cudnn_deterministic": True,
        "cudnn_tf32": False,
        "deterministic_algorithms": True,
        "deterministic_warn_only": False,
        "float32_matmul_precision": "highest",
        "matmul_tf32": False,
        "preferred_blas_library": "_BlasBackend.Cublas",
        "preferred_linalg_library": "_LinalgBackend.Default",
        "python_hash_seed": str(train_seed),
    }
    mismatches = {
        name: {"expected": expected, "observed": environment.get(name)}
        for name, expected in expected_determinism.items()
        if environment.get(name) != expected
    }
    require(not mismatches, f"checkpoint was not trained under strict deterministic controls: {mismatches}")
    authenticated_runtime = environment.get("authenticated_runtime")
    require(isinstance(authenticated_runtime, dict), "training environment has no authenticated runtime")
    require(
        set(authenticated_runtime)
        == {
            "algorithm_override_environment",
            "environment",
            "nccl_environment",
            "static_environment_sha256",
            "torchrun",
            "train_venv",
        },
        "training authenticated-runtime fields differ",
    )
    require(
        authenticated_runtime.get("algorithm_override_environment") == {},
        "training environment contains unpinned algorithm overrides",
    )
    require(
        authenticated_runtime.get("nccl_environment") == {},
        "training environment contains unpinned NCCL overrides",
    )
    expected_venv = authenticated_runtime.get("train_venv")
    train_venv = Path(os.environ.get("DUO_VLA_TRAIN_VENV", ""))
    require(train_venv.is_absolute(), "serving process has no canonical train-venv path")
    require(
        train_venv.resolve() == (Path(os.environ["DUO_VLA_CACHE_ROOT"]) / "venvs/train").resolve(),
        "serving train-venv path differs from the cache-root contract",
    )
    live_venv = content_address_train_venv(train_venv)
    require_matching_train_venv(expected_venv, live_venv)
    expected_environment = authenticated_runtime.get("environment")
    require(isinstance(expected_environment, dict), "training environment has no static process environment")
    expected_training_environment = {
        **REQUIRED_SERVING_ENVIRONMENT,
        "DUO_VLA_CACHE_ROOT": str(Path(os.environ["DUO_VLA_CACHE_ROOT"]).resolve()),
        "DUO_VLA_PROJECT_ROOT": str(project_root.resolve()),
        "DUO_VLA_TRAIN_VENV": str(train_venv.resolve()),
        "HF_HOME": str(Path(os.environ["HF_HOME"]).resolve()),
        "PYTHONHASHSEED": str(train_seed),
    }
    require(
        expected_environment == dict(sorted(expected_training_environment.items())),
        "checkpoint training process environment differs from the canonical launcher",
    )
    require(
        authenticated_runtime.get("static_environment_sha256")
        == static_environment_identity(expected_environment)["sha256"],
        "checkpoint training static-environment SHA-256 mismatch",
    )
    require(
        expected_environment.get("DUO_VLA_PROJECT_ROOT") == str(project_root.resolve()),
        "checkpoint training project root differs from serving",
    )
    require(
        authenticated_runtime.get("torchrun")
        == {
            "group_world_size": 1,
            "local_rank_equals_rank": True,
            "local_world_size": 2,
            "role_world_size": 2,
            "world_size": 2,
        },
        "checkpoint training torchrun topology differs",
    )
    return environment


def resolve_checkpoint(
    checkpoint_dir: Path,
    *,
    train_seed_override: int | None,
    model_snapshot_report: dict[str, Any],
) -> tuple[
    dict[str, Any],
    Path,
    Path,
    dict[str, Any],
    int,
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
]:
    from duo_vla.checkpointing import load_checkpoint_manifest
    from duo_vla.data.libero_stats import load_libero_normalizers
    from duo_vla.policy_contract import validate_manifest_policy_contract
    from duo_vla.prefix_geometry import CameraGeometry, SnapshotTreeIdentity, load_prefix_geometry_contract
    from duo_vla.run_config import canonical_config_sha256, load_verified_resolved_config

    checkpoint_dir = checkpoint_dir.resolve()
    manifest = load_checkpoint_manifest(checkpoint_dir, verify_hashes=True)
    require(manifest.get("kind") == "resumable-libero-training", "checkpoint kind is not strict LIBERO training")
    require(manifest.get("model_id") == MODEL_ID, "checkpoint model identity mismatch")
    require(manifest.get("model_revision") == MODEL_REVISION, "checkpoint model revision mismatch")
    require(manifest.get("dataset_id") == DATASET_ID, "checkpoint dataset identity mismatch")
    require(manifest.get("dataset_revision") == DATASET_REVISION, "checkpoint dataset revision mismatch")

    resolved_artifact = manifest.get("artifacts", {}).get("resolved_config")
    require(
        isinstance(resolved_artifact, dict) and isinstance(resolved_artifact.get("path"), str),
        "checkpoint has no resolved configuration artifact",
    )
    resolved_config_path = checkpoint_dir / resolved_artifact["path"]
    require(
        resolved_config_path.resolve().is_relative_to(checkpoint_dir),
        "resolved configuration artifact escapes the checkpoint directory",
    )
    config_sha256 = manifest.get("config_sha256")
    require(isinstance(config_sha256, str) and len(config_sha256) == 64, "checkpoint config SHA-256 is invalid")
    resolved_config, _ = load_verified_resolved_config(
        resolved_config_path,
        expected_sha256=config_sha256,
    )
    manifest_source_sha256 = manifest.get("source_tree_sha256")
    resolved_source_sha256 = resolved_config.get("source_tree_sha256")
    require(
        isinstance(manifest_source_sha256, str)
        and len(manifest_source_sha256) == 64
        and resolved_source_sha256 == manifest_source_sha256,
        "checkpoint and resolved config source-tree identities disagree",
    )
    policy_contract = validate_manifest_policy_contract(manifest, resolved_config)
    policy_contract_dict = policy_contract.to_dict()
    require(
        manifest.get("policy_contract_sha256") == canonical_config_sha256(policy_contract_dict),
        "checkpoint policy contract SHA-256 mismatch",
    )
    model_config = resolved_config.get("model")
    benchmark_config = resolved_config.get("benchmark")
    require(isinstance(model_config, dict), "resolved config has no model table")
    require(isinstance(benchmark_config, dict), "resolved config has no benchmark table")
    require(model_config.get("id") == MODEL_ID, "resolved config model identity mismatch")
    require(model_config.get("revision") == MODEL_REVISION, "resolved config model revision mismatch")
    artifact_trees = resolved_config.get("artifact_trees")
    require(isinstance(artifact_trees, dict), "resolved config has no artifact tree identities")
    expected_dataset_identity = {
        "dataset_content_inventory_sha256": DATASET_CONTENT_INVENTORY_SHA256,
        "dataset_files_verified": DATASET_FILES_VERIFIED,
        "dataset_total_bytes": DATASET_TOTAL_BYTES,
        "dataset_tree_sha256": DATASET_TREE_SHA256,
    }
    for name, expected in expected_dataset_identity.items():
        require(artifact_trees.get(name) == expected, f"resolved config {name} differs from the qualified dataset")
        require(manifest.get(name) == expected, f"checkpoint {name} differs from the qualified dataset")
    expected_model_identity = {
        "model_content_inventory_sha256": model_snapshot_report.get("content_inventory_sha256"),
        "model_files_verified": model_snapshot_report.get("files_verified"),
        "model_total_bytes": model_snapshot_report.get("total_bytes"),
        "model_tree_sha256": model_snapshot_report.get("tree_metadata_sha256"),
    }
    for name, expected in expected_model_identity.items():
        require(artifact_trees.get(name) == expected, f"resolved config {name} differs from the authenticated snapshot")
        require(manifest.get(name) == expected, f"checkpoint {name} differs from the authenticated snapshot")
    require(
        artifact_trees.get("model_tree_sha256") == model_snapshot_report.get("tree_metadata_sha256"),
        "resolved config model tree identity differs from the authenticated snapshot",
    )
    require(
        artifact_trees.get("model_content_inventory_sha256") == model_snapshot_report.get("content_inventory_sha256"),
        "resolved config model content inventory differs from the authenticated snapshot",
    )
    require(
        manifest.get("model_tree_sha256") == model_snapshot_report.get("tree_metadata_sha256"),
        "checkpoint model tree identity differs from the authenticated snapshot",
    )
    require(
        manifest.get("model_content_inventory_sha256") == model_snapshot_report.get("content_inventory_sha256"),
        "checkpoint model content inventory differs from the authenticated snapshot",
    )
    require(benchmark_config.get("dataset_id") == DATASET_ID, "resolved config dataset identity mismatch")
    require(
        benchmark_config.get("dataset_revision") == DATASET_REVISION,
        "resolved config dataset revision mismatch",
    )
    require(resolved_config.get("protocol") == PROTOCOL, "resolved config LIBERO protocol mismatch")
    require(
        policy_contract.action_horizon == ACTION_HORIZON and policy_contract.action_dim == ACTION_DIM,
        "resolved policy action shape differs from IPC",
    )
    require(benchmark_config.get("state_dimension") == STATE_DIM, "resolved policy state shape differs from IPC")
    execution_geometry = _execution_geometry_from_config(resolved_config)
    manifest_geometry_mismatches = {
        name: {"expected": expected, "observed": manifest.get(name)}
        for name, expected in execution_geometry.items()
        if manifest.get(name) != expected
    }
    require(
        not manifest_geometry_mismatches,
        f"checkpoint fixed-batch execution geometry differs: {manifest_geometry_mismatches}",
    )

    prefix_artifact = manifest.get("artifacts", {}).get("prefix_geometry")
    require(
        isinstance(prefix_artifact, dict) and isinstance(prefix_artifact.get("path"), str),
        "checkpoint has no prefix geometry artifact",
    )
    prefix_geometry_path = checkpoint_dir / prefix_artifact["path"]
    require(
        prefix_geometry_path.resolve().is_relative_to(checkpoint_dir),
        "prefix geometry artifact escapes the checkpoint directory",
    )
    model_identity = SnapshotTreeIdentity.from_huggingface_report(MODEL_ID, model_snapshot_report)
    prefix_geometry = load_prefix_geometry_contract(
        prefix_geometry_path,
        expected_content_sha256=str(execution_geometry["prefix_geometry_content_sha256"]),
        expected_model_identity=model_identity,
        expected_processor_identity=model_identity,
        expected_ordered_cameras=(
            CameraGeometry(LIBERO_PREFIX_CAMERA_NAMES[0], *IMAGE_SHAPE[:2]),
            CameraGeometry(LIBERO_PREFIX_CAMERA_NAMES[1], *IMAGE_SHAPE[:2]),
        ),
        expected_fixed_physical_prefix_width=int(execution_geometry["fixed_physical_prefix_width"]),
    )

    artifact = manifest.get("artifacts", {}).get("normalization")
    require(
        isinstance(artifact, dict) and isinstance(artifact.get("path"), str), "checkpoint has no normalization artifact"
    )
    normalization_path = checkpoint_dir / artifact["path"]
    require(
        normalization_path.resolve().is_relative_to(checkpoint_dir),
        "normalization artifact escapes the checkpoint directory",
    )
    _, _, normalization_manifest = load_libero_normalizers(
        normalization_path,
        expected_revision=DATASET_REVISION,
    )
    observed_normalization_hash = normalization_manifest.get("content_sha256")
    require(observed_normalization_hash == NORMALIZATION_SHA256, "normalization content pin mismatch")
    checkpoint_normalization_hash = manifest.get("normalization_sha256")
    if checkpoint_normalization_hash is None:
        checkpoint_normalization_hash = manifest.get("normalization_content_sha256")
    require(
        checkpoint_normalization_hash == observed_normalization_hash,
        "checkpoint and normalization artifact content hashes disagree",
    )

    recorded_seed = manifest.get("run_seed")
    require(type(recorded_seed) is int and recorded_seed in {0, 1, 2}, "checkpoint run_seed must be one of {0,1,2}")
    if train_seed_override is not None:
        require(train_seed_override == recorded_seed, "--train-seed differs from checkpoint run_seed")
    train_seed = recorded_seed
    training_environment = _authenticated_training_environment(
        resolved_config,
        manifest,
        project_root=Path(__file__).resolve().parents[1],
        train_seed=train_seed,
    )
    checkpoint_report = {
        **expected_dataset_identity,
        **expected_model_identity,
        "kind": manifest.get("kind"),
        "manifest_sha256": sha256_file(checkpoint_dir / "manifest.json"),
        "path": str(checkpoint_dir),
        "policy_contract": policy_contract_dict,
        "policy_contract_sha256": manifest["policy_contract_sha256"],
        "source_tree_sha256": manifest.get("source_tree_sha256"),
        "train_seed": train_seed,
        "training_execution_environment": training_environment,
        "training_execution_environment_sha256": manifest["execution_environment_sha256"],
        "train_venv": training_environment["authenticated_runtime"]["train_venv"],
        "execution_geometry": execution_geometry,
    }
    return (
        manifest,
        normalization_path,
        prefix_geometry_path,
        prefix_geometry,
        train_seed,
        checkpoint_report,
        resolved_config,
        policy_contract_dict,
    )


def train_runtime_preflight(
    project_root: Path,
    checkpoint_dir: Path,
    *,
    train_seed_override: int | None,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    Path,
    Path,
    dict[str, Any],
    int,
    dict[str, Any],
    dict[str, Any],
]:
    require(sys.version_info[:2] == (3, 11), f"policy server requires Python 3.11, found {sys.version.split()[0]}")
    package_versions = {name: importlib.metadata.version(name) for name in EXPECTED_TRAIN_PACKAGES}
    require(package_versions == EXPECTED_TRAIN_PACKAGES, f"train package pin mismatch: {package_versions}")
    lock_path = project_root / "uv.lock"
    require(lock_path.is_file(), f"missing train lockfile: {lock_path}")
    lock_sha256 = sha256_file(lock_path)
    require(lock_sha256 == TRAIN_LOCK_SHA256, f"train lockfile SHA-256 mismatch: {lock_sha256}")
    snapshot_report = model_snapshot_preflight()
    (
        manifest,
        normalization_path,
        prefix_geometry_path,
        prefix_geometry,
        train_seed,
        checkpoint_report,
        resolved_config,
        policy_contract,
    ) = resolve_checkpoint(
        checkpoint_dir,
        train_seed_override=train_seed_override,
        model_snapshot_report=snapshot_report,
    )
    report = {
        "checkpoint": checkpoint_report,
        "dataset": {"id": DATASET_ID, "revision": DATASET_REVISION},
        "execution_geometry": checkpoint_report["execution_geometry"],
        "lock_sha256": lock_sha256,
        "model": snapshot_report,
        "normalization_content_sha256": NORMALIZATION_SHA256,
        "packages": package_versions,
        "policy_contract": policy_contract,
        "python": sys.version.split()[0],
        "status": "ok",
    }
    return (
        report,
        manifest,
        normalization_path,
        prefix_geometry_path,
        prefix_geometry,
        train_seed,
        resolved_config,
        policy_contract,
    )


def _health_payload(
    *,
    mode: str,
    train_seed: int,
    checkpoint_report: dict[str, Any] | None,
    policy_contract: dict[str, Any],
    latency_runtime_sha256: str | None = None,
    serving_runtime_sha256: str | None = None,
) -> dict[str, Any]:
    wire_contract = {name: policy_contract[name] for name in ("objective", "sampler", "nfe", "inference_seed_behavior")}
    validate_wire_policy_contract(wire_contract, allow_fake=mode == "fake")
    for name, value in (
        ("latency_runtime_sha256", latency_runtime_sha256),
        ("serving_runtime_sha256", serving_runtime_sha256),
    ):
        if mode == "real":
            require(
                isinstance(value, str)
                and len(value) == 64
                and all(character in "0123456789abcdef" for character in value),
                f"real policy {name} is invalid",
            )
        else:
            require(value is None, f"fake policy cannot claim {name}")
    return {
        "action_dim": ACTION_DIM,
        "action_horizon": ACTION_HORIZON,
        "checkpoint": checkpoint_report,
        "dataset_revision": DATASET_REVISION,
        "execution_geometry": checkpoint_report["execution_geometry"] if mode == "real" else None,
        "latency_runtime_sha256": latency_runtime_sha256,
        "mode": mode,
        "model_revision": MODEL_REVISION if mode == "real" else None,
        "normalization_content_sha256": NORMALIZATION_SHA256 if mode == "real" else None,
        "prefix_cache_scope": "request",
        "protocol": PROTOCOL,
        "state_dim": STATE_DIM,
        "serving_runtime_sha256": serving_runtime_sha256,
        "train_seed": train_seed,
        **wire_contract,
    }


def select_serving_policy_contract(
    checkpoint_policy_contract: dict[str, Any],
    *,
    flow_steps_override: int | None,
) -> dict[str, Any]:
    """Select runtime NFE without changing the authenticated training contract."""

    selected = {
        name: checkpoint_policy_contract[name] for name in ("objective", "sampler", "nfe", "inference_seed_behavior")
    }
    if selected["objective"] == "rectified_flow":
        if flow_steps_override is not None:
            require(type(flow_steps_override) is int, "--flow-steps must be an integer")
            selected["nfe"] = flow_steps_override
        require(selected["nfe"] in {1, 5, 10}, "rectified-flow serving NFE must be one of {1, 5, 10}")
    elif selected["objective"] == "direct_regression":
        require(flow_steps_override is None, "--flow-steps is forbidden for a direct-regression checkpoint")
        require(selected["nfe"] == 1, "direct-regression checkpoint must use one function evaluation")
    else:
        raise RuntimeError(f"unsupported checkpoint objective: {selected['objective']!r}")
    validate_wire_policy_contract(selected)
    return selected


def _evaluation_identity_echo(request: dict[str, Any]) -> dict[str, Any]:
    episode = request["episode"]
    return {
        "evaluation_seed": request["evaluation_seed"],
        "reset_id": episode["reset_id"],
        "reset_source": episode["reset_source"],
        "reset_state_sha256": episode["reset_state_sha256"],
    }


class FakePolicy:
    """Deterministic policy used only to exercise IPC and evaluator control flow."""

    contract: ClassVar[dict[str, Any]] = {
        "objective": "test_fake",
        "sampler": "seeded_test_normal",
        "nfe": 0,
        "inference_seed_behavior": "episode_identity_test_generator",
    }

    def predict(self, request: dict[str, Any]) -> dict[str, Any]:
        started = time.perf_counter()
        generator = np.random.default_rng(request["inference_seed"])
        actions = generator.normal(0.0, 0.05, size=(ACTION_HORIZON, ACTION_DIM)).astype(np.float32)
        actions[:, 6] = np.where(actions[:, 6] >= 0, 1.0, -1.0)
        return {
            "actions": actions.tolist(),
            "inference_seed": request["inference_seed"],
            "normalized_clip_fraction": 0.0,
            "policy_seconds": time.perf_counter() - started,
            **_evaluation_identity_echo(request),
            **self.contract,
        }


class RealPolicy:
    """TP-sharded DiffusionGemma plus the trained Duo-VLA action interface."""

    def __init__(
        self,
        checkpoint_dir: Path,
        normalization_path: Path,
        device: Any,
        *,
        resolved_config: dict[str, Any],
        policy_contract: dict[str, Any],
        prefix_geometry: dict[str, Any],
    ) -> None:
        import torch
        from PIL import Image
        from transformers import AutoProcessor

        from duo_vla.action_interface import ActionInputProjector, VelocityHead
        from duo_vla.backbones.diffusion_gemma import (
            DiffusionGemmaActionDecoder,
            encode_diffusion_gemma_prefix,
        )
        from duo_vla.backbones.loading import DEFAULT_DIFFUSION_GEMMA_SPEC, load_diffusion_gemma_bf16_tp
        from duo_vla.backbones.sample_isolated_experts import (
            install_sample_isolated_grouped_mm_experts,
            verify_sample_isolated_grouped_mm_experts,
        )
        from duo_vla.checkpointing import load_interface_state_dict, load_lora_checkpoint
        from duo_vla.config import ActionInterfaceConfig
        from duo_vla.data.libero_stats import load_libero_normalizers
        from duo_vla.flow import euler_sample
        from duo_vla.modeling import DuoVLADenoiser
        from duo_vla.prefix_geometry import apply_fixed_prefix_chat_template

        self.torch = torch
        self.Image = Image
        self.encode_prefix = encode_diffusion_gemma_prefix
        self.euler_sample = euler_sample
        self.apply_fixed_prefix_chat_template = apply_fixed_prefix_chat_template
        self.device = device
        self.policy_contract = policy_contract
        self.execution_geometry = _execution_geometry_from_config(resolved_config)
        self.prefix_geometry = prefix_geometry
        self.prefix_valid_lengths = {
            record["instruction"]: int(record["valid_prefix_length"])
            for record in prefix_geometry["instruction_inventory"]["records"]
        }
        self.processor = AutoProcessor.from_pretrained(
            DEFAULT_DIFFUSION_GEMMA_SPEC.model_id,
            revision=DEFAULT_DIFFUSION_GEMMA_SPEC.revision,
            local_files_only=True,
        )
        self.model = load_diffusion_gemma_bf16_tp(local_files_only=True, tp_size=2)
        install_sample_isolated_grouped_mm_experts(
            self.model,
            physical_batch_size=PHYSICAL_BATCH_SIZE,
        )
        verify_sample_isolated_grouped_mm_experts(
            self.model,
            physical_batch_size=PHYSICAL_BATCH_SIZE,
        )
        backend = DiffusionGemmaActionDecoder.from_block_diffusion_model(self.model)
        self.adapted, _ = load_lora_checkpoint(
            checkpoint_dir,
            self.model,
            is_trainable=False,
            validate_decoder_contract=True,
            expected_rank=int(resolved_config["lora"]["rank"]),
        )
        verify_sample_isolated_grouped_mm_experts(
            self.model,
            physical_batch_size=PHYSICAL_BATCH_SIZE,
        )
        action_config = resolved_config["action"]
        benchmark_config = resolved_config["benchmark"]
        interface_config = ActionInterfaceConfig(
            hidden_size=2816,
            state_dim=int(benchmark_config["state_dimension"]),
            action_horizon=int(action_config["horizon"]),
            action_dim=int(action_config["dimension"]),
            timestep_embedding_dim=int(action_config["timestep_embedding_dimension"]),
            timestep_scale=float(action_config["timestep_scale"]),
            timestep_max_period=float(action_config["timestep_max_period"]),
            output_init_std=float(action_config["output_head_initialization_std"]),
        )
        projector = ActionInputProjector(interface_config).to(device)
        head = VelocityHead(interface_config.hidden_size, interface_config.action_dim).to(device)
        load_interface_state_dict(
            checkpoint_dir / "interface.safetensors",
            {"action_projector": projector, "velocity_head": head},
        )
        self.denoiser = DuoVLADenoiser(projector, backend, head).eval()
        self.adapted.eval()
        self.model.model.encoder.eval()
        state_normalizer, action_normalizer, _ = load_libero_normalizers(
            normalization_path,
            expected_revision=DATASET_REVISION,
        )
        self.state_normalizer = state_normalizer.to(device=device, dtype=torch.float32)
        self.action_normalizer = action_normalizer

    def _processor_inputs(self, request: dict[str, Any]) -> dict[str, Any]:
        observation = request["observation"]
        instruction = request["instruction"]
        require(
            instruction in self.prefix_valid_lengths, "instruction is absent from the authenticated prefix inventory"
        )
        agentview = np.asarray(observation["agentview_rgb"])
        wrist = np.asarray(observation["wrist_rgb"])
        for name, values in zip(LIBERO_PREFIX_CAMERA_NAMES, (agentview, wrist), strict=True):
            require(
                values.shape == IMAGE_SHAPE and values.dtype == np.uint8,
                f"{name} must be uint8{IMAGE_SHAPE}",
            )
        conversations = [
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": self.Image.fromarray(agentview)},
                        {"type": "image", "image": self.Image.fromarray(wrist)},
                        {"type": "text", "text": instruction},
                    ],
                }
            ]
            for _ in range(PHYSICAL_BATCH_SIZE)
        ]
        values = self.apply_fixed_prefix_chat_template(
            self.processor,
            conversations,
            fixed_physical_prefix_width=int(self.execution_geometry["fixed_physical_prefix_width"]),
            padding_side=str(self.prefix_geometry["tokenization"]["padding_side"]),
            expected_batch_size=PHYSICAL_BATCH_SIZE,
            images_per_prefix=len(LIBERO_PREFIX_CAMERA_NAMES),
        )
        expected_length = self.prefix_valid_lengths[instruction]
        observed_lengths = tuple(int(value) for value in values["attention_mask"].bool().sum(dim=1).tolist())
        require(
            observed_lengths == (expected_length,) * PHYSICAL_BATCH_SIZE,
            "processor valid-prefix lengths differ from the authenticated geometry",
        )
        return dict(values.to(self.device))

    def _require_bitwise_replicas(self, values: Any, *, name: str) -> None:
        torch = self.torch
        require(
            isinstance(values, torch.Tensor) and values.ndim >= 1 and values.shape[0] == PHYSICAL_BATCH_SIZE,
            f"{name} must have physical batch {PHYSICAL_BATCH_SIZE}",
        )
        require(bool(torch.isfinite(values).all()), f"{name} contains non-finite values")
        reference = values[0].contiguous().view(torch.uint8)
        for index in range(1, PHYSICAL_BATCH_SIZE):
            replica = values[index].contiguous().view(torch.uint8)
            require(torch.equal(replica, reference), f"{name} replica {index} differs bitwise from row 0")

    def predict(self, request: dict[str, Any]) -> dict[str, Any]:
        torch = self.torch
        torch.cuda.synchronize(self.device)
        started = time.perf_counter()
        observation = request["observation"]
        state_row = torch.from_numpy(observation["state"]).to(device=self.device, dtype=torch.float32).unsqueeze(0)
        normalized_state = self.state_normalizer.normalize(state_row).expand(PHYSICAL_BATCH_SIZE, -1).clone()
        prefix_inputs = self._processor_inputs(request)
        prefix = self.encode_prefix(self.model, prefix_inputs)
        valid = torch.ones((PHYSICAL_BATCH_SIZE, ACTION_HORIZON), device=self.device, dtype=torch.bool)
        objective = self.policy_contract["objective"]
        if objective == "rectified_flow":
            generator = torch.Generator(device=self.device).manual_seed(request["inference_seed"])
            noise_row = torch.randn(
                (1, ACTION_HORIZON, ACTION_DIM),
                device=self.device,
                dtype=normalized_state.dtype,
                generator=generator,
            )
            noise = noise_row.expand(PHYSICAL_BATCH_SIZE, -1, -1).clone()

            def velocity(actions: Any, timesteps: Any) -> Any:
                prediction = self.denoiser(
                    actions,
                    timesteps,
                    normalized_state,
                    prefix_cache=prefix.past_key_values,
                    prefix_attention_mask=prefix.attention_mask,
                    action_valid_mask=valid,
                )
                self._require_bitwise_replicas(prediction, name="flow velocity")
                return prediction

        elif objective != "direct_regression":
            raise RuntimeError(f"unsupported loaded policy objective: {objective!r}")

        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            if objective == "rectified_flow":
                raw_normalized = self.euler_sample(
                    velocity,
                    initial_noise=noise,
                    num_steps=self.policy_contract["nfe"],
                )
            else:
                raw_normalized = self.denoiser(
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
        self._require_bitwise_replicas(raw_normalized, name="normalized action")
        normalized_row = raw_normalized[0]
        normalized_clip_fraction = float((normalized_row.abs() > 1.0).float().mean().item())
        actions = self.action_normalizer.unnormalize(normalized_row.clamp(-1.0, 1.0))
        torch.cuda.synchronize(self.device)
        policy_seconds = time.perf_counter() - started
        values = actions.detach().float().cpu().numpy().astype(np.float32, copy=True)
        return {
            "actions": values.tolist(),
            "inference_seed": request["inference_seed"],
            "normalized_clip_fraction": normalized_clip_fraction,
            "policy_seconds": policy_seconds,
            **_evaluation_identity_echo(request),
            **{name: self.policy_contract[name] for name in ("objective", "sampler", "nfe", "inference_seed_behavior")},
        }


def _dispatch_local(
    request: dict[str, Any],
    *,
    policy: FakePolicy | RealPolicy,
    health_payload: dict[str, Any],
) -> dict[str, Any]:
    operation = request["operation"]
    if operation == "health":
        return make_success_response(request, **health_payload)
    if operation == "shutdown":
        return make_success_response(request, stopped=True)
    if request["train_seed"] != health_payload["train_seed"]:
        raise ValueError("request train_seed differs from the loaded checkpoint")
    prediction = policy.predict(request)
    for name, expected in _evaluation_identity_echo(request).items():
        require(name in prediction and prediction[name] == expected, f"policy response {name} did not echo the request")
    return make_success_response(request, **prediction)


def _canonical_prediction(value: dict[str, Any]) -> str:
    comparable = {key: item for key, item in value.items() if key != "policy_seconds"}
    return json.dumps(comparable, allow_nan=False, separators=(",", ":"), sort_keys=True)


def run_fake_server(socket_path: Path, *, train_seed: int) -> None:
    policy = FakePolicy()
    health = _health_payload(
        mode="fake",
        train_seed=train_seed,
        checkpoint_report=None,
        policy_contract=policy.contract,
    )

    def dispatch(request: dict[str, Any]) -> dict[str, Any]:
        return _dispatch_local(request, policy=policy, health_payload=health)

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
    checkpoint_report: dict[str, Any],
    resolved_config: dict[str, Any],
    policy_contract: dict[str, Any],
    prefix_geometry: dict[str, Any],
) -> None:
    import torch
    import torch.distributed as dist

    project_root = Path(__file__).resolve().parents[1]
    serving_runtime, serving_runtime_sha256 = configure_and_identify_serving_runtime(
        torch,
        project_root=project_root,
        checkpoint_report=checkpoint_report,
    )
    _, latency_runtime_sha256 = latency_runtime_identity(serving_runtime)
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", device_id=device)
    require(dist.get_world_size() == 2, "real LIBERO policy serving requires TP world size 2")
    rank = dist.get_rank()
    try:
        runtime_hashes: list[str | None] = [None] * dist.get_world_size()
        dist.all_gather_object(runtime_hashes, serving_runtime_sha256)
        require(
            runtime_hashes == [serving_runtime_sha256] * dist.get_world_size(),
            f"TP ranks have different serving runtime identities: {runtime_hashes}",
        )
        latency_runtime_hashes: list[str | None] = [None] * dist.get_world_size()
        dist.all_gather_object(latency_runtime_hashes, latency_runtime_sha256)
        require(
            latency_runtime_hashes == [latency_runtime_sha256] * dist.get_world_size(),
            f"TP ranks have different latency runtime identities: {latency_runtime_hashes}",
        )
        policy = RealPolicy(
            checkpoint_dir,
            normalization_path,
            device,
            resolved_config=resolved_config,
            policy_contract=policy_contract,
            prefix_geometry=prefix_geometry,
        )
        health = _health_payload(
            mode="real",
            train_seed=train_seed,
            checkpoint_report=checkpoint_report,
            policy_contract=policy_contract,
            latency_runtime_sha256=latency_runtime_sha256,
            serving_runtime_sha256=serving_runtime_sha256,
        )
        dist.barrier()

        def execute(request: dict[str, Any]) -> dict[str, Any]:
            local_result: dict[str, Any] | None = None
            local_error: str | None = None
            try:
                local_result = _dispatch_local(request, policy=policy, health_payload=health)
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
                assert all(result is not None for result in results)
                concrete = [result for result in results if result is not None]
                reference = _canonical_prediction(concrete[0])
                if not all(_canonical_prediction(result) == reference for result in concrete[1:]):
                    return make_error_response(request, RuntimeError("TP ranks produced different action chunks"))
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
                            "checkpoint": str(checkpoint_dir),
                            "mode": "real",
                            "nfe": policy_contract["nfe"],
                            "objective": policy_contract["objective"],
                            **checkpoint_report["execution_geometry"],
                            "latency_runtime_sha256": latency_runtime_sha256,
                            "sampler": policy_contract["sampler"],
                            "serving_runtime": serving_runtime,
                            "serving_runtime_sha256": serving_runtime_sha256,
                            "socket": str(socket_path),
                            "status": "ready",
                            "train_seed": train_seed,
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
                require(request is not None, "rank 0 broadcast an empty policy request")
                execute(request)
                if request["operation"] == "shutdown":
                    break
    finally:
        dist.destroy_process_group()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", nargs="?", type=Path)
    parser.add_argument(
        "--socket",
        type=Path,
        default=Path(os.environ.get("DUO_VLA_CACHE_ROOT", "/root/.cache/duo-vla")) / "run/libero-policy.sock",
    )
    parser.add_argument("--train-seed", type=int)
    parser.add_argument(
        "--flow-steps",
        type=int,
        choices=(1, 5, 10),
        help="Override Euler NFE for a rectified-flow checkpoint; forbidden for direct regression.",
    )
    parser.add_argument(
        "--fake-policy", action="store_true", help="serve deterministic fake chunks without loading torch"
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="verify exact pins/checkpoint and CUDA runtime without loading model weights",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    _validate_serving_process_environment(project_root)
    if args.fake_policy:
        require(args.checkpoint is None, "fake policy does not accept a checkpoint")
        require(args.train_seed is not None and 0 <= args.train_seed < 2**63, "fake policy requires --train-seed")
        require(args.flow_steps is None, "fake policy does not accept --flow-steps")
        require(not args.preflight_only, "--preflight-only is only for the real policy")
        run_fake_server(args.socket, train_seed=args.train_seed)
        return

    require(args.checkpoint is not None, "real policy requires a checkpoint directory")
    (
        report,
        _,
        normalization_path,
        _,
        prefix_geometry,
        train_seed,
        resolved_config,
        policy_contract,
    ) = train_runtime_preflight(project_root, args.checkpoint, train_seed_override=args.train_seed)
    serving_policy_contract = select_serving_policy_contract(
        policy_contract,
        flow_steps_override=args.flow_steps,
    )
    report["serving_policy_contract"] = serving_policy_contract
    if args.preflight_only:
        import torch

        serving_runtime, serving_runtime_sha256 = configure_and_identify_serving_runtime(
            torch,
            project_root=project_root,
            checkpoint_report=report["checkpoint"],
        )
        _, latency_runtime_sha256 = latency_runtime_identity(serving_runtime)
        report["serving_runtime"] = serving_runtime
        report["serving_runtime_sha256"] = serving_runtime_sha256
        report["latency_runtime_sha256"] = latency_runtime_sha256
        print(json.dumps(report, indent=2, sort_keys=True))
        return
    run_distributed_server(
        args.socket,
        checkpoint_dir=args.checkpoint.resolve(),
        normalization_path=normalization_path,
        train_seed=train_seed,
        checkpoint_report=report["checkpoint"],
        resolved_config=resolved_config,
        policy_contract=serving_policy_contract,
        prefix_geometry=prefix_geometry,
    )


if __name__ == "__main__":
    main()
