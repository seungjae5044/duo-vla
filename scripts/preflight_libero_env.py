#!/usr/bin/env python3
"""Validate the pinned LIBERO simulator and construct one EGL environment."""

from __future__ import annotations

import argparse
import base64
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
import site
import stat
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

EXPECTED_MANIFEST_PACKAGES = {
    "cmake": "4.1.3",
    "hf-egl-probe": "1.0.2",
    "hf-libero": "0.1.4",
    "huggingface-hub": "1.29.0",
    "mujoco": "3.8.1",
    "numpy": "2.2.6",
    "robosuite": "1.4.0",
    "torch": "2.11.0+cpu",
    "torchvision": "0.26.0+cpu",
}
EXPECTED_PACKAGES = {
    **EXPECTED_MANIFEST_PACKAGES,
    "PyOpenGL": "3.1.10",
    "PyYAML": "6.0.3",
    "bddl": "1.0.1",
    "glfw": "2.10.2",
    "gymnasium": "1.3.0",
    "hydra-core": "1.3.5",
    "llvmlite": "0.49.0",
    "numba": "0.67.0",
    "omegaconf": "2.3.1",
    "opencv-python": "5.0.0.93",
    "robomimic": "0.2.0",
    "scipy": "1.18.1",
    "setuptools": "81.0.0",
}
EXPECTED_SOURCE_REVISION = "8561c60eea2fb93096146f240194649df73d8b1e"
EXPECTED_SOURCE_URL = "https://github.com/huggingface/LIBERO.git"
EXPECTED_ASSETS_REVISION = "0b3ea86be5fe169d0fd036ae63d1070ec09e90f6"
EXPECTED_ASSETS_REPO = "lerobot/libero-assets"
EXPECTED_ASSETS_IDENTITY = {
    "file_count": 586,
    "root_sha256": "8d0757e06f484cef06e08349a1d8bc2790fea5831e36d82e00174508856c659f",
    "total_bytes": 422_320_936,
}
EXPECTED_EVAL_LOCK_SHA256 = "81210a8d7bd58233ca24bd94fd2700e243329f267fd7c25e65c1f41497d57a3c"
EXPECTED_INSTALLED_DISTRIBUTIONS_SHA256 = "03d681d41088af5db6e97148ba9318ae03e19dd932355ae1a961b91a522950d9"
EXPECTED_TASK_INVENTORY_SHA256 = "d00c211a09f34003089ba5a4dbbbb0e11af2543f4bba9cb1901a04a2a25e0117"
SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
CAMERA_KEYS = ("agentview_image", "robot0_eye_in_hand_image")
CAMERA_NAMES = ("agentview", "robot0_eye_in_hand")
ATTESTATION_SCHEMA = "duo-vla-libero-simulator-attestation-v3"
REQUIRED_EVALUATOR_ENVIRONMENT = {
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
_ALGORITHM_ENVIRONMENT_PREFIXES = (
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
_ALLOWED_ALGORITHM_ENVIRONMENT = frozenset(REQUIRED_EVALUATOR_ENVIRONMENT)
_INJECTION_ENVIRONMENT_PREFIXES = ("LD_", "MALLOC_", "OPENSSL_", "PYTHON")
_ALLOWED_INJECTION_ENVIRONMENT = {
    "PYTHONHASHSEED",
    "PYTHONNOUSERSITE",
    "PYTHONPATH",
    "PYTHONSAFEPATH",
    "PYTHONDONTWRITEBYTECODE",
}
_CRITICAL_MODULE_DISTRIBUTIONS = {
    "OpenGL": "PyOpenGL",
    "bddl": "bddl",
    "cv2": "opencv-python",
    "glfw": "glfw",
    "gymnasium": "gymnasium",
    "hydra": "hydra-core",
    "libero": "hf-libero",
    "llvmlite": "llvmlite",
    "mujoco": "mujoco",
    "numpy": "numpy",
    "numba": "numba",
    "omegaconf": "omegaconf",
    "robomimic": "robomimic",
    "robosuite": "robosuite",
    "scipy": "scipy",
    "torch": "torch",
    "yaml": "PyYAML",
}
_RECORD_VERIFIED_DISTRIBUTIONS = (
    "PyOpenGL",
    "PyYAML",
    "bddl",
    "glfw",
    "gymnasium",
    "hf-libero",
    "hydra-core",
    "llvmlite",
    "mujoco",
    "numpy",
    "numba",
    "omegaconf",
    "opencv-python",
    "robomimic",
    "robosuite",
    "scipy",
    "setuptools",
)
_EXPECTED_DISTRIBUTION_INVENTORIES = {
    "PyOpenGL": "0f4939ee29546291f1548bd5761467abde8f13e1e996ed7f0f2ad46848cd79b9",
    "PyYAML": "fbd5d7a73d666238bbc5d51846c2d0c4b7065ea59b533fd4460e8150853ef0bf",
    "bddl": "8270d06f251b58a85049d6531e18c2cbb6c460d4304edbe1216a8b08c8c4b149",
    "glfw": "e730b3c99de3bf0145d855d8fb62284cec0b6a17ae0a5b12426c87df2a8acb1c",
    "gymnasium": "1b78a20c646883ea2d097113f5724fee11f8732ec8c1eacd92f7424541a0f04a",
    "hf-libero": "f010487a985de3e8ce59448887e9aa067294a3b6225ace6bf3c0f71b7327e0db",
    "hydra-core": "764d54731d23a4e2e7fecf44fbc1022a7da8e7c85879912a253e5d9017696b97",
    "llvmlite": "abd239d163551c38ec6da92073bced68d0a6549790e784e74a12c9e814a7bd14",
    "mujoco": "712fb2f4e6f0b691951761233735f3c35497a1d9075e1ea903dd0d37a35d991b",
    "numpy": "8744c53c2a84c80d48a2c7eebeefb4ced8598c4a3d6e37c90f88e0c1672030ca",
    "numba": "d98d44f7569289973f838974274985294ee18e025a0873c8950c598de2784075",
    "omegaconf": "b36941cf9e1a06d9cf555232e9940a4663f60c11bd784151f824b7c0eaef7198",
    "opencv-python": "28532619d2092f4ab7bb15a2f43521422b9e03d7e1994753a6581f5e386e457e",
    "robomimic": "aa5325c87fa33e14e361c68e61a66d0828488284666ce9f3445ed1bb0bf3f140",
    "robosuite": "fbd0c86b30aba8647ca479db2ec161a9a0aff9e6f95dba2c501fd5d0cba6535a",
    "scipy": "e45ddd2e68aa80f7f1a6b4a768334216b00d60b355e32d56103ed7d49568b927",
    "setuptools": "4f4ed57de0b5863db11387b203b75e1001bea4e65977d22ea63422fb17cb4553",
}
_EXPECTED_STARTUP_FILES = {
    "_virtualenv.pth": "69ac3d8f27e679c81b94ab30b3b56e9cd138219b1ba94a1fa3606d5a76a1433d",
    "_virtualenv.py": "6cf30c56faf2a55228914dbbd17f8088ed371ebb08f5e7fa6fd931f913fcaf1d",
    "distutils-precedence.pth": "2638ce9e2500e572a5e0de7faed6661eb569d1b696fcba07b0dd223da5f5d224",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def activate_project_source_root(project_root: Path) -> Path:
    source_path = project_root / "src"
    source_identity = os.lstat(source_path)
    require(stat.S_ISDIR(source_identity.st_mode), "project src import root must be a real directory")
    source_root = source_path.resolve()
    entries = list(os.scandir(source_root))
    require(
        len(entries) == 1 and entries[0].name == "duo_vla" and entries[0].is_dir(follow_symlinks=False),
        "project src import root must contain only the real duo_vla package directory",
    )

    def visit(directory: Path, *, in_cache: bool) -> None:
        for entry in os.scandir(directory):
            mode = entry.stat(follow_symlinks=False).st_mode
            path = directory / entry.name
            if stat.S_ISDIR(mode):
                visit(path, in_cache=in_cache or entry.name == "__pycache__")
            elif stat.S_ISREG(mode):
                allowed = entry.name.endswith(".pyc") if in_cache else entry.name.endswith(".py")
                require(allowed, f"project package import tree contains a forbidden file: {path}")
            else:
                raise RuntimeError(f"project package import tree contains a linked or special entry: {path}")

    visit(source_root / "duo_vla", in_cache=False)
    source_text = str(source_root)
    sys.path[:] = [source_text] + [
        entry for entry in sys.path if str(Path(entry or os.getcwd()).resolve()) != source_text
    ]
    return source_root


def validate_project_module_origins(source_root: Path, required_modules: set[str]) -> dict[str, str]:
    package_root = (source_root / "duo_vla").resolve(strict=True)
    loaded = {name: module for name, module in sys.modules.items() if name == "duo_vla" or name.startswith("duo_vla.")}
    missing = sorted(required_modules - set(loaded))
    require(not missing, f"required checkout modules are not loaded: {missing}")
    origins: dict[str, str] = {}
    for name, module in sorted(loaded.items()):
        origin = getattr(getattr(module, "__spec__", None), "origin", None)
        module_file = getattr(module, "__file__", None)
        require(
            isinstance(origin, str) and isinstance(module_file, str),
            f"checkout module has no file origin: {name}",
        )
        resolved_origin = Path(origin).resolve(strict=True)
        resolved_file = Path(module_file).resolve(strict=True)
        require(
            resolved_origin == resolved_file and resolved_file.is_relative_to(package_root),
            f"checkout module origin escapes authenticated source root: {name}",
        )
        origins[name] = str(resolved_file)
    return origins


def _expected_eval_sys_path(project_root: Path, venv_root: Path) -> list[str]:
    version = f"python{sys.version_info.major}.{sys.version_info.minor}"
    compact_version = f"python{sys.version_info.major}{sys.version_info.minor}"
    return [
        str((project_root / "src").resolve()),
        str(Path(sys.base_prefix) / "lib" / f"{compact_version}.zip"),
        str(Path(sys.base_prefix) / "lib" / version),
        str(Path(sys.base_exec_prefix) / "lib" / version / "lib-dynload"),
        str(venv_root / "lib" / version / "site-packages"),
    ]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, allow_nan=False, separators=(",", ":"), sort_keys=True).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, value in pairs:
        if name in result:
            raise ValueError(f"duplicate simulator manifest field {name!r}")
        result[name] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite simulator manifest constant {value}")


def _require_finite_json_numbers(value: Any, *, name: str) -> None:
    if isinstance(value, float):
        require(math.isfinite(value), f"{name} contains a non-finite number")
    elif isinstance(value, dict):
        for key, item in value.items():
            _require_finite_json_numbers(item, name=f"{name}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _require_finite_json_numbers(item, name=f"{name}[{index}]")


def load_strict_manifest(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_unique_json_object,
            parse_constant=_reject_json_constant,
        )
    except (OSError, UnicodeError, ValueError) as exc:
        raise RuntimeError(f"simulator manifest is not strict finite UTF-8 JSON: {path}") from exc
    require(isinstance(value, dict), "simulator manifest root must be an object")
    _require_finite_json_numbers(value, name="simulator manifest")
    require(
        set(value) == {"assets", "environment", "paths", "schema", "source", "training_data_downloaded"},
        "simulator manifest top-level fields changed",
    )
    return value


def validate_process_environment(project_root: Path, cache_root: Path) -> dict[str, Any]:
    """Require the closed launcher environment before importing simulator code."""

    source_root = activate_project_source_root(project_root)
    validate_project_module_origins(source_root, {"duo_vla"} if "duo_vla" in sys.modules else set())
    forbidden = ("GCONV_PATH", "GLIBC_TUNABLES", "LOCPATH")
    present_forbidden = [name for name in forbidden if os.environ.get(name)]
    injection_overrides = sorted(
        name
        for name in os.environ
        if name.startswith(_INJECTION_ENVIRONMENT_PREFIXES) and name not in _ALLOWED_INJECTION_ENVIRONMENT
    )
    algorithm_overrides = sorted(
        name
        for name in os.environ
        if name.startswith(_ALGORITHM_ENVIRONMENT_PREFIXES) and name not in _ALLOWED_ALGORITHM_ENVIRONMENT
    )
    require(
        not present_forbidden and not injection_overrides and not algorithm_overrides,
        "LIBERO evaluator environment contains injection/algorithm overrides: "
        f"forbidden={present_forbidden}, injection_overrides={injection_overrides}, "
        f"algorithm_overrides={algorithm_overrides}",
    )
    observed = {name: os.environ.get(name) for name in REQUIRED_EVALUATOR_ENVIRONMENT}
    require(observed == REQUIRED_EVALUATOR_ENVIRONMENT, f"LIBERO evaluator environment differs: {observed}")
    expected_prefix = (cache_root / "venvs/libero-eval").resolve()
    require(Path(sys.prefix).resolve() == expected_prefix, f"LIBERO evaluator must run from {expected_prefix}")
    require(sys.flags.safe_path == 1, "LIBERO evaluator requires Python safe-path mode")
    require(sys.flags.dont_write_bytecode == 1 and sys.dont_write_bytecode, "LIBERO evaluator requires -B")
    require(sys.flags.no_user_site == 1 and not site.ENABLE_USER_SITE, "LIBERO evaluator requires no user site")
    require(sys.pycache_prefix == "/dev/null", "LIBERO evaluator requires an impossible pycache lookup prefix")
    expected_invocation_flags = ["-P", "-B", "-X", "pycache_prefix=/dev/null"]
    require(
        sys.orig_argv[1:5] == expected_invocation_flags,
        "LIBERO evaluator must be invoked with the exact safe Python flags",
    )
    expected_sys_path = _expected_eval_sys_path(project_root, expected_prefix)
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
        **observed,
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
    return {
        "environment": expected_environment,
        "python_base_exec_prefix": sys.base_exec_prefix,
        "python_base_prefix": sys.base_prefix,
        "python_executable": sys.executable,
        "python_flags": {
            "dont_write_bytecode": bool(sys.dont_write_bytecode),
            "no_user_site": bool(sys.flags.no_user_site),
            "safe_path": bool(sys.flags.safe_path),
        },
        "python_invocation_flags": expected_invocation_flags,
        "python_prefix": str(expected_prefix),
        "python_pycache_prefix": sys.pycache_prefix,
        "python_version": sys.version.split()[0],
        "sys_path": expected_sys_path,
    }


def git_source_identity(source_path: Path) -> dict[str, Any]:
    def git(*args: str) -> str:
        completed = subprocess.run(
            ("git", "-C", str(source_path), *args),
            check=False,
            capture_output=True,
            text=True,
        )
        require(completed.returncode == 0, f"cannot authenticate LIBERO Git source: {completed.stderr.strip()}")
        return completed.stdout.strip()

    require((source_path / ".git").is_dir(), f"LIBERO source is not a Git checkout: {source_path}")
    revision = git("rev-parse", "HEAD")
    require(revision == EXPECTED_SOURCE_REVISION, "live LIBERO source revision mismatch")
    origin_url = git("config", "--get", "remote.origin.url")
    require(origin_url == EXPECTED_SOURCE_URL, "live LIBERO source origin URL mismatch")
    require(not git("status", "--porcelain=v1", "--untracked-files=all"), "live LIBERO source checkout is modified")
    return {
        "origin_url": origin_url,
        "path": str(source_path.resolve()),
        "revision": revision,
        "tree": git("rev-parse", "HEAD^{tree}"),
    }


def tree_content_identity(root: Path) -> dict[str, Any]:
    """Hash every file/symlink path and content below an immutable asset root."""

    require(root.is_dir(), f"content tree is missing: {root}")
    digest = hashlib.sha256()
    file_count = 0
    total_bytes = 0
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            payload = os.readlink(path).encode("utf-8")
            kind = b"L"
        elif path.is_file():
            payload_digest = hashlib.sha256()
            with path.open("rb") as handle:
                while block := handle.read(8 * 1024 * 1024):
                    payload_digest.update(block)
                    total_bytes += len(block)
            payload = payload_digest.digest()
            kind = b"F"
        else:
            continue
        file_count += 1
        digest.update(kind)
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(payload)
    require(file_count > 0, f"content tree is empty: {root}")
    return {"file_count": file_count, "root_sha256": digest.hexdigest(), "total_bytes": total_bytes}


def module_origin_identity(venv_root: Path) -> dict[str, dict[str, str]]:
    identities: dict[str, dict[str, str]] = {}
    site_packages = venv_root / "lib/python3.12/site-packages"
    for module_name, distribution_name in _CRITICAL_MODULE_DISTRIBUTIONS.items():
        spec = importlib.util.find_spec(module_name)
        require(spec is not None and isinstance(spec.origin, str), f"cannot locate critical module {module_name}")
        origin = Path(spec.origin).resolve()
        expected_origin = (site_packages / module_name / "__init__.py").resolve()
        require(origin == expected_origin, f"critical module {module_name} is shadowed by {origin}")
        distribution = importlib.metadata.distribution(distribution_name)
        metadata = distribution.read_text("METADATA")
        record = distribution.read_text("RECORD")
        require(
            metadata is not None and record is not None,
            f"distribution metadata is incomplete: {distribution_name}",
        )
        identities[module_name] = {
            "distribution": distribution_name,
            "metadata_sha256": hashlib.sha256(metadata.encode("utf-8")).hexdigest(),
            "origin": str(origin),
            "record_sha256": hashlib.sha256(record.encode("utf-8")).hexdigest(),
            "version": distribution.version,
        }
    return identities


def installed_distribution_identity() -> dict[str, Any]:
    distributions = sorted(
        (
            {"name": distribution.metadata["Name"], "version": distribution.version}
            for distribution in importlib.metadata.distributions()
        ),
        key=lambda value: (value["name"].lower(), value["name"], value["version"]),
    )
    identity = canonical_sha256(distributions)
    require(
        len(distributions) == 109 and identity == EXPECTED_INSTALLED_DISTRIBUTIONS_SHA256,
        "installed evaluator distribution closure differs from the frozen lock environment",
    )
    return {"count": len(distributions), "inventory_sha256": identity, "packages": distributions}


def verify_site_packages_inventory(venv_root: Path, expected_assets_path: Path) -> dict[str, Any]:
    """Reject startup hooks, shadow modules, and files absent from installed RECORD inventories."""

    site_packages = venv_root / "lib/python3.12/site-packages"
    require(site_packages.is_dir(), f"evaluator site-packages is missing: {site_packages}")
    require(importlib.util.find_spec("sitecustomize") is None, "sitecustomize injection is present")
    require(importlib.util.find_spec("usercustomize") is None, "usercustomize injection is present")
    startup_files = {path.name: sha256_file(path) for path in sorted(site_packages.glob("*.pth"))}
    startup_files["_virtualenv.py"] = sha256_file(site_packages / "_virtualenv.py")
    require(startup_files == _EXPECTED_STARTUP_FILES, "evaluator Python startup files changed")

    declared: set[str] = set()
    for distribution in importlib.metadata.distributions():
        files = distribution.files
        require(files is not None, f"distribution has no installed-file inventory: {distribution.metadata['Name']}")
        for entry in files:
            unresolved = Path(distribution.locate_file(entry))
            try:
                relative = unresolved.relative_to(site_packages)
            except ValueError:
                continue
            declared.add(relative.as_posix())

    unexpected: list[str] = []
    for path in site_packages.rglob("*"):
        if not path.is_file() or path.is_symlink():
            continue
        relative = path.relative_to(site_packages).as_posix()
        if relative in declared or relative in {"_virtualenv.pth", "_virtualenv.py"}:
            continue
        if "/__pycache__/" in f"/{relative}" and relative.endswith((".nbc", ".nbi", ".pyc", ".pyo")):
            continue
        unexpected.append(relative)
    require(not unexpected, f"unregistered evaluator site-packages files are present: {unexpected[:10]}")

    symlinks = [path for path in site_packages.rglob("*") if path.is_symlink()]
    expected_asset_link = site_packages / "libero/libero/assets"
    require(symlinks == [expected_asset_link], f"unexpected evaluator site-packages symlinks: {symlinks}")
    require(expected_asset_link.resolve() == expected_assets_path, "LIBERO package asset link target changed")
    return {
        "declared_files": len(declared),
        "startup_files": startup_files,
        "symlinks": {"libero/libero/assets": str(expected_assets_path)},
        "unregistered_files": 0,
    }


def verify_distribution_records(venv_root: Path) -> dict[str, dict[str, Any]]:
    reports: dict[str, dict[str, Any]] = {}
    for distribution_name in _RECORD_VERIFIED_DISTRIBUTIONS:
        distribution = importlib.metadata.distribution(distribution_name)
        files = distribution.files
        require(files is not None, f"distribution has no RECORD inventory: {distribution_name}")
        checked = 0
        total_bytes = 0
        inventory: list[dict[str, Any]] = []
        for entry in sorted(files, key=str):
            if entry.hash is None:
                continue
            path = Path(distribution.locate_file(entry)).resolve()
            require(path.is_relative_to(venv_root), f"distribution file escapes evaluator venv: {path}")
            require(path.is_file(), f"distribution file is missing: {path}")
            try:
                digest = hashlib.new(entry.hash.mode)
            except ValueError as exc:
                raise RuntimeError(f"unsupported RECORD hash {entry.hash.mode!r}") from exc
            with path.open("rb") as handle:
                while block := handle.read(8 * 1024 * 1024):
                    digest.update(block)
                    total_bytes += len(block)
            encoded = base64.urlsafe_b64encode(digest.digest()).rstrip(b"=").decode("ascii")
            require(encoded == entry.hash.value, f"installed distribution file hash mismatch: {path}")
            checked += 1
            inventory_hash = f"{entry.hash.mode}={entry.hash.value}"
            if str(entry).startswith("../../../bin/"):
                payload = path.read_bytes()
                shebang, separator, remainder = payload.partition(b"\n")
                allowed_shebangs = {
                    f"#!{venv_root}/bin/python".encode(),
                    f"#!{venv_root}/bin/python3".encode(),
                }
                require(
                    separator == b"\n" and shebang in allowed_shebangs,
                    f"evaluator entry point has an unexpected interpreter: {path}",
                )
                normalized = b"#!<VENV>/bin/python\n" + remainder
                normalized_digest = hashlib.sha256(normalized).digest()
                normalized_value = base64.urlsafe_b64encode(normalized_digest).rstrip(b"=").decode("ascii")
                inventory_hash = f"sha256={normalized_value}"
            inventory.append({"hash": inventory_hash, "path": str(entry)})
        require(checked > 0, f"distribution has no hashed RECORD files: {distribution_name}")
        reports[distribution_name] = {
            "inventory_sha256": canonical_sha256(inventory),
            "verified_files": checked,
            "verified_total_bytes": total_bytes,
        }
        require(
            reports[distribution_name]["inventory_sha256"] == _EXPECTED_DISTRIBUTION_INVENTORIES[distribution_name],
            f"installed distribution inventory differs from the frozen wheel: {distribution_name}",
        )
    return reports


def source_file_identities(project_root: Path) -> dict[str, str]:
    paths = {
        "bridge": project_root / "scripts/libero_bridge.py",
        "evaluator": project_root / "scripts/evaluate_libero.py",
        "launcher": project_root / "scripts/run_libero_eval.sh",
        "preflight": Path(__file__),
        "preflight_launcher": project_root / "scripts/run_libero_preflight.sh",
        "replay_contract": project_root / "src/duo_vla/libero_replay_evidence.py",
        "replay_binder_launcher": project_root / "scripts/run_bind_libero_expert_replay.sh",
        "replay_collector_launcher": project_root / "scripts/run_collect_libero_expert_replay.sh",
        "replay_qualification": project_root / "scripts/qualify_libero_expert_replay.py",
        "replay_qualification_launcher": project_root / "scripts/run_qualify_libero_expert_replay.sh",
    }
    return {name: sha256_file(path) for name, path in paths.items()}


def validate_observation(observation: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key in CAMERA_KEYS:
        require(key in observation, f"missing LIBERO camera observation {key}")
        image = np.asarray(observation[key])
        require(image.shape == (256, 256, 3), f"unexpected {key} shape: {image.shape}")
        require(image.dtype == np.uint8, f"unexpected {key} dtype: {image.dtype}")
        result[key] = {
            "dtype": str(image.dtype),
            "pixels_sha256": hashlib.sha256(np.ascontiguousarray(image).tobytes()).hexdigest(),
            "shape": list(image.shape),
        }

    position = np.asarray(observation["robot0_eef_pos"])
    quaternion = np.asarray(observation["robot0_eef_quat"])
    gripper = np.asarray(observation["robot0_gripper_qpos"])
    require(position.shape == (3,), f"unexpected EEF position shape: {position.shape}")
    require(quaternion.shape == (4,), f"unexpected EEF quaternion shape: {quaternion.shape}")
    require(gripper.shape == (2,), f"unexpected gripper state shape: {gripper.shape}")
    require(np.isfinite(np.concatenate((position, quaternion, gripper))).all(), "non-finite robot state")
    return result


def opengl_runtime_identity(mujoco: Any) -> dict[str, str]:
    """Identify the live EGL renderer/driver through a fresh pinned MuJoCo context."""

    from OpenGL import GL

    context = mujoco.GLContext(8, 8)
    try:
        context.make_current()
        fields = {
            "renderer": GL.GL_RENDERER,
            "shading_language_version": GL.GL_SHADING_LANGUAGE_VERSION,
            "vendor": GL.GL_VENDOR,
            "version": GL.GL_VERSION,
        }
        result: dict[str, str] = {}
        for name, identifier in fields.items():
            value = GL.glGetString(identifier)
            require(isinstance(value, bytes) and bool(value), f"OpenGL did not report {name}")
            result[name] = value.decode("ascii")
        return result
    finally:
        context.free()


def check_no_training_data(dataset_path: Path) -> None:
    unexpected = [path for path in dataset_path.rglob("*") if path.is_file()]
    require(not unexpected, f"training/demo data unexpectedly present under simulator dataset path: {unexpected[:3]}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--imports-only",
        action="store_true",
        help="verify pins, suite metadata, fixed states, and EGL selection without constructing MuJoCo",
    )
    parser.add_argument("--output-json", type=Path, help="write the pure attestation JSON to a new file")
    return parser.parse_args()


def emit_report(output: dict[str, Any], output_path: Path | None) -> None:
    text = json.dumps(output, allow_nan=False, indent=2, sort_keys=True) + "\n"
    if output_path is not None:
        destination = output_path.resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("x", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
    print(text, end="")


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    cache_root = Path(os.environ.get("DUO_VLA_CACHE_ROOT", "/root/.cache/duo-vla"))
    source_root = activate_project_source_root(project_root)
    process_identity = validate_process_environment(project_root, cache_root)
    from duo_vla.runtime_integrity import content_address_eval_venv, require_matching_eval_venv

    validate_project_module_origins(
        source_root,
        {"duo_vla", "duo_vla.runtime_integrity"},
    )

    eval_venv_identity = content_address_eval_venv(cache_root / "venvs/libero-eval")
    runtime_root = cache_root / "simulators/libero"
    manifest_path = runtime_root / "manifest.json"
    require(manifest_path.is_file(), f"missing simulator manifest: {manifest_path}")
    manifest = load_strict_manifest(manifest_path)
    require(manifest["schema"] == "duo-vla-libero-simulator-v1", "unexpected simulator manifest schema")
    source_manifest = manifest["source"]
    require(isinstance(source_manifest, dict), "simulator source manifest must be an object")
    require(set(source_manifest) == {"path", "revision", "url"}, "simulator source manifest fields changed")
    expected_source_path = (runtime_root / "source").resolve()
    require(Path(source_manifest["path"]).resolve() == expected_source_path, "LIBERO source path changed")
    require(source_manifest["revision"] == EXPECTED_SOURCE_REVISION, "LIBERO source pin mismatch")
    require(source_manifest["url"] == EXPECTED_SOURCE_URL, "LIBERO source URL mismatch")

    assets_manifest = manifest["assets"]
    require(isinstance(assets_manifest, dict), "simulator assets manifest must be an object")
    require(
        set(assets_manifest) == {"path", "repo_id", "repo_type", "revision"},
        "simulator assets manifest fields changed",
    )
    expected_assets_path = (runtime_root / "assets" / EXPECTED_ASSETS_REVISION).resolve()
    require(Path(assets_manifest["path"]).resolve() == expected_assets_path, "LIBERO assets path changed")
    require(assets_manifest["revision"] == EXPECTED_ASSETS_REVISION, "LIBERO asset pin mismatch")
    require(assets_manifest["repo_id"] == EXPECTED_ASSETS_REPO, "LIBERO assets repository changed")
    require(assets_manifest["repo_type"] == "dataset", "LIBERO assets repository type changed")

    environment_manifest = manifest["environment"]
    require(isinstance(environment_manifest, dict), "simulator environment manifest must be an object")
    require(
        set(environment_manifest) == {"packages", "python", "uv_lock_sha256"},
        "simulator environment manifest fields changed",
    )
    require(
        environment_manifest["packages"] == EXPECTED_MANIFEST_PACKAGES,
        "simulator manifest package pins changed",
    )
    require(environment_manifest["python"] == "3.12.13", "simulator manifest Python pin changed")

    paths_manifest = manifest["paths"]
    require(isinstance(paths_manifest, dict), "simulator paths manifest must be an object")
    require(set(paths_manifest) == {"config", "datasets"}, "simulator paths manifest fields changed")
    expected_config_path = (runtime_root / "config/config.yaml").resolve()
    expected_dataset_path = (cache_root / "data/libero/simulator-datasets").resolve()
    require(Path(paths_manifest["config"]).resolve() == expected_config_path, "LIBERO config path changed")
    require(Path(paths_manifest["datasets"]).resolve() == expected_dataset_path, "LIBERO dataset path changed")
    require(manifest["training_data_downloaded"] is False, "simulator manifest claims training data was downloaded")
    lock_path = project_root / "envs/libero-eval/uv.lock"
    lock_sha256 = sha256_file(lock_path)
    require(lock_sha256 == EXPECTED_EVAL_LOCK_SHA256, f"LIBERO evaluator lock SHA-256 mismatch: {lock_sha256}")
    require(
        environment_manifest["uv_lock_sha256"] == lock_sha256,
        "simulator manifest and current evaluator lock differ",
    )
    source_identity = git_source_identity(expected_source_path)

    require(os.environ.get("MUJOCO_GL") == "egl", "MUJOCO_GL must be exactly 'egl'")
    require(os.environ.get("PYOPENGL_PLATFORM") == "egl", "PYOPENGL_PLATFORM must be exactly 'egl'")
    python_version = sys.version.split()[0]
    require(python_version == environment_manifest["python"], f"unexpected Python version: {python_version}")
    package_versions = {name: importlib.metadata.version(name) for name in EXPECTED_PACKAGES}
    require(package_versions == EXPECTED_PACKAGES, f"package pin mismatch: {package_versions}")
    installed_distributions = installed_distribution_identity()

    import mujoco.gl_context
    import robosuite
    import torch
    from libero.libero import benchmark, get_assets_path, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    require(mujoco.gl_context.GLContext.__module__ == "mujoco.egl", "MuJoCo did not select its EGL backend")
    require(robosuite.__version__ == "1.4.0", f"unexpected robosuite import: {robosuite.__version__}")

    assets_path = Path(get_assets_path()).resolve()
    require(assets_path == expected_assets_path, "LIBERO did not resolve the pinned assets")
    assets_identity = tree_content_identity(assets_path)
    require(assets_identity == EXPECTED_ASSETS_IDENTITY, "LIBERO asset content identity mismatch")
    dataset_path = Path(get_libero_path("datasets")).resolve()
    require(dataset_path == expected_dataset_path, "LIBERO did not resolve the isolated simulator dataset path")
    package_root = Path(importlib.util.find_spec("libero.libero").origin).resolve().parent
    require(
        Path(get_libero_path("benchmark_root")).resolve() == package_root,
        "LIBERO benchmark root does not come from the pinned package",
    )
    require(
        Path(get_libero_path("bddl_files")).resolve() == package_root / "bddl_files",
        "LIBERO BDDL root does not come from the pinned package",
    )
    require(
        Path(get_libero_path("init_states")).resolve() == package_root / "init_files",
        "LIBERO reset-state root does not come from the pinned package",
    )
    check_no_training_data(dataset_path)
    venv_root = Path(sys.prefix).resolve()
    site_packages_identity = verify_site_packages_inventory(venv_root, expected_assets_path)
    module_origins = module_origin_identity(venv_root)
    distribution_records = verify_distribution_records(venv_root)

    benchmark_map = benchmark.get_benchmark_dict()
    suites: dict[str, Any] = {}
    fixed_state_counts: dict[str, list[int]] = {}
    task_inventory: list[dict[str, Any]] = []
    for suite_name in SUITES:
        suite = benchmark_map[suite_name]()
        require(suite.n_tasks == 10, f"{suite_name} has {suite.n_tasks} tasks, expected 10")
        suites[suite_name] = suite
        counts = []
        for task_id in range(suite.n_tasks):
            task = suite.get_task(task_id)
            states = np.asarray(suite.get_task_init_states(task_id))
            counts.append(len(states))
            require(len(states) == 50, f"{suite_name} task {task_id} has {len(states)} fixed states, expected 50")
            bddl_path = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
            require(bddl_path.is_file(), f"LIBERO task BDDL is missing: {bddl_path}")
            state_hashes = [hashlib.sha256(np.ascontiguousarray(state).tobytes()).hexdigest() for state in states]
            task_inventory.append(
                {
                    "bddl_sha256": sha256_file(bddl_path),
                    "instruction": task.language,
                    "reset_count": len(states),
                    "reset_dtype": str(states.dtype),
                    "reset_shape": list(states.shape[1:]),
                    "reset_state_sha256": state_hashes,
                    "suite": suite_name,
                    "task_id": task_id,
                    "task_name": task.name,
                }
            )
        fixed_state_counts[suite_name] = counts

    task_inventory_sha256 = canonical_sha256(task_inventory)
    require(task_inventory_sha256 == EXPECTED_TASK_INVENTORY_SHA256, "LIBERO task/reset inventory changed")
    output: dict[str, Any] = {
        "schema": ATTESTATION_SCHEMA,
        "status": "ok",
        "backend": mujoco.gl_context.GLContext.__module__,
        "egl_device": os.environ.get("MUJOCO_EGL_DEVICE_ID", "0"),
        "packages": package_versions,
        "torch": torch.__version__,
        "assets": {"path": str(assets_path), **assets_identity},
        "eval_venv_identity": eval_venv_identity,
        "distribution_records": distribution_records,
        "evaluator_lock_sha256": lock_sha256,
        "installed_distributions": installed_distributions,
        "manifest_sha256": sha256_file(manifest_path),
        "module_origins": module_origins,
        "opengl": None,
        "process": process_identity,
        "project_sources": source_file_identities(project_root),
        "source": source_identity,
        "site_packages": site_packages_identity,
        "suites": {
            name: {"tasks": suites[name].n_tasks, "fixed_states_per_task": counts}
            for name, counts in fixed_state_counts.items()
        },
        "task_inventory_count": len(task_inventory),
        "task_inventory_sha256": task_inventory_sha256,
        "task_inventory": task_inventory,
        "training_data_files": 0,
        "environment_constructed": False,
    }
    if args.imports_only:
        require_matching_eval_venv(
            eval_venv_identity,
            content_address_eval_venv(cache_root / "venvs/libero-eval"),
        )
        validate_project_module_origins(
            source_root,
            {"duo_vla", "duo_vla.runtime_integrity"},
        )
        emit_report(output, args.output_json)
        return

    output["opengl"] = opengl_runtime_identity(mujoco)

    suite = suites["libero_spatial"]
    task = suite.get_task(0)
    bddl_file = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    initial_state = suite.get_task_init_states(0)[0]
    environment = OffScreenRenderEnv(
        bddl_file_name=str(bddl_file),
        camera_heights=256,
        camera_widths=256,
        camera_names=list(CAMERA_NAMES),
        control_freq=20,
        render_gpu_device_id=int(os.environ.get("MUJOCO_EGL_DEVICE_ID", "0")),
    )
    try:
        environment.seed(7)

        def fixed_reset() -> tuple[np.ndarray, dict[str, Any]]:
            environment.seed(7)
            environment.reset()
            observation = environment.set_init_state(initial_state)
            copied = {
                key: np.asarray(value).copy() if isinstance(value, np.ndarray) else value
                for key, value in observation.items()
            }
            return environment.get_sim_state().copy(), copied

        first_state, first_observation = fixed_reset()
        second_state, second_observation = fixed_reset()
        require(np.array_equal(first_state, second_state), "fixed-state simulator reset is not deterministic")
        for camera_key in CAMERA_KEYS:
            first_image = np.asarray(first_observation[camera_key]).astype(np.int16)
            second_image = np.asarray(second_observation[camera_key]).astype(np.int16)
            pixel_delta = np.abs(first_image - second_image)
            require(
                np.array_equal(first_image, second_image),
                f"fixed-state {camera_key} reset is not deterministic: "
                f"changed_values={np.count_nonzero(pixel_delta)}, max_delta={pixel_delta.max()}, "
                f"mean_delta={pixel_delta.mean():.6f}",
            )
        cameras = validate_observation(second_observation)
        action = np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0], dtype=np.float32)
        observation, reward, done, info = environment.step(action)
        post_step_cameras = validate_observation(observation)
        require(environment.env.action_dim == 7, f"unexpected LIBERO action dimension: {environment.env.action_dim}")
        require(np.isclose(environment.env.control_timestep, 0.05), "LIBERO control frequency is not 20 Hz")
        output["environment_constructed"] = True
        output["smoke"] = {
            "suite": "libero_spatial",
            "task_id": 0,
            "task": task.name,
            "cameras": cameras,
            "action_dimension": environment.env.action_dim,
            "control_frequency_hz": round(1.0 / environment.env.control_timestep),
            "post_step_reward": float(reward),
            "post_step_done": bool(done),
            "post_step_info": info,
            "post_step_cameras": post_step_cameras,
            "fixed_reset_deterministic": True,
        }
    finally:
        environment.close()
    require_matching_eval_venv(
        eval_venv_identity,
        content_address_eval_venv(cache_root / "venvs/libero-eval"),
    )
    validate_project_module_origins(
        source_root,
        {"duo_vla", "duo_vla.runtime_integrity"},
    )
    emit_report(output, args.output_json)


if __name__ == "__main__":
    main()
