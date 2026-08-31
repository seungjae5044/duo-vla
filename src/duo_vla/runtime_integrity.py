"""Fail-closed process and virtual-environment identity helpers.

The tree walker deliberately never follows symlinks.  It binds each opened
directory/file descriptor to the corresponding directory entry before and
after reading so a concurrent replacement cannot silently change the reported
identity.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

TRAIN_VENV_IDENTITY_SCHEMA = "duo-vla-train-venv-identity-v2"
EVAL_VENV_IDENTITY_SCHEMA = "duo-vla-eval-venv-identity-v1"
BASE_PYTHON_RUNTIME_IDENTITY_SCHEMA = "duo-vla-base-python-runtime-identity-v1"

_IDENTITY_FIELDS = ("st_dev", "st_ino", "st_mode", "st_size", "st_mtime_ns", "st_ctime_ns", "st_nlink")
_DIRECTORY_OPEN_FLAGS = os.O_RDONLY | os.O_NONBLOCK | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
_FILE_OPEN_FLAGS = os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW | os.O_CLOEXEC
_TORCHRUN_ENVIRONMENT_FIELDS = frozenset(
    {
        "GROUP_RANK",
        "GROUP_WORLD_SIZE",
        "LOCAL_RANK",
        "LOCAL_WORLD_SIZE",
        "MASTER_ADDR",
        "MASTER_PORT",
        "RANK",
        "ROLE_NAME",
        "ROLE_RANK",
        "ROLE_WORLD_SIZE",
        "TORCHELASTIC_ERROR_FILE",
        "TORCHELASTIC_MAX_RESTARTS",
        "TORCHELASTIC_RESTART_COUNT",
        "TORCHELASTIC_RUN_ID",
        "TORCHELASTIC_SIGNALS_TO_HANDLE",
        "TORCHELASTIC_USE_AGENT_STORE",
        "WORLD_SIZE",
    }
)


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, allow_nan=False, separators=(",", ":"), sort_keys=True).encode()
    return hashlib.sha256(payload).hexdigest()


def _same_identity(left: os.stat_result, right: os.stat_result) -> bool:
    return all(getattr(left, field) == getattr(right, field) for field in _IDENTITY_FIELDS)


def _open_bound_directory(parent_fd: int, name: str, *, context: str) -> tuple[int, os.stat_result]:
    before = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    if not stat.S_ISDIR(before.st_mode):
        raise RuntimeError(f"train-venv ancestor must be a real directory: {context}")
    descriptor = os.open(name, _DIRECTORY_OPEN_FLAGS, dir_fd=parent_fd)
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISDIR(opened.st_mode) or not _same_identity(before, opened):
            raise RuntimeError(f"train-venv ancestor changed while opening: {context}")
        return descriptor, opened
    except BaseException:
        os.close(descriptor)
        raise


def _verify_directory_binding(
    parent_fd: int,
    name: str,
    descriptor: int,
    expected: os.stat_result,
    *,
    context: str,
) -> None:
    after_descriptor = os.fstat(descriptor)
    after_path = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    if not _same_identity(expected, after_descriptor) or not _same_identity(expected, after_path):
        raise RuntimeError(f"train-venv ancestor changed while reading: {context}")


def _open_absolute_directory_chain(path: Path) -> tuple[list[int], list[tuple[int, str, int, os.stat_result, str]]]:
    if not path.is_absolute():
        raise RuntimeError(f"train venv must be absolute: {path}")
    descriptors = [os.open("/", _DIRECTORY_OPEN_FLAGS)]
    bindings: list[tuple[int, str, int, os.stat_result, str]] = []
    try:
        for index, name in enumerate(path.parts[1:]):
            context = "/" + "/".join(path.parts[1 : index + 2])
            child_fd, child_identity = _open_bound_directory(descriptors[-1], name, context=context)
            bindings.append((descriptors[-1], name, child_fd, child_identity, context))
            descriptors.append(child_fd)
        return descriptors, bindings
    except BaseException:
        for descriptor in reversed(descriptors):
            os.close(descriptor)
        raise


def _verify_absolute_directory_chain(
    descriptors: Sequence[int],
    bindings: Sequence[tuple[int, str, int, os.stat_result, str]],
) -> None:
    if not _same_identity(os.fstat(descriptors[0]), os.stat("/", follow_symlinks=False)):
        raise RuntimeError("filesystem root changed while hashing the train venv")
    for parent_fd, name, child_fd, expected, context in reversed(bindings):
        _verify_directory_binding(parent_fd, name, child_fd, expected, context=context)


def _hash_regular_file(parent_fd: int, name: str, before: os.stat_result, *, context: str) -> tuple[str, int]:
    descriptor = os.open(name, _FILE_OPEN_FLAGS, dir_fd=parent_fd)
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode) or not _same_identity(before, opened):
            raise RuntimeError(f"train-venv file changed while opening: {context}")
        digest = hashlib.sha256()
        size = 0
        while block := os.read(descriptor, 8 * 1024 * 1024):
            digest.update(block)
            size += len(block)
        after_descriptor = os.fstat(descriptor)
        after_path = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        if not _same_identity(opened, after_descriptor) or not _same_identity(opened, after_path):
            raise RuntimeError(f"train-venv file changed while reading: {context}")
        if size != opened.st_size:
            raise RuntimeError(f"train-venv file size changed while reading: {context}")
        return digest.hexdigest(), size
    finally:
        os.close(descriptor)


def _read_symlink(parent_fd: int, name: str, before: os.stat_result, *, context: str) -> str:
    target = os.readlink(name, dir_fd=parent_fd)
    after = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    if not stat.S_ISLNK(after.st_mode) or not _same_identity(before, after):
        raise RuntimeError(f"train-venv symlink changed while reading: {context}")
    if os.readlink(name, dir_fd=parent_fd) != target:
        raise RuntimeError(f"train-venv symlink target changed while reading: {context}")
    return target


def _is_site_packages_root(parts: tuple[str, ...]) -> bool:
    return bool(parts) and parts[-1] in {"site-packages", "dist-packages"}


def _is_forbidden_customization_hook(prefix: tuple[str, ...], name: str) -> bool:
    if not _is_site_packages_root(prefix):
        return False
    lowered = name.lower()
    return any(lowered == stem or lowered.startswith(f"{stem}.") for stem in ("sitecustomize", "usercustomize"))


def _walk_venv_directory(
    descriptor: int,
    prefix: tuple[str, ...],
    records: list[dict[str, Any]],
    startup_hooks: list[dict[str, Any]],
) -> None:
    directory_before = os.fstat(descriptor)
    names_before = sorted(os.listdir(descriptor))
    for name in names_before:
        context = "/".join((*prefix, name))
        observed = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
        mode = stat.S_IMODE(observed.st_mode)
        if _is_forbidden_customization_hook(prefix, name):
            raise RuntimeError(f"train venv contains a forbidden Python startup hook: {context}")
        if stat.S_ISDIR(observed.st_mode):
            records.append({"mode": mode, "path": context, "type": "directory"})
            child_fd, child_identity = _open_bound_directory(descriptor, name, context=context)
            try:
                _walk_venv_directory(child_fd, (*prefix, name), records, startup_hooks)
                _verify_directory_binding(descriptor, name, child_fd, child_identity, context=context)
            finally:
                os.close(child_fd)
        elif stat.S_ISREG(observed.st_mode):
            digest, size = _hash_regular_file(descriptor, name, observed, context=context)
            record = {"bytes": size, "mode": mode, "path": context, "sha256": digest, "type": "file"}
            records.append(record)
            if _is_site_packages_root(prefix) and name.lower().endswith(".pth"):
                startup_hooks.append(record)
        elif stat.S_ISLNK(observed.st_mode):
            target = _read_symlink(descriptor, name, observed, context=context)
            record = {"mode": mode, "path": context, "target": target, "type": "symlink"}
            records.append(record)
            if _is_site_packages_root(prefix) and name.lower().endswith(".pth"):
                startup_hooks.append(record)
        else:
            raise RuntimeError(f"train-venv entry must be a regular file, symlink, or real directory: {context}")
    if names_before != sorted(os.listdir(descriptor)) or not _same_identity(directory_before, os.fstat(descriptor)):
        raise RuntimeError(f"train-venv directory changed during inventory: {'/'.join(prefix) or '.'}")


def _content_address_venv(root: str | Path, *, schema: str) -> dict[str, Any]:
    """Return an exact, no-follow identity for every venv file and symlink."""

    root_path = Path(os.path.abspath(root))
    descriptors, bindings = _open_absolute_directory_chain(root_path)
    root_fd = descriptors[-1]
    try:
        root_before = os.fstat(root_fd)
        records: list[dict[str, Any]] = [{"mode": stat.S_IMODE(root_before.st_mode), "path": ".", "type": "directory"}]
        startup_hooks: list[dict[str, Any]] = []
        _walk_venv_directory(root_fd, (), records, startup_hooks)
        records.sort(key=lambda record: str(record["path"]))
        startup_hooks.sort(key=lambda record: str(record["path"]))
        metadata_records = [{name: value for name, value in record.items() if name != "sha256"} for record in records]
        content_inventory_sha256 = canonical_sha256(records)
        tree_metadata_sha256 = canonical_sha256(metadata_records)
        files_verified = sum(record["type"] == "file" for record in records)
        symlinks_verified = sum(record["type"] == "symlink" for record in records)
        total_bytes = sum(int(record.get("bytes", 0)) for record in records)
        startup_hooks_sha256 = canonical_sha256(startup_hooks)
        base_python_runtime = content_address_base_python_runtime(root_path)
        root_sha256 = canonical_sha256(
            {
                "base_python_runtime_root_sha256": base_python_runtime["root_sha256"],
                "content_inventory_sha256": content_inventory_sha256,
                "files_verified": files_verified,
                "schema": schema,
                "startup_hooks_sha256": startup_hooks_sha256,
                "symlinks_verified": symlinks_verified,
                "total_bytes": total_bytes,
                "tree_metadata_sha256": tree_metadata_sha256,
            }
        )
        if not _same_identity(root_before, os.fstat(root_fd)):
            raise RuntimeError(f"train venv root changed while hashing: {root_path}")
        _verify_absolute_directory_chain(descriptors, bindings)
        return {
            "base_python_runtime": base_python_runtime,
            "content_inventory_sha256": content_inventory_sha256,
            "files_verified": files_verified,
            "root": str(root_path),
            "root_sha256": root_sha256,
            "schema": schema,
            "startup_hooks": [record["path"] for record in startup_hooks],
            "startup_hooks_sha256": startup_hooks_sha256,
            "symlinks_verified": symlinks_verified,
            "total_bytes": total_bytes,
            "tree_metadata_sha256": tree_metadata_sha256,
        }
    finally:
        for descriptor in reversed(descriptors):
            os.close(descriptor)


def content_address_train_venv(root: str | Path) -> dict[str, Any]:
    return _content_address_venv(root, schema=TRAIN_VENV_IDENTITY_SCHEMA)


def content_address_eval_venv(root: str | Path) -> dict[str, Any]:
    return _content_address_venv(root, schema=EVAL_VENV_IDENTITY_SCHEMA)


def _content_address_tree(root_path: Path, *, schema: str) -> dict[str, Any]:
    descriptors, bindings = _open_absolute_directory_chain(root_path)
    root_fd = descriptors[-1]
    try:
        root_before = os.fstat(root_fd)
        records: list[dict[str, Any]] = [{"mode": stat.S_IMODE(root_before.st_mode), "path": ".", "type": "directory"}]
        startup_hooks: list[dict[str, Any]] = []
        _walk_venv_directory(root_fd, (), records, startup_hooks)
        records.sort(key=lambda record: str(record["path"]))
        startup_hooks.sort(key=lambda record: str(record["path"]))
        metadata_records = [{name: value for name, value in record.items() if name != "sha256"} for record in records]
        identity = {
            "content_inventory_sha256": canonical_sha256(records),
            "files_verified": sum(record["type"] == "file" for record in records),
            "schema": schema,
            "startup_hooks": [record["path"] for record in startup_hooks],
            "startup_hooks_sha256": canonical_sha256(startup_hooks),
            "symlinks_verified": sum(record["type"] == "symlink" for record in records),
            "total_bytes": sum(int(record.get("bytes", 0)) for record in records),
            "tree_metadata_sha256": canonical_sha256(metadata_records),
        }
        if not _same_identity(root_before, os.fstat(root_fd)):
            raise RuntimeError(f"base Python root changed while hashing: {root_path}")
        _verify_absolute_directory_chain(descriptors, bindings)
        return identity
    finally:
        for descriptor in reversed(descriptors):
            os.close(descriptor)


def _stable_regular_file(path: Path, *, context: str) -> dict[str, Any]:
    parent = path.parent
    parent_descriptors, parent_bindings = _open_absolute_directory_chain(parent)
    parent_fd = parent_descriptors[-1]
    try:
        before = os.stat(path.name, dir_fd=parent_fd, follow_symlinks=False)
        if not stat.S_ISREG(before.st_mode):
            raise RuntimeError(f"{context} must be a regular file: {path}")
        digest, size = _hash_regular_file(parent_fd, path.name, before, context=context)
        _verify_absolute_directory_chain(parent_descriptors, parent_bindings)
        return {"bytes": size, "sha256": digest}
    finally:
        for descriptor in reversed(parent_descriptors):
            os.close(descriptor)


def content_address_base_python_runtime(venv_root: str | Path) -> dict[str, Any]:
    """Bind a venv launcher to the complete external CPython base installation."""

    venv_path = Path(os.path.abspath(venv_root))
    launcher = venv_path / "bin/python"
    launcher_before = os.lstat(launcher)
    if not stat.S_ISLNK(launcher_before.st_mode):
        raise RuntimeError(f"venv Python launcher must be a symbolic link: {launcher}")
    launcher_target = os.readlink(launcher)
    resolved_executable = launcher.resolve(strict=True)
    base_prefix = resolved_executable.parent.parent
    if resolved_executable.parent != base_prefix / "bin":
        raise RuntimeError("resolved venv Python executable is not below the canonical base bin directory")

    pyvenv_path = venv_path / "pyvenv.cfg"
    pyvenv_identity = _stable_regular_file(pyvenv_path, context="pyvenv.cfg")
    try:
        pyvenv_text = pyvenv_path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise RuntimeError(f"cannot read UTF-8 pyvenv.cfg: {pyvenv_path}") from exc
    homes = [
        line.partition("=")[2].strip() for line in pyvenv_text.splitlines() if line.partition("=")[0].strip() == "home"
    ]
    if len(homes) != 1 or not homes[0]:
        raise RuntimeError("pyvenv.cfg must contain exactly one non-empty home entry")
    configured_home = Path(homes[0])
    if not configured_home.is_absolute():
        raise RuntimeError("pyvenv.cfg home must be absolute")
    configured_home_resolved = configured_home.resolve(strict=True)
    if configured_home_resolved != base_prefix / "bin":
        raise RuntimeError("pyvenv.cfg home does not resolve to the executable base bin directory")

    executable_identity = _stable_regular_file(resolved_executable, context="resolved base Python executable")
    tree_identity = _content_address_tree(base_prefix, schema=BASE_PYTHON_RUNTIME_IDENTITY_SCHEMA)
    launcher_after = os.lstat(launcher)
    if not _same_identity(launcher_before, launcher_after) or os.readlink(launcher) != launcher_target:
        raise RuntimeError("venv Python launcher changed while authenticating its base runtime")
    if launcher.resolve(strict=True) != resolved_executable:
        raise RuntimeError("venv Python launcher resolution changed while authenticating its base runtime")
    if _stable_regular_file(pyvenv_path, context="pyvenv.cfg") != pyvenv_identity:
        raise RuntimeError("pyvenv.cfg changed while authenticating the base runtime")

    identity = {
        **tree_identity,
        "base_prefix": str(base_prefix),
        "configured_home": str(configured_home),
        "configured_home_resolved": str(configured_home_resolved),
        "pyvenv_cfg_bytes": pyvenv_identity["bytes"],
        "pyvenv_cfg_sha256": pyvenv_identity["sha256"],
        "resolved_executable": str(resolved_executable),
        "resolved_executable_bytes": executable_identity["bytes"],
        "resolved_executable_sha256": executable_identity["sha256"],
        "venv_python": str(launcher),
        "venv_python_link_target": launcher_target,
        "venv_root": str(venv_path),
    }
    identity["root_sha256"] = canonical_sha256(identity)
    return identity


def require_matching_base_python_runtime(expected: Any, observed: dict[str, Any]) -> dict[str, Any]:
    fields = {
        "base_prefix",
        "configured_home",
        "configured_home_resolved",
        "content_inventory_sha256",
        "files_verified",
        "pyvenv_cfg_bytes",
        "pyvenv_cfg_sha256",
        "resolved_executable",
        "resolved_executable_bytes",
        "resolved_executable_sha256",
        "root_sha256",
        "schema",
        "startup_hooks",
        "startup_hooks_sha256",
        "symlinks_verified",
        "total_bytes",
        "tree_metadata_sha256",
        "venv_python",
        "venv_python_link_target",
        "venv_root",
    }
    if not isinstance(expected, dict) or set(expected) != fields:
        raise RuntimeError("base-Python runtime identity fields differ")
    if set(observed) != fields:
        raise RuntimeError("live base-Python runtime identity fields differ")
    if expected.get("schema") != BASE_PYTHON_RUNTIME_IDENTITY_SCHEMA:
        raise RuntimeError("base-Python runtime identity schema differs")
    for name in (
        "content_inventory_sha256",
        "pyvenv_cfg_sha256",
        "resolved_executable_sha256",
        "root_sha256",
        "startup_hooks_sha256",
        "tree_metadata_sha256",
    ):
        value = expected.get(name)
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise RuntimeError(f"base-Python runtime {name} is invalid")
    for name in ("files_verified", "pyvenv_cfg_bytes", "resolved_executable_bytes", "symlinks_verified", "total_bytes"):
        value = expected.get(name)
        if type(value) is not int or value < 0:
            raise RuntimeError(f"base-Python runtime {name} is invalid")
    unsigned = {name: value for name, value in expected.items() if name != "root_sha256"}
    if canonical_sha256(unsigned) != expected["root_sha256"]:
        raise RuntimeError("base-Python runtime semantic root hash differs")
    if expected != observed:
        raise RuntimeError("live base-Python runtime differs from its recorded identity")
    return observed


def require_matching_train_venv(expected: Any, observed: dict[str, Any]) -> dict[str, Any]:
    """Validate the strict schema and require a byte-for-byte environment match."""

    fields = {
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
    if not isinstance(expected, dict) or set(expected) != fields:
        raise RuntimeError("checkpoint train-venv identity fields differ")
    if set(observed) != fields:
        raise RuntimeError("live train-venv identity fields differ")
    for name in (
        "content_inventory_sha256",
        "root_sha256",
        "startup_hooks_sha256",
        "tree_metadata_sha256",
    ):
        value = expected.get(name)
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise RuntimeError(f"checkpoint train-venv {name} is invalid")
    if expected.get("schema") != TRAIN_VENV_IDENTITY_SCHEMA:
        raise RuntimeError("checkpoint train-venv schema differs")
    require_matching_base_python_runtime(expected.get("base_python_runtime"), observed.get("base_python_runtime", {}))
    for name in ("files_verified", "symlinks_verified", "total_bytes"):
        value = expected.get(name)
        if type(value) is not int or value < 0:
            raise RuntimeError(f"checkpoint train-venv {name} is invalid")
    root_payload = {
        "base_python_runtime_root_sha256": expected["base_python_runtime"]["root_sha256"],
        "content_inventory_sha256": expected["content_inventory_sha256"],
        "files_verified": expected["files_verified"],
        "schema": expected["schema"],
        "startup_hooks_sha256": expected["startup_hooks_sha256"],
        "symlinks_verified": expected["symlinks_verified"],
        "total_bytes": expected["total_bytes"],
        "tree_metadata_sha256": expected["tree_metadata_sha256"],
    }
    if canonical_sha256(root_payload) != expected["root_sha256"]:
        raise RuntimeError("checkpoint train-venv semantic root hash differs")
    if expected != observed:
        raise RuntimeError("live train venv differs from the checkpoint training environment")
    return observed


def require_matching_eval_venv(expected: Any, observed: dict[str, Any]) -> dict[str, Any]:
    fields = {
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
    if not isinstance(expected, dict) or set(expected) != fields or set(observed) != fields:
        raise RuntimeError("eval-venv identity fields differ")
    if expected.get("schema") != EVAL_VENV_IDENTITY_SCHEMA:
        raise RuntimeError("eval-venv identity schema differs")
    require_matching_base_python_runtime(expected.get("base_python_runtime"), observed.get("base_python_runtime", {}))
    for name in (
        "content_inventory_sha256",
        "root_sha256",
        "startup_hooks_sha256",
        "tree_metadata_sha256",
    ):
        value = expected.get(name)
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise RuntimeError(f"eval-venv {name} is invalid")
    for name in ("files_verified", "symlinks_verified", "total_bytes"):
        value = expected.get(name)
        if type(value) is not int or value < 0:
            raise RuntimeError(f"eval-venv {name} is invalid")
    root_payload = {
        "base_python_runtime_root_sha256": expected["base_python_runtime"]["root_sha256"],
        "content_inventory_sha256": expected["content_inventory_sha256"],
        "files_verified": expected["files_verified"],
        "schema": expected["schema"],
        "startup_hooks_sha256": expected["startup_hooks_sha256"],
        "symlinks_verified": expected["symlinks_verified"],
        "total_bytes": expected["total_bytes"],
        "tree_metadata_sha256": expected["tree_metadata_sha256"],
    }
    if canonical_sha256(root_payload) != expected["root_sha256"]:
        raise RuntimeError("eval-venv semantic root hash differs")
    if expected != observed:
        raise RuntimeError("live eval venv differs from its recorded identity")
    return observed


def validate_torchrun_rank_environment(
    environment: Mapping[str, str],
    *,
    required: bool,
) -> dict[str, Any]:
    """Validate only torchrun-created rank state; caller rank variables are scrubbed by launchers."""

    topology = {
        "group_world_size": 1,
        "local_rank_equals_rank": True,
        "local_world_size": 2,
        "role_world_size": 2,
        "world_size": 2,
    }
    present = _TORCHRUN_ENVIRONMENT_FIELDS & set(environment)
    if not present:
        if required:
            raise RuntimeError("canonical torchrun rank environment is missing")
        return topology
    missing = sorted(_TORCHRUN_ENVIRONMENT_FIELDS - set(environment))
    if missing:
        raise RuntimeError(f"canonical torchrun rank environment is incomplete: {missing}")
    rank = environment["RANK"]
    local_rank = environment["LOCAL_RANK"]
    required_values = {
        "GROUP_RANK": "0",
        "GROUP_WORLD_SIZE": "1",
        "LOCAL_WORLD_SIZE": "2",
        "ROLE_NAME": "default",
        "ROLE_RANK": rank,
        "ROLE_WORLD_SIZE": "2",
        "TORCHELASTIC_MAX_RESTARTS": "0",
        "TORCHELASTIC_RESTART_COUNT": "0",
        "WORLD_SIZE": "2",
    }
    mismatches = {
        name: {"expected": value, "observed": environment.get(name)}
        for name, value in required_values.items()
        if environment.get(name) != value
    }
    if rank not in {"0", "1"} or local_rank != rank:
        mismatches["rank"] = {"expected": "RANK=LOCAL_RANK in {0,1}", "observed": [rank, local_rank]}
    port = environment["MASTER_PORT"]
    if not port.isdecimal() or not 1 <= int(port) <= 65535:
        mismatches["MASTER_PORT"] = {"expected": "1..65535", "observed": port}
    if not environment["MASTER_ADDR"]:
        mismatches["MASTER_ADDR"] = {"expected": "non-empty", "observed": environment["MASTER_ADDR"]}
    if not environment["TORCHELASTIC_RUN_ID"]:
        mismatches["TORCHELASTIC_RUN_ID"] = {"expected": "non-empty", "observed": ""}
    if environment["TORCHELASTIC_USE_AGENT_STORE"] not in {"True", "False"}:
        mismatches["TORCHELASTIC_USE_AGENT_STORE"] = {
            "expected": "True or False",
            "observed": environment["TORCHELASTIC_USE_AGENT_STORE"],
        }
    error_file = environment["TORCHELASTIC_ERROR_FILE"]
    if error_file and not Path(error_file).is_absolute():
        mismatches["TORCHELASTIC_ERROR_FILE"] = {"expected": "empty or absolute", "observed": error_file}
    if not environment["TORCHELASTIC_SIGNALS_TO_HANDLE"]:
        mismatches["TORCHELASTIC_SIGNALS_TO_HANDLE"] = {"expected": "non-empty", "observed": ""}
    if mismatches:
        raise RuntimeError(f"torchrun rank environment differs from TP=2 standalone launch: {mismatches}")
    return topology


def static_environment_identity(environment: Mapping[str, str]) -> dict[str, Any]:
    """Return a canonical identity for the rank-independent launcher environment."""

    ordered = dict(sorted(environment.items()))
    return {"environment": ordered, "sha256": canonical_sha256(ordered)}
