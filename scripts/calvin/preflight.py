#!/usr/bin/env python3
"""Fail-closed checks for the pinned official CALVIN evaluator runtime."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import json
import os
import platform
import re
import site
import sqlite3
import stat
import struct
import subprocess
import sys
import sysconfig
import zlib
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

CALVIN_REVISION = "fa03f01f19c65920e18cf37398a9ce859274af76"
CALVIN_ENV_REVISION = "1431a46bd36bde5903fb6345e68b5ccc30def666"
CALVIN_TACTO_REVISION = "dd53360d9a8c186f0d6439372ec0be0fa5e21731"
SEQUENCE_SHA256 = "90191d9ac76baecb4f292ab766bbbf3ae65dbf43a83bbe5c376d99db10fd6446"
PYTHON_VERSION = "3.8.20"

ATTESTATION_SCHEMA = "duovla-calvin-official-runtime-data-attestation-v2"
RUNTIME_ATTESTATION_SCHEMA = "duovla-calvin-evaluator-runtime-attestation-v2"
DATASET_MANIFEST_SCHEMA = "duo-vla-calvin-dataset-manifest-v4"
MEMBER_INDEX_SCHEMA = "duo-vla-calvin-member-index-v2"
ARCHIVE_READER_SCHEMA = "duo-vla-calvin-archive-reader-v1"
STATE_ACTION_SIDECAR_SCHEMA = "duo-vla-calvin-state-action-sidecar-v1"
STORAGE_IDENTITY_SCHEMA = "duo-vla-calvin-storage-identity-v1"
LEGACY_DATASET_MANIFEST_SCHEMA = "duo-vla-calvin-dataset-manifest-v3"
LEGACY_MEMBER_INDEX_SCHEMA = "duo-vla-calvin-member-index-v1"
ARCHIVE_SHA256 = "c2036c67eb4c06966af1d1e1665bdb572c69e1404f5e77ffd46b384ff2b79f74"
ARCHIVE_BYTES = 555_309_812_705
ARCHIVE_URL = "http://calvin.cs.uni-freiburg.de/dataset/task_ABC_D.zip"
CHECKSUM_URL = "http://calvin.cs.uni-freiburg.de/dataset/sha256sum.txt"
ARCHIVE_NAME = "task_ABC_D.zip"
MEMBER_INDEX_NAME = "task_ABC_D.members-v2.sqlite3"
LEGACY_MEMBER_INDEX_NAME = "task_ABC_D.members.sqlite3"
MANIFEST_NAME = "task_ABC_D.manifest.json"

CENTRAL_DIRECTORY_OFFSET = 555_080_601_096
CENTRAL_DIRECTORY_BYTES = 229_211_511
CENTRAL_DIRECTORY_SHA256 = "b4f79bda7f6b966b51aa419badd0f7db7a8972a7b58d6d342af60aceff0ea31b"
ZIP64_TRAILER_BYTES = 98
ZIP64_TRAILER_SHA256 = "c25191e6eb9871238d396e9cfecbc96cb75ba78566654c04a6bdec0bfa359ffd"
ZIP64_VERSION_MADE = 798
MEMBER_COUNT = 1_894_126
FILE_MEMBER_COUNT = 1_894_106
NPZ_MEMBER_COUNT = 1_894_067
DIRECTORY_MEMBER_COUNT = 20
ARCHIVE_DIRECT_VERIFICATION = (
    "full-archive-sha256+streamed-central-inventory+zip64-local-header+raw-deflate-eof-length-crc32-logical-sha256"
)
LIVE_ARCHIVE_VERIFICATION = "live-size+central-directory-sha256+zip64-trailer; full SHA-256 verified by v4 preparation"

VALIDATION_ANNOTATIONS_SHA256 = "d14b0bf960f65158c5815b10ff11d2464073f26ac9b62c9f54e5c90ca352ccfa"
TASK_ORACLE_SHA256 = "6e905de3ca05118efdd8a51f8a7756ec6e61ffdb2b9b6a2843f0b7e0e9e51dcf"

DATASET_CRITICAL_FILES = (
    "training/ep_start_end_ids.npy",
    "training/lang_annotations/auto_lang_ann.npy",
    "training/scene_info.npy",
    "training/.hydra/merged_config.yaml",
    "validation/ep_start_end_ids.npy",
    "validation/.hydra/merged_config.yaml",
)
VALIDATION_CRITICAL_FILES = (
    "validation/ep_start_end_ids.npy",
    "validation/.hydra/merged_config.yaml",
)
TRAINING_METADATA_FILES = (
    "ep_start_end_ids.npy",
    "lang_annotations/auto_lang_ann.npy",
    "scene_info.npy",
    ".hydra/merged_config.yaml",
)

_V4_MANIFEST_FIELDS = {
    "archive",
    "checksum_url",
    "content_sha256",
    "critical_files",
    "dataset",
    "schema",
    "storage",
}
_V3_MANIFEST_FIELDS = {
    "archive",
    "checksum_url",
    "content_sha256",
    "critical_files",
    "dataset",
    "extraction",
    "schema",
}
_MEMBERS_TABLE_SQL = (
    "CREATE TABLE members("
    "path TEXT PRIMARY KEY, "
    "kind INTEGER NOT NULL CHECK(kind IN (0,1)), "
    "split TEXT CHECK(split IS NULL OR split IN ('training','validation')), "
    "global_index INTEGER CHECK(global_index IS NULL OR global_index >= 0), "
    "local_header_offset INTEGER NOT NULL CHECK(local_header_offset >= 0), "
    "data_offset INTEGER NOT NULL CHECK(data_offset >= -1), "
    "compressed_bytes INTEGER NOT NULL CHECK(compressed_bytes >= 0), "
    "logical_bytes INTEGER NOT NULL CHECK(logical_bytes >= 0), "
    "method INTEGER NOT NULL, "
    "version_needed INTEGER NOT NULL CHECK(version_needed >= 10 AND version_needed <= 65535), "
    "flags INTEGER NOT NULL, "
    "crc32 INTEGER NOT NULL CHECK(crc32 >= 0 AND crc32 <= 4294967295), "
    "logical_sha256 BLOB NOT NULL CHECK(length(logical_sha256) = 32), "
    "state_action_row INTEGER CHECK(state_action_row IS NULL OR state_action_row >= 0)"
    ") WITHOUT ROWID"
)
_METADATA_TABLE_SQL = "CREATE TABLE metadata(name TEXT PRIMARY KEY, value TEXT NOT NULL) WITHOUT ROWID"
_LOCAL_HEADER_INDEX_SQL = "CREATE INDEX members_local_header ON members(local_header_offset)"
_EPISODE_INDEX_SQL = "CREATE UNIQUE INDEX members_episode ON members(split,global_index) WHERE global_index IS NOT NULL"
_PHYSICAL_INDEX_SQL = "CREATE INDEX members_physical ON members(data_offset)"
_EXPECTED_MEMBER_COLUMNS = [
    (0, "path", "TEXT", 1, None, 1),
    (1, "kind", "INTEGER", 1, None, 0),
    (2, "split", "TEXT", 0, None, 0),
    (3, "global_index", "INTEGER", 0, None, 0),
    (4, "local_header_offset", "INTEGER", 1, None, 0),
    (5, "data_offset", "INTEGER", 1, None, 0),
    (6, "compressed_bytes", "INTEGER", 1, None, 0),
    (7, "logical_bytes", "INTEGER", 1, None, 0),
    (8, "method", "INTEGER", 1, None, 0),
    (9, "version_needed", "INTEGER", 1, None, 0),
    (10, "flags", "INTEGER", 1, None, 0),
    (11, "crc32", "INTEGER", 1, None, 0),
    (12, "logical_sha256", "BLOB", 1, None, 0),
    (13, "state_action_row", "INTEGER", 0, None, 0),
]
_EXPECTED_METADATA_COLUMNS = [
    (0, "name", "TEXT", 1, None, 1),
    (1, "value", "TEXT", 1, None, 0),
]
_INDEX_METADATA_BASE_NAMES = {
    "archive_bytes",
    "archive_root",
    "archive_sha256",
    "central_directory_bytes",
    "central_directory_offset",
    "central_directory_sha256",
    "central_directory_zip64",
    "compressed_bytes",
    "directory_member_count",
    "file_member_count",
    "member_count",
    "member_inventory_sha256",
    "npz_member_count",
    "reader_schema",
    "schema",
    "state_action_sidecar_schema",
    "state_action_sidecar_status",
    "status",
    "uncompressed_bytes",
}
_LOCAL_HEADER = struct.Struct("<I5H3I2H")
_EXTRA_HEADER = struct.Struct("<HH")
_LOCAL_SIGNATURE = 0x04034B50
_ZIP64_EXTRA_ID = 0x0001
_EXTENDED_TIMESTAMP_EXTRA_ID = 0x5455
_UNIX_UID_GID_EXTRA_ID = 0x7875
_ALLOWED_EXTRA_IDS = {_ZIP64_EXTRA_ID, _EXTENDED_TIMESTAMP_EXTRA_ID, _UNIX_UID_GID_EXTRA_ID}
_UINT32_MAX = (1 << 32) - 1
_EMPTY_SHA256 = hashlib.sha256(b"").digest()
_ZERO_SHA256 = b"\0" * 32
_EPISODE_MEMBER_RE = re.compile(r"^(training|validation)/episode_([0-9]{7})\.npz$")
_ZIP64_EOCD_PREFIX = struct.Struct("<IQ")
_ZIP64_EOCD_BODY = struct.Struct("<HHIIQQQQ")
_ZIP64_LOCATOR = struct.Struct("<IIQI")
_EOCD = struct.Struct("<IHHHHIIH")
_ZIP64_EOCD_SIGNATURE = 0x06064B50
_ZIP64_LOCATOR_SIGNATURE = 0x07064B50
_EOCD_SIGNATURE = 0x06054B50

_SCRIPT_SOURCE_NAMES = ("evaluate_calvin.py", "calvin_bridge.py", "preflight.py")
_CANONICAL_STATIC_ENVIRONMENT = {
    "CUDA_VISIBLE_DEVICES": "0",
    "EGL_PLATFORM": "surfaceless",
    "EGL_VISIBLE_DEVICES": "0",
    "LANG": "C.UTF-8",
    "LC_ALL": "C.UTF-8",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "OMP_DYNAMIC": "FALSE",
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "PYOPENGL_PLATFORM": "egl",
    "PYTHONHASHSEED": "0",
    "PYTHONNOUSERSITE": "1",
    "TZ": "UTC",
}
_CANONICAL_DYNAMIC_ENVIRONMENT = ("CALVIN_SOURCE_ROOT", "DUO_VLA_CACHE_ROOT", "PATH")
_FORBIDDEN_EVALUATOR_ENVIRONMENT = (
    "DISPLAY",
    "LD_LIBRARY_PATH",
    "LD_PRELOAD",
    "NVIDIA_VISIBLE_DEVICES",
    "PYTHONPATH",
)
_RUNTIME_ENVIRONMENT_NAMES = tuple(
    sorted(
        set(_CANONICAL_STATIC_ENVIRONMENT) | set(_CANONICAL_DYNAMIC_ENVIRONMENT) | set(_FORBIDDEN_EVALUATOR_ENVIRONMENT)
    )
)
_PACKAGE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*\Z")
_SCRIPT_DIR = Path(__file__).resolve().parent


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def canonical_evaluator_environment() -> dict[str, str]:
    """Return the one environment accepted by official simulator processes."""

    source_root = os.environ.get("CALVIN_SOURCE_ROOT")
    cache_root = os.environ.get("DUO_VLA_CACHE_ROOT")
    require(isinstance(source_root, str) and bool(source_root), "CALVIN_SOURCE_ROOT must be set")
    require(isinstance(cache_root, str) and bool(cache_root), "DUO_VLA_CACHE_ROOT must be set")
    require(
        source_root == str(Path(source_root).resolve()) and Path(source_root).is_dir(),
        "CALVIN_SOURCE_ROOT must be an existing canonical absolute directory",
    )
    require(
        cache_root == str(Path(cache_root).resolve()) and Path(cache_root).is_dir(),
        "DUO_VLA_CACHE_ROOT must be an existing canonical absolute directory",
    )
    environment = {
        **_CANONICAL_STATIC_ENVIRONMENT,
        "CALVIN_SOURCE_ROOT": source_root,
        "DUO_VLA_CACHE_ROOT": cache_root,
        "PATH": f"{Path(sys.prefix).resolve() / 'bin'}:/usr/bin:/bin",
    }
    return environment


def require_canonical_evaluator_runtime() -> dict[str, str]:
    """Reject inherited variables, unsafe flags, and local import injection."""

    expected = canonical_evaluator_environment()
    observed = dict(os.environ)
    require(
        observed == expected,
        "CALVIN evaluator environment differs from the canonical env-i contract: "
        f"missing={sorted(set(expected) - set(observed))}, "
        f"extra={sorted(set(observed) - set(expected))}, "
        f"changed={sorted(name for name in set(expected) & set(observed) if expected[name] != observed[name])}",
    )
    required_flags = {
        "dont_write_bytecode": 1,
        "hash_randomization": 0,
        "ignore_environment": 0,
        "isolated": 0,
        "no_site": 0,
        "no_user_site": 1,
        "optimize": 0,
    }
    for name, expected_value in required_flags.items():
        require(getattr(sys.flags, name) == expected_value, f"CALVIN evaluator runtime flag {name} is not canonical")
    require(site.ENABLE_USER_SITE is False, "CALVIN evaluator user-site imports must be disabled")
    require(importlib.util.find_spec("sitecustomize") is None, "sitecustomize injection is present")
    require(importlib.util.find_spec("usercustomize") is None, "usercustomize injection is present")
    require(
        bool(sys.path) and Path(sys.path[0]).resolve() == _SCRIPT_DIR,
        "CALVIN evaluator script directory is not the first import path",
    )
    return expected


def canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(value, allow_nan=False, ensure_ascii=True, separators=(",", ":"), sort_keys=True).encode(
            "ascii"
        )
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"value is not finite canonical JSON: {exc}") from exc


def _content_sha256(value: Mapping[str, Any], hash_field: str) -> str:
    payload = {name: item for name, item in value.items() if name != hash_field}
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def _valid_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _stable_stat(value: os.stat_result) -> tuple[int, int, int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_nlink,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


class _PinnedDirectoryPath:
    """Keep every absolute directory component open and no-follow bound."""

    def __init__(self, path: Path, descriptors: list[int], names: list[str], identities: list[tuple[int, ...]]):
        self.path = path
        self._descriptors = descriptors
        self._names = names
        self._identities = identities

    @classmethod
    def open(cls, path: Path) -> _PinnedDirectoryPath:
        absolute = Path(os.path.abspath(str(path)))
        require(absolute.is_absolute(), f"attestation directory must be absolute: {path}")
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | os.O_DIRECTORY | os.O_NOFOLLOW
        descriptors = [os.open("/", flags)]
        names: list[str] = []
        identities = [_stable_stat(os.fstat(descriptors[0]))]
        try:
            for component in absolute.parts[1:]:
                require(component not in ("", ".", "..") and "/" not in component, "unsafe path component")
                try:
                    child = os.open(component, flags, dir_fd=descriptors[-1])
                except OSError as exc:
                    raise RuntimeError(f"cannot no-follow open attestation directory: {absolute}") from exc
                descriptors.append(child)
                metadata = os.fstat(child)
                require(stat.S_ISDIR(metadata.st_mode), f"attestation path component is not a directory: {absolute}")
                bound = os.stat(component, dir_fd=descriptors[-2], follow_symlinks=False)
                require(
                    _stable_stat(metadata) == _stable_stat(bound),
                    f"attestation directory binding raced: {absolute}",
                )
                names.append(component)
                identities.append(_stable_stat(metadata))
            result = cls(absolute, descriptors, names, identities)
            result.assert_bound()
            return result
        except BaseException:
            for descriptor in reversed(descriptors):
                os.close(descriptor)
            raise

    @property
    def descriptor(self) -> int:
        return self._descriptors[-1]

    def assert_bound(self) -> None:
        for index, descriptor in enumerate(self._descriptors):
            require(
                _stable_stat(os.fstat(descriptor)) == self._identities[index],
                f"attestation directory inode changed: {self.path}",
            )
            if index:
                observed = os.stat(self._names[index - 1], dir_fd=self._descriptors[index - 1], follow_symlinks=False)
                require(
                    _stable_stat(observed) == self._identities[index],
                    f"attestation directory path binding changed: {self.path}",
                )

    def open_regular(self, relative: str) -> _PinnedRegularFile:
        components = relative.split("/")
        require(
            components and all(part not in ("", ".", "..") and "/" not in part for part in components),
            "unsafe relative path",
        )
        directory_flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | os.O_DIRECTORY | os.O_NOFOLLOW
        file_flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | os.O_NOFOLLOW
        directory_descriptors = [os.dup(self.descriptor)]
        directory_names: list[str] = []
        directory_identities = [_stable_stat(os.fstat(directory_descriptors[0]))]
        descriptor = -1
        try:
            for component in components[:-1]:
                try:
                    child = os.open(component, directory_flags, dir_fd=directory_descriptors[-1])
                except OSError as exc:
                    raise RuntimeError(f"cannot no-follow open attestation path component: {relative}") from exc
                directory_descriptors.append(child)
                metadata = os.fstat(child)
                require(stat.S_ISDIR(metadata.st_mode), f"attestation path component is not a directory: {relative}")
                bound = os.stat(component, dir_fd=directory_descriptors[-2], follow_symlinks=False)
                require(_stable_stat(metadata) == _stable_stat(bound), f"attestation path component raced: {relative}")
                directory_names.append(component)
                directory_identities.append(_stable_stat(metadata))
            try:
                descriptor = os.open(components[-1], file_flags, dir_fd=directory_descriptors[-1])
            except OSError as exc:
                raise RuntimeError(f"cannot no-follow open attestation file: {relative}") from exc
            metadata = os.fstat(descriptor)
            require(
                stat.S_ISREG(metadata.st_mode) and metadata.st_nlink == 1,
                f"attestation file must be a single-link regular file: {relative}",
            )
            bound = os.stat(components[-1], dir_fd=directory_descriptors[-1], follow_symlinks=False)
            require(_stable_stat(metadata) == _stable_stat(bound), f"attestation file binding raced: {relative}")
            return _PinnedRegularFile(
                owner=self,
                relative=relative,
                descriptor=descriptor,
                identity=_stable_stat(metadata),
                directory_descriptors=directory_descriptors,
                directory_names=directory_names,
                directory_identities=directory_identities,
                basename=components[-1],
            )
        except BaseException:
            if descriptor >= 0:
                os.close(descriptor)
            for directory_descriptor in reversed(directory_descriptors):
                os.close(directory_descriptor)
            raise

    def exact_inventory(self) -> tuple[set[str], set[str]]:
        files: set[str] = set()
        directories: set[str] = set()

        def visit(descriptor: int, prefix: str) -> None:
            with os.scandir(descriptor) as entries:
                snapshot = list(entries)
            for entry in snapshot:
                require(entry.name not in ("", ".", "..") and "/" not in entry.name, "unsafe projected name")
                relative = f"{prefix}/{entry.name}" if prefix else entry.name
                metadata = entry.stat(follow_symlinks=False)
                if stat.S_ISDIR(metadata.st_mode):
                    child = os.open(
                        entry.name,
                        os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | os.O_DIRECTORY | os.O_NOFOLLOW,
                        dir_fd=descriptor,
                    )
                    try:
                        require(
                            _stable_stat(os.fstat(child)) == _stable_stat(metadata),
                            f"projected directory binding raced: {relative}",
                        )
                        directories.add(relative)
                        visit(child, relative)
                    finally:
                        os.close(child)
                elif stat.S_ISREG(metadata.st_mode):
                    require(metadata.st_nlink == 1, f"projected metadata has external hard links: {relative}")
                    files.add(relative)
                else:
                    raise RuntimeError(f"projected metadata contains a symlink or special file: {relative}")

        visit(self.descriptor, "")
        self.assert_bound()
        return files, directories

    def close(self) -> None:
        descriptors, self._descriptors = self._descriptors, []
        for descriptor in reversed(descriptors):
            os.close(descriptor)


class _PinnedRegularFile:
    def __init__(
        self,
        *,
        owner: _PinnedDirectoryPath,
        relative: str,
        descriptor: int,
        identity: tuple[int, ...],
        directory_descriptors: list[int],
        directory_names: list[str],
        directory_identities: list[tuple[int, ...]],
        basename: str,
    ) -> None:
        self.owner = owner
        self.relative = relative
        self.descriptor = descriptor
        self.identity = identity
        self.directory_descriptors = directory_descriptors
        self.directory_names = directory_names
        self.directory_identities = directory_identities
        self.basename = basename

    @property
    def path(self) -> Path:
        return self.owner.path / self.relative

    def assert_bound(self) -> None:
        self.owner.assert_bound()
        require(_stable_stat(os.fstat(self.descriptor)) == self.identity, f"attestation file changed: {self.path}")
        for index, descriptor in enumerate(self.directory_descriptors):
            require(
                _stable_stat(os.fstat(descriptor)) == self.directory_identities[index],
                f"attestation file parent changed: {self.path}",
            )
            if index:
                observed = os.stat(
                    self.directory_names[index - 1],
                    dir_fd=self.directory_descriptors[index - 1],
                    follow_symlinks=False,
                )
                require(
                    _stable_stat(observed) == self.directory_identities[index],
                    f"attestation file parent binding changed: {self.path}",
                )
        observed = os.stat(self.basename, dir_fd=self.directory_descriptors[-1], follow_symlinks=False)
        require(_stable_stat(observed) == self.identity, f"attestation file path binding changed: {self.path}")

    def digest(
        self,
        *,
        crc32: bool = False,
        collect: bool = False,
        maximum_bytes: int | None = None,
    ) -> tuple[dict[str, Any], bytes | None]:
        self.assert_bound()
        expected_size = self.identity[4]
        require(maximum_bytes is None or expected_size <= maximum_bytes, f"attestation file is too large: {self.path}")
        digest = hashlib.sha256()
        checksum = 0
        size = 0
        chunks = [] if collect else None
        os.lseek(self.descriptor, 0, os.SEEK_SET)
        while True:
            block = os.read(self.descriptor, 8 * 1024 * 1024)
            if not block:
                break
            size += len(block)
            digest.update(block)
            if crc32:
                checksum = zlib.crc32(block, checksum)
            if chunks is not None:
                chunks.append(block)
        self.assert_bound()
        require(size == expected_size, f"attestation file changed while being read: {self.path}")
        result = {"bytes": size, "path": str(self.path), "sha256": digest.hexdigest()}
        if crc32:
            result["crc32"] = checksum & 0xFFFFFFFF
        return result, b"".join(chunks) if chunks is not None else None

    def pread(self, count: int, offset: int) -> bytes:
        require(
            count >= 0 and offset >= 0 and offset <= self.identity[4] - count,
            f"attestation read escapes file: {self.path}",
        )
        raw = os.pread(self.descriptor, count, offset)
        require(len(raw) == count, f"short attestation read: {self.path}")
        return raw

    def close(self) -> None:
        if self.descriptor >= 0:
            os.close(self.descriptor)
            self.descriptor = -1
        for descriptor in reversed(self.directory_descriptors):
            os.close(descriptor)
        self.directory_descriptors = []


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(8 * 1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _file_identity(path: Path, *, crc32: bool = False) -> dict[str, Any]:
    flags = os.O_RDONLY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(str(path), flags)
    except OSError as exc:
        raise RuntimeError(f"required attestation file is unavailable: {path}") from exc
    digest = hashlib.sha256()
    checksum = 0
    size = 0
    try:
        before = os.fstat(descriptor)
        require(stat.S_ISREG(before.st_mode), f"required attestation file is not regular: {path}")
        with os.fdopen(descriptor, "rb") as handle:
            descriptor = -1
            while True:
                block = handle.read(8 * 1024 * 1024)
                if not block:
                    break
                size += len(block)
                digest.update(block)
                if crc32:
                    checksum = zlib.crc32(block, checksum)
            after = os.fstat(handle.fileno())
        stable_fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
        require(
            all(getattr(before, name) == getattr(after, name) for name in stable_fields) and size == after.st_size,
            f"required attestation file changed while being hashed: {path}",
        )
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    result = {"bytes": size, "path": str(path.resolve()), "sha256": digest.hexdigest()}
    if crc32:
        result["crc32"] = checksum & 0xFFFFFFFF
    return result


def _source_file_identity(path: Path) -> dict[str, Any]:
    flags = os.O_RDONLY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(str(path), flags)
    except OSError as exc:
        raise RuntimeError(f"required attestation file is unavailable: {path}") from exc
    digest = hashlib.sha256()
    size = 0
    try:
        before = os.fstat(descriptor)
        require(stat.S_ISREG(before.st_mode), f"required attestation file is not regular: {path}")
        with os.fdopen(descriptor, "rb") as handle:
            descriptor = -1
            while True:
                block = handle.read(8 * 1024 * 1024)
                if not block:
                    break
                size += len(block)
                digest.update(block)
            after = os.fstat(handle.fileno())
        stable_fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
        require(
            all(getattr(before, name) == getattr(after, name) for name in stable_fields) and size == after.st_size,
            f"required attestation file changed while being hashed: {path}",
        )
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    return {"bytes": size, "path": str(path.resolve()), "sha256": digest.hexdigest()}


def evaluator_source_identities(script_dir: Path | None = None) -> dict[str, dict[str, Any]]:
    root = (script_dir or _SCRIPT_DIR).resolve()
    return {name: _source_file_identity(root / name) for name in _SCRIPT_SOURCE_NAMES}


def require_evaluator_sources_unchanged(expected: Mapping[str, Any]) -> None:
    require(set(expected) == set(_SCRIPT_SOURCE_NAMES), "evaluator source snapshot names changed")
    observed: dict[str, dict[str, Any]] = {}
    for name in _SCRIPT_SOURCE_NAMES:
        identity = expected[name]
        require(isinstance(identity, Mapping), f"evaluator source identity {name} must be an object")
        require(set(identity) == {"bytes", "path", "sha256"}, f"evaluator source identity {name} fields changed")
        path = identity["path"]
        require(isinstance(path, str), f"evaluator source identity {name} path is invalid")
        observed[name] = _source_file_identity(Path(path))
    require(observed == expected, "attested evaluator sources changed after import-time snapshot")


# This is intentionally captured before any CALVIN module import or runtime
# inspection.  Later attestations must describe these exact bytes.
_IMPORT_EVALUATOR_SOURCE_IDENTITIES = evaluator_source_identities(_SCRIPT_DIR)


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, value in pairs:
        if name in result:
            raise ValueError(f"duplicate JSON field {name!r}")
        result[name] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant {value}")


def _decode_strict_json(raw: bytes, path: Path) -> dict[str, Any]:
    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, ValueError) as exc:
        raise RuntimeError(f"attestation input is not strict finite UTF-8 JSON: {path}") from exc
    require(isinstance(value, dict), f"attestation JSON root must be an object: {path}")
    return value


def _read_strict_json(path: Path) -> tuple[dict[str, Any], str]:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise RuntimeError(f"attestation input is unavailable: {path}") from exc
    return _decode_strict_json(raw, path), hashlib.sha256(raw).hexdigest()


def _git_output(root: Path, *args: str) -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(root), *args],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(f"cannot inspect CALVIN checkout {root}: git {' '.join(args)}") from exc


def verify_checkout(source_root: Path) -> dict[str, dict[str, str]]:
    root = source_root.resolve()
    repositories = (
        ("calvin", root, CALVIN_REVISION),
        ("calvin_env", root / "calvin_env", CALVIN_ENV_REVISION),
        ("tacto", root / "calvin_env" / "tacto", CALVIN_TACTO_REVISION),
    )
    result: dict[str, dict[str, str]] = {}
    for name, path, expected_revision in repositories:
        require(path.is_dir(), f"CALVIN checkout component is missing: {path}")
        revision = _git_output(path, "rev-parse", "--verify", "HEAD")
        require(
            revision == expected_revision,
            f"CALVIN {name} revision mismatch: expected {expected_revision}, found {revision}",
        )
        dirty = _git_output(path, "status", "--porcelain=v1", "--untracked-files=all")
        require(not dirty, f"CALVIN {name} checkout is not clean: {dirty}")
        result[name] = {"path": str(path), "revision": revision, "status": "clean"}
    return result


def _parse_constraints(path: Path) -> tuple[dict[str, str], str]:
    require(path.is_file(), f"CALVIN evaluator constraints are missing: {path}")
    packages: dict[str, str] = {}
    canonical_names: set[str] = set()
    for line_number, source_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = source_line.split("#", 1)[0].strip()
        if not line:
            continue
        name, separator, version = line.partition("==")
        require(
            separator == "==" and "==" not in version and bool(_PACKAGE_NAME.fullmatch(name)) and bool(version),
            f"constraint line {line_number} is not an exact name==version pin",
        )
        normalized = re.sub(r"[-_.]+", "-", name).lower()
        require(normalized not in canonical_names, f"duplicate constrained package: {name}")
        canonical_names.add(normalized)
        packages[name] = version
    require(bool(packages), "CALVIN evaluator constraints contain no package pins")
    return packages, _sha256_file(path)


def verify_packages(constraints_path: Path | None = None) -> dict[str, Any]:
    from importlib import metadata

    path = (constraints_path or Path(__file__).resolve().with_name("constraints-py38.txt")).resolve()
    expected, constraints_sha256 = _parse_constraints(path)
    actual: dict[str, str] = {}
    missing: list[str] = []
    for name in expected:
        try:
            actual[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            missing.append(name)
    require(not missing, f"CALVIN evaluator packages are missing: {missing}")
    mismatches = {
        name: {"expected": expected[name], "actual": actual[name]}
        for name in expected
        if actual[name] != expected[name]
    }
    require(not mismatches, f"CALVIN package-version mismatch: {mismatches}")
    return {
        "constraints_path": str(path),
        "constraints_sha256": constraints_sha256,
        "packages": actual,
    }


def verify_module_origins(source_root: Path) -> dict[str, dict[str, str]]:
    root = source_root.resolve()
    expected = {
        "calvin_agent": (root / "calvin_models" / "calvin_agent" / "__init__.py").resolve(),
        "calvin_env": (root / "calvin_env" / "calvin_env" / "__init__.py").resolve(),
    }
    result: dict[str, dict[str, str]] = {}
    for name, expected_path in expected.items():
        require(expected_path.is_file(), f"expected {name} module is missing: {expected_path}")
        module = importlib.import_module(name)
        module_file = getattr(module, "__file__", None)
        require(isinstance(module_file, str), f"imported {name} module has no file origin")
        observed = Path(module_file).resolve()
        require(observed == expected_path, f"{name} imported from unexpected path: {observed}")
        result[name] = {"origin": str(observed), "sha256": _sha256_file(observed)}
    return result


def verify_official_yaml_files(source_root: Path) -> dict[str, dict[str, Any]]:
    conf = source_root.resolve() / "calvin_models" / "conf"
    files = {
        "task_oracle": (
            conf / "callbacks" / "rollout" / "tasks" / "new_playtable_tasks.yaml",
            TASK_ORACLE_SHA256,
        ),
        "validation_annotations": (
            conf / "annotations" / "new_playtable_validation.yaml",
            VALIDATION_ANNOTATIONS_SHA256,
        ),
    }
    result: dict[str, dict[str, Any]] = {}
    for name, (path, expected_sha256) in files.items():
        identity = _file_identity(path)
        require(identity["sha256"] == expected_sha256, f"pinned CALVIN {name} YAML hash mismatch")
        result[name] = identity
    return result


def runtime_identity() -> dict[str, Any]:
    canonical_environment = require_canonical_evaluator_runtime()
    version = platform.python_version()
    require(version == PYTHON_VERSION, f"CALVIN evaluator requires Python {PYTHON_VERSION}, found {version}")
    uname = platform.uname()
    flags = {
        name: getattr(sys.flags, name)
        for name in (
            "bytes_warning",
            "debug",
            "dev_mode",
            "dont_write_bytecode",
            "hash_randomization",
            "ignore_environment",
            "inspect",
            "interactive",
            "isolated",
            "no_site",
            "no_user_site",
            "optimize",
            "quiet",
            "utf8_mode",
            "verbose",
        )
    }
    return {
        "byteorder": sys.byteorder,
        "environment": {
            name: canonical_environment.get(name) if name in canonical_environment else None
            for name in _RUNTIME_ENVIRONMENT_NAMES
        },
        "executable": str(Path(sys.executable).resolve()),
        "libc": list(platform.libc_ver()),
        "machine": uname.machine,
        "node": uname.node,
        "os_name": os.name,
        "platform": platform.platform(),
        "processor": uname.processor,
        "python_compiler": platform.python_compiler(),
        "python_build": list(platform.python_build()),
        "python_implementation": platform.python_implementation(),
        "python_soabi": sysconfig.get_config_var("SOABI"),
        "python_sys_version": sys.version,
        "python_version": version,
        "python_version_info": list(sys.version_info),
        "runtime_flags": flags,
        "sqlite_version": sqlite3.sqlite_version,
        "release": uname.release,
        "sys_path": list(sys.path),
        "system": uname.system,
        "version": uname.version,
        "zlib_version": zlib.ZLIB_RUNTIME_VERSION,
    }


def inspect_dataset(dataset_root: Path | None, require_dataset: bool) -> dict[str, Any]:
    if dataset_root is None:
        if require_dataset:
            raise RuntimeError("--require-dataset needs --dataset-root")
        return {"present": False, "path": None, "reason": "dataset root not configured"}
    root = dataset_root.resolve()
    required = [
        root / "training",
        root / "validation",
        root / "training" / "lang_annotations" / "auto_lang_ann.npy",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing and require_dataset:
        raise RuntimeError(f"CALVIN ABC_D dataset is incomplete: {missing}")
    return {"present": not missing, "path": str(root), "missing": missing}


def _verify_dataset_identity_v3(dataset_root: Path) -> dict[str, Any]:
    """Legacy extracted-tree parity verifier; never usable for official scoring."""

    root = dataset_root.resolve()
    require(root.is_dir() and root.name == "task_ABC_D", f"invalid CALVIN dataset root: {root}")
    manifest_path = root.parent / MANIFEST_NAME
    manifest, manifest_file_sha256 = _read_strict_json(manifest_path)
    expected_root_fields = {
        "archive",
        "checksum_url",
        "content_sha256",
        "critical_files",
        "dataset",
        "extraction",
        "schema",
    }
    require(set(manifest) == expected_root_fields, "CALVIN dataset manifest fields differ from v3")
    require(manifest["schema"] == LEGACY_DATASET_MANIFEST_SCHEMA, "unsupported legacy CALVIN dataset manifest schema")
    require(manifest["dataset"] == "task_ABC_D", "CALVIN dataset manifest dataset identity mismatch")
    require(manifest["checksum_url"] == CHECKSUM_URL, "CALVIN dataset checksum URL mismatch")
    require(_valid_sha256(manifest["content_sha256"]), "CALVIN dataset manifest content hash is invalid")
    require(
        manifest["content_sha256"] == _content_sha256(manifest, "content_sha256"),
        "CALVIN dataset manifest content hash mismatch",
    )

    archive = manifest["archive"]
    require(isinstance(archive, dict), "CALVIN dataset manifest archive identity is missing")
    require(
        set(archive) == {"bytes", "member_inventory", "sha256", "uncompressed_bytes", "url"},
        "CALVIN dataset manifest archive fields differ",
    )
    require(
        archive["bytes"] == ARCHIVE_BYTES and archive["sha256"] == ARCHIVE_SHA256 and archive["url"] == ARCHIVE_URL,
        "CALVIN dataset archive identity mismatch",
    )
    inventory = archive["member_inventory"]
    inventory_fields = {
        "compressed_bytes",
        "file_member_count",
        "member_count",
        "npz_member_count",
        "sha256",
        "uncompressed_bytes",
    }
    require(isinstance(inventory, dict) and set(inventory) == inventory_fields, "CALVIN member inventory is invalid")
    for name in inventory_fields - {"sha256"}:
        require(type(inventory[name]) is int and inventory[name] > 0, f"CALVIN member inventory {name} is invalid")
    require(_valid_sha256(inventory["sha256"]), "CALVIN member inventory SHA-256 is invalid")
    require(
        archive["uncompressed_bytes"] == inventory["uncompressed_bytes"],
        "CALVIN archive uncompressed byte totals differ",
    )
    require(
        inventory["npz_member_count"] <= inventory["file_member_count"] <= inventory["member_count"],
        "CALVIN member inventory counts are inconsistent",
    )

    critical = manifest["critical_files"]
    require(
        isinstance(critical, dict) and set(critical) == set(DATASET_CRITICAL_FILES),
        "CALVIN dataset critical-file inventory mismatch",
    )
    require(
        all(_valid_sha256(value) for value in critical.values()),
        "CALVIN dataset critical-file hash is invalid",
    )

    extraction = manifest["extraction"]
    require(
        isinstance(extraction, dict) and set(extraction) == {"file_members_verified", "member_index", "verification"},
        "CALVIN extraction contract fields differ",
    )
    require(
        extraction["file_members_verified"] == inventory["file_member_count"]
        and extraction["verification"] == "size-and-crc32-against-every-pinned-zip-member",
        "CALVIN extraction verification contract mismatch",
    )
    member_index = extraction["member_index"]
    require(
        isinstance(member_index, dict) and set(member_index) == {"bytes", "path", "schema", "sha256"},
        "CALVIN member-index identity fields differ",
    )
    index_path = root.parent / LEGACY_MEMBER_INDEX_NAME
    require(
        member_index["path"] == index_path.name
        and member_index["schema"] == LEGACY_MEMBER_INDEX_SCHEMA
        and type(member_index["bytes"]) is int
        and member_index["bytes"] > 0
        and _valid_sha256(member_index["sha256"]),
        "CALVIN member-index identity is invalid",
    )
    index_identity = _file_identity(index_path)
    require(
        index_identity["bytes"] == member_index["bytes"] and index_identity["sha256"] == member_index["sha256"],
        "CALVIN member-index file differs from the extraction manifest",
    )

    validation: dict[str, dict[str, Any]] = {}
    uri = index_path.as_uri() + "?mode=ro&immutable=1"
    try:
        with sqlite3.connect(uri, uri=True) as connection:
            tables = {
                row[0]
                for row in connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
                if isinstance(row[0], str)
            }
            metadata = dict(connection.execute("SELECT name, value FROM metadata"))
            row_count = connection.execute("SELECT count(*) FROM members").fetchone()
            integrity = connection.execute("PRAGMA integrity_check").fetchone()
            require(tables == {"members", "metadata"}, "CALVIN member-index table inventory differs")
            require(integrity == ("ok",), "CALVIN member-index integrity check failed")
            require(
                metadata
                == {
                    "file_member_count": str(inventory["file_member_count"]),
                    "schema": LEGACY_MEMBER_INDEX_SCHEMA,
                }
                and row_count == (inventory["file_member_count"],),
                "CALVIN member-index database metadata differs from the manifest",
            )
            for relative in VALIDATION_CRITICAL_FILES:
                identity = _file_identity(root / relative, crc32=True)
                row = connection.execute(
                    "SELECT bytes, crc32, sha256 FROM members WHERE path = ?",
                    (relative,),
                ).fetchone()
                require(
                    row is not None
                    and len(row) == 3
                    and type(row[0]) is int
                    and type(row[1]) is int
                    and _valid_sha256(row[2]),
                    f"CALVIN member index has no valid validation identity: {relative}",
                )
                require(
                    identity["bytes"] == row[0]
                    and identity["crc32"] == row[1]
                    and identity["sha256"] == row[2]
                    and identity["sha256"] == critical[relative],
                    f"CALVIN validation critical file differs from manifest/member index: {relative}",
                )
                validation[relative] = identity
    except sqlite3.Error as exc:
        raise RuntimeError(f"cannot authenticate CALVIN member index: {index_path}") from exc

    validation_rows = [
        {
            "bytes": validation[name]["bytes"],
            "crc32": validation[name]["crc32"],
            "path": name,
            "sha256": validation[name]["sha256"],
        }
        for name in VALIDATION_CRITICAL_FILES
    ]
    return {
        "archive": {
            "bytes": ARCHIVE_BYTES,
            "sha256": ARCHIVE_SHA256,
            "url": ARCHIVE_URL,
        },
        "dataset_root": str(root),
        "manifest": {
            "content_sha256": manifest["content_sha256"],
            "file_sha256": manifest_file_sha256,
            "path": str(manifest_path),
            "schema": LEGACY_DATASET_MANIFEST_SCHEMA,
        },
        "member_index": {
            **member_index,
            "path": str(index_path),
            "validation_rows_sha256": hashlib.sha256(canonical_json_bytes(validation_rows)).hexdigest(),
        },
        "member_inventory": dict(inventory),
        "storage_mode": "legacy-extracted-parity-only",
        "validation_critical_files": validation,
    }


def _canonical_metadata_uint(metadata: Mapping[str, str], name: str) -> int:
    value = metadata.get(name)
    require(
        isinstance(value, str) and re.fullmatch(r"0|[1-9][0-9]*", value) is not None,
        f"CALVIN member-index metadata is not canonical ASCII decimal: {name}",
    )
    parsed = int(value)
    require(parsed <= (1 << 63) - 1, f"CALVIN member-index metadata exceeds SQLite integer range: {name}")
    return parsed


def _validate_archive_direct_manifest(manifest: Mapping[str, Any]) -> None:
    require(set(manifest) == _V4_MANIFEST_FIELDS, "CALVIN v4 manifest root fields differ")
    require(manifest.get("schema") == DATASET_MANIFEST_SCHEMA, "unsupported CALVIN v4 manifest schema")
    require(manifest.get("dataset") == "task_ABC_D", "CALVIN v4 dataset identity mismatch")
    require(manifest.get("checksum_url") == CHECKSUM_URL, "CALVIN v4 checksum URL mismatch")
    require(_valid_sha256(manifest.get("content_sha256")), "CALVIN v4 manifest content hash is invalid")
    require(
        manifest["content_sha256"] == _content_sha256(manifest, "content_sha256"),
        "CALVIN v4 manifest content hash mismatch",
    )

    archive = manifest.get("archive")
    require(isinstance(archive, dict), "CALVIN v4 archive identity is missing")
    require(
        set(archive) == {"bytes", "central_directory", "member_inventory", "path", "sha256", "url"},
        "CALVIN v4 archive fields differ",
    )
    require(
        archive.get("bytes") == ARCHIVE_BYTES
        and archive.get("path") == ARCHIVE_NAME
        and archive.get("sha256") == ARCHIVE_SHA256
        and archive.get("url") == ARCHIVE_URL,
        "CALVIN v4 archive identity mismatch",
    )
    central = archive.get("central_directory")
    require(isinstance(central, dict), "CALVIN v4 central-directory identity is missing")
    require(
        set(central) == {"bytes", "entries", "offset", "sha256", "zip64"},
        "CALVIN v4 central-directory fields differ",
    )
    require(
        central
        == {
            "bytes": CENTRAL_DIRECTORY_BYTES,
            "entries": MEMBER_COUNT,
            "offset": CENTRAL_DIRECTORY_OFFSET,
            "sha256": CENTRAL_DIRECTORY_SHA256,
            "zip64": True,
        },
        "CALVIN v4 central-directory identity differs from the pinned archive",
    )
    inventory = archive.get("member_inventory")
    require(isinstance(inventory, dict), "CALVIN v4 member inventory is missing")
    require(
        set(inventory)
        == {
            "compressed_bytes",
            "directory_member_count",
            "file_member_count",
            "member_count",
            "npz_member_count",
            "sha256",
            "uncompressed_bytes",
        },
        "CALVIN v4 member-inventory fields differ",
    )
    require(
        inventory.get("member_count") == MEMBER_COUNT
        and inventory.get("file_member_count") == FILE_MEMBER_COUNT
        and inventory.get("directory_member_count") == DIRECTORY_MEMBER_COUNT
        and inventory.get("npz_member_count") == NPZ_MEMBER_COUNT,
        "CALVIN v4 member counts differ from the pinned archive",
    )
    for name in ("compressed_bytes", "uncompressed_bytes"):
        require(type(inventory.get(name)) is int and inventory[name] > 0, f"CALVIN v4 inventory {name} is invalid")
    require(_valid_sha256(inventory.get("sha256")), "CALVIN v4 member-inventory SHA-256 is invalid")

    critical = manifest.get("critical_files")
    require(
        isinstance(critical, dict) and set(critical) == set(DATASET_CRITICAL_FILES),
        "CALVIN v4 critical metadata inventory differs",
    )
    for relative, identity in critical.items():
        require(
            isinstance(identity, dict) and set(identity) == {"bytes", "crc32", "sha256"},
            f"CALVIN v4 critical identity fields differ: {relative}",
        )
        require(
            type(identity.get("bytes")) is int
            and identity["bytes"] >= 0
            and type(identity.get("crc32")) is int
            and 0 <= identity["crc32"] <= 0xFFFFFFFF
            and _valid_sha256(identity.get("sha256")),
            f"CALVIN v4 critical identity is invalid: {relative}",
        )

    storage = manifest.get("storage")
    require(isinstance(storage, dict), "CALVIN v4 storage contract is missing")
    require(
        set(storage)
        == {"derived_artifacts", "materialized_files", "member_index", "mode", "reader_schema", "verification"},
        "CALVIN v4 storage fields differ",
    )
    require(
        storage.get("mode") == "archive-direct"
        and storage.get("reader_schema") == ARCHIVE_READER_SCHEMA
        and storage.get("verification") == ARCHIVE_DIRECT_VERIFICATION,
        "CALVIN v4 archive-direct reader contract differs",
    )
    require(
        storage.get("materialized_files") == list(DATASET_CRITICAL_FILES),
        "CALVIN v4 materialized metadata inventory differs",
    )
    require(
        storage.get("derived_artifacts")
        == {
            "state_action_sidecar": None,
            "state_action_sidecar_schema_hook": STATE_ACTION_SIDECAR_SCHEMA,
        },
        "CALVIN v4 P0 derived-artifact contract differs",
    )
    index = storage.get("member_index")
    require(
        isinstance(index, dict)
        and set(index) == {"bytes", "path", "schema", "sha256"}
        and index.get("path") == MEMBER_INDEX_NAME
        and index.get("schema") == MEMBER_INDEX_SCHEMA
        and type(index.get("bytes")) is int
        and index["bytes"] > 0
        and _valid_sha256(index.get("sha256")),
        "CALVIN v4 member-index identity is invalid",
    )


def _validate_zip64_tail(archive: _PinnedRegularFile) -> dict[str, Any]:
    central_end = CENTRAL_DIRECTORY_OFFSET + CENTRAL_DIRECTORY_BYTES
    require(
        central_end + ZIP64_TRAILER_BYTES == ARCHIVE_BYTES,
        "pinned CALVIN ZIP64 trailer geometry is inconsistent",
    )
    raw = archive.pread(ZIP64_TRAILER_BYTES, central_end)
    trailer_sha256 = hashlib.sha256(raw).hexdigest()
    require(trailer_sha256 == ZIP64_TRAILER_SHA256, "CALVIN ZIP64 trailer SHA-256 differs")
    signature, record_size = _ZIP64_EOCD_PREFIX.unpack_from(raw, 0)
    require(signature == _ZIP64_EOCD_SIGNATURE and record_size == 44, "CALVIN ZIP64 EOCD prefix differs")
    version_made, version_needed, disk, central_disk, disk_entries, total_entries, size, offset = (
        _ZIP64_EOCD_BODY.unpack_from(raw, _ZIP64_EOCD_PREFIX.size)
    )
    require(
        version_made == ZIP64_VERSION_MADE
        and version_needed == 45
        and disk == 0
        and central_disk == 0
        and disk_entries == MEMBER_COUNT
        and total_entries == MEMBER_COUNT
        and size == CENTRAL_DIRECTORY_BYTES
        and offset == CENTRAL_DIRECTORY_OFFSET,
        "CALVIN ZIP64 EOCD body differs from the pinned central directory",
    )
    locator_offset = _ZIP64_EOCD_PREFIX.size + _ZIP64_EOCD_BODY.size
    locator_signature, locator_disk, zip64_offset, locator_disks = _ZIP64_LOCATOR.unpack_from(raw, locator_offset)
    require(
        locator_signature == _ZIP64_LOCATOR_SIGNATURE
        and locator_disk == 0
        and zip64_offset == central_end
        and locator_disks == 1,
        "CALVIN ZIP64 locator differs",
    )
    eocd_offset = locator_offset + _ZIP64_LOCATOR.size
    eocd = _EOCD.unpack_from(raw, eocd_offset)
    require(
        eocd
        == (
            _EOCD_SIGNATURE,
            0,
            0,
            0xFFFF,
            0xFFFF,
            CENTRAL_DIRECTORY_BYTES,
            0xFFFFFFFF,
            0,
        ),
        "CALVIN classic EOCD does not carry the exact ZIP64 sentinels",
    )
    return {
        "classic_eocd_sentinels": True,
        "record_bytes": len(raw),
        "sha256": trailer_sha256,
        "version_made": version_made,
        "version_needed": version_needed,
        "zip64_eocd_offset": central_end,
    }


def _hash_archive_central_directory(archive: _PinnedRegularFile) -> str:
    archive.assert_bound()
    digest = hashlib.sha256()
    offset = CENTRAL_DIRECTORY_OFFSET
    remaining = CENTRAL_DIRECTORY_BYTES
    while remaining:
        block_size = min(8 * 1024 * 1024, remaining)
        digest.update(archive.pread(block_size, offset))
        offset += block_size
        remaining -= block_size
    archive.assert_bound()
    return digest.hexdigest()


def _validate_member_index_schema(connection: sqlite3.Connection) -> dict[str, Any]:
    objects = {
        (row[0], row[1]): row[2]
        for row in connection.execute(
            "SELECT type,name,sql FROM sqlite_master WHERE name NOT LIKE 'sqlite_%' ORDER BY type,name"
        )
    }
    expected = {
        ("index", "members_episode"): _EPISODE_INDEX_SQL,
        ("index", "members_local_header"): _LOCAL_HEADER_INDEX_SQL,
        ("index", "members_physical"): _PHYSICAL_INDEX_SQL,
        ("table", "members"): _MEMBERS_TABLE_SQL,
        ("table", "metadata"): _METADATA_TABLE_SQL,
    }
    require(objects == expected, "CALVIN v2 member-index SQLite object inventory differs")
    require(
        list(connection.execute("PRAGMA table_info(members)")) == _EXPECTED_MEMBER_COLUMNS,
        "CALVIN v2 member-index members columns differ",
    )
    require(
        list(connection.execute("PRAGMA table_info(metadata)")) == _EXPECTED_METADATA_COLUMNS,
        "CALVIN v2 member-index metadata columns differ",
    )
    index_list = {
        (row[1], row[2], row[3], row[4])
        for row in connection.execute("PRAGMA index_list(members)")
        if not row[1].startswith("sqlite_")
    }
    require(
        index_list
        == {
            ("members_episode", 1, "c", 1),
            ("members_local_header", 0, "c", 0),
            ("members_physical", 0, "c", 0),
        },
        "CALVIN v2 member-index index properties differ",
    )
    for name, columns in {
        "members_episode": ["split", "global_index"],
        "members_local_header": ["local_header_offset"],
        "members_physical": ["data_offset"],
    }.items():
        require(
            [row[2] for row in connection.execute(f"PRAGMA index_info({name})")] == columns,
            f"CALVIN v2 member-index columns differ: {name}",
        )
    require(
        connection.execute("PRAGMA application_id").fetchone() == (1145853251,),
        "CALVIN index application ID differs",
    )
    require(connection.execute("PRAGMA user_version").fetchone() == (2,), "CALVIN index user version differs")
    require(connection.execute("PRAGMA integrity_check").fetchone() == ("ok",), "CALVIN index integrity check failed")
    return {"objects": [[kind, name, sql] for (kind, name), sql in sorted(objects.items())]}


def _validate_v2_member_rows(connection: sqlite3.Connection) -> tuple[dict[str, int], dict[str, tuple[Any, ...]]]:
    query = (
        "SELECT path,kind,split,global_index,local_header_offset,data_offset,compressed_bytes,logical_bytes,"
        "method,version_needed,flags,crc32,logical_sha256,state_action_row "
        "FROM members ORDER BY local_header_offset"
    )
    summary = {
        "compressed_bytes": 0,
        "directory_member_count": 0,
        "file_member_count": 0,
        "member_count": 0,
        "npz_member_count": 0,
        "uncompressed_bytes": 0,
    }
    critical_rows: dict[str, tuple[Any, ...]] = {}
    previous_end = 0
    directory_paths: set[str] = set()
    required_parent_paths: set[str] = set()
    for row in connection.execute(query):
        require(len(row) == 14, "CALVIN v2 member-index row width differs")
        (
            path,
            kind,
            split,
            global_index,
            local_header_offset,
            data_offset,
            compressed_bytes,
            logical_bytes,
            method,
            version_needed,
            flags,
            crc32,
            logical_sha256,
            state_action_row,
        ) = row
        require(
            type(path) is str
            and type(kind) is int
            and kind in (0, 1)
            and type(local_header_offset) is int
            and type(data_offset) is int
            and type(compressed_bytes) is int
            and type(logical_bytes) is int
            and type(method) is int
            and type(version_needed) is int
            and type(flags) is int
            and type(crc32) is int
            and type(logical_sha256) is bytes
            and len(logical_sha256) == 32
            and (split is None or type(split) is str)
            and (global_index is None or type(global_index) is int)
            and (state_action_row is None or type(state_action_row) is int),
            "CALVIN v2 member-index row types differ",
        )
        try:
            path.encode("ascii")
        except UnicodeEncodeError as exc:
            raise RuntimeError("CALVIN v2 member-index path is not canonical ASCII") from exc
        if path:
            components = path.split("/")
            require(
                not path.startswith("/")
                and not path.endswith("/")
                and "\\" not in path
                and "\0" not in path
                and all(component not in ("", ".", "..") and ":" not in component for component in components),
                "CALVIN v2 member-index path is not canonical",
            )
            required_parent_paths.update("/".join(components[:depth]) for depth in range(1, len(components)))
        else:
            require(kind == 1, "only the CALVIN archive root row may have an empty path")
        episode = _EPISODE_MEMBER_RE.fullmatch(path) if kind == 0 else None
        expected_role = (episode.group(1), int(episode.group(2))) if episode is not None else (None, None)
        require(
            (split, global_index) == expected_role,
            "CALVIN v2 member-index split/global identity differs from its canonical path",
        )
        require(
            local_header_offset >= previous_end
            and data_offset >= local_header_offset + _LOCAL_HEADER.size
            and compressed_bytes >= 0
            and logical_bytes >= 0
            and 0 <= crc32 <= _UINT32_MAX
            and data_offset <= CENTRAL_DIRECTORY_OFFSET - compressed_bytes
            and state_action_row is None,
            "CALVIN v2 member-index numeric/physical contract differs",
        )
        require(flags == 0, "CALVIN v2 member-index contains unsupported ZIP flags")
        if kind == 1:
            require(
                method == 0
                and version_needed >= 10
                and compressed_bytes == 0
                and logical_bytes == 0
                and crc32 == 0
                and logical_sha256 == _EMPTY_SHA256,
                "CALVIN v2 member-index directory row violates the storage contract",
            )
            summary["directory_member_count"] += 1
            directory_paths.add(path)
        else:
            require(
                method == 8 and version_needed >= 20 and logical_sha256 != _ZERO_SHA256,
                "CALVIN v2 member-index file row violates the DEFLATE contract",
            )
            summary["file_member_count"] += 1
            summary["npz_member_count"] += int(path.endswith(".npz"))
        previous_end = data_offset + compressed_bytes
        summary["member_count"] += 1
        summary["compressed_bytes"] += compressed_bytes
        summary["uncompressed_bytes"] += logical_bytes
        if path in DATASET_CRITICAL_FILES:
            critical_rows[path] = row
    require(
        required_parent_paths <= directory_paths,
        "CALVIN v2 member index is missing a declared non-root parent directory",
    )
    require(previous_end <= CENTRAL_DIRECTORY_OFFSET, "CALVIN v2 member payload overlaps the central directory")
    require(set(critical_rows) == set(DATASET_CRITICAL_FILES), "CALVIN v2 critical row inventory differs")
    return summary, critical_rows


def _parse_local_extra_fields(raw: bytes, relative: str) -> dict[int, bytes]:
    fields: dict[int, bytes] = {}
    cursor = 0
    while cursor < len(raw):
        require(len(raw) - cursor >= _EXTRA_HEADER.size, f"truncated local extra-field header: {relative}")
        field_id, field_size = _EXTRA_HEADER.unpack_from(raw, cursor)
        cursor += _EXTRA_HEADER.size
        require(field_size <= len(raw) - cursor, f"truncated local extra-field payload: {relative}")
        require(field_id not in fields, f"duplicate local extra-field ID: {relative}")
        require(field_id in _ALLOWED_EXTRA_IDS, f"unsupported local extra-field ID: {relative}")
        fields[field_id] = raw[cursor : cursor + field_size]
        cursor += field_size
    timestamp = fields.get(_EXTENDED_TIMESTAMP_EXTRA_ID)
    if timestamp is not None:
        require(timestamp and not (timestamp[0] & ~0x07), f"malformed local timestamp extra field: {relative}")
        timestamp_count = (timestamp[0] & 1) + ((timestamp[0] >> 1) & 1) + ((timestamp[0] >> 2) & 1)
        require(
            len(timestamp) >= 1 + 4 * min(timestamp_count, 1),
            f"truncated local timestamp extra field: {relative}",
        )
    uid_gid = fields.get(_UNIX_UID_GID_EXTRA_ID)
    if uid_gid is not None:
        require(len(uid_gid) >= 3 and uid_gid[0] == 1, f"malformed local UID/GID extra field: {relative}")
        uid_size = uid_gid[1]
        gid_size_offset = 2 + uid_size
        require(
            uid_size > 0 and gid_size_offset < len(uid_gid),
            f"malformed local UID/GID extra field: {relative}",
        )
        gid_size = uid_gid[gid_size_offset]
        require(
            gid_size > 0 and gid_size_offset + 1 + gid_size == len(uid_gid),
            f"malformed local UID/GID extra field: {relative}",
        )
    return fields


def _resolve_local_sizes(
    logical_32: int,
    compressed_32: int,
    extras: Mapping[int, bytes],
    relative: str,
) -> tuple[int, int]:
    logical_required = logical_32 == _UINT32_MAX
    compressed_required = compressed_32 == _UINT32_MAX
    payload = extras.get(_ZIP64_EXTRA_ID)
    if payload is None:
        require(
            not logical_required and not compressed_required,
            f"critical local header is missing ZIP64 sizes: {relative}",
        )
        return logical_32, compressed_32
    require(logical_required or compressed_required, f"critical local header has unnecessary ZIP64 sizes: {relative}")
    cursor = 0

    def consume() -> int:
        nonlocal cursor
        require(cursor + 8 <= len(payload), f"critical local ZIP64 sizes are truncated: {relative}")
        value = struct.unpack_from("<Q", payload, cursor)[0]
        cursor += 8
        return value

    logical = consume() if logical_required else logical_32
    compressed = consume() if compressed_required else compressed_32
    require(cursor == len(payload), f"critical local ZIP64 sizes have trailing bytes: {relative}")
    return logical, compressed


def _authenticate_indexed_archive_member(
    archive: _PinnedRegularFile,
    row: tuple[Any, ...],
    relative: str,
) -> dict[str, Any]:
    (
        path,
        kind,
        split,
        global_index,
        local_header_offset,
        data_offset,
        compressed_bytes,
        logical_bytes,
        method,
        version_needed,
        flags,
        crc32,
        logical_sha256,
        state_action_row,
    ) = row
    require(path == relative and kind == 0, f"CALVIN critical index row identity differs: {relative}")
    require(
        split is None and global_index is None and state_action_row is None,
        f"CALVIN critical row role differs: {relative}",
    )
    integer_values = (
        local_header_offset,
        data_offset,
        compressed_bytes,
        logical_bytes,
        method,
        version_needed,
        flags,
        crc32,
    )
    require(all(type(value) is int for value in integer_values), f"CALVIN critical row types differ: {relative}")
    require(
        isinstance(logical_sha256, bytes) and len(logical_sha256) == 32,
        f"CALVIN critical row logical SHA-256 differs: {relative}",
    )
    require(
        local_header_offset >= 0
        and data_offset >= local_header_offset + _LOCAL_HEADER.size
        and compressed_bytes >= 0
        and logical_bytes >= 0
        and data_offset <= CENTRAL_DIRECTORY_OFFSET - compressed_bytes,
        f"CALVIN critical row range escapes the archive: {relative}",
    )
    header_raw = archive.pread(_LOCAL_HEADER.size, local_header_offset)
    header = _LOCAL_HEADER.unpack(header_raw)
    require(header[0] == _LOCAL_SIGNATURE, f"CALVIN critical local-header signature differs: {relative}")
    require(
        header[1] == version_needed and header[2] == flags and header[3] == method,
        f"CALVIN critical local-header/index fields differ: {relative}",
    )
    variable_size = header[9] + header[10]
    variable = archive.pread(variable_size, local_header_offset + _LOCAL_HEADER.size)
    name_bytes = variable[: header[9]]
    expected_name = f"task_ABC_D/{relative}".encode("ascii")
    require(name_bytes == expected_name, f"CALVIN critical local-header path differs: {relative}")
    extras = _parse_local_extra_fields(variable[header[9] :], relative)
    uses_zip64 = header[7] == _UINT32_MAX or header[8] == _UINT32_MAX
    require(not uses_zip64 or version_needed >= 45, f"CALVIN critical ZIP64 version is invalid: {relative}")
    local_logical, local_compressed = _resolve_local_sizes(header[8], header[7], extras, relative)
    require(
        header[6] == crc32 and local_compressed == compressed_bytes and local_logical == logical_bytes,
        f"CALVIN critical local-header CRC/size fields differ from the index: {relative}",
    )
    require(
        data_offset == local_header_offset + _LOCAL_HEADER.size + variable_size,
        f"CALVIN critical data offset differs from its local header: {relative}",
    )
    require(
        flags == 0 and method == 8 and version_needed >= 20, f"CALVIN critical DEFLATE contract differs: {relative}"
    )

    digest = hashlib.sha256()
    checksum = 0
    output_bytes = 0

    def consume(block: bytes) -> None:
        nonlocal checksum, output_bytes
        output_bytes += len(block)
        require(output_bytes <= logical_bytes, f"CALVIN critical member expands beyond its index: {relative}")
        checksum = zlib.crc32(block, checksum)
        digest.update(block)

    decompressor = zlib.decompressobj(-zlib.MAX_WBITS)
    cursor = data_offset
    remaining = compressed_bytes
    logical_remaining = logical_bytes
    try:
        while remaining:
            count = min(1024 * 1024, remaining)
            compressed = archive.pread(count, cursor)
            cursor += count
            remaining -= count
            pending = compressed
            while pending:
                previous_length = len(pending)
                decoded = decompressor.decompress(pending, logical_remaining + 1)
                logical_remaining -= len(decoded)
                pending = decompressor.unconsumed_tail
                consume(decoded)
                require(
                    not pending or len(pending) != previous_length or bool(decoded),
                    f"CALVIN critical DEFLATE decoder made no progress: {relative}",
                )
        flushed = decompressor.flush(logical_remaining + 1)
        logical_remaining -= len(flushed)
        consume(flushed)
        require(
            decompressor.eof
            and not decompressor.unused_data
            and not decompressor.unconsumed_tail
            and logical_remaining == 0,
            f"CALVIN critical deflate stream is not exact: {relative}",
        )
    except zlib.error as exc:
        raise RuntimeError(f"CALVIN critical deflate stream is invalid: {relative}") from exc
    checksum &= 0xFFFFFFFF
    require(output_bytes == logical_bytes, f"CALVIN critical logical byte length differs: {relative}")
    require(checksum == crc32, f"CALVIN critical CRC32 differs: {relative}")
    require(digest.digest() == logical_sha256, f"CALVIN critical logical SHA-256 differs: {relative}")
    return {"bytes": output_bytes, "crc32": checksum, "sha256": digest.hexdigest()}


def _v4_storage_identity(
    manifest: Mapping[str, Any],
    *,
    manifest_file_sha256: str,
) -> dict[str, Any]:
    payload = {
        "archive": {
            "bytes": manifest["archive"]["bytes"],
            "path": manifest["archive"]["path"],
            "sha256": manifest["archive"]["sha256"],
            "url": manifest["archive"]["url"],
        },
        "central_directory": dict(manifest["archive"]["central_directory"]),
        "checksum_url": manifest["checksum_url"],
        "manifest": {
            "content_sha256": manifest["content_sha256"],
            "file_sha256": manifest_file_sha256,
            "schema": manifest["schema"],
        },
        "member_index": dict(manifest["storage"]["member_index"]),
        "member_inventory": dict(manifest["archive"]["member_inventory"]),
        "mode": manifest["storage"]["mode"],
        "reader_schema": manifest["storage"]["reader_schema"],
        "schema": STORAGE_IDENTITY_SCHEMA,
    }
    payload["content_sha256"] = hashlib.sha256(canonical_json_bytes(payload)).hexdigest()
    return payload


def _v4_calvin_identity(
    manifest: Mapping[str, Any],
    storage_identity: Mapping[str, Any],
) -> dict[str, Any]:
    metadata_hashes = {
        f"training/{relative}": manifest["critical_files"][f"training/{relative}"]["sha256"]
        for relative in TRAINING_METADATA_FILES
    }
    metadata_sha256 = hashlib.sha256(canonical_json_bytes(metadata_hashes)).hexdigest()
    member_index = manifest["storage"]["member_index"]
    return {
        "archive_bytes": manifest["archive"]["bytes"],
        "archive_sha256": manifest["archive"]["sha256"],
        "central_directory_sha256": manifest["archive"]["central_directory"]["sha256"],
        "dataset_manifest_file_sha256": storage_identity["manifest"]["file_sha256"],
        "dataset_manifest_schema": manifest["schema"],
        "dataset_manifest_sha256": manifest["content_sha256"],
        "member_index": {
            "bytes": member_index["bytes"],
            "path": member_index["path"],
            "schema": member_index["schema"],
            "sha256": member_index["sha256"],
        },
        "member_inventory_sha256": manifest["archive"]["member_inventory"]["sha256"],
        "metadata_files": list(TRAINING_METADATA_FILES),
        "metadata_sha256": metadata_sha256,
        "name": "task_ABC_D",
        "reader_schema": manifest["storage"]["reader_schema"],
        "split": "training",
        "storage_identity_sha256": storage_identity["content_sha256"],
        "storage_mode": manifest["storage"]["mode"],
    }


def validate_official_dataset_identity(value: Any) -> dict[str, Any]:
    """Validate the persisted v2 official dataset-attestation payload."""

    required = {
        "archive",
        "archive_bytes",
        "archive_sha256",
        "calvin_identity",
        "central_directory_sha256",
        "critical_files",
        "critical_rows_sha256",
        "dataset_manifest_file_sha256",
        "dataset_manifest_schema",
        "dataset_manifest_sha256",
        "dataset_root",
        "manifest",
        "member_index",
        "member_index_attestation",
        "member_index_bytes",
        "member_index_path",
        "member_index_schema",
        "member_index_sha256",
        "member_inventory",
        "member_inventory_sha256",
        "metadata_files",
        "metadata_sha256",
        "reader_schema",
        "storage",
        "storage_identity",
        "storage_identity_sha256",
        "storage_mode",
        "validation_critical_files",
    }
    require(isinstance(value, dict) and set(value) == required, "official CALVIN dataset-attestation fields differ")
    calvin = value["calvin_identity"]
    calvin_fields = {
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
    require(
        isinstance(calvin, dict) and set(calvin) == calvin_fields, "official CALVIN permanent identity fields differ"
    )
    member_index = calvin.get("member_index")
    require(
        isinstance(member_index, dict)
        and set(member_index) == {"bytes", "path", "schema", "sha256"}
        and type(member_index.get("bytes")) is int
        and member_index["bytes"] > 0
        and member_index.get("path") == MEMBER_INDEX_NAME
        and member_index.get("schema") == MEMBER_INDEX_SCHEMA
        and _valid_sha256(member_index.get("sha256")),
        "official CALVIN permanent member-index identity differs",
    )
    inventory = value.get("member_inventory")
    inventory_fields = {
        "compressed_bytes",
        "directory_member_count",
        "file_member_count",
        "member_count",
        "npz_member_count",
        "sha256",
        "uncompressed_bytes",
    }
    require(
        isinstance(inventory, dict)
        and set(inventory) == inventory_fields
        and inventory.get("member_count") == MEMBER_COUNT
        and inventory.get("file_member_count") == FILE_MEMBER_COUNT
        and inventory.get("directory_member_count") == DIRECTORY_MEMBER_COUNT
        and inventory.get("npz_member_count") == NPZ_MEMBER_COUNT
        and type(inventory.get("compressed_bytes")) is int
        and inventory["compressed_bytes"] > 0
        and type(inventory.get("uncompressed_bytes")) is int
        and inventory["uncompressed_bytes"] > 0
        and _valid_sha256(inventory.get("sha256")),
        "official CALVIN member inventory differs",
    )
    expected_scalars = {
        "archive_bytes": ARCHIVE_BYTES,
        "archive_sha256": ARCHIVE_SHA256,
        "central_directory_sha256": CENTRAL_DIRECTORY_SHA256,
        "dataset_manifest_schema": DATASET_MANIFEST_SCHEMA,
        "member_inventory_sha256": inventory["sha256"],
        "name": "task_ABC_D",
        "reader_schema": ARCHIVE_READER_SCHEMA,
        "split": "training",
        "storage_mode": "archive-direct",
    }
    require(
        all(calvin.get(name) == expected for name, expected in expected_scalars.items()),
        "official CALVIN permanent identity scalar differs",
    )
    for name in (
        "dataset_manifest_file_sha256",
        "dataset_manifest_sha256",
        "metadata_sha256",
        "storage_identity_sha256",
    ):
        require(_valid_sha256(calvin.get(name)), f"official CALVIN permanent identity hash is invalid: {name}")
    require(calvin.get("metadata_files") == list(TRAINING_METADATA_FILES), "official CALVIN metadata file list differs")

    critical_files = value.get("critical_files")
    require(
        isinstance(critical_files, dict) and set(critical_files) == set(DATASET_CRITICAL_FILES),
        "official CALVIN critical-file inventory differs",
    )
    for relative, identity in critical_files.items():
        require(
            isinstance(identity, dict)
            and set(identity) == {"bytes", "crc32", "sha256"}
            and type(identity.get("bytes")) is int
            and identity["bytes"] >= 0
            and type(identity.get("crc32")) is int
            and 0 <= identity["crc32"] <= _UINT32_MAX
            and _valid_sha256(identity.get("sha256")),
            f"official CALVIN critical-file identity differs: {relative}",
        )
    metadata_hashes = {
        f"training/{relative}": critical_files[f"training/{relative}"]["sha256"] for relative in TRAINING_METADATA_FILES
    }
    require(
        hashlib.sha256(canonical_json_bytes(metadata_hashes)).hexdigest() == calvin["metadata_sha256"],
        "official CALVIN metadata identity differs from its critical files",
    )

    aliases = {
        "archive_bytes": calvin["archive_bytes"],
        "archive_sha256": calvin["archive_sha256"],
        "central_directory_sha256": calvin["central_directory_sha256"],
        "dataset_manifest_file_sha256": calvin["dataset_manifest_file_sha256"],
        "dataset_manifest_schema": calvin["dataset_manifest_schema"],
        "dataset_manifest_sha256": calvin["dataset_manifest_sha256"],
        "member_index_bytes": member_index["bytes"],
        "member_index_path": member_index["path"],
        "member_index_schema": member_index["schema"],
        "member_index_sha256": member_index["sha256"],
        "member_inventory_sha256": calvin["member_inventory_sha256"],
        "metadata_files": calvin["metadata_files"],
        "metadata_sha256": calvin["metadata_sha256"],
        "reader_schema": calvin["reader_schema"],
        "storage_identity_sha256": calvin["storage_identity_sha256"],
        "storage_mode": calvin["storage_mode"],
    }
    require(all(value.get(name) == expected for name, expected in aliases.items()), "official CALVIN aliases differ")
    require(value.get("member_index") == member_index, "official CALVIN nested member-index alias differs")

    storage_identity = value["storage_identity"]
    storage_fields = {
        "archive",
        "central_directory",
        "checksum_url",
        "content_sha256",
        "manifest",
        "member_index",
        "member_inventory",
        "mode",
        "reader_schema",
        "schema",
    }
    require(
        isinstance(storage_identity, dict) and set(storage_identity) == storage_fields,
        "official CALVIN storage identity fields differ",
    )
    unsigned_storage = {name: item for name, item in storage_identity.items() if name != "content_sha256"}
    require(
        storage_identity.get("content_sha256") == hashlib.sha256(canonical_json_bytes(unsigned_storage)).hexdigest()
        and storage_identity["content_sha256"] == calvin["storage_identity_sha256"],
        "official CALVIN storage identity content hash differs",
    )
    require(
        storage_identity.get("schema") == STORAGE_IDENTITY_SCHEMA
        and storage_identity.get("mode") == "archive-direct"
        and storage_identity.get("reader_schema") == ARCHIVE_READER_SCHEMA
        and storage_identity.get("checksum_url") == CHECKSUM_URL
        and storage_identity.get("archive")
        == {"bytes": ARCHIVE_BYTES, "path": ARCHIVE_NAME, "sha256": ARCHIVE_SHA256, "url": ARCHIVE_URL}
        and storage_identity.get("central_directory")
        == {
            "bytes": CENTRAL_DIRECTORY_BYTES,
            "entries": MEMBER_COUNT,
            "offset": CENTRAL_DIRECTORY_OFFSET,
            "sha256": CENTRAL_DIRECTORY_SHA256,
            "zip64": True,
        }
        and storage_identity.get("member_index") == member_index
        and storage_identity.get("member_inventory") == value["member_inventory"],
        "official CALVIN storage identity cross-binding differs",
    )
    manifest_identity = storage_identity.get("manifest")
    require(
        manifest_identity
        == {
            "content_sha256": calvin["dataset_manifest_sha256"],
            "file_sha256": calvin["dataset_manifest_file_sha256"],
            "schema": DATASET_MANIFEST_SCHEMA,
        },
        "official CALVIN storage/manifest identity differs",
    )
    reconstructed_manifest = {
        "archive": {
            **storage_identity["archive"],
            "central_directory": storage_identity["central_directory"],
            "member_inventory": storage_identity["member_inventory"],
        },
        "checksum_url": storage_identity["checksum_url"],
        "content_sha256": calvin["dataset_manifest_sha256"],
        "critical_files": critical_files,
        "dataset": "task_ABC_D",
        "schema": DATASET_MANIFEST_SCHEMA,
        "storage": {
            "derived_artifacts": {
                "state_action_sidecar": None,
                "state_action_sidecar_schema_hook": STATE_ACTION_SIDECAR_SCHEMA,
            },
            "materialized_files": list(DATASET_CRITICAL_FILES),
            "member_index": member_index,
            "mode": "archive-direct",
            "reader_schema": ARCHIVE_READER_SCHEMA,
            "verification": ARCHIVE_DIRECT_VERIFICATION,
        },
    }
    require(
        _content_sha256(reconstructed_manifest, "content_sha256") == calvin["dataset_manifest_sha256"],
        "official CALVIN manifest content hash differs from its bound identities",
    )
    require(
        value.get("storage")
        == {
            "materialized_files": list(DATASET_CRITICAL_FILES),
            "mode": "archive-direct",
            "reader_schema": ARCHIVE_READER_SCHEMA,
        },
        "official CALVIN storage dispatch identity differs",
    )

    root = value.get("dataset_root")
    require(
        isinstance(root, str) and Path(root).is_absolute() and Path(root).name == "task_ABC_D", "dataset root differs"
    )
    archive_report = value.get("archive")
    require(
        isinstance(archive_report, dict)
        and set(archive_report)
        == {
            "bytes",
            "central_directory_bytes",
            "central_directory_sha256",
            "full_sha256_contract",
            "path",
            "verification",
            "zip64",
        }
        and archive_report["bytes"] == ARCHIVE_BYTES
        and archive_report["central_directory_bytes"] == CENTRAL_DIRECTORY_BYTES
        and archive_report["central_directory_sha256"] == CENTRAL_DIRECTORY_SHA256
        and archive_report["full_sha256_contract"] == ARCHIVE_SHA256
        and archive_report["path"] == str(Path(root).parent / ARCHIVE_NAME)
        and archive_report["verification"] == LIVE_ARCHIVE_VERIFICATION,
        "official CALVIN live archive attestation differs",
    )
    zip64 = archive_report["zip64"]
    require(
        isinstance(zip64, dict)
        and set(zip64)
        == {
            "classic_eocd_sentinels",
            "record_bytes",
            "sha256",
            "version_made",
            "version_needed",
            "zip64_eocd_offset",
        }
        and zip64["classic_eocd_sentinels"] is True
        and zip64["record_bytes"] == ZIP64_TRAILER_BYTES
        and zip64["sha256"] == ZIP64_TRAILER_SHA256
        and zip64["version_made"] == ZIP64_VERSION_MADE
        and zip64["version_needed"] == 45
        and zip64["zip64_eocd_offset"] == CENTRAL_DIRECTORY_OFFSET + CENTRAL_DIRECTORY_BYTES,
        "official CALVIN live ZIP64 attestation differs",
    )
    manifest = value.get("manifest")
    require(
        isinstance(manifest, dict)
        and manifest
        == {
            "content_sha256": calvin["dataset_manifest_sha256"],
            "file_sha256": calvin["dataset_manifest_file_sha256"],
            "path": str(Path(root).parent / MANIFEST_NAME),
            "schema": DATASET_MANIFEST_SCHEMA,
        },
        "official CALVIN manifest attestation identity differs",
    )
    index_attestation = value.get("member_index_attestation")
    require(
        isinstance(index_attestation, dict)
        and set(index_attestation)
        == {
            "bytes",
            "metadata_content_sha256",
            "path",
            "schema",
            "sha256",
            "sqlite_schema_content_sha256",
            "validation_rows_sha256",
        }
        and index_attestation["bytes"] == member_index["bytes"]
        and index_attestation["path"] == str(Path(root).parent / MEMBER_INDEX_NAME)
        and index_attestation["schema"] == MEMBER_INDEX_SCHEMA
        and index_attestation["sha256"] == member_index["sha256"]
        and all(
            _valid_sha256(index_attestation[name])
            for name in ("metadata_content_sha256", "sqlite_schema_content_sha256", "validation_rows_sha256")
        ),
        "official CALVIN member-index attestation cross-binding differs",
    )
    validation = value.get("validation_critical_files")
    require(
        isinstance(validation, dict) and set(validation) == set(VALIDATION_CRITICAL_FILES),
        "official CALVIN validation metadata inventory differs",
    )
    for relative, identity in validation.items():
        require(
            isinstance(identity, dict)
            and set(identity) == {"bytes", "crc32", "path", "sha256"}
            and type(identity.get("bytes")) is int
            and identity["bytes"] >= 0
            and type(identity.get("crc32")) is int
            and 0 <= identity["crc32"] <= _UINT32_MAX
            and identity.get("path") == str(Path(root) / relative)
            and _valid_sha256(identity.get("sha256")),
            f"official CALVIN validation metadata identity differs: {relative}",
        )
        require(
            {name: identity[name] for name in ("bytes", "crc32", "sha256")} == critical_files[relative],
            f"official CALVIN projected/archive metadata identities differ: {relative}",
        )
    validation_rows = [
        {name: validation[relative][name] for name in ("bytes", "crc32", "path", "sha256")}
        for relative in VALIDATION_CRITICAL_FILES
    ]
    require(
        hashlib.sha256(canonical_json_bytes(validation_rows)).hexdigest()
        == index_attestation["validation_rows_sha256"],
        "official CALVIN projected validation-row hash differs",
    )
    critical_rows = [{"path": relative, **critical_files[relative]} for relative in DATASET_CRITICAL_FILES]
    require(
        value.get("critical_rows_sha256") == hashlib.sha256(canonical_json_bytes(critical_rows)).hexdigest(),
        "official CALVIN critical-row hash differs",
    )
    return dict(value)


def _verify_dataset_identity_v4(dataset_root: Path) -> dict[str, Any]:
    root = Path(os.path.abspath(str(dataset_root)))
    require(root.name == "task_ABC_D", f"invalid CALVIN archive-direct dataset root: {root}")
    parent = _PinnedDirectoryPath.open(root.parent)
    projection = None
    manifest_file = archive_file = index_file = None
    try:
        projection = _PinnedDirectoryPath.open(root)
        require(
            _stable_stat(os.fstat(projection.descriptor))
            == _stable_stat(os.stat(root.name, dir_fd=parent.descriptor, follow_symlinks=False)),
            "CALVIN projected-root parent binding differs",
        )
        files, directories = projection.exact_inventory()
        expected_directories = {
            "training",
            "training/.hydra",
            "training/lang_annotations",
            "validation",
            "validation/.hydra",
        }
        require(files == set(DATASET_CRITICAL_FILES), "CALVIN v4 projected file inventory differs")
        require(directories == expected_directories, "CALVIN v4 projected directory inventory differs")

        manifest_file = parent.open_regular(MANIFEST_NAME)
        manifest_identity, manifest_raw = manifest_file.digest(collect=True, maximum_bytes=16 * 1024 * 1024)
        assert manifest_raw is not None
        manifest = _decode_strict_json(manifest_raw, manifest_file.path)
        _validate_archive_direct_manifest(manifest)

        archive_file = parent.open_regular(ARCHIVE_NAME)
        require(archive_file.identity[4] == ARCHIVE_BYTES, "CALVIN archive-direct ZIP byte length differs")
        central_sha256 = _hash_archive_central_directory(archive_file)
        require(central_sha256 == CENTRAL_DIRECTORY_SHA256, "CALVIN live central-directory SHA-256 differs")
        zip64 = _validate_zip64_tail(archive_file)

        index_file = parent.open_regular(MEMBER_INDEX_NAME)
        index_identity, _unused = index_file.digest()
        index_contract = manifest["storage"]["member_index"]
        require(
            index_identity["bytes"] == index_contract["bytes"] and index_identity["sha256"] == index_contract["sha256"],
            "CALVIN live v2 member index differs from the manifest",
        )
        uri = f"file:/proc/self/fd/{index_file.descriptor}?mode=ro&immutable=1"
        try:
            with sqlite3.connect(uri, uri=True) as connection:
                connection.execute("PRAGMA query_only=ON")
                connection.execute("PRAGMA trusted_schema=OFF")
                schema_report = _validate_member_index_schema(connection)
                metadata = dict(connection.execute("SELECT name,value FROM metadata"))
                expected_metadata_names = _INDEX_METADATA_BASE_NAMES | {
                    f"critical_sha256:{relative}" for relative in DATASET_CRITICAL_FILES
                }
                require(
                    set(metadata) == expected_metadata_names
                    and all(type(name) is str and type(value) is str for name, value in metadata.items()),
                    "CALVIN v2 member-index metadata inventory differs",
                )
                numeric_names = {
                    "archive_bytes",
                    "central_directory_bytes",
                    "central_directory_offset",
                    "central_directory_zip64",
                    "compressed_bytes",
                    "directory_member_count",
                    "file_member_count",
                    "member_count",
                    "npz_member_count",
                    "uncompressed_bytes",
                }
                numeric = {name: _canonical_metadata_uint(metadata, name) for name in numeric_names}
                inventory = manifest["archive"]["member_inventory"]
                require(
                    metadata["schema"] == MEMBER_INDEX_SCHEMA
                    and metadata["reader_schema"] == ARCHIVE_READER_SCHEMA
                    and metadata["state_action_sidecar_schema"] == STATE_ACTION_SIDECAR_SCHEMA
                    and metadata["state_action_sidecar_status"] == "absent"
                    and metadata["status"] == "complete"
                    and metadata["archive_root"] == "task_ABC_D"
                    and metadata["archive_sha256"] == ARCHIVE_SHA256
                    and metadata["central_directory_sha256"] == CENTRAL_DIRECTORY_SHA256
                    and metadata["member_inventory_sha256"] == inventory["sha256"],
                    "CALVIN v2 member-index generation metadata differs",
                )
                expected_numeric = {
                    "archive_bytes": ARCHIVE_BYTES,
                    "central_directory_bytes": CENTRAL_DIRECTORY_BYTES,
                    "central_directory_offset": CENTRAL_DIRECTORY_OFFSET,
                    "central_directory_zip64": 1,
                    "compressed_bytes": inventory["compressed_bytes"],
                    "directory_member_count": DIRECTORY_MEMBER_COUNT,
                    "file_member_count": FILE_MEMBER_COUNT,
                    "member_count": MEMBER_COUNT,
                    "npz_member_count": NPZ_MEMBER_COUNT,
                    "uncompressed_bytes": inventory["uncompressed_bytes"],
                }
                require(numeric == expected_numeric, "CALVIN v2 member-index numeric metadata differs")
                semantic_summary, critical_rows = _validate_v2_member_rows(connection)
                require(
                    semantic_summary
                    == {
                        "compressed_bytes": inventory["compressed_bytes"],
                        "directory_member_count": DIRECTORY_MEMBER_COUNT,
                        "file_member_count": FILE_MEMBER_COUNT,
                        "member_count": MEMBER_COUNT,
                        "npz_member_count": NPZ_MEMBER_COUNT,
                        "uncompressed_bytes": inventory["uncompressed_bytes"],
                    },
                    "CALVIN v2 member-index semantic aggregates differ",
                )

                projected: dict[str, dict[str, Any]] = {}
                archive_rows: list[dict[str, Any]] = []
                for relative in DATASET_CRITICAL_FILES:
                    row = critical_rows[relative]
                    archived = _authenticate_indexed_archive_member(archive_file, row, relative)
                    expected = manifest["critical_files"][relative]
                    require(archived == expected, f"CALVIN archive critical identity differs: {relative}")
                    require(
                        metadata[f"critical_sha256:{relative}"] == expected["sha256"],
                        f"CALVIN index/manifest critical identity differs: {relative}",
                    )
                    projected_file = projection.open_regular(relative)
                    try:
                        identity, _raw = projected_file.digest(crc32=True)
                    finally:
                        projected_file.close()
                    require(
                        {name: identity[name] for name in ("bytes", "crc32", "sha256")} == expected,
                        f"CALVIN projected critical metadata differs: {relative}",
                    )
                    projected[relative] = identity
                    archive_rows.append({"path": relative, **archived})

                metadata_report = [[name, metadata[name]] for name in sorted(metadata)]
                schema_report["content_sha256"] = hashlib.sha256(canonical_json_bytes(schema_report)).hexdigest()
                index_report = {
                    **index_identity,
                    "metadata_content_sha256": hashlib.sha256(canonical_json_bytes(metadata_report)).hexdigest(),
                    "schema": MEMBER_INDEX_SCHEMA,
                    "sqlite_schema_content_sha256": schema_report["content_sha256"],
                    "validation_rows_sha256": hashlib.sha256(
                        canonical_json_bytes(
                            [
                                {name: projected[relative][name] for name in ("bytes", "crc32", "path", "sha256")}
                                for relative in VALIDATION_CRITICAL_FILES
                            ]
                        )
                    ).hexdigest(),
                }
        except sqlite3.Error as exc:
            raise RuntimeError(f"cannot authenticate CALVIN v2 member index: {index_file.path}") from exc

        archive_file.assert_bound()
        index_file.assert_bound()
        manifest_file.assert_bound()
        projection.assert_bound()
        parent.assert_bound()
        storage_identity = _v4_storage_identity(
            manifest,
            manifest_file_sha256=manifest_identity["sha256"],
        )
        calvin_identity = _v4_calvin_identity(manifest, storage_identity)
        permanent_index = calvin_identity["member_index"]
        result = {
            "archive": {
                "bytes": ARCHIVE_BYTES,
                "central_directory_bytes": CENTRAL_DIRECTORY_BYTES,
                "central_directory_sha256": central_sha256,
                "full_sha256_contract": ARCHIVE_SHA256,
                "path": str(archive_file.path),
                "verification": LIVE_ARCHIVE_VERIFICATION,
                "zip64": zip64,
            },
            "archive_bytes": calvin_identity["archive_bytes"],
            "archive_sha256": calvin_identity["archive_sha256"],
            "calvin_identity": calvin_identity,
            "central_directory_sha256": calvin_identity["central_directory_sha256"],
            "critical_files": {
                relative: dict(manifest["critical_files"][relative]) for relative in DATASET_CRITICAL_FILES
            },
            "critical_rows_sha256": hashlib.sha256(canonical_json_bytes(archive_rows)).hexdigest(),
            "dataset_manifest_file_sha256": calvin_identity["dataset_manifest_file_sha256"],
            "dataset_manifest_schema": calvin_identity["dataset_manifest_schema"],
            "dataset_manifest_sha256": calvin_identity["dataset_manifest_sha256"],
            "dataset_root": str(root),
            "manifest": {
                "content_sha256": manifest["content_sha256"],
                "file_sha256": manifest_identity["sha256"],
                "path": str(manifest_file.path),
                "schema": DATASET_MANIFEST_SCHEMA,
            },
            "member_index": permanent_index,
            "member_index_attestation": index_report,
            "member_index_bytes": permanent_index["bytes"],
            "member_index_path": permanent_index["path"],
            "member_index_schema": permanent_index["schema"],
            "member_index_sha256": permanent_index["sha256"],
            "member_inventory": dict(manifest["archive"]["member_inventory"]),
            "member_inventory_sha256": calvin_identity["member_inventory_sha256"],
            "metadata_files": calvin_identity["metadata_files"],
            "metadata_sha256": calvin_identity["metadata_sha256"],
            "reader_schema": calvin_identity["reader_schema"],
            "storage": {
                "materialized_files": list(DATASET_CRITICAL_FILES),
                "mode": "archive-direct",
                "reader_schema": ARCHIVE_READER_SCHEMA,
            },
            "storage_identity": storage_identity,
            "storage_identity_sha256": storage_identity["content_sha256"],
            "storage_mode": calvin_identity["storage_mode"],
            "validation_critical_files": {relative: projected[relative] for relative in VALIDATION_CRITICAL_FILES},
        }
        return validate_official_dataset_identity(result)
    finally:
        for pinned in (index_file, archive_file, manifest_file):
            if pinned is not None:
                pinned.close()
        if projection is not None:
            projection.close()
        parent.close()


def verify_dataset_identity(dataset_root: Path, *, allow_legacy_v3: bool = False) -> dict[str, Any]:
    """Dispatch solely from an exact manifest schema/root-field pair.

    The official path accepts only v4 archive-direct storage.  The opt-in v3
    result exists for extracted-tree parity diagnostics and cannot be promoted
    into an official runtime/data attestation.
    """

    root = Path(os.path.abspath(str(dataset_root)))
    manifest_path = root.parent / MANIFEST_NAME
    manifest, _digest = _read_strict_json(manifest_path)
    schema = manifest.get("schema")
    fields = set(manifest)
    if schema == DATASET_MANIFEST_SCHEMA and fields == _V4_MANIFEST_FIELDS:
        return _verify_dataset_identity_v4(root)
    if schema == LEGACY_DATASET_MANIFEST_SCHEMA and fields == _V3_MANIFEST_FIELDS:
        require(allow_legacy_v3, "legacy CALVIN v3 is parity-only and forbidden for official attestation")
        return _verify_dataset_identity_v3(root)
    raise RuntimeError(
        f"CALVIN manifest schema/root inventory is unsupported or ambiguous: schema={schema!r}, fields={sorted(fields)}"
    )


def build_runtime_attestation(source_root: Path, *, script_dir: Path | None = None) -> dict[str, Any]:
    root = source_root.resolve()
    require(
        str(root) == os.environ.get("CALVIN_SOURCE_ROOT"),
        "--source-root differs from canonical CALVIN_SOURCE_ROOT",
    )
    evaluator_root = (script_dir or _SCRIPT_DIR).resolve()
    source_snapshot = (
        _IMPORT_EVALUATOR_SOURCE_IDENTITIES
        if evaluator_root == _SCRIPT_DIR
        else evaluator_source_identities(evaluator_root)
    )
    require_evaluator_sources_unchanged(source_snapshot)
    # Everything before module-origin verification is raw-byte, package-metadata,
    # platform, or Git inspection. No Hydra/OmegaConf/environment code is used.
    runtime = runtime_identity()
    checkout = verify_checkout(root)
    packages = verify_packages(evaluator_root / "constraints-py38.txt")
    official_yaml = verify_official_yaml_files(root)
    sources = evaluator_source_identities(evaluator_root)
    require(sources == source_snapshot, "runtime attestation sources differ from import-time snapshot")
    modules = verify_module_origins(root)
    require_evaluator_sources_unchanged(source_snapshot)
    payload: dict[str, Any] = {
        "checkout": checkout,
        "modules": modules,
        "official_yaml": official_yaml,
        "packages": packages,
        "runtime": runtime,
        "schema": RUNTIME_ATTESTATION_SCHEMA,
        "sources": {name: dict(identity) for name, identity in source_snapshot.items()},
    }
    payload["content_sha256"] = _content_sha256(payload, "content_sha256")
    return payload


def build_official_attestation(
    source_root: Path,
    dataset_root: Path,
    *,
    script_dir: Path | None = None,
) -> dict[str, Any]:
    evaluator_root = (script_dir or _SCRIPT_DIR).resolve()
    source_snapshot = (
        _IMPORT_EVALUATOR_SOURCE_IDENTITIES
        if evaluator_root == _SCRIPT_DIR
        else evaluator_source_identities(evaluator_root)
    )
    require_evaluator_sources_unchanged(source_snapshot)
    # Authenticate the validation data using only raw files/stdlib before even
    # importing the top-level CALVIN packages for their origin check.
    dataset = verify_dataset_identity(dataset_root)
    require(
        dataset.get("storage")
        == {
            "materialized_files": list(DATASET_CRITICAL_FILES),
            "mode": "archive-direct",
            "reader_schema": ARCHIVE_READER_SCHEMA,
        },
        "official CALVIN attestation requires the exact v4 archive-direct storage contract",
    )
    runtime = build_runtime_attestation(source_root, script_dir=script_dir)
    require(
        runtime.get("schema") == RUNTIME_ATTESTATION_SCHEMA
        and runtime.get("content_sha256") == _content_sha256(runtime, "content_sha256"),
        "CALVIN runtime attestation is invalid",
    )
    require(runtime.get("sources") == source_snapshot, "official attestation sources changed during construction")
    payload: dict[str, Any] = {
        "dataset": dataset,
        "runtime": runtime,
        "schema": ATTESTATION_SCHEMA,
    }
    payload["attestation_sha256"] = _content_sha256(payload, "attestation_sha256")
    require_evaluator_sources_unchanged(source_snapshot)
    return payload


def canonical_sequence_sha256(sequences: Sequence[Any]) -> str:
    return hashlib.sha256(canonical_json_bytes(sequences)).hexdigest()


def verify_sequences(source_root: Path) -> dict[str, Any]:
    module = importlib.import_module("calvin_agent.evaluation.multistep_sequences")
    module_path = Path(module.__file__).resolve()
    expected_path = source_root.resolve() / "calvin_models" / "calvin_agent" / "evaluation" / "multistep_sequences.py"
    require(module_path == expected_path, f"CALVIN sequence generator imported from unexpected path: {module_path}")
    sequences = module.get_sequences(1000)
    digest = canonical_sequence_sha256(sequences)
    require(
        len(sequences) == 1000 and digest == SEQUENCE_SHA256,
        f"official sequence contract changed: count={len(sequences)}, sha256={digest}",
    )
    first_state, first_tasks = sequences[0]
    expected_tasks = [
        "rotate_blue_block_right",
        "move_slider_right",
        "lift_red_block_slider",
        "place_in_slider",
        "turn_off_lightbulb",
    ]
    require(list(first_tasks) == expected_tasks, f"unexpected first official sequence: {first_tasks}")
    return {"count": len(sequences), "sha256": digest, "first_state": first_state, "first_tasks": first_tasks}


def run_egl_smoke(source_root: Path) -> dict[str, Any]:
    import hydra
    import numpy as np
    from hydra import compose, initialize_config_dir

    conf_root = source_root / "calvin_env" / "conf"
    data_root = source_root / "calvin_env" / "data"
    with initialize_config_dir(config_dir=str(conf_root), job_name="duo_vla_calvin_preflight"):
        config = compose(
            config_name="config_data_collection",
            overrides=[
                "cameras=static_and_gripper",
                "scene=calvin_scene_D_eval",
                "use_vr=false",
                f"data_path={data_root}",
            ],
        )
    env = None
    try:
        env = hydra.utils.instantiate(config.env, show_gui=False, use_vr=False, use_scene_info=True)
        observation = env.reset()
        static = observation["rgb_obs"]["rgb_static"]
        gripper = observation["rgb_obs"]["rgb_gripper"]
        robot_obs = observation["robot_obs"]
        scene_obs = observation["scene_obs"]
        require(
            static.shape == (200, 200, 3) and static.dtype == np.uint8,
            f"unexpected CALVIN static image: {static.shape} {static.dtype}",
        )
        require(
            gripper.shape == (84, 84, 3) and gripper.dtype == np.uint8,
            f"unexpected CALVIN gripper image: {gripper.shape} {gripper.dtype}",
        )
        require(
            robot_obs.shape == (15,) and scene_obs.shape == (24,),
            f"unexpected CALVIN state shapes: {robot_obs.shape}, {scene_obs.shape}",
        )
        action = np.asarray([0.5, -0.5, 0.25, 0.2, -0.2, 0.1, 1.0], dtype=np.float32)
        original = action.copy()
        env.step(action)
        expected = original.copy()
        expected[:3] *= 0.02
        expected[3:6] *= 0.05
        require(np.array_equal(action, expected), f"official action mutation/scaling changed: {action}")
        return {
            "action_mutates_input": True,
            "control_frequency_hz": int(env.control_freq),
            "gripper_image": list(gripper.shape),
            "robot_obs": list(robot_obs.shape),
            "scene_obs": list(scene_obs.shape),
            "static_image": list(static.shape),
        }
    finally:
        if env is not None:
            env.close()
            env.cid = -1
            env.p = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--require-dataset", action="store_true")
    parser.add_argument(
        "--legacy-v3-parity",
        action="store_true",
        help="verify an extracted v3 tree for parity only; emits no official attestation",
    )
    parser.add_argument(
        "--egl",
        action="store_true",
        help="instantiate scene D, render both cameras, and take one step",
    )
    return parser.parse_args()


def main() -> None:
    require_evaluator_sources_unchanged(_IMPORT_EVALUATOR_SOURCE_IDENTITIES)
    require_canonical_evaluator_runtime()
    args = parse_args()
    # Reserve the original stdout for the single JSON report, then leave fd 1
    # directed to stderr. Native CALVIN/PyBullet libraries write diagnostics
    # to C stdout both during execution and process teardown; they must not
    # corrupt the machine-readable attestation stream.
    sys.stdout.flush()
    report_descriptor = os.dup(sys.stdout.fileno())
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    try:
        source_root = args.source_root.resolve()
        if args.legacy_v3_parity:
            require(args.dataset_root is not None, "--legacy-v3-parity requires --dataset-root")
            require(not args.egl, "--legacy-v3-parity cannot construct the official D environment")
            runtime_attestation = build_runtime_attestation(source_root)
            dataset = verify_dataset_identity(args.dataset_root.resolve(), allow_legacy_v3=True)
            require(
                dataset.get("storage_mode") == "legacy-extracted-parity-only",
                "--legacy-v3-parity requires an exact v3 manifest/root inventory",
            )
            official_attestation = None
        elif args.dataset_root is None:
            runtime_attestation = build_runtime_attestation(source_root)
            dataset = inspect_dataset(None, args.require_dataset)
            official_attestation = None
        else:
            official_attestation = build_official_attestation(source_root, args.dataset_root)
            runtime_attestation = official_attestation["runtime"]
            dataset = {"path": str(args.dataset_root.resolve()), "present": True}
        # Sequence generation and EGL are intentionally after every raw-byte,
        # checkout, package, module-origin, and (when configured) data check.
        result: dict[str, Any] = {
            "attestation": official_attestation,
            "dataset": dataset,
            "egl": run_egl_smoke(source_root) if args.egl else {"checked": False},
            "runtime_attestation": runtime_attestation,
            "sequences": verify_sequences(source_root),
        }
        require_evaluator_sources_unchanged(_IMPORT_EVALUATOR_SOURCE_IDENTITIES)
        payload = (json.dumps(result, allow_nan=False, indent=2, sort_keys=True) + "\n").encode("utf-8")
        while payload:
            written = os.write(report_descriptor, payload)
            require(written > 0, "CALVIN preflight report write made no progress")
            payload = payload[written:]
    finally:
        os.close(report_descriptor)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"CALVIN preflight failed: {type(error).__name__}: {error}", file=sys.stderr)
        raise SystemExit(1) from error
