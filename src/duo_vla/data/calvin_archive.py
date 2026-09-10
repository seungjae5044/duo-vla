"""Authenticated, extraction-free access to the pinned CALVIN ABC->D ZIP.

This module deliberately does not use :mod:`zipfile` for production archive
access.  The official archive has almost two million entries and most central
directory offsets are ZIP64 values.  Keeping ``ZipInfo`` objects alive is both
memory-heavy and an unnecessarily broad parsing surface.  Preparation instead
streams the authenticated central directory into an immutable SQLite index;
runtime reads use only pinned file descriptors and ``pread``.

The v4 manifest and v2 member index are intentionally separate from the legacy
v3 extracted-dataset contract.  No function in this module upgrades, deletes,
or blesses an existing v3 publication in place.
"""

from __future__ import annotations

import contextlib
import copy
import ctypes
import dataclasses
import errno
import fcntl
import hashlib
import json
import os
import re
import secrets
import sqlite3
import stat
import struct
import zlib
from collections.abc import Iterator, Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final

CALVIN_ARCHIVE_BYTES: Final = 555_309_812_705
CALVIN_ARCHIVE_SHA256: Final = "c2036c67eb4c06966af1d1e1665bdb572c69e1404f5e77ffd46b384ff2b79f74"
CALVIN_ARCHIVE_URL: Final = "http://calvin.cs.uni-freiburg.de/dataset/task_ABC_D.zip"
CALVIN_CHECKSUM_URL: Final = "http://calvin.cs.uni-freiburg.de/dataset/sha256sum.txt"
CALVIN_ARCHIVE_NAME: Final = "task_ABC_D.zip"
CALVIN_DATASET_NAME: Final = "task_ABC_D"

CALVIN_MANIFEST_SCHEMA: Final = "duo-vla-calvin-dataset-manifest-v4"
CALVIN_MEMBER_INDEX_SCHEMA: Final = "duo-vla-calvin-member-index-v2"
CALVIN_ARCHIVE_READER_SCHEMA: Final = "duo-vla-calvin-archive-reader-v1"
CALVIN_STATE_ACTION_SIDECAR_SCHEMA: Final = "duo-vla-calvin-state-action-sidecar-v1"

CALVIN_INDEX_NAME: Final = "task_ABC_D.members-v2.sqlite3"
CALVIN_MANIFEST_NAME: Final = "task_ABC_D.manifest.json"

CALVIN_CRITICAL_FILES: Final = (
    "training/ep_start_end_ids.npy",
    "training/lang_annotations/auto_lang_ann.npy",
    "training/scene_info.npy",
    "training/.hydra/merged_config.yaml",
    "validation/ep_start_end_ids.npy",
    "validation/.hydra/merged_config.yaml",
)

OFFICIAL_CENTRAL_DIRECTORY_OFFSET: Final = 555_080_601_096
OFFICIAL_CENTRAL_DIRECTORY_BYTES: Final = 229_211_511
OFFICIAL_CENTRAL_DIRECTORY_SHA256: Final = "b4f79bda7f6b966b51aa419badd0f7db7a8972a7b58d6d342af60aceff0ea31b"
OFFICIAL_MEMBER_COUNT: Final = 1_894_126
OFFICIAL_FILE_MEMBER_COUNT: Final = 1_894_106
OFFICIAL_NPZ_MEMBER_COUNT: Final = 1_894_067
OFFICIAL_DIRECTORY_MEMBER_COUNT: Final = 20

_EOCD_SIGNATURE = 0x06054B50
_ZIP64_EOCD_SIGNATURE = 0x06064B50
_ZIP64_LOCATOR_SIGNATURE = 0x07064B50
_CENTRAL_SIGNATURE = 0x02014B50
_LOCAL_SIGNATURE = 0x04034B50
_ZIP64_EXTRA_ID = 0x0001
_EXTENDED_TIMESTAMP_EXTRA_ID = 0x5455
_UNIX_UID_GID_EXTRA_ID = 0x7875
_ALLOWED_EXTRA_IDS = frozenset({_ZIP64_EXTRA_ID, _EXTENDED_TIMESTAMP_EXTRA_ID, _UNIX_UID_GID_EXTRA_ID})
_UINT16_MAX = (1 << 16) - 1
_UINT32_MAX = (1 << 32) - 1
_SQLITE_INT_MAX = (1 << 63) - 1
_HASH_CHUNK_BYTES = 8 * 1024 * 1024
_DEFLATE_CHUNK_BYTES = 1024 * 1024
_MAX_EOCD_TAIL_BYTES = 22 + _UINT16_MAX
_MAX_MEMBER_NAME_BYTES = 4096
_MAX_RUNTIME_MEMBER_BYTES = 1024 * 1024 * 1024
_ZERO_SHA256 = b"\0" * 32
_EMPTY_SHA256 = hashlib.sha256(b"").digest()
_EPISODE_MEMBER_RE = re.compile(r"^(training|validation)/episode_([0-9]{7})\.npz$")

_KIND_FILE = 0
_KIND_DIRECTORY = 1
_AT_FDCWD = -100
_RENAME_NOREPLACE = 1
_AT_EMPTY_PATH = 0x1000
_VERIFICATION_CONTRACT = (
    "full-archive-sha256+streamed-central-inventory+zip64-local-header+raw-deflate-eof-length-crc32-logical-sha256"
)

_EOCD_STRUCT = struct.Struct("<IHHHHIIH")
_ZIP64_LOCATOR_STRUCT = struct.Struct("<IIQI")
_ZIP64_EOCD_PREFIX_STRUCT = struct.Struct("<IQ")
_ZIP64_EOCD_BODY_STRUCT = struct.Struct("<HHIIQQQQ")
_CENTRAL_STRUCT = struct.Struct("<I6H3I5H2I")
_LOCAL_STRUCT = struct.Struct("<I5H3I2H")
_EXTRA_HEADER_STRUCT = struct.Struct("<HH")


class CalvinArchiveError(RuntimeError):
    """Base class for archive-direct failures."""


class CalvinArchiveValidationError(CalvinArchiveError):
    """The opened archive, index, or manifest violated the pinned contract."""


class CalvinArchivePublicationError(CalvinArchiveError):
    """A staged v4 generation could not be published without replacement."""


@dataclasses.dataclass(frozen=True, slots=True)
class FileIdentity:
    device: int
    inode: int
    mode: int
    link_count: int
    size: int
    mtime_ns: int
    ctime_ns: int

    @classmethod
    def from_stat(cls, value: os.stat_result) -> FileIdentity:
        return cls(
            device=value.st_dev,
            inode=value.st_ino,
            mode=value.st_mode,
            link_count=value.st_nlink,
            size=value.st_size,
            mtime_ns=value.st_mtime_ns,
            ctime_ns=value.st_ctime_ns,
        )

    def as_ephemeral_capability(self) -> dict[str, int]:
        """Return the exact live-file identity used for one job-local handoff."""

        return {
            "device": self.device,
            "inode": self.inode,
            "mode": self.mode,
            "link_count": self.link_count,
            "size": self.size,
            "mtime_ns": self.mtime_ns,
            "ctime_ns": self.ctime_ns,
        }


_EPHEMERAL_FILE_IDENTITY_FIELDS: Final = frozenset(
    {"device", "inode", "mode", "link_count", "size", "mtime_ns", "ctime_ns"}
)


def _parse_ephemeral_file_identity(value: Mapping[str, object]) -> FileIdentity:
    if not isinstance(value, Mapping) or set(value) != _EPHEMERAL_FILE_IDENTITY_FIELDS:
        raise ValueError("ephemeral archive file identity has an invalid field inventory")
    if any(type(value[name]) is not int for name in _EPHEMERAL_FILE_IDENTITY_FIELDS):
        raise ValueError("ephemeral archive file identity fields must be exact integers")
    identity = FileIdentity(
        device=value["device"],
        inode=value["inode"],
        mode=value["mode"],
        link_count=value["link_count"],
        size=value["size"],
        mtime_ns=value["mtime_ns"],
        ctime_ns=value["ctime_ns"],
    )
    if (
        identity.device < 0
        or identity.inode <= 0
        or identity.link_count != 1
        or identity.size < 0
        or identity.mtime_ns < 0
        or identity.ctime_ns < 0
        or not stat.S_ISREG(identity.mode)
    ):
        raise ValueError("ephemeral archive file identity is not a canonical single-link regular file")
    return identity


@dataclasses.dataclass(frozen=True, slots=True)
class DirectoryIdentity:
    device: int
    inode: int
    mode: int

    @classmethod
    def from_stat(cls, value: os.stat_result) -> DirectoryIdentity:
        return cls(value.st_dev, value.st_ino, value.st_mode)


class PinnedDirectoryPath:
    """An absolute directory path retained as a no-follow descriptor chain."""

    def __init__(
        self,
        path: Path,
        descriptors: list[int],
        names: list[str],
        identities: list[DirectoryIdentity],
    ) -> None:
        self.path = path
        self._descriptors = descriptors
        self._names = names
        self._identities = identities
        self._closed = False

    @property
    def descriptor(self) -> int:
        if self._closed:
            raise CalvinArchiveValidationError(f"pinned directory path is closed: {self.path}")
        return self._descriptors[-1]

    @classmethod
    def open(cls, path: str | Path) -> PinnedDirectoryPath:
        absolute = Path(os.path.abspath(os.fspath(path)))
        components = absolute.parts[1:]
        flags = os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW
        descriptors: list[int] = []
        names: list[str] = []
        identities: list[DirectoryIdentity] = []
        try:
            root_descriptor = os.open("/", flags)
            descriptors.append(root_descriptor)
            identities.append(DirectoryIdentity.from_stat(os.fstat(root_descriptor)))
            for component in components:
                descriptor = os.open(component, flags, dir_fd=descriptors[-1])
                descriptors.append(descriptor)
                names.append(component)
                identities.append(DirectoryIdentity.from_stat(os.fstat(descriptor)))
            pinned = cls(absolute, descriptors, names, identities)
            pinned.assert_bound()
            return pinned
        except BaseException as exc:
            for descriptor in reversed(descriptors):
                os.close(descriptor)
            if isinstance(exc, CalvinArchiveValidationError):
                raise
            raise CalvinArchiveValidationError(f"cannot no-follow pin directory ancestor chain: {absolute}") from exc

    def assert_bound(self) -> None:
        if self._closed:
            raise CalvinArchiveValidationError(f"pinned directory path is closed: {self.path}")
        for index, (descriptor, expected) in enumerate(zip(self._descriptors, self._identities, strict=True)):
            current = DirectoryIdentity.from_stat(os.fstat(descriptor))
            if current != expected or not stat.S_ISDIR(current.mode):
                raise CalvinArchiveValidationError(f"pinned directory inode changed: {self.path}")
            if index:
                try:
                    path_stat = os.stat(
                        self._names[index - 1],
                        dir_fd=self._descriptors[index - 1],
                        follow_symlinks=False,
                    )
                except OSError as exc:
                    raise CalvinArchiveValidationError(
                        f"pinned directory path binding disappeared: {self.path}"
                    ) from exc
                if DirectoryIdentity.from_stat(path_stat) != expected or not stat.S_ISDIR(path_stat.st_mode):
                    raise CalvinArchiveValidationError(f"pinned directory path binding changed: {self.path}")

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            for descriptor in reversed(self._descriptors):
                os.close(descriptor)

    def __enter__(self) -> PinnedDirectoryPath:
        self.assert_bound()
        return self

    def __exit__(self, *_exc_info: object) -> None:
        self.close()


@dataclasses.dataclass(frozen=True, slots=True)
class CentralDirectoryContract:
    offset: int
    size: int
    sha256: str
    member_count: int
    file_member_count: int
    directory_member_count: int
    npz_member_count: int


OFFICIAL_CENTRAL_DIRECTORY_CONTRACT: Final = CentralDirectoryContract(
    offset=OFFICIAL_CENTRAL_DIRECTORY_OFFSET,
    size=OFFICIAL_CENTRAL_DIRECTORY_BYTES,
    sha256=OFFICIAL_CENTRAL_DIRECTORY_SHA256,
    member_count=OFFICIAL_MEMBER_COUNT,
    file_member_count=OFFICIAL_FILE_MEMBER_COUNT,
    directory_member_count=OFFICIAL_DIRECTORY_MEMBER_COUNT,
    npz_member_count=OFFICIAL_NPZ_MEMBER_COUNT,
)


@dataclasses.dataclass(frozen=True, slots=True)
class CentralDirectoryInfo:
    offset: int
    size: int
    entry_count: int
    trailer_offset: int
    eocd_offset: int
    zip64: bool
    sha256: str


@dataclasses.dataclass(frozen=True, slots=True)
class InventorySummary:
    member_count: int
    file_member_count: int
    directory_member_count: int
    npz_member_count: int
    compressed_bytes: int
    uncompressed_bytes: int
    sha256: str


@dataclasses.dataclass(frozen=True, slots=True)
class PreparedCalvinArchive:
    archive_path: Path
    dataset_root: Path
    index_path: Path
    manifest_path: Path
    manifest: Mapping[str, Any]
    publication_warnings: tuple[str, ...] = ()


@dataclasses.dataclass(frozen=True, slots=True)
class ArchiveMemberRecord:
    path: str
    kind: int
    split: str | None
    global_index: int | None
    local_header_offset: int
    data_offset: int
    compressed_bytes: int
    logical_bytes: int
    method: int
    version_needed: int
    flags: int
    crc32: int
    logical_sha256: bytes
    state_action_row: int | None

    @property
    def is_file(self) -> bool:
        return self.kind == _KIND_FILE


class PinnedRegularFile:
    """A no-follow regular file whose opened inode remains the authority."""

    def __init__(self, path: Path, descriptor: int, identity: FileIdentity) -> None:
        self.path = path
        self.descriptor = descriptor
        self.identity = identity
        self._closed = False

    @classmethod
    def open(cls, path: str | Path) -> PinnedRegularFile:
        resolved = Path(os.path.abspath(os.fspath(path)))
        parent = PinnedDirectoryPath.open(resolved.parent)
        try:
            pinned = cls.open_at(parent.descriptor, resolved.name, display_path=resolved)
            parent.assert_bound()
            return pinned
        finally:
            parent.close()

    @classmethod
    def open_at(cls, parent_descriptor: int, name: str, *, display_path: Path) -> PinnedRegularFile:
        if not name or "/" in name or name in (".", ".."):
            raise ValueError("pinned relative file name must be one canonical component")
        flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW
        try:
            descriptor = os.open(name, flags, dir_fd=parent_descriptor)
        except OSError as exc:
            raise CalvinArchiveValidationError(f"cannot no-follow open regular file: {display_path}") from exc
        try:
            identity = FileIdentity.from_stat(os.fstat(descriptor))
            if not stat.S_ISREG(identity.mode) or identity.link_count != 1:
                raise CalvinArchiveValidationError(
                    f"pinned file must be regular with exactly one hard link: {display_path}"
                )
            pinned = cls(display_path, descriptor, identity)
            pinned.assert_bound_at(parent_descriptor, name)
            return pinned
        except BaseException:
            os.close(descriptor)
            raise

    def assert_unchanged(self) -> None:
        if self._closed:
            raise CalvinArchiveValidationError(f"pinned file is already closed: {self.path}")
        current = FileIdentity.from_stat(os.fstat(self.descriptor))
        if current != self.identity:
            raise CalvinArchiveValidationError(f"pinned file identity changed during use: {self.path}")

    def assert_bound_at(self, parent_descriptor: int, name: str) -> None:
        self.assert_unchanged()
        try:
            path_identity = FileIdentity.from_stat(os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False))
        except OSError as exc:
            raise CalvinArchiveValidationError(f"pinned file path binding disappeared: {self.path}") from exc
        if path_identity != self.identity:
            raise CalvinArchiveValidationError(f"pinned file path binding changed: {self.path}")

    def sha256(self) -> str:
        self.assert_unchanged()
        result = _sha256_fd(self.descriptor, self.identity.size)
        self.assert_unchanged()
        return result

    def close(self) -> None:
        if not self._closed:
            os.close(self.descriptor)
            self._closed = True

    def __enter__(self) -> PinnedRegularFile:
        if self._closed:
            raise CalvinArchiveValidationError(f"pinned file is already closed: {self.path}")
        return self

    def __exit__(self, *_exc_info: object) -> None:
        self.close()


@dataclasses.dataclass(slots=True)
class _PinnedPrepareLock:
    root_descriptor: int
    descriptor: int
    identity: FileIdentity
    display_path: Path

    def assert_bound(self) -> None:
        current = FileIdentity.from_stat(os.fstat(self.descriptor))
        try:
            path_identity = FileIdentity.from_stat(
                os.stat(
                    ".task_ABC_D.prepare.lock",
                    dir_fd=self.root_descriptor,
                    follow_symlinks=False,
                )
            )
        except OSError as exc:
            raise CalvinArchivePublicationError("prepare lock path binding disappeared") from exc
        if (
            current != self.identity
            or path_identity != self.identity
            or not stat.S_ISREG(current.mode)
            or current.link_count != 1
        ):
            raise CalvinArchivePublicationError("prepare lock path identity changed")


def _directory_binding_identity(
    parent_descriptor: int,
    name: str,
    descriptor: int,
    *,
    label: str,
) -> DirectoryIdentity:
    current = DirectoryIdentity.from_stat(os.fstat(descriptor))
    try:
        bound = DirectoryIdentity.from_stat(os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False))
    except OSError as exc:
        raise CalvinArchivePublicationError(f"{label} directory binding disappeared") from exc
    if current != bound or not stat.S_ISDIR(current.mode):
        raise CalvinArchivePublicationError(f"{label} directory binding changed")
    return current


def _same_pinned_file_object(
    value: os.stat_result,
    expected: FileIdentity,
    *,
    link_count: int,
) -> bool:
    return (
        value.st_dev == expected.device
        and value.st_ino == expected.inode
        and value.st_mode == expected.mode
        and value.st_nlink == link_count
        and value.st_size == expected.size
        and value.st_mtime_ns == expected.mtime_ns
        and stat.S_ISREG(value.st_mode)
    )


def _pread_exact(descriptor: int, size: int, offset: int, *, label: str) -> bytes:
    if size < 0 or offset < 0 or size > _SQLITE_INT_MAX or offset > _SQLITE_INT_MAX - size:
        raise CalvinArchiveValidationError(f"invalid {label} byte range")
    result = bytearray()
    cursor = offset
    remaining = size
    while remaining:
        try:
            block = os.pread(descriptor, remaining, cursor)
        except InterruptedError:
            continue
        except OSError as exc:
            raise CalvinArchiveValidationError(f"cannot pread {label}") from exc
        if not block:
            raise CalvinArchiveValidationError(f"short pread while reading {label}")
        result.extend(block)
        cursor += len(block)
        remaining -= len(block)
    return bytes(result)


def _sha256_fd(descriptor: int, size: int) -> str:
    digest = hashlib.sha256()
    cursor = 0
    while cursor < size:
        block = _pread_exact(
            descriptor,
            min(_HASH_CHUNK_BYTES, size - cursor),
            cursor,
            label="file for SHA-256",
        )
        digest.update(block)
        cursor += len(block)
    return digest.hexdigest()


def _sha256_range(descriptor: int, offset: int, size: int) -> str:
    digest = hashlib.sha256()
    cursor = offset
    remaining = size
    while remaining:
        block = _pread_exact(
            descriptor,
            min(_HASH_CHUNK_BYTES, remaining),
            cursor,
            label="central-directory range",
        )
        digest.update(block)
        cursor += len(block)
        remaining -= len(block)
    return digest.hexdigest()


def _canonical_json_bytes(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(payload, allow_nan=False, separators=(",", ":"), sort_keys=True).encode("utf-8")


def _content_sha256(payload: Mapping[str, Any]) -> str:
    content = dict(payload)
    content.pop("content_sha256", None)
    return hashlib.sha256(_canonical_json_bytes(content)).hexdigest()


def _valid_sha256(value: object) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _canonical_metadata_uint(metadata: Mapping[str, str], name: str) -> int:
    value = metadata.get(name)
    if type(value) is not str or re.fullmatch(r"0|[1-9][0-9]*", value) is None:
        raise CalvinArchiveValidationError(f"member-index metadata is not canonical ASCII decimal: {name}")
    parsed = int(value)
    if parsed > _SQLITE_INT_MAX:
        raise CalvinArchiveValidationError(f"member-index metadata exceeds the signed integer range: {name}")
    return parsed


def _checked_sqlite_integer(value: int, *, label: str) -> int:
    if value < 0 or value > _SQLITE_INT_MAX:
        raise CalvinArchiveValidationError(f"{label} is outside SQLite's signed integer range")
    return value


def _find_eocd(descriptor: int, archive_size: int) -> tuple[int, tuple[int, ...]]:
    if archive_size < _EOCD_STRUCT.size:
        raise CalvinArchiveValidationError("ZIP archive is too short for EOCD")
    tail_size = min(archive_size, _MAX_EOCD_TAIL_BYTES)
    tail_offset = archive_size - tail_size
    tail = _pread_exact(descriptor, tail_size, tail_offset, label="ZIP EOCD tail")
    signature = struct.pack("<I", _EOCD_SIGNATURE)
    candidates: list[tuple[int, tuple[int, ...]]] = []
    search_at = 0
    while True:
        relative = tail.find(signature, search_at)
        if relative < 0:
            break
        if relative + _EOCD_STRUCT.size <= len(tail):
            fields = _EOCD_STRUCT.unpack_from(tail, relative)
            comment_length = fields[-1]
            if relative + _EOCD_STRUCT.size + comment_length == len(tail):
                candidates.append((tail_offset + relative, fields))
        search_at = relative + 1
    if len(candidates) != 1:
        raise CalvinArchiveValidationError("ZIP archive must contain exactly one unambiguous terminal EOCD")
    return candidates[0]


def _read_central_directory_info(descriptor: int, archive_size: int) -> CentralDirectoryInfo:
    eocd_offset, fields = _find_eocd(descriptor, archive_size)
    (
        signature,
        disk_number,
        central_disk,
        entries_on_disk_16,
        entries_total_16,
        central_size_32,
        central_offset_32,
        _comment_length,
    ) = fields
    if signature != _EOCD_SIGNATURE or disk_number != 0 or central_disk != 0:
        raise CalvinArchiveValidationError("multi-disk or malformed ZIP EOCD is not allowed")
    if entries_on_disk_16 != entries_total_16:
        raise CalvinArchiveValidationError("ZIP EOCD per-disk and total entry counts differ")

    uses_zip64 = (
        entries_on_disk_16 == _UINT16_MAX
        or entries_total_16 == _UINT16_MAX
        or central_size_32 == _UINT32_MAX
        or central_offset_32 == _UINT32_MAX
    )
    if uses_zip64:
        locator_offset = eocd_offset - _ZIP64_LOCATOR_STRUCT.size
        if locator_offset < 0:
            raise CalvinArchiveValidationError("ZIP64 EOCD locator is missing")
        locator = _ZIP64_LOCATOR_STRUCT.unpack(
            _pread_exact(
                descriptor,
                _ZIP64_LOCATOR_STRUCT.size,
                locator_offset,
                label="ZIP64 EOCD locator",
            )
        )
        locator_signature, zip64_disk, zip64_offset, total_disks = locator
        if locator_signature != _ZIP64_LOCATOR_SIGNATURE or zip64_disk != 0 or total_disks != 1:
            raise CalvinArchiveValidationError("multi-disk or malformed ZIP64 locator is not allowed")
        prefix = _ZIP64_EOCD_PREFIX_STRUCT.unpack(
            _pread_exact(
                descriptor,
                _ZIP64_EOCD_PREFIX_STRUCT.size,
                zip64_offset,
                label="ZIP64 EOCD prefix",
            )
        )
        zip64_signature, record_body_size = prefix
        if zip64_signature != _ZIP64_EOCD_SIGNATURE or record_body_size != _ZIP64_EOCD_BODY_STRUCT.size:
            raise CalvinArchiveValidationError("malformed ZIP64 EOCD record")
        record_end = zip64_offset + _ZIP64_EOCD_PREFIX_STRUCT.size + record_body_size
        if record_end != locator_offset:
            raise CalvinArchiveValidationError("ZIP64 EOCD record does not end at its locator")
        body = _ZIP64_EOCD_BODY_STRUCT.unpack(
            _pread_exact(
                descriptor,
                _ZIP64_EOCD_BODY_STRUCT.size,
                zip64_offset + _ZIP64_EOCD_PREFIX_STRUCT.size,
                label="ZIP64 EOCD body",
            )
        )
        (
            _version_made,
            version_needed,
            zip64_disk_number,
            zip64_central_disk,
            entries_on_disk,
            entry_count,
            central_size,
            central_offset,
        ) = body
        if version_needed < 45:
            raise CalvinArchiveValidationError("ZIP64 EOCD has an invalid version-needed value")
        if zip64_disk_number != 0 or zip64_central_disk != 0 or entries_on_disk != entry_count:
            raise CalvinArchiveValidationError("multi-disk ZIP64 archives are not allowed")
        legacy_zip64_pairs = (
            (entries_on_disk_16, entries_on_disk, _UINT16_MAX, "per-disk entry count"),
            (entries_total_16, entry_count, _UINT16_MAX, "total entry count"),
            (central_size_32, central_size, _UINT32_MAX, "central-directory size"),
            (central_offset_32, central_offset, _UINT32_MAX, "central-directory offset"),
        )
        for legacy_value, zip64_value, sentinel, label in legacy_zip64_pairs:
            if legacy_value != sentinel and legacy_value != zip64_value:
                raise CalvinArchiveValidationError(f"classic/ZIP64 EOCD {label} differs")
        trailer_offset = zip64_offset
    else:
        entry_count = entries_total_16
        central_size = central_size_32
        central_offset = central_offset_32
        trailer_offset = eocd_offset
        if eocd_offset >= _ZIP64_LOCATOR_STRUCT.size:
            possible_locator = _pread_exact(
                descriptor,
                4,
                eocd_offset - _ZIP64_LOCATOR_STRUCT.size,
                label="possible unexpected ZIP64 locator",
            )
            if struct.unpack("<I", possible_locator)[0] == _ZIP64_LOCATOR_SIGNATURE:
                raise CalvinArchiveValidationError("ZIP64 locator is present without ZIP64 EOCD sentinels")

    _checked_sqlite_integer(entry_count, label="central-directory entry count")
    _checked_sqlite_integer(central_size, label="central-directory size")
    _checked_sqlite_integer(central_offset, label="central-directory offset")
    if entry_count <= 0 or central_size <= 0:
        raise CalvinArchiveValidationError("ZIP central directory must be non-empty")
    if central_offset + central_size != trailer_offset:
        raise CalvinArchiveValidationError("ZIP central directory is not contiguous with its authenticated trailer")
    central_sha256 = _sha256_range(descriptor, central_offset, central_size)
    return CentralDirectoryInfo(
        offset=central_offset,
        size=central_size,
        entry_count=entry_count,
        trailer_offset=trailer_offset,
        eocd_offset=eocd_offset,
        zip64=uses_zip64,
        sha256=central_sha256,
    )


def _parse_extra_fields(raw: bytes, *, label: str) -> dict[int, bytes]:
    fields: dict[int, bytes] = {}
    cursor = 0
    while cursor < len(raw):
        if len(raw) - cursor < _EXTRA_HEADER_STRUCT.size:
            raise CalvinArchiveValidationError(f"truncated {label} extra-field header")
        field_id, field_size = _EXTRA_HEADER_STRUCT.unpack_from(raw, cursor)
        cursor += _EXTRA_HEADER_STRUCT.size
        if field_size > len(raw) - cursor:
            raise CalvinArchiveValidationError(f"truncated {label} extra-field payload")
        if field_id in fields:
            raise CalvinArchiveValidationError(f"duplicate {label} extra-field id 0x{field_id:04x}")
        if field_id not in _ALLOWED_EXTRA_IDS:
            raise CalvinArchiveValidationError(f"unsupported {label} extra-field id 0x{field_id:04x}")
        value = raw[cursor : cursor + field_size]
        fields[field_id] = value
        cursor += field_size
    timestamp = fields.get(_EXTENDED_TIMESTAMP_EXTRA_ID)
    if timestamp is not None:
        if not timestamp or timestamp[0] & ~0x07:
            raise CalvinArchiveValidationError(f"malformed {label} extended-timestamp extra field")
        timestamp_count = (timestamp[0] & 1) + ((timestamp[0] >> 1) & 1) + ((timestamp[0] >> 2) & 1)
        if len(timestamp) < 1 + 4 * min(timestamp_count, 1):
            raise CalvinArchiveValidationError(f"truncated {label} extended-timestamp extra field")
    uid_gid = fields.get(_UNIX_UID_GID_EXTRA_ID)
    if uid_gid is not None:
        if len(uid_gid) < 3 or uid_gid[0] != 1:
            raise CalvinArchiveValidationError(f"malformed {label} Unix UID/GID extra field")
        uid_size = uid_gid[1]
        gid_size_offset = 2 + uid_size
        if uid_size == 0 or gid_size_offset >= len(uid_gid):
            raise CalvinArchiveValidationError(f"malformed {label} Unix UID/GID extra field")
        gid_size = uid_gid[gid_size_offset]
        if gid_size == 0 or gid_size_offset + 1 + gid_size != len(uid_gid):
            raise CalvinArchiveValidationError(f"malformed {label} Unix UID/GID extra field")
    return fields


def _consume_zip64_value(payload: bytes, cursor: int, width: int, *, label: str) -> tuple[int, int]:
    if width not in (4, 8) or cursor + width > len(payload):
        raise CalvinArchiveValidationError(f"missing {label} value in ZIP64 extra field")
    format_string = "<I" if width == 4 else "<Q"
    return struct.unpack_from(format_string, payload, cursor)[0], cursor + width


def _resolve_central_zip64(
    *,
    logical_32: int,
    compressed_32: int,
    local_offset_32: int,
    disk_start_16: int,
    extras: Mapping[int, bytes],
) -> tuple[int, int, int, int]:
    required = (
        logical_32 == _UINT32_MAX,
        compressed_32 == _UINT32_MAX,
        local_offset_32 == _UINT32_MAX,
        disk_start_16 == _UINT16_MAX,
    )
    payload = extras.get(_ZIP64_EXTRA_ID)
    if payload is None:
        if any(required):
            raise CalvinArchiveValidationError("central entry is missing its required ZIP64 extra field")
        return logical_32, compressed_32, local_offset_32, disk_start_16
    if not any(required):
        raise CalvinArchiveValidationError("central entry contains an unnecessary ZIP64 extra field")
    cursor = 0
    logical = logical_32
    compressed = compressed_32
    local_offset = local_offset_32
    disk_start = disk_start_16
    if required[0]:
        logical, cursor = _consume_zip64_value(payload, cursor, 8, label="logical size")
    if required[1]:
        compressed, cursor = _consume_zip64_value(payload, cursor, 8, label="compressed size")
    if required[2]:
        local_offset, cursor = _consume_zip64_value(payload, cursor, 8, label="local-header offset")
    if required[3]:
        disk_start, cursor = _consume_zip64_value(payload, cursor, 4, label="disk start")
    if cursor != len(payload):
        raise CalvinArchiveValidationError("central ZIP64 extra field has trailing bytes")
    return logical, compressed, local_offset, disk_start


def _validate_central_zip64_version(
    *,
    version_needed: int,
    logical_32: int,
    compressed_32: int,
) -> None:
    """Require ZIP64 extraction version only when a member size needs ZIP64.

    The pinned official archive uses ZIP64 solely to extend central-directory
    local-header offsets after the first 4 GiB while retaining the member's
    DEFLATE extraction version 20. The offset extension is parsed and bounded
    independently and does not change how the member itself is extracted.
    """

    if (logical_32 == _UINT32_MAX or compressed_32 == _UINT32_MAX) and version_needed < 45:
        raise CalvinArchiveValidationError("ZIP64 member size requires version-needed 45 or newer")


def _resolve_local_zip64(
    *,
    logical_32: int,
    compressed_32: int,
    extras: Mapping[int, bytes],
) -> tuple[int, int]:
    logical_required = logical_32 == _UINT32_MAX
    compressed_required = compressed_32 == _UINT32_MAX
    payload = extras.get(_ZIP64_EXTRA_ID)
    if payload is None:
        if logical_required or compressed_required:
            raise CalvinArchiveValidationError("local header is missing its required ZIP64 extra field")
        return logical_32, compressed_32
    if not logical_required and not compressed_required:
        raise CalvinArchiveValidationError("local header contains an unnecessary ZIP64 extra field")
    cursor = 0
    logical = logical_32
    compressed = compressed_32
    if logical_required:
        logical, cursor = _consume_zip64_value(payload, cursor, 8, label="local logical size")
    if compressed_required:
        compressed, cursor = _consume_zip64_value(payload, cursor, 8, label="local compressed size")
    if cursor != len(payload):
        raise CalvinArchiveValidationError("local ZIP64 extra field has trailing bytes")
    return logical, compressed


def _member_paths(
    raw_name: bytes,
    *,
    archive_root: str,
    external_attributes: int,
    version_made: int,
) -> tuple[str, int]:
    if not raw_name or len(raw_name) > _MAX_MEMBER_NAME_BYTES:
        raise CalvinArchiveValidationError("ZIP member name length is invalid")
    try:
        decoded = raw_name.decode("ascii")
    except UnicodeDecodeError as exc:
        raise CalvinArchiveValidationError("ZIP member names must be ASCII") from exc
    if "\0" in decoded or "\\" in decoded or decoded.startswith("/"):
        raise CalvinArchiveValidationError(f"unsafe ZIP member path: {decoded!r}")
    prefix = f"{archive_root}/"
    if not decoded.startswith(prefix):
        raise CalvinArchiveValidationError(f"ZIP member is outside {prefix!r}: {decoded!r}")
    is_directory_name = decoded.endswith("/")
    relative_with_optional_slash = decoded[len(prefix) :]
    relative = relative_with_optional_slash[:-1] if is_directory_name else relative_with_optional_slash
    components = [] if not relative else relative.split("/")
    if any(component in ("", ".", "..") for component in components):
        raise CalvinArchiveValidationError(f"unsafe ZIP member path components: {decoded!r}")
    if any(":" in component for component in components):
        raise CalvinArchiveValidationError(f"drive-like ZIP member component is forbidden: {decoded!r}")
    if (version_made >> 8) != 3:
        raise CalvinArchiveValidationError("ZIP entries must carry Unix external attributes")
    unix_mode = (external_attributes >> 16) & 0xFFFF
    file_type = stat.S_IFMT(unix_mode)
    dos_directory = bool(external_attributes & 0x10)
    if is_directory_name:
        if file_type != stat.S_IFDIR or not dos_directory:
            raise CalvinArchiveValidationError(f"directory member has non-directory attributes: {decoded!r}")
        kind = _KIND_DIRECTORY
    else:
        if file_type != stat.S_IFREG or dos_directory:
            raise CalvinArchiveValidationError(f"file member has non-regular attributes: {decoded!r}")
        kind = _KIND_FILE
    return relative, kind


def _member_raw_name(path: str, kind: int, *, archive_root: str) -> bytes:
    if not path:
        if kind != _KIND_DIRECTORY:
            raise CalvinArchiveValidationError("archive root entry must be a directory")
        value = f"{archive_root}/"
    else:
        value = f"{archive_root}/{path}"
        if kind == _KIND_DIRECTORY:
            value += "/"
    return value.encode("ascii")


def _episode_identity(path: str, kind: int) -> tuple[str | None, int | None]:
    if kind != _KIND_FILE:
        return None, None
    match = _EPISODE_MEMBER_RE.fullmatch(path)
    if match is None:
        return None, None
    return match.group(1), int(match.group(2))


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


def _create_stage_database(path: str) -> sqlite3.Connection:
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA journal_mode=OFF")
    connection.execute("PRAGMA synchronous=OFF")
    connection.execute("PRAGMA temp_store=MEMORY")
    connection.execute("PRAGMA trusted_schema=OFF")
    connection.execute("PRAGMA application_id=1145853251")
    connection.execute("PRAGMA user_version=2")
    connection.execute(_MEMBERS_TABLE_SQL)
    connection.execute(_METADATA_TABLE_SQL)
    return connection


def _inventory_digest_update(
    digest: Any,
    *,
    raw_name: bytes,
    kind: int,
    flags: int,
    method: int,
    version_needed: int,
    crc32: int,
    compressed_bytes: int,
    logical_bytes: int,
    local_header_offset: int,
    external_attributes: int,
) -> None:
    digest.update(struct.pack("<I", len(raw_name)))
    digest.update(raw_name)
    digest.update(
        struct.pack(
            "<BHHHIQQQI",
            kind,
            flags,
            method,
            version_needed,
            crc32,
            compressed_bytes,
            logical_bytes,
            local_header_offset,
            external_attributes,
        )
    )


def _stream_central_directory(
    archive: PinnedRegularFile,
    connection: sqlite3.Connection,
    central: CentralDirectoryInfo,
    *,
    archive_root: str,
) -> InventorySummary:
    cursor = central.offset
    central_end = central.offset + central.size
    directory_paths: set[str] = set()
    required_directory_paths: set[str] = set()
    inventory_digest = hashlib.sha256()
    file_count = 0
    directory_count = 0
    npz_count = 0
    compressed_total = 0
    logical_total = 0
    for entry_number in range(central.entry_count):
        fixed = _pread_exact(
            archive.descriptor,
            _CENTRAL_STRUCT.size,
            cursor,
            label=f"central entry {entry_number}",
        )
        fields = _CENTRAL_STRUCT.unpack(fixed)
        (
            signature,
            version_made,
            version_needed,
            flags,
            method,
            _modified_time,
            _modified_date,
            crc32,
            compressed_32,
            logical_32,
            name_length,
            extra_length,
            comment_length,
            disk_start_16,
            _internal_attributes,
            external_attributes,
            local_offset_32,
        ) = fields
        if signature != _CENTRAL_SIGNATURE:
            raise CalvinArchiveValidationError(f"invalid central signature at entry {entry_number}")
        if flags != 0:
            raise CalvinArchiveValidationError("CALVIN ZIP general-purpose flags must be exactly zero")
        if name_length == 0 or comment_length != 0:
            raise CalvinArchiveValidationError("ZIP members require a name and must not have comments")
        variable_length = name_length + extra_length + comment_length
        if cursor + _CENTRAL_STRUCT.size + variable_length > central_end:
            raise CalvinArchiveValidationError("central entry extends beyond the authenticated directory")
        variable = _pread_exact(
            archive.descriptor,
            variable_length,
            cursor + _CENTRAL_STRUCT.size,
            label=f"central entry {entry_number} variable fields",
        )
        raw_name = variable[:name_length]
        extra_raw = variable[name_length : name_length + extra_length]
        extras = _parse_extra_fields(extra_raw, label="central")
        logical_bytes, compressed_bytes, local_header_offset, disk_start = _resolve_central_zip64(
            logical_32=logical_32,
            compressed_32=compressed_32,
            local_offset_32=local_offset_32,
            disk_start_16=disk_start_16,
            extras=extras,
        )
        for value, label in (
            (logical_bytes, "member logical size"),
            (compressed_bytes, "member compressed size"),
            (local_header_offset, "member local-header offset"),
        ):
            _checked_sqlite_integer(value, label=label)
        if disk_start != 0 or local_header_offset >= central.offset:
            raise CalvinArchiveValidationError("member uses another disk or points into the central directory")
        path, kind = _member_paths(
            raw_name,
            archive_root=archive_root,
            external_attributes=external_attributes,
            version_made=version_made,
        )
        parent = path.rpartition("/")[0] if path else ""
        if parent:
            required_directory_paths.add(parent)
        if kind == _KIND_DIRECTORY:
            if method != 0 or version_needed < 10 or logical_bytes != 0 or compressed_bytes != 0 or crc32 != 0:
                raise CalvinArchiveValidationError("directory entries must be empty and STORED")
            directory_paths.add(path)
            directory_count += 1
        else:
            if method != 8 or version_needed < 20:
                raise CalvinArchiveValidationError("file entries must use supported raw DEFLATE")
            file_count += 1
            if path.endswith(".npz"):
                npz_count += 1
        _validate_central_zip64_version(
            version_needed=version_needed,
            logical_32=logical_32,
            compressed_32=compressed_32,
        )
        split, global_index = _episode_identity(path, kind)
        try:
            connection.execute(
                "INSERT INTO members("
                "path,kind,split,global_index,local_header_offset,data_offset,compressed_bytes,logical_bytes,"
                "method,version_needed,flags,crc32,logical_sha256,state_action_row"
                ") VALUES(?,?,?,?,?,-1,?,?,?,?,?,?,?,NULL)",
                (
                    path,
                    kind,
                    split,
                    global_index,
                    local_header_offset,
                    compressed_bytes,
                    logical_bytes,
                    method,
                    version_needed,
                    flags,
                    crc32,
                    _ZERO_SHA256,
                ),
            )
        except sqlite3.IntegrityError as exc:
            raise CalvinArchiveValidationError(f"duplicate or colliding ZIP member path: {path!r}") from exc
        _inventory_digest_update(
            inventory_digest,
            raw_name=raw_name,
            kind=kind,
            flags=flags,
            method=method,
            version_needed=version_needed,
            crc32=crc32,
            compressed_bytes=compressed_bytes,
            logical_bytes=logical_bytes,
            local_header_offset=local_header_offset,
            external_attributes=external_attributes,
        )
        compressed_total += compressed_bytes
        logical_total += logical_bytes
        cursor += _CENTRAL_STRUCT.size + variable_length
        if entry_number and entry_number % 50_000 == 0:
            connection.commit()
    if cursor != central_end:
        raise CalvinArchiveValidationError("central entry count does not consume the exact central directory")
    missing_directories = required_directory_paths - directory_paths
    if missing_directories:
        first = min(missing_directories)
        raise CalvinArchiveValidationError(f"ZIP member parent directory is not declared: {first!r}")
    connection.commit()
    return InventorySummary(
        member_count=central.entry_count,
        file_member_count=file_count,
        directory_member_count=directory_count,
        npz_member_count=npz_count,
        compressed_bytes=compressed_total,
        uncompressed_bytes=logical_total,
        sha256=inventory_digest.hexdigest(),
    )


def _local_header_data_offset(
    descriptor: int,
    record: ArchiveMemberRecord,
    *,
    archive_root: str,
    central_offset: int,
) -> int:
    fixed = _pread_exact(
        descriptor,
        _LOCAL_STRUCT.size,
        record.local_header_offset,
        label=f"local header for {record.path!r}",
    )
    (
        signature,
        version_needed,
        flags,
        method,
        _modified_time,
        _modified_date,
        crc32,
        compressed_32,
        logical_32,
        name_length,
        extra_length,
    ) = _LOCAL_STRUCT.unpack(fixed)
    if signature != _LOCAL_SIGNATURE:
        raise CalvinArchiveValidationError(f"invalid local-header signature for {record.path!r}")
    if flags != record.flags or flags != 0 or method != record.method:
        raise CalvinArchiveValidationError(f"local/central flag or method mismatch for {record.path!r}")
    if version_needed != record.version_needed:
        raise CalvinArchiveValidationError(f"local/central version-needed mismatch for {record.path!r}")
    variable_size = name_length + extra_length
    if record.local_header_offset > _SQLITE_INT_MAX - _LOCAL_STRUCT.size - variable_size:
        raise CalvinArchiveValidationError(f"local-header offset overflow for {record.path!r}")
    variable = _pread_exact(
        descriptor,
        variable_size,
        record.local_header_offset + _LOCAL_STRUCT.size,
        label=f"local variable fields for {record.path!r}",
    )
    raw_name = variable[:name_length]
    if raw_name != _member_raw_name(record.path, record.kind, archive_root=archive_root):
        raise CalvinArchiveValidationError(f"local/central member name mismatch for {record.path!r}")
    extras = _parse_extra_fields(variable[name_length:], label="local")
    uses_local_zip64 = logical_32 == _UINT32_MAX or compressed_32 == _UINT32_MAX
    if uses_local_zip64 and version_needed < 45:
        raise CalvinArchiveValidationError(f"ZIP64 local header requires version-needed 45: {record.path!r}")
    logical_bytes, compressed_bytes = _resolve_local_zip64(
        logical_32=logical_32,
        compressed_32=compressed_32,
        extras=extras,
    )
    if logical_bytes != record.logical_bytes or compressed_bytes != record.compressed_bytes or crc32 != record.crc32:
        raise CalvinArchiveValidationError(f"local/central size or CRC mismatch for {record.path!r}")
    data_offset = record.local_header_offset + _LOCAL_STRUCT.size + variable_size
    if data_offset > central_offset or record.compressed_bytes > central_offset - data_offset:
        raise CalvinArchiveValidationError(f"member data range escapes the ZIP payload: {record.path!r}")
    return data_offset


def _write_all(descriptor: int, payload: bytes, *, label: str) -> None:
    view = memoryview(payload)
    while view:
        try:
            written = os.write(descriptor, view)
        except InterruptedError:
            continue
        except OSError as exc:
            raise CalvinArchiveValidationError(f"cannot write {label}") from exc
        if written <= 0:
            raise CalvinArchiveValidationError(f"short write for {label}")
        view = view[written:]


def _inflate_member(
    descriptor: int,
    record: ArchiveMemberRecord,
    *,
    data_offset: int,
    expected_logical_sha256: bytes | None,
    output_descriptor: int | None = None,
    collect: bool = False,
    maximum_logical_bytes: int = _MAX_RUNTIME_MEMBER_BYTES,
) -> tuple[bytes | None, bytes]:
    if record.kind != _KIND_FILE or record.method != 8 or record.flags != 0:
        raise CalvinArchiveValidationError(f"member is not an allowed DEFLATE file: {record.path!r}")
    if record.logical_bytes > maximum_logical_bytes:
        raise CalvinArchiveValidationError(f"member exceeds the bounded logical-size limit: {record.path!r}")
    decompressor = zlib.decompressobj(-zlib.MAX_WBITS)
    digest = hashlib.sha256()
    checksum = 0
    output = bytearray() if collect else None
    compressed_cursor = data_offset
    compressed_remaining = record.compressed_bytes
    logical_remaining = record.logical_bytes

    def consume(block: bytes) -> None:
        nonlocal checksum, logical_remaining
        if len(block) > logical_remaining:
            raise CalvinArchiveValidationError(f"raw DEFLATE output exceeds declared size: {record.path!r}")
        logical_remaining -= len(block)
        checksum = zlib.crc32(block, checksum)
        digest.update(block)
        if output_descriptor is not None:
            _write_all(output_descriptor, block, label=f"projected member {record.path!r}")
        if output is not None:
            output.extend(block)

    while compressed_remaining:
        compressed = _pread_exact(
            descriptor,
            min(_DEFLATE_CHUNK_BYTES, compressed_remaining),
            compressed_cursor,
            label=f"compressed member {record.path!r}",
        )
        compressed_cursor += len(compressed)
        compressed_remaining -= len(compressed)
        pending = compressed
        while pending:
            previous_length = len(pending)
            decoded = decompressor.decompress(pending, logical_remaining + 1)
            pending = decompressor.unconsumed_tail
            consume(decoded)
            if pending and len(pending) == previous_length and not decoded:
                raise CalvinArchiveValidationError(f"raw DEFLATE decoder made no progress: {record.path!r}")
    flushed = decompressor.flush(logical_remaining + 1)
    consume(flushed)
    if not decompressor.eof or decompressor.unused_data or decompressor.unconsumed_tail or logical_remaining != 0:
        raise CalvinArchiveValidationError(f"raw DEFLATE stream did not terminate exactly: {record.path!r}")
    if checksum & _UINT32_MAX != record.crc32:
        raise CalvinArchiveValidationError(f"CRC32 mismatch for member: {record.path!r}")
    logical_sha256 = digest.digest()
    if expected_logical_sha256 is not None and logical_sha256 != expected_logical_sha256:
        raise CalvinArchiveValidationError(f"logical SHA-256 mismatch for member: {record.path!r}")
    return (bytes(output) if output is not None else None), logical_sha256


def _row_to_record(row: tuple[Any, ...], *, allow_unscanned: bool = False) -> ArchiveMemberRecord:
    if len(row) != 14:
        raise CalvinArchiveValidationError("member-index query returned an invalid row width")
    record = ArchiveMemberRecord(*row)
    if (
        not isinstance(record.path, str)
        or type(record.kind) is not int
        or record.kind not in (_KIND_FILE, _KIND_DIRECTORY)
        or type(record.local_header_offset) is not int
        or type(record.data_offset) is not int
        or type(record.compressed_bytes) is not int
        or type(record.logical_bytes) is not int
        or type(record.method) is not int
        or type(record.version_needed) is not int
        or type(record.flags) is not int
        or type(record.crc32) is not int
        or type(record.logical_sha256) is not bytes
        or len(record.logical_sha256) != 32
        or (record.split is not None and type(record.split) is not str)
        or (record.global_index is not None and type(record.global_index) is not int)
        or (record.state_action_row is not None and type(record.state_action_row) is not int)
    ):
        raise CalvinArchiveValidationError("member-index row contains invalid types")
    try:
        record.path.encode("ascii")
    except UnicodeEncodeError as exc:
        raise CalvinArchiveValidationError("member-index path is not canonical ASCII") from exc
    if record.path:
        if (
            record.path.startswith("/")
            or record.path.endswith("/")
            or "\\" in record.path
            or "\0" in record.path
            or any(component in ("", ".", "..") for component in record.path.split("/"))
        ):
            raise CalvinArchiveValidationError("member-index path is not canonical")
    elif record.kind != _KIND_DIRECTORY:
        raise CalvinArchiveValidationError("only the directory root may have an empty member-index path")
    expected_split, expected_global_index = _episode_identity(record.path, record.kind)
    if (record.split, record.global_index) != (expected_split, expected_global_index):
        raise CalvinArchiveValidationError("member-index split/global identity differs from its canonical path")
    if (
        record.local_header_offset < 0
        or (
            record.data_offset < record.local_header_offset + _LOCAL_STRUCT.size
            and not (allow_unscanned and record.data_offset == -1)
        )
        or record.compressed_bytes < 0
        or record.logical_bytes < 0
        or not 0 <= record.crc32 <= _UINT32_MAX
        or (record.global_index is not None and record.global_index < 0)
        or (record.state_action_row is not None and record.state_action_row < 0)
    ):
        raise CalvinArchiveValidationError("member-index row contains an invalid numeric range")
    if record.flags != 0:
        raise CalvinArchiveValidationError("member-index row contains unsupported ZIP flags")
    if record.kind == _KIND_DIRECTORY:
        if (
            record.method != 0
            or record.version_needed < 10
            or record.compressed_bytes != 0
            or record.logical_bytes != 0
            or record.crc32 != 0
            or (
                record.logical_sha256 != _EMPTY_SHA256
                and not (allow_unscanned and record.logical_sha256 == _ZERO_SHA256)
            )
            or record.state_action_row is not None
        ):
            raise CalvinArchiveValidationError("member-index directory row violates the storage contract")
    elif record.method != 8 or record.version_needed < 20:
        raise CalvinArchiveValidationError("member-index file row violates the DEFLATE contract")
    return record


_MEMBER_SELECT = (
    "SELECT path,kind,split,global_index,local_header_offset,data_offset,compressed_bytes,logical_bytes,"
    "method,version_needed,flags,crc32,logical_sha256,state_action_row FROM members"
)


def _open_directory_at(parent_descriptor: int, name: str, *, label: str) -> int:
    try:
        descriptor = os.open(
            name,
            os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW,
            dir_fd=parent_descriptor,
        )
    except OSError as exc:
        raise CalvinArchiveValidationError(f"cannot no-follow open {label} directory: {name}") from exc
    if not stat.S_ISDIR(os.fstat(descriptor).st_mode):
        os.close(descriptor)
        raise CalvinArchiveValidationError(f"{label} entry is not a directory: {name}")
    return descriptor


def _make_projection_directories(stage_descriptor: int) -> int:
    try:
        os.mkdir(CALVIN_DATASET_NAME, 0o700, dir_fd=stage_descriptor)
    except OSError as exc:
        raise CalvinArchiveValidationError("cannot create private projection root") from exc
    root_descriptor = _open_directory_at(
        stage_descriptor,
        CALVIN_DATASET_NAME,
        label="private projection root",
    )
    for relative in CALVIN_CRITICAL_FILES:
        current_descriptor = os.dup(root_descriptor)
        for component in Path(relative).parts[:-1]:
            try:
                os.mkdir(component, 0o700, dir_fd=current_descriptor)
            except FileExistsError:
                pass
            except OSError as exc:
                os.close(current_descriptor)
                os.close(root_descriptor)
                raise CalvinArchiveValidationError(f"cannot create private projection directory: {component}") from exc
            next_descriptor = _open_directory_at(
                current_descriptor,
                component,
                label="private projection",
            )
            os.close(current_descriptor)
            current_descriptor = next_descriptor
        os.close(current_descriptor)
    return root_descriptor


def _open_projection_file(root_descriptor: int, relative: str) -> int:
    if relative not in CALVIN_CRITICAL_FILES:
        raise CalvinArchiveValidationError(f"refusing to project non-critical member: {relative}")
    directory_descriptor = os.dup(root_descriptor)
    components = Path(relative).parts
    try:
        for component in components[:-1]:
            next_descriptor = _open_directory_at(
                directory_descriptor,
                component,
                label="private projection",
            )
            os.close(directory_descriptor)
            directory_descriptor = next_descriptor
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW
        return os.open(components[-1], flags, 0o600, dir_fd=directory_descriptor)
    except OSError as exc:
        raise CalvinArchiveValidationError(f"cannot exclusively create projected member: {relative}") from exc
    finally:
        os.close(directory_descriptor)


def _scan_members_and_project(
    archive: PinnedRegularFile,
    connection: sqlite3.Connection,
    central: CentralDirectoryInfo,
    projection_descriptor: int,
    *,
    archive_root: str,
) -> dict[str, dict[str, Any]]:
    connection.execute(_LOCAL_HEADER_INDEX_SQL)
    connection.commit()
    critical: dict[str, dict[str, Any]] = {}
    previous_data_end = 0
    cursor = connection.execute(f"{_MEMBER_SELECT} ORDER BY local_header_offset")
    for member_number, row in enumerate(cursor):
        record = _row_to_record(row, allow_unscanned=True)
        if record.local_header_offset < previous_data_end:
            raise CalvinArchiveValidationError(f"overlapping local member interval: {record.path!r}")
        data_offset = _local_header_data_offset(
            archive.descriptor,
            record,
            archive_root=archive_root,
            central_offset=central.offset,
        )
        previous_data_end = data_offset + record.compressed_bytes
        projected_descriptor: int | None = None
        if record.kind == _KIND_DIRECTORY:
            logical_sha256 = _EMPTY_SHA256
        else:
            if record.path in CALVIN_CRITICAL_FILES:
                projected_descriptor = _open_projection_file(projection_descriptor, record.path)
            try:
                _unused, logical_sha256 = _inflate_member(
                    archive.descriptor,
                    record,
                    data_offset=data_offset,
                    expected_logical_sha256=None,
                    output_descriptor=projected_descriptor,
                    collect=False,
                )
                if projected_descriptor is not None:
                    os.fchmod(projected_descriptor, 0o444)
                    os.fsync(projected_descriptor)
            finally:
                if projected_descriptor is not None:
                    os.close(projected_descriptor)
            if record.path in CALVIN_CRITICAL_FILES:
                critical[record.path] = {
                    "bytes": record.logical_bytes,
                    "crc32": record.crc32,
                    "sha256": logical_sha256.hex(),
                }
        connection.execute(
            "UPDATE members SET data_offset=?, logical_sha256=? WHERE path=?",
            (data_offset, logical_sha256, record.path),
        )
        if member_number and member_number % 10_000 == 0:
            connection.commit()
            archive.assert_unchanged()
    connection.commit()
    if previous_data_end > central.offset:
        raise CalvinArchiveValidationError("last member overlaps the central directory")
    if set(critical) != set(CALVIN_CRITICAL_FILES):
        missing = sorted(set(CALVIN_CRITICAL_FILES) - set(critical))
        raise CalvinArchiveValidationError(f"archive is missing critical metadata members: {missing}")
    unauthenticated = connection.execute(
        "SELECT count(*) FROM members WHERE data_offset < 0 OR logical_sha256=?",
        (_ZERO_SHA256,),
    ).fetchone()
    if unauthenticated != (0,):
        raise CalvinArchiveValidationError("member scan did not authenticate every index row")
    if connection.execute("SELECT count(*) FROM members WHERE state_action_row IS NOT NULL").fetchone() != (0,):
        raise CalvinArchiveValidationError("P0 member index unexpectedly contains state/action sidecar rows")
    try:
        connection.execute(_EPISODE_INDEX_SQL)
        connection.execute(_PHYSICAL_INDEX_SQL)
        connection.commit()
    except sqlite3.IntegrityError as exc:
        raise CalvinArchiveValidationError("duplicate CALVIN split/global frame identity") from exc
    return critical


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


def _validate_index_schema(connection: sqlite3.Connection) -> None:
    objects = {
        (row[0], row[1]): row[2]
        for row in connection.execute(
            "SELECT type,name,sql FROM sqlite_schema WHERE name NOT LIKE 'sqlite_%' ORDER BY type,name"
        )
    }
    expected_objects = {
        ("index", "members_episode"): _EPISODE_INDEX_SQL,
        ("index", "members_local_header"): _LOCAL_HEADER_INDEX_SQL,
        ("index", "members_physical"): _PHYSICAL_INDEX_SQL,
        ("table", "members"): _MEMBERS_TABLE_SQL,
        ("table", "metadata"): _METADATA_TABLE_SQL,
    }
    if objects != expected_objects:
        raise CalvinArchiveValidationError("member-index SQLite object inventory differs from v2")
    if list(connection.execute("PRAGMA table_info(members)")) != _EXPECTED_MEMBER_COLUMNS:
        raise CalvinArchiveValidationError("member-index members columns differ from v2")
    if list(connection.execute("PRAGMA table_info(metadata)")) != _EXPECTED_METADATA_COLUMNS:
        raise CalvinArchiveValidationError("member-index metadata columns differ from v2")
    index_list = {
        (row[1], row[2], row[3], row[4])
        for row in connection.execute("PRAGMA index_list(members)")
        if not row[1].startswith("sqlite_")
    }
    if index_list != {
        ("members_episode", 1, "c", 1),
        ("members_local_header", 0, "c", 0),
        ("members_physical", 0, "c", 0),
    }:
        raise CalvinArchiveValidationError("member-index index properties differ from v2")
    expected_index_columns = {
        "members_episode": ["split", "global_index"],
        "members_local_header": ["local_header_offset"],
        "members_physical": ["data_offset"],
    }
    for name, expected_columns in expected_index_columns.items():
        columns = [row[2] for row in connection.execute(f"PRAGMA index_info({name})")]
        if columns != expected_columns:
            raise CalvinArchiveValidationError(f"member-index {name} columns differ from v2")
    if connection.execute("PRAGMA application_id").fetchone() != (1145853251,):
        raise CalvinArchiveValidationError("member-index SQLite application id differs")
    if connection.execute("PRAGMA user_version").fetchone() != (2,):
        raise CalvinArchiveValidationError("member-index SQLite user version differs")
    if connection.execute("PRAGMA integrity_check").fetchone() != ("ok",):
        raise CalvinArchiveValidationError("member-index SQLite integrity check failed")


def _metadata_rows(
    *,
    archive_sha256: str,
    archive_size: int,
    archive_root: str,
    central: CentralDirectoryInfo,
    inventory: InventorySummary,
    critical: Mapping[str, Mapping[str, Any]],
) -> dict[str, str]:
    result = {
        "archive_bytes": str(archive_size),
        "archive_root": archive_root,
        "archive_sha256": archive_sha256,
        "central_directory_bytes": str(central.size),
        "central_directory_offset": str(central.offset),
        "central_directory_sha256": central.sha256,
        "central_directory_zip64": "1" if central.zip64 else "0",
        "compressed_bytes": str(inventory.compressed_bytes),
        "directory_member_count": str(inventory.directory_member_count),
        "file_member_count": str(inventory.file_member_count),
        "member_count": str(inventory.member_count),
        "member_inventory_sha256": inventory.sha256,
        "npz_member_count": str(inventory.npz_member_count),
        "reader_schema": CALVIN_ARCHIVE_READER_SCHEMA,
        "schema": CALVIN_MEMBER_INDEX_SCHEMA,
        "state_action_sidecar_schema": CALVIN_STATE_ACTION_SIDECAR_SCHEMA,
        "state_action_sidecar_status": "absent",
        "status": "complete",
        "uncompressed_bytes": str(inventory.uncompressed_bytes),
    }
    for relative in CALVIN_CRITICAL_FILES:
        result[f"critical_sha256:{relative}"] = str(critical[relative]["sha256"])
    return result


def _finalize_database(connection: sqlite3.Connection, metadata: Mapping[str, str]) -> None:
    connection.executemany("INSERT INTO metadata(name,value) VALUES(?,?)", sorted(metadata.items()))
    connection.commit()
    _validate_index_schema(connection)
    expected_count = int(metadata["member_count"])
    if connection.execute("SELECT count(*) FROM members").fetchone() != (expected_count,):
        raise CalvinArchiveValidationError("member-index row count differs from authenticated inventory")
    connection.execute("PRAGMA synchronous=FULL")
    connection.commit()


def _validate_central_contract(
    central: CentralDirectoryInfo,
    inventory: InventorySummary,
    expected: CentralDirectoryContract | None,
) -> None:
    if central.entry_count != inventory.member_count:
        raise CalvinArchiveValidationError("central directory and inventory entry counts differ")
    if expected == OFFICIAL_CENTRAL_DIRECTORY_CONTRACT and not central.zip64:
        raise CalvinArchiveValidationError("the official CALVIN archive must use its pinned ZIP64 trailer")
    if expected is None:
        return
    actual = CentralDirectoryContract(
        offset=central.offset,
        size=central.size,
        sha256=central.sha256,
        member_count=inventory.member_count,
        file_member_count=inventory.file_member_count,
        directory_member_count=inventory.directory_member_count,
        npz_member_count=inventory.npz_member_count,
    )
    if actual != expected:
        raise CalvinArchiveValidationError("ZIP central-directory contract differs from the pinned inventory")


def _build_manifest(
    *,
    archive_name: str,
    archive_size: int,
    archive_sha256: str,
    archive_url: str,
    checksum_url: str,
    central: CentralDirectoryInfo,
    inventory: InventorySummary,
    critical: Mapping[str, Mapping[str, Any]],
    index_bytes: int,
    index_sha256: str,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "archive": {
            "bytes": archive_size,
            "central_directory": {
                "bytes": central.size,
                "entries": central.entry_count,
                "offset": central.offset,
                "sha256": central.sha256,
                "zip64": central.zip64,
            },
            "member_inventory": {
                "compressed_bytes": inventory.compressed_bytes,
                "directory_member_count": inventory.directory_member_count,
                "file_member_count": inventory.file_member_count,
                "member_count": inventory.member_count,
                "npz_member_count": inventory.npz_member_count,
                "sha256": inventory.sha256,
                "uncompressed_bytes": inventory.uncompressed_bytes,
            },
            "path": archive_name,
            "sha256": archive_sha256,
            "url": archive_url,
        },
        "checksum_url": checksum_url,
        "critical_files": {name: dict(critical[name]) for name in CALVIN_CRITICAL_FILES},
        "dataset": CALVIN_DATASET_NAME,
        "schema": CALVIN_MANIFEST_SCHEMA,
        "storage": {
            "derived_artifacts": {
                "state_action_sidecar": None,
                "state_action_sidecar_schema_hook": CALVIN_STATE_ACTION_SIDECAR_SCHEMA,
            },
            "materialized_files": list(CALVIN_CRITICAL_FILES),
            "member_index": {
                "bytes": index_bytes,
                "path": CALVIN_INDEX_NAME,
                "schema": CALVIN_MEMBER_INDEX_SCHEMA,
                "sha256": index_sha256,
            },
            "mode": "archive-direct",
            "reader_schema": CALVIN_ARCHIVE_READER_SCHEMA,
            "verification": _VERIFICATION_CONTRACT,
        },
    }
    payload["content_sha256"] = _content_sha256(payload)
    return payload


@dataclasses.dataclass(slots=True)
class _PublicationFile:
    descriptor: int
    identity: FileIdentity
    sha256: str
    display_path: Path
    source_parent_descriptor: int | None
    source_name: str | None

    @property
    def anonymous(self) -> bool:
        return self.source_name is None

    def assert_source_bound(self) -> None:
        current = FileIdentity.from_stat(os.fstat(self.descriptor))
        if current != self.identity:
            raise CalvinArchivePublicationError(f"staged publication inode changed: {self.display_path}")
        if self.anonymous:
            if current.link_count != 0:
                raise CalvinArchivePublicationError("anonymous staged publication unexpectedly has a link")
            return
        if self.source_parent_descriptor is None or self.source_name is None:
            raise AssertionError("named publication source is missing its parent binding")
        try:
            bound = FileIdentity.from_stat(
                os.stat(
                    self.source_name,
                    dir_fd=self.source_parent_descriptor,
                    follow_symlinks=False,
                )
            )
        except OSError as exc:
            raise CalvinArchivePublicationError(f"staged source binding disappeared: {self.display_path}") from exc
        if bound != self.identity:
            raise CalvinArchivePublicationError(f"staged source binding changed: {self.display_path}")

    def close(self) -> None:
        os.close(self.descriptor)


def _publication_file_from_named(
    parent_descriptor: int,
    name: str,
    *,
    display_path: Path,
) -> _PublicationFile:
    pinned = PinnedRegularFile.open_at(parent_descriptor, name, display_path=display_path)
    try:
        digest = pinned.sha256()
        os.fsync(pinned.descriptor)
        pinned.assert_bound_at(parent_descriptor, name)
        return _PublicationFile(
            descriptor=pinned.descriptor,
            identity=pinned.identity,
            sha256=digest,
            display_path=display_path,
            source_parent_descriptor=parent_descriptor,
            source_name=name,
        )
    except BaseException:
        pinned.close()
        raise


def _write_manifest_source(
    _root_descriptor: int,
    stage_descriptor: int,
    stage_path: Path,
    payload: Mapping[str, Any],
) -> _PublicationFile:
    raw = (json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n").encode("utf-8")
    # Some filesystems accept O_TMPFILE creation but reject the later
    # linkat(AT_EMPTY_PATH) with ENOENT.  Discovering that incompatibility at
    # the final commit point is too late: the projection and index have
    # already been published.  The stage is private, pinned, and exclusively
    # created, so a named source inside it has the same no-overwrite commit
    # semantics while remaining portable across those filesystems.
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW
    try:
        named_descriptor = os.open(CALVIN_MANIFEST_NAME, flags, 0o600, dir_fd=stage_descriptor)
    except OSError as exc:
        raise CalvinArchivePublicationError("cannot create named manifest fallback source") from exc
    try:
        _write_all(named_descriptor, raw, label="named manifest fallback source")
        os.fchmod(named_descriptor, 0o444)
        os.fsync(named_descriptor)
    finally:
        os.close(named_descriptor)
    return _publication_file_from_named(
        stage_descriptor,
        CALVIN_MANIFEST_NAME,
        display_path=stage_path / CALVIN_MANIFEST_NAME,
    )


def _fsync_directory_descriptor(descriptor: int, *, label: str) -> None:
    try:
        os.fsync(descriptor)
    except OSError as exc:
        raise CalvinArchivePublicationError(f"cannot fsync {label} directory") from exc


def _seal_projection_tree(descriptor: int, *, prefix: str = "") -> None:
    with os.scandir(os.dup(descriptor)) as scanner:
        names = sorted(entry.name for entry in scanner)
    for name in names:
        relative = f"{prefix}/{name}" if prefix else name
        try:
            value = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
        except OSError as exc:
            raise CalvinArchiveValidationError(f"cannot stat private projected metadata entry: {relative}") from exc
        if stat.S_ISDIR(value.st_mode):
            child = _open_directory_at(descriptor, name, label="private projection")
            try:
                _seal_projection_tree(child, prefix=relative)
            finally:
                os.close(child)
        elif stat.S_ISREG(value.st_mode):
            try:
                file_descriptor = os.open(
                    name,
                    os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW,
                    dir_fd=descriptor,
                )
            except OSError as exc:
                raise CalvinArchiveValidationError(f"cannot open private projected metadata: {relative}") from exc
            try:
                identity = FileIdentity.from_stat(os.fstat(file_descriptor))
                if not stat.S_ISREG(identity.mode) or identity.link_count != 1:
                    raise CalvinArchiveValidationError(
                        f"projected metadata is not a single-link regular file: {relative}"
                    )
                os.fsync(file_descriptor)
            finally:
                os.close(file_descriptor)
        else:
            raise CalvinArchiveValidationError(f"private projection contains a non-regular entry: {relative}")
    os.fchmod(descriptor, 0o555)
    _fsync_directory_descriptor(descriptor, label=f"projection {prefix or '/'}")


def _renameat2_noreplace(
    source_parent_descriptor: int,
    source_name: str,
    destination_parent_descriptor: int,
    destination_name: str,
    *,
    label: str,
) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise CalvinArchivePublicationError("renameat2(RENAME_NOREPLACE) is unavailable")
    renameat2.argtypes = (ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint)
    renameat2.restype = ctypes.c_int
    source_raw = os.fsencode(source_name)
    destination_raw = os.fsencode(destination_name)
    result = renameat2(
        source_parent_descriptor,
        source_raw,
        destination_parent_descriptor,
        destination_raw,
        _RENAME_NOREPLACE,
    )
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number == errno.EEXIST:
        raise CalvinArchivePublicationError(f"refusing to replace existing publication: {destination_name}")
    try:
        message = os.strerror(error_number)
    except ValueError:
        message = f"errno {error_number}"
    raise CalvinArchivePublicationError(f"cannot publish staged {label} {destination_name}: {message}")


def _linkat_empty_path(source_descriptor: int, destination_descriptor: int, destination_name: str) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    linkat = getattr(libc, "linkat", None)
    if linkat is None:
        raise CalvinArchivePublicationError("linkat(AT_EMPTY_PATH) is unavailable")
    linkat.argtypes = (ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_int)
    linkat.restype = ctypes.c_int
    result = linkat(
        source_descriptor,
        b"",
        destination_descriptor,
        os.fsencode(destination_name),
        _AT_EMPTY_PATH,
    )
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number == errno.EEXIST:
        raise CalvinArchivePublicationError(f"refusing to replace existing publication: {destination_name}")
    raise CalvinArchivePublicationError(
        f"cannot publish anonymous source {destination_name}: {os.strerror(error_number)}"
    )


def _published_file_matches(
    source: _PublicationFile,
    destination_descriptor: int,
    destination_name: str,
) -> bool:
    try:
        destination_stat = os.stat(
            destination_name,
            dir_fd=destination_descriptor,
            follow_symlinks=False,
        )
        descriptor_stat = os.fstat(source.descriptor)
        if not _same_pinned_file_object(
            destination_stat, source.identity, link_count=1
        ) or not _same_pinned_file_object(descriptor_stat, source.identity, link_count=1):
            return False
        return _sha256_fd(source.descriptor, source.identity.size) == source.sha256
    except (OSError, CalvinArchiveValidationError):
        return False


def _publish_file_noreplace(
    source: _PublicationFile,
    destination_descriptor: int,
    destination_name: str,
) -> tuple[str, ...]:
    source.assert_source_bound()
    if source.anonymous:
        _linkat_empty_path(source.descriptor, destination_descriptor, destination_name)
    else:
        if source.source_parent_descriptor is None or source.source_name is None:
            raise AssertionError("named publication source is missing its binding")
        _renameat2_noreplace(
            source.source_parent_descriptor,
            source.source_name,
            destination_descriptor,
            destination_name,
            label="file",
        )
    if not _published_file_matches(source, destination_descriptor, destination_name):
        raise CalvinArchivePublicationError(f"published destination differs from its pinned source: {destination_name}")
    warnings: list[str] = []
    if not source.anonymous and source.source_parent_descriptor is not None and source.source_name is not None:
        try:
            os.stat(
                source.source_name,
                dir_fd=source.source_parent_descriptor,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            pass
        except OSError as exc:
            warnings.append(f"cannot determine post-move source residue: {exc}")
        else:
            warnings.append(f"foreign post-move source residue retained: {source.source_name}")
    return tuple(warnings)


def _publish_directory_noreplace(
    source_parent_descriptor: int,
    source_name: str,
    source_descriptor: int,
    destination_parent_descriptor: int,
    destination_name: str,
) -> tuple[str, ...]:
    expected = _directory_binding_identity(
        source_parent_descriptor,
        source_name,
        source_descriptor,
        label="staged projection",
    )
    _renameat2_noreplace(
        source_parent_descriptor,
        source_name,
        destination_parent_descriptor,
        destination_name,
        label="directory",
    )
    try:
        destination = DirectoryIdentity.from_stat(
            os.stat(
                destination_name,
                dir_fd=destination_parent_descriptor,
                follow_symlinks=False,
            )
        )
    except OSError as exc:
        raise CalvinArchivePublicationError("published projection binding disappeared") from exc
    current = DirectoryIdentity.from_stat(os.fstat(source_descriptor))
    if destination != expected or current != expected or not stat.S_ISDIR(destination.mode):
        raise CalvinArchivePublicationError("published projection differs from its pinned source")
    warnings: list[str] = []
    try:
        os.stat(source_name, dir_fd=source_parent_descriptor, follow_symlinks=False)
    except FileNotFoundError:
        pass
    except OSError as exc:
        warnings.append(f"cannot determine post-move projection residue: {exc}")
    else:
        warnings.append(f"foreign post-move projection residue retained: {source_name}")
    return tuple(warnings)


@contextlib.contextmanager
def _exclusive_prepare_lock(data_root: PinnedDirectoryPath) -> Iterator[_PinnedPrepareLock]:
    lock_name = ".task_ABC_D.prepare.lock"
    descriptor = os.open(
        lock_name,
        os.O_RDWR | os.O_CREAT | os.O_CLOEXEC | os.O_NOFOLLOW,
        0o600,
        dir_fd=data_root.descriptor,
    )
    try:
        identity = FileIdentity.from_stat(os.fstat(descriptor))
        if not stat.S_ISREG(identity.mode) or identity.link_count != 1:
            raise CalvinArchivePublicationError("prepare lock must be a single-link regular file")
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise CalvinArchivePublicationError("another CALVIN archive preparation holds the lock") from exc
        lock = _PinnedPrepareLock(
            root_descriptor=data_root.descriptor,
            descriptor=descriptor,
            identity=identity,
            display_path=data_root.path / lock_name,
        )
        data_root.assert_bound()
        lock.assert_bound()
        yield lock
    finally:
        os.close(descriptor)


def _entry_exists_at(parent_descriptor: int, name: str) -> bool:
    try:
        os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
    except FileNotFoundError:
        return False
    except OSError as exc:
        raise CalvinArchivePublicationError(f"cannot inspect publication entry: {name}") from exc
    return True


def _directory_names(descriptor: int) -> tuple[str, ...]:
    try:
        with os.scandir(os.dup(descriptor)) as scanner:
            return tuple(sorted(entry.name for entry in scanner))
    except OSError as exc:
        raise CalvinArchivePublicationError("cannot enumerate pinned data-root directory") from exc


def prepare_calvin_archive(
    archive_path: str | Path,
    data_root: str | Path,
    *,
    expected_archive_bytes: int = CALVIN_ARCHIVE_BYTES,
    expected_archive_sha256: str = CALVIN_ARCHIVE_SHA256,
    expected_central_directory: CentralDirectoryContract | None = OFFICIAL_CENTRAL_DIRECTORY_CONTRACT,
    archive_root: str = CALVIN_DATASET_NAME,
    archive_url: str = CALVIN_ARCHIVE_URL,
    checksum_url: str = CALVIN_CHECKSUM_URL,
) -> PreparedCalvinArchive:
    """Build and atomically publish a v4 archive-direct generation.

    The caller must supply the expected archive hash for non-official fixtures.
    Existing roots, indexes, or manifests are never replaced.  A failure after
    publishing the root or index but before the manifest intentionally leaves
    an uncommitted generation that a later invocation refuses to bless.
    """

    root = Path(os.path.abspath(os.fspath(data_root)))
    archive_file = Path(os.path.abspath(os.fspath(archive_path)))
    if not _valid_sha256(expected_archive_sha256):
        raise ValueError("expected archive SHA-256 must be lowercase hexadecimal")
    if expected_archive_bytes <= 0:
        raise ValueError("expected archive byte length must be positive")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", archive_root) or "/" in archive_root:
        raise ValueError("archive_root must be one safe ASCII path component")
    dataset_path = root / CALVIN_DATASET_NAME
    index_path = root / CALVIN_INDEX_NAME
    manifest_path = root / CALVIN_MANIFEST_NAME
    with PinnedDirectoryPath.open(root) as pinned_root:  # noqa: SIM117
        with PinnedDirectoryPath.open(archive_file.parent) as archive_parent:
            with _exclusive_prepare_lock(pinned_root) as prepare_lock:
                pinned_root.assert_bound()
                archive_parent.assert_bound()
                if _entry_exists_at(archive_parent.descriptor, f"{archive_file.name}.aria2"):
                    raise CalvinArchivePublicationError("archive download control file is still present")
                stale_stages = tuple(
                    name
                    for name in _directory_names(pinned_root.descriptor)
                    if name.startswith(".task_ABC_D.archive-direct.")
                )
                if stale_stages:
                    raise CalvinArchivePublicationError(
                        f"stale archive-direct staging artifact requires explicit recovery: {root / stale_stages[0]}"
                    )
                for name in (CALVIN_DATASET_NAME, CALVIN_INDEX_NAME, CALVIN_MANIFEST_NAME):
                    if _entry_exists_at(pinned_root.descriptor, name):
                        raise CalvinArchivePublicationError(
                            f"refusing to replace existing CALVIN publication: {root / name}"
                        )
                stage_name = f".task_ABC_D.archive-direct.{os.getpid()}.{secrets.token_hex(16)}"
                try:
                    os.mkdir(stage_name, 0o700, dir_fd=pinned_root.descriptor)
                except OSError as exc:
                    raise CalvinArchivePublicationError("cannot create private archive-direct stage") from exc
                stage_path = root / stage_name
                stage_descriptor = _open_directory_at(
                    pinned_root.descriptor,
                    stage_name,
                    label="private archive-direct stage",
                )
                projection_descriptor: int | None = None
                index_source: _PublicationFile | None = None
                manifest_source: _PublicationFile | None = None
                warnings: list[str] = []
                manifest_committed = False
                try:
                    _directory_binding_identity(
                        pinned_root.descriptor,
                        stage_name,
                        stage_descriptor,
                        label="private stage",
                    )
                    projection_descriptor = _make_projection_directories(stage_descriptor)
                    with PinnedRegularFile.open_at(
                        archive_parent.descriptor,
                        archive_file.name,
                        display_path=archive_file,
                    ) as archive:
                        if archive.identity.size != expected_archive_bytes:
                            raise CalvinArchiveValidationError(
                                "archive byte length differs: "
                                f"expected={expected_archive_bytes} actual={archive.identity.size}"
                            )
                        archive_sha256 = archive.sha256()
                        archive.assert_bound_at(archive_parent.descriptor, archive_file.name)
                        archive_parent.assert_bound()
                        if archive_sha256 != expected_archive_sha256:
                            raise CalvinArchiveValidationError("archive SHA-256 differs from the pinned checksum")
                        central = _read_central_directory_info(
                            archive.descriptor,
                            archive.identity.size,
                        )
                        stage_index_path = f"/proc/self/fd/{stage_descriptor}/{CALVIN_INDEX_NAME}"
                        connection = _create_stage_database(stage_index_path)
                        try:
                            inventory = _stream_central_directory(
                                archive,
                                connection,
                                central,
                                archive_root=archive_root,
                            )
                            _validate_central_contract(
                                central,
                                inventory,
                                expected_central_directory,
                            )
                            critical = _scan_members_and_project(
                                archive,
                                connection,
                                central,
                                projection_descriptor,
                                archive_root=archive_root,
                            )
                            metadata = _metadata_rows(
                                archive_sha256=archive_sha256,
                                archive_size=archive.identity.size,
                                archive_root=archive_root,
                                central=central,
                                inventory=inventory,
                                critical=critical,
                            )
                            _finalize_database(connection, metadata)
                        finally:
                            connection.close()
                        archive.assert_unchanged()
                        archive.assert_bound_at(archive_parent.descriptor, archive_file.name)
                        index_chmod_descriptor = os.open(
                            CALVIN_INDEX_NAME,
                            os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW,
                            dir_fd=stage_descriptor,
                        )
                        try:
                            os.fchmod(index_chmod_descriptor, 0o444)
                            os.fsync(index_chmod_descriptor)
                        finally:
                            os.close(index_chmod_descriptor)
                        index_source = _publication_file_from_named(
                            stage_descriptor,
                            CALVIN_INDEX_NAME,
                            display_path=stage_path / CALVIN_INDEX_NAME,
                        )
                        _seal_projection_tree(projection_descriptor)
                        manifest = _build_manifest(
                            archive_name=archive_file.name,
                            archive_size=expected_archive_bytes,
                            archive_sha256=expected_archive_sha256,
                            archive_url=archive_url,
                            checksum_url=checksum_url,
                            central=central,
                            inventory=inventory,
                            critical=critical,
                            index_bytes=index_source.identity.size,
                            index_sha256=index_source.sha256,
                        )
                        manifest_source = _write_manifest_source(
                            pinned_root.descriptor,
                            stage_descriptor,
                            stage_path,
                            manifest,
                        )
                        _fsync_directory_descriptor(stage_descriptor, label="private stage")

                        pinned_root.assert_bound()
                        archive_parent.assert_bound()
                        archive.assert_bound_at(archive_parent.descriptor, archive_file.name)
                        prepare_lock.assert_bound()
                        index_source.assert_source_bound()
                        manifest_source.assert_source_bound()
                        _directory_binding_identity(
                            pinned_root.descriptor,
                            stage_name,
                            stage_descriptor,
                            label="private stage",
                        )
                        warnings.extend(
                            _publish_directory_noreplace(
                                stage_descriptor,
                                CALVIN_DATASET_NAME,
                                projection_descriptor,
                                pinned_root.descriptor,
                                CALVIN_DATASET_NAME,
                            )
                        )
                        _fsync_directory_descriptor(pinned_root.descriptor, label="data root")
                        warnings.extend(
                            _publish_file_noreplace(
                                index_source,
                                pinned_root.descriptor,
                                CALVIN_INDEX_NAME,
                            )
                        )
                        _fsync_directory_descriptor(pinned_root.descriptor, label="data root")

                        pinned_root.assert_bound()
                        archive_parent.assert_bound()
                        archive.assert_bound_at(archive_parent.descriptor, archive_file.name)
                        prepare_lock.assert_bound()
                        try:
                            manifest_source.assert_source_bound()
                            manifest_warnings = _publish_file_noreplace(
                                manifest_source,
                                pinned_root.descriptor,
                                CALVIN_MANIFEST_NAME,
                            )
                        except CalvinArchivePublicationError as exc:
                            if not _published_file_matches(
                                manifest_source,
                                pinned_root.descriptor,
                                CALVIN_MANIFEST_NAME,
                            ):
                                raise
                            manifest_committed = True
                            warnings.append(f"manifest commit validated after a post-publication error: {exc}")
                        else:
                            warnings.extend(manifest_warnings)
                            manifest_committed = True
                        try:
                            _fsync_directory_descriptor(pinned_root.descriptor, label="data root")
                        except CalvinArchivePublicationError as exc:
                            warnings.append(f"manifest is committed but final directory fsync failed: {exc}")
                        for label, assertion in (
                            ("data-root path", pinned_root.assert_bound),
                            ("archive-parent path", archive_parent.assert_bound),
                            (
                                "archive path",
                                lambda: archive.assert_bound_at(
                                    archive_parent.descriptor,
                                    archive_file.name,
                                ),
                            ),
                            ("prepare-lock path", prepare_lock.assert_bound),
                        ):
                            try:
                                assertion()
                            except (CalvinArchiveValidationError, CalvinArchivePublicationError) as exc:
                                warnings.append(f"manifest is committed but {label} rebinding failed: {exc}")
                    try:
                        _directory_binding_identity(
                            pinned_root.descriptor,
                            stage_name,
                            stage_descriptor,
                            label="private stage before cleanup",
                        )
                    except CalvinArchivePublicationError as exc:
                        warnings.append(
                            "private stage binding became ambiguous after commit; "
                            f"residue retained without deletion: {exc}"
                        )
                    else:
                        try:
                            os.rmdir(stage_name, dir_fd=pinned_root.descriptor)
                        except OSError as exc:
                            warnings.append(f"private stage residue retained after commit: {exc}")
                        else:
                            try:
                                _fsync_directory_descriptor(
                                    pinned_root.descriptor,
                                    label="data root",
                                )
                            except CalvinArchivePublicationError as exc:
                                warnings.append(f"stage removal fsync failed after commit: {exc}")
                    for label, assertion in (
                        ("data-root path at return", pinned_root.assert_bound),
                        ("archive-parent path at return", archive_parent.assert_bound),
                        ("prepare-lock path at return", prepare_lock.assert_bound),
                    ):
                        try:
                            assertion()
                        except (CalvinArchiveValidationError, CalvinArchivePublicationError) as exc:
                            warnings.append(f"manifest is committed but {label} rebinding failed: {exc}")
                    return PreparedCalvinArchive(
                        archive_path=archive_file,
                        dataset_root=dataset_path,
                        index_path=index_path,
                        manifest_path=manifest_path,
                        manifest=manifest,
                        publication_warnings=tuple(warnings),
                    )
                except BaseException as exc:
                    if manifest_committed:
                        raise AssertionError("post-commit work must convert failures to warnings") from exc
                    # No staged or published object is deleted here. A source or
                    # destination substitution is therefore fail-closed residue,
                    # never an excuse to unlink a potentially foreign entry.
                    raise
                finally:
                    if manifest_source is not None:
                        manifest_source.close()
                    if index_source is not None:
                        index_source.close()
                    if projection_descriptor is not None:
                        os.close(projection_descriptor)
                    os.close(stage_descriptor)


def _parse_strict_json(raw: bytes, *, label: str) -> dict[str, Any]:
    def no_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise CalvinArchiveValidationError(f"duplicate JSON key: {key!r}")
            result[key] = value
        return result

    try:
        payload = json.loads(raw, object_pairs_hook=no_duplicate_pairs)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CalvinArchiveValidationError(f"cannot parse strict JSON file: {label}") from exc
    if not isinstance(payload, dict):
        raise CalvinArchiveValidationError(f"JSON root must be an object: {label}")
    return payload


def _read_strict_json_at(
    parent_descriptor: int,
    name: str,
    *,
    display_path: Path,
    maximum_bytes: int = 16 * 1024 * 1024,
) -> dict[str, Any]:
    with PinnedRegularFile.open_at(
        parent_descriptor,
        name,
        display_path=display_path,
    ) as pinned:
        if pinned.identity.size > maximum_bytes:
            raise CalvinArchiveValidationError(f"JSON file exceeds bounded size: {display_path}")
        raw = _pread_exact(
            pinned.descriptor,
            pinned.identity.size,
            0,
            label=f"JSON file {name}",
        )
        pinned.assert_bound_at(parent_descriptor, name)
    return _parse_strict_json(raw, label=str(display_path))


def _read_strict_json_file(
    path: Path,
    *,
    maximum_bytes: int = 16 * 1024 * 1024,
) -> dict[str, Any]:
    resolved = Path(os.path.abspath(os.fspath(path)))
    with PinnedDirectoryPath.open(resolved.parent) as parent:
        return _read_strict_json_at(
            parent.descriptor,
            resolved.name,
            display_path=resolved,
            maximum_bytes=maximum_bytes,
        )


_EXPECTED_PROJECTION_DIRECTORIES = frozenset(
    {
        "training",
        "training/.hydra",
        "training/lang_annotations",
        "validation",
        "validation/.hydra",
    }
)


def _projection_inventory(root_descriptor: int) -> tuple[frozenset[str], frozenset[str]]:
    directories: set[str] = set()
    files: set[str] = set()
    pending = [(os.dup(root_descriptor), "")]
    try:
        while pending:
            directory_descriptor, prefix = pending.pop()
            try:
                with os.scandir(os.dup(directory_descriptor)) as entries:
                    names = sorted(entry.name for entry in entries)
                for name in names:
                    relative = f"{prefix}/{name}" if prefix else name
                    value = os.stat(name, dir_fd=directory_descriptor, follow_symlinks=False)
                    if stat.S_ISDIR(value.st_mode):
                        directories.add(relative)
                        child = _open_directory_at(
                            directory_descriptor,
                            name,
                            label="metadata projection",
                        )
                        pending.append((child, relative))
                    elif stat.S_ISREG(value.st_mode):
                        files.add(relative)
                    elif stat.S_ISLNK(value.st_mode):
                        raise CalvinArchiveValidationError(f"metadata projection contains a symlink: {relative}")
                    else:
                        raise CalvinArchiveValidationError(
                            f"metadata projection contains a non-regular entry: {relative}"
                        )
            finally:
                os.close(directory_descriptor)
    finally:
        for pending_descriptor, _prefix in pending:
            os.close(pending_descriptor)
    return frozenset(directories), frozenset(files)


def _authenticate_projected_metadata(
    data_root: PinnedDirectoryPath,
    critical: Mapping[str, Any],
) -> Mapping[str, bytes]:
    data_root.assert_bound()
    projection_descriptor = _open_directory_at(
        data_root.descriptor,
        CALVIN_DATASET_NAME,
        label="metadata projection root",
    )
    projection_identity = DirectoryIdentity.from_stat(os.fstat(projection_descriptor))
    try:
        expected_inventory = (
            _EXPECTED_PROJECTION_DIRECTORIES,
            frozenset(CALVIN_CRITICAL_FILES),
        )
        if _projection_inventory(projection_descriptor) != expected_inventory:
            raise CalvinArchiveValidationError("metadata projection filesystem inventory differs from v4")
        authenticated: dict[str, bytes] = {}
        for relative in CALVIN_CRITICAL_FILES:
            expected = critical.get(relative)
            if not isinstance(expected, dict) or set(expected) != {"bytes", "crc32", "sha256"}:
                raise CalvinArchiveValidationError(f"critical metadata identity is malformed: {relative}")
            if (
                type(expected["bytes"]) is not int
                or not 0 <= expected["bytes"] <= _MAX_RUNTIME_MEMBER_BYTES
                or type(expected["crc32"]) is not int
                or not 0 <= expected["crc32"] <= _UINT32_MAX
                or not _valid_sha256(expected["sha256"])
            ):
                raise CalvinArchiveValidationError(f"critical metadata identity values are invalid: {relative}")
            components = Path(relative).parts
            directory_descriptors = [os.dup(projection_descriptor)]
            directory_bindings: list[tuple[int, str, int, DirectoryIdentity]] = []
            descriptor: int | None = None
            try:
                for component in components[:-1]:
                    parent_descriptor = directory_descriptors[-1]
                    child = _open_directory_at(
                        parent_descriptor,
                        component,
                        label="metadata projection",
                    )
                    identity = DirectoryIdentity.from_stat(os.fstat(child))
                    directory_bindings.append((parent_descriptor, component, child, identity))
                    directory_descriptors.append(child)
                descriptor = os.open(
                    components[-1],
                    os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW,
                    dir_fd=directory_descriptors[-1],
                )
                identity = FileIdentity.from_stat(os.fstat(descriptor))
                if not stat.S_ISREG(identity.mode) or identity.link_count != 1:
                    raise CalvinArchiveValidationError(
                        f"projected metadata must be a single-link regular file: {relative}"
                    )
                if identity.size != expected["bytes"]:
                    raise CalvinArchiveValidationError(f"projected metadata byte length differs: {relative}")
                raw = _pread_exact(
                    descriptor,
                    identity.size,
                    0,
                    label=f"projected metadata {relative}",
                )
                if FileIdentity.from_stat(os.fstat(descriptor)) != identity:
                    raise CalvinArchiveValidationError(f"projected metadata changed during read: {relative}")
                leaf_binding = FileIdentity.from_stat(
                    os.stat(
                        components[-1],
                        dir_fd=directory_descriptors[-1],
                        follow_symlinks=False,
                    )
                )
                if leaf_binding != identity:
                    raise CalvinArchiveValidationError(f"projected metadata path binding changed: {relative}")
                for parent_descriptor, component, child, expected_directory in directory_bindings:
                    current = DirectoryIdentity.from_stat(os.fstat(child))
                    path_value = DirectoryIdentity.from_stat(
                        os.stat(
                            component,
                            dir_fd=parent_descriptor,
                            follow_symlinks=False,
                        )
                    )
                    if current != expected_directory or path_value != expected_directory:
                        raise CalvinArchiveValidationError(f"projected metadata component binding changed: {relative}")
                if (
                    hashlib.sha256(raw).hexdigest() != expected["sha256"]
                    or zlib.crc32(raw) & _UINT32_MAX != expected["crc32"]
                ):
                    raise CalvinArchiveValidationError(f"projected metadata content differs: {relative}")
                authenticated[relative] = raw
            except OSError as exc:
                raise CalvinArchiveValidationError(f"cannot no-follow open projected metadata: {relative}") from exc
            finally:
                if descriptor is not None:
                    os.close(descriptor)
                for directory_descriptor in reversed(directory_descriptors):
                    os.close(directory_descriptor)
        if _projection_inventory(projection_descriptor) != expected_inventory:
            raise CalvinArchiveValidationError("metadata projection inventory changed during authentication")
        current_projection = DirectoryIdentity.from_stat(os.fstat(projection_descriptor))
        bound_projection = DirectoryIdentity.from_stat(
            os.stat(
                CALVIN_DATASET_NAME,
                dir_fd=data_root.descriptor,
                follow_symlinks=False,
            )
        )
        data_root.assert_bound()
        if current_projection != projection_identity or bound_projection != projection_identity:
            raise CalvinArchiveValidationError("metadata projection root binding changed during authentication")
        return MappingProxyType(authenticated)
    finally:
        os.close(projection_descriptor)


def _load_calvin_archive_manifest_from_pinned_root(
    pinned_root: PinnedDirectoryPath,
    *,
    expected_archive_bytes: int = CALVIN_ARCHIVE_BYTES,
    expected_archive_sha256: str = CALVIN_ARCHIVE_SHA256,
    expected_central_directory: CentralDirectoryContract | None = OFFICIAL_CENTRAL_DIRECTORY_CONTRACT,
    expected_archive_url: str = CALVIN_ARCHIVE_URL,
    expected_checksum_url: str = CALVIN_CHECKSUM_URL,
) -> tuple[dict[str, Any], Mapping[str, bytes]]:
    root = pinned_root.path
    pinned_root.assert_bound()
    manifest = _read_strict_json_at(
        pinned_root.descriptor,
        CALVIN_MANIFEST_NAME,
        display_path=root / CALVIN_MANIFEST_NAME,
    )
    if set(manifest) != {
        "archive",
        "checksum_url",
        "content_sha256",
        "critical_files",
        "dataset",
        "schema",
        "storage",
    }:
        raise CalvinArchiveValidationError("CALVIN archive-direct manifest root fields differ")
    if manifest.get("schema") != CALVIN_MANIFEST_SCHEMA or manifest.get("dataset") != CALVIN_DATASET_NAME:
        raise CalvinArchiveValidationError("unsupported CALVIN archive-direct manifest")
    if manifest.get("content_sha256") != _content_sha256(manifest):
        raise CalvinArchiveValidationError("CALVIN archive-direct manifest content hash differs")
    archive = manifest.get("archive")
    storage = manifest.get("storage")
    critical = manifest.get("critical_files")
    if not isinstance(archive, dict) or not isinstance(storage, dict) or not isinstance(critical, dict):
        raise CalvinArchiveValidationError("CALVIN archive-direct manifest sections are missing")
    if set(archive) != {
        "bytes",
        "central_directory",
        "member_inventory",
        "path",
        "sha256",
        "url",
    }:
        raise CalvinArchiveValidationError("CALVIN archive identity fields differ")
    if (
        archive.get("bytes") != expected_archive_bytes
        or archive.get("sha256") != expected_archive_sha256
        or archive.get("url") != expected_archive_url
        or manifest.get("checksum_url") != expected_checksum_url
    ):
        raise CalvinArchiveValidationError("CALVIN archive identity differs from the pinned contract")
    if archive.get("path") != CALVIN_ARCHIVE_NAME:
        raise CalvinArchiveValidationError("CALVIN archive basename differs")
    central = archive.get("central_directory")
    inventory = archive.get("member_inventory")
    if not isinstance(central, dict) or not isinstance(inventory, dict):
        raise CalvinArchiveValidationError("CALVIN central-directory inventory is missing")
    if set(central) != {"bytes", "entries", "offset", "sha256", "zip64"}:
        raise CalvinArchiveValidationError("CALVIN central-directory fields differ")
    if set(inventory) != {
        "compressed_bytes",
        "directory_member_count",
        "file_member_count",
        "member_count",
        "npz_member_count",
        "sha256",
        "uncompressed_bytes",
    }:
        raise CalvinArchiveValidationError("CALVIN member-inventory fields differ")
    integer_inventory = {
        "compressed_bytes",
        "directory_member_count",
        "file_member_count",
        "member_count",
        "npz_member_count",
        "uncompressed_bytes",
    }
    if (
        type(central.get("bytes")) is not int
        or central["bytes"] <= 0
        or type(central.get("entries")) is not int
        or central["entries"] <= 0
        or type(central.get("offset")) is not int
        or central["offset"] <= 0
        or type(central.get("zip64")) is not bool
        or not _valid_sha256(central.get("sha256"))
        or any(type(inventory.get(name)) is not int or inventory[name] < 0 for name in integer_inventory)
        or not _valid_sha256(inventory.get("sha256"))
        or central["entries"] != inventory["member_count"]
        or inventory["file_member_count"] + inventory["directory_member_count"] != inventory["member_count"]
        or inventory["npz_member_count"] > inventory["file_member_count"]
    ):
        raise CalvinArchiveValidationError("CALVIN central/member inventory values are invalid")
    if expected_central_directory is not None:
        actual_central = CentralDirectoryContract(
            offset=central.get("offset"),
            size=central.get("bytes"),
            sha256=central.get("sha256"),
            member_count=inventory.get("member_count"),
            file_member_count=inventory.get("file_member_count"),
            directory_member_count=inventory.get("directory_member_count"),
            npz_member_count=inventory.get("npz_member_count"),
        )
        if actual_central != expected_central_directory:
            raise CalvinArchiveValidationError("CALVIN central-directory contract differs")
    if set(storage) != {
        "derived_artifacts",
        "materialized_files",
        "member_index",
        "mode",
        "reader_schema",
        "verification",
    }:
        raise CalvinArchiveValidationError("CALVIN archive-direct storage fields differ")
    if (
        storage.get("mode") != "archive-direct"
        or storage.get("reader_schema") != CALVIN_ARCHIVE_READER_SCHEMA
        or storage.get("verification") != _VERIFICATION_CONTRACT
    ):
        raise CalvinArchiveValidationError("CALVIN archive-direct storage contract differs")
    if storage.get("materialized_files") != list(CALVIN_CRITICAL_FILES) or set(critical) != set(CALVIN_CRITICAL_FILES):
        raise CalvinArchiveValidationError("CALVIN critical metadata inventory differs")
    derived = storage.get("derived_artifacts")
    if derived != {
        "state_action_sidecar": None,
        "state_action_sidecar_schema_hook": CALVIN_STATE_ACTION_SIDECAR_SCHEMA,
    }:
        raise CalvinArchiveValidationError("CALVIN P0 derived-artifact contract differs")
    member_index = storage.get("member_index")
    if (
        not isinstance(member_index, dict)
        or set(member_index) != {"bytes", "path", "schema", "sha256"}
        or member_index.get("schema") != CALVIN_MEMBER_INDEX_SCHEMA
        or member_index.get("path") != CALVIN_INDEX_NAME
        or type(member_index.get("bytes")) is not int
        or member_index["bytes"] <= 0
        or not _valid_sha256(member_index.get("sha256"))
    ):
        raise CalvinArchiveValidationError("CALVIN v2 member-index manifest identity differs")
    if expected_central_directory == OFFICIAL_CENTRAL_DIRECTORY_CONTRACT and central["zip64"] is not True:
        raise CalvinArchiveValidationError("the official CALVIN manifest must authenticate ZIP64")
    authenticated_metadata = _authenticate_projected_metadata(pinned_root, critical)
    pinned_root.assert_bound()
    return manifest, authenticated_metadata


def load_calvin_archive_manifest(
    data_root: str | Path,
    *,
    expected_archive_bytes: int = CALVIN_ARCHIVE_BYTES,
    expected_archive_sha256: str = CALVIN_ARCHIVE_SHA256,
    expected_central_directory: CentralDirectoryContract | None = OFFICIAL_CENTRAL_DIRECTORY_CONTRACT,
    expected_archive_url: str = CALVIN_ARCHIVE_URL,
    expected_checksum_url: str = CALVIN_CHECKSUM_URL,
) -> dict[str, Any]:
    with CalvinArchiveReader.from_manifest(
        data_root,
        expected_archive_bytes=expected_archive_bytes,
        expected_archive_sha256=expected_archive_sha256,
        expected_central_directory=expected_central_directory,
        expected_archive_url=expected_archive_url,
        expected_checksum_url=expected_checksum_url,
    ) as reader:
        if reader._authenticated_manifest is None:
            raise AssertionError("manifest-owned reader is missing its authenticated manifest")
        return dict(reader._authenticated_manifest)


def _validate_index_member_rows(
    connection: sqlite3.Connection,
    *,
    central_offset: int,
) -> dict[str, int]:
    summary = {
        "compressed_bytes": 0,
        "directory_member_count": 0,
        "file_member_count": 0,
        "member_count": 0,
        "npz_member_count": 0,
        "uncompressed_bytes": 0,
    }
    previous_end = 0
    directory_paths: set[str] = set()
    required_directory_paths: set[str] = set()
    for row in connection.execute(f"{_MEMBER_SELECT} ORDER BY local_header_offset"):
        record = _row_to_record(row)
        if record.local_header_offset < previous_end:
            raise CalvinArchiveValidationError(f"member-index physical ranges overlap: {record.path!r}")
        if record.data_offset > central_offset or record.compressed_bytes > central_offset - record.data_offset:
            raise CalvinArchiveValidationError(
                f"member-index payload range escapes the authenticated ZIP: {record.path!r}"
            )
        previous_end = record.data_offset + record.compressed_bytes
        summary["member_count"] += 1
        summary["compressed_bytes"] += record.compressed_bytes
        summary["uncompressed_bytes"] += record.logical_bytes
        if record.kind == _KIND_DIRECTORY:
            summary["directory_member_count"] += 1
            directory_paths.add(record.path)
        else:
            summary["file_member_count"] += 1
            summary["npz_member_count"] += int(record.path.endswith(".npz"))
        parent = record.path.rpartition("/")[0] if record.path else ""
        if parent:
            required_directory_paths.add(parent)
    missing_directories = required_directory_paths - directory_paths
    if missing_directories:
        first = min(missing_directories)
        raise CalvinArchiveValidationError(f"member index is missing a declared parent directory: {first!r}")
    if previous_end > central_offset:
        raise CalvinArchiveValidationError("member-index payload overlaps the central directory")
    return summary


class CalvinArchiveReader:
    """Pinned archive/index session for exact authenticated member reads."""

    def __init__(
        self,
        archive: PinnedRegularFile,
        index: PinnedRegularFile,
        connection: sqlite3.Connection,
        *,
        archive_root: str,
        central_offset: int,
        metadata: Mapping[str, str],
        pinned_root: PinnedDirectoryPath | None = None,
        archive_name: str | None = None,
        index_name: str | None = None,
        authenticated_metadata: Mapping[str, bytes] | None = None,
        authenticated_manifest: Mapping[str, Any] | None = None,
    ) -> None:
        self._archive = archive
        self._index = index
        self._connection = connection
        self.archive_root = archive_root
        self.central_offset = central_offset
        self._metadata = dict(metadata)
        self._pinned_root = pinned_root
        self._archive_name = archive_name
        self._index_name = index_name
        self._authenticated_metadata = MappingProxyType(dict(authenticated_metadata or {}))
        self._authenticated_manifest = (
            MappingProxyType(dict(authenticated_manifest)) if authenticated_manifest is not None else None
        )
        self._closed = False

    @classmethod
    def _open_pinned(
        cls,
        archive: PinnedRegularFile,
        index: PinnedRegularFile,
        *,
        expected_archive_bytes: int,
        expected_archive_sha256: str,
        expected_index_bytes: int,
        expected_index_sha256: str,
        pinned_root: PinnedDirectoryPath | None = None,
        archive_name: str | None = None,
        index_name: str | None = None,
        authenticated_metadata: Mapping[str, bytes] | None = None,
        authenticated_manifest: Mapping[str, Any] | None = None,
        expected_archive_file_identity: FileIdentity | None = None,
    ) -> CalvinArchiveReader:
        connection: sqlite3.Connection | None = None
        try:
            if archive.identity.size != expected_archive_bytes:
                raise CalvinArchiveValidationError("pinned archive differs from the authenticated reader contract")
            if expected_archive_file_identity is None:
                if archive.sha256() != expected_archive_sha256:
                    raise CalvinArchiveValidationError("pinned archive differs from the authenticated reader contract")
            elif archive.identity != expected_archive_file_identity:
                raise CalvinArchiveValidationError("pinned archive differs from the ephemeral file identity")
            central = _read_central_directory_info(
                archive.descriptor,
                archive.identity.size,
            )
            if index.identity.size != expected_index_bytes or index.sha256() != expected_index_sha256:
                raise CalvinArchiveValidationError("pinned member index differs from the manifest")
            uri = f"file:/proc/self/fd/{index.descriptor}?mode=ro&immutable=1"
            connection = sqlite3.connect(uri, uri=True)
            connection.execute("PRAGMA query_only=ON")
            connection.execute("PRAGMA trusted_schema=OFF")
            _validate_index_schema(connection)
            metadata = dict(connection.execute("SELECT name,value FROM metadata"))
            expected_metadata_names = _INDEX_METADATA_BASE_NAMES | {
                f"critical_sha256:{relative}" for relative in CALVIN_CRITICAL_FILES
            }
            if set(metadata) != expected_metadata_names or any(
                type(name) is not str or type(value) is not str for name, value in metadata.items()
            ):
                raise CalvinArchiveValidationError("member-index metadata inventory differs from v2")
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
            if numeric["central_directory_zip64"] not in (0, 1):
                raise CalvinArchiveValidationError("member-index ZIP64 metadata must be canonical zero or one")
            hash_names = {
                "archive_sha256",
                "central_directory_sha256",
                "member_inventory_sha256",
            } | {f"critical_sha256:{relative}" for relative in CALVIN_CRITICAL_FILES}
            if any(not _valid_sha256(metadata[name]) for name in hash_names):
                raise CalvinArchiveValidationError("member-index metadata contains an invalid SHA-256")
            if (
                metadata["schema"] != CALVIN_MEMBER_INDEX_SCHEMA
                or metadata["reader_schema"] != CALVIN_ARCHIVE_READER_SCHEMA
                or metadata["status"] != "complete"
                or metadata["state_action_sidecar_schema"] != CALVIN_STATE_ACTION_SIDECAR_SCHEMA
                or metadata["state_action_sidecar_status"] != "absent"
                or metadata["archive_sha256"] != expected_archive_sha256
                or numeric["archive_bytes"] != expected_archive_bytes
            ):
                raise CalvinArchiveValidationError("member-index generation identity differs")
            archive_root = metadata["archive_root"]
            if not re.fullmatch(r"[A-Za-z0-9_.-]+", archive_root):
                raise CalvinArchiveValidationError("member-index archive root is invalid")
            if (
                numeric["central_directory_offset"] != central.offset
                or numeric["central_directory_bytes"] != central.size
                or numeric["central_directory_zip64"] != int(central.zip64)
                or metadata["central_directory_sha256"] != central.sha256
                or numeric["member_count"] != central.entry_count
            ):
                raise CalvinArchiveValidationError("member-index central-directory identity differs from the archive")
            row_summary = _validate_index_member_rows(
                connection,
                central_offset=central.offset,
            )
            if any(numeric[name] != value for name, value in row_summary.items()):
                raise CalvinArchiveValidationError("member-index aggregate metadata differs from its semantic rows")
            if connection.execute("SELECT count(*) FROM members WHERE state_action_row IS NOT NULL").fetchone() != (0,):
                raise CalvinArchiveValidationError("P0 member index contains unauthorized sidecar rows")
            archive.assert_unchanged()
            index.assert_unchanged()
            if pinned_root is not None:
                pinned_root.assert_bound()
                if archive_name is None or index_name is None:
                    raise AssertionError("root-owned reader requires archive and index basenames")
                archive.assert_bound_at(pinned_root.descriptor, archive_name)
                index.assert_bound_at(pinned_root.descriptor, index_name)
            return cls(
                archive,
                index,
                connection,
                archive_root=archive_root,
                central_offset=central.offset,
                metadata=metadata,
                pinned_root=pinned_root,
                archive_name=archive_name,
                index_name=index_name,
                authenticated_metadata=authenticated_metadata,
                authenticated_manifest=authenticated_manifest,
            )
        except BaseException:
            if connection is not None:
                connection.close()
            index.close()
            archive.close()
            if pinned_root is not None:
                pinned_root.close()
            raise

    @classmethod
    def open(
        cls,
        archive_path: str | Path,
        index_path: str | Path,
        *,
        expected_archive_bytes: int,
        expected_archive_sha256: str,
        expected_index_bytes: int,
        expected_index_sha256: str,
    ) -> CalvinArchiveReader:
        if not _valid_sha256(expected_archive_sha256) or not _valid_sha256(expected_index_sha256):
            raise ValueError("reader expected hashes must be lowercase SHA-256 values")
        archive = PinnedRegularFile.open(archive_path)
        try:
            index = PinnedRegularFile.open(index_path)
        except BaseException:
            archive.close()
            raise
        return cls._open_pinned(
            archive,
            index,
            expected_archive_bytes=expected_archive_bytes,
            expected_archive_sha256=expected_archive_sha256,
            expected_index_bytes=expected_index_bytes,
            expected_index_sha256=expected_index_sha256,
        )

    @classmethod
    def from_manifest(
        cls,
        data_root: str | Path,
        *,
        expected_archive_bytes: int = CALVIN_ARCHIVE_BYTES,
        expected_archive_sha256: str = CALVIN_ARCHIVE_SHA256,
        expected_central_directory: CentralDirectoryContract | None = OFFICIAL_CENTRAL_DIRECTORY_CONTRACT,
        expected_archive_url: str = CALVIN_ARCHIVE_URL,
        expected_checksum_url: str = CALVIN_CHECKSUM_URL,
    ) -> CalvinArchiveReader:
        return cls._from_manifest(
            data_root,
            expected_archive_bytes=expected_archive_bytes,
            expected_archive_sha256=expected_archive_sha256,
            expected_central_directory=expected_central_directory,
            expected_archive_url=expected_archive_url,
            expected_checksum_url=expected_checksum_url,
            expected_archive_file_identity=None,
        )

    @classmethod
    def from_manifest_fast(
        cls,
        data_root: str | Path,
        *,
        expected_archive_file_identity: Mapping[str, object],
        expected_archive_bytes: int = CALVIN_ARCHIVE_BYTES,
        expected_archive_sha256: str = CALVIN_ARCHIVE_SHA256,
        expected_central_directory: CentralDirectoryContract | None = OFFICIAL_CENTRAL_DIRECTORY_CONTRACT,
        expected_archive_url: str = CALVIN_ARCHIVE_URL,
        expected_checksum_url: str = CALVIN_CHECKSUM_URL,
    ) -> CalvinArchiveReader:
        """Open from a freshly broadcast, job-local archive identity.

        This path deliberately skips only the second full-archive SHA-256 pass.
        The caller must obtain ``expected_archive_file_identity`` from a reader
        opened with :meth:`from_manifest` in the same trusted process group. It
        is not a persistent authenticity credential. The no-follow path chain,
        exact live inode identity, central-directory digest, full index digest
        and semantics, projected metadata, and archive-member parity remain
        verified and pinned for the reader lifetime.
        """

        identity = _parse_ephemeral_file_identity(expected_archive_file_identity)
        if identity.size != expected_archive_bytes:
            raise ValueError("ephemeral archive file identity has the wrong archive size")
        return cls._from_manifest(
            data_root,
            expected_archive_bytes=expected_archive_bytes,
            expected_archive_sha256=expected_archive_sha256,
            expected_central_directory=expected_central_directory,
            expected_archive_url=expected_archive_url,
            expected_checksum_url=expected_checksum_url,
            expected_archive_file_identity=identity,
        )

    @classmethod
    def _from_manifest(
        cls,
        data_root: str | Path,
        *,
        expected_archive_bytes: int,
        expected_archive_sha256: str,
        expected_central_directory: CentralDirectoryContract | None,
        expected_archive_url: str,
        expected_checksum_url: str,
        expected_archive_file_identity: FileIdentity | None,
    ) -> CalvinArchiveReader:
        pinned_root = PinnedDirectoryPath.open(data_root)
        root = pinned_root.path
        try:
            manifest, authenticated_metadata = _load_calvin_archive_manifest_from_pinned_root(
                pinned_root,
                expected_archive_bytes=expected_archive_bytes,
                expected_archive_sha256=expected_archive_sha256,
                expected_central_directory=expected_central_directory,
                expected_archive_url=expected_archive_url,
                expected_checksum_url=expected_checksum_url,
            )
        except BaseException:
            pinned_root.close()
            raise
        archive = manifest["archive"]
        index = manifest["storage"]["member_index"]
        if (
            not isinstance(archive.get("path"), str)
            or Path(archive["path"]).name != archive["path"]
            or not isinstance(index.get("path"), str)
            or Path(index["path"]).name != index["path"]
        ):
            pinned_root.close()
            raise CalvinArchiveValidationError("manifest archive/index paths must be basenames")
        try:
            archive_file = PinnedRegularFile.open_at(
                pinned_root.descriptor,
                archive["path"],
                display_path=root / archive["path"],
            )
        except BaseException:
            pinned_root.close()
            raise
        try:
            index_file = PinnedRegularFile.open_at(
                pinned_root.descriptor,
                index["path"],
                display_path=root / index["path"],
            )
        except BaseException:
            archive_file.close()
            pinned_root.close()
            raise
        reader = cls._open_pinned(
            archive_file,
            index_file,
            expected_archive_bytes=archive["bytes"],
            expected_archive_sha256=archive["sha256"],
            expected_index_bytes=index["bytes"],
            expected_index_sha256=index["sha256"],
            pinned_root=pinned_root,
            archive_name=archive["path"],
            index_name=index["path"],
            authenticated_metadata=authenticated_metadata,
            authenticated_manifest=manifest,
            expected_archive_file_identity=expected_archive_file_identity,
        )
        try:
            central = archive["central_directory"]
            inventory = archive["member_inventory"]
            expected_metadata = {
                "central_directory_bytes": str(central["bytes"]),
                "central_directory_offset": str(central["offset"]),
                "central_directory_sha256": central["sha256"],
                "central_directory_zip64": "1" if central["zip64"] else "0",
                "compressed_bytes": str(inventory["compressed_bytes"]),
                "directory_member_count": str(inventory["directory_member_count"]),
                "file_member_count": str(inventory["file_member_count"]),
                "member_count": str(inventory["member_count"]),
                "member_inventory_sha256": inventory["sha256"],
                "npz_member_count": str(inventory["npz_member_count"]),
                "uncompressed_bytes": str(inventory["uncompressed_bytes"]),
            }
            for name, value in expected_metadata.items():
                if reader._metadata.get(name) != value:
                    raise CalvinArchiveValidationError(f"manifest/index identity differs: {name}")
            for relative in CALVIN_CRITICAL_FILES:
                if (
                    reader._metadata.get(f"critical_sha256:{relative}")
                    != manifest["critical_files"][relative]["sha256"]
                ):
                    raise CalvinArchiveValidationError(f"manifest/index critical identity differs: {relative}")
                if reader.read_member_bytes(relative) != authenticated_metadata[relative]:
                    raise CalvinArchiveValidationError(
                        f"projected metadata differs from authenticated archive bytes: {relative}"
                    )
            return reader
        except BaseException:
            reader.close()
            raise

    def _ensure_open(self) -> None:
        if self._closed:
            raise CalvinArchiveValidationError("CALVIN archive reader is closed")
        if self._pinned_root is not None:
            if self._archive_name is None or self._index_name is None:
                raise AssertionError("root-owned reader is missing publication basenames")
            self._pinned_root.assert_bound()
            self._archive.assert_bound_at(
                self._pinned_root.descriptor,
                self._archive_name,
            )
            self._index.assert_bound_at(
                self._pinned_root.descriptor,
                self._index_name,
            )

    @property
    def verified_archive_path(self) -> Path:
        """The display path whose inode is pinned by this reader session."""

        self._ensure_open()
        return self._archive.path

    @property
    def archive_file_identity(self) -> dict[str, int]:
        """Return a defensive job-local capability for same-run rank handoff."""

        self._ensure_open()
        self._archive.assert_unchanged()
        if self._pinned_root is not None:
            if self._archive_name is None:
                raise AssertionError("root-owned reader is missing its archive basename")
            self._archive.assert_bound_at(self._pinned_root.descriptor, self._archive_name)
        return self._archive.identity.as_ephemeral_capability()

    @property
    def authenticated_metadata_bytes(self) -> Mapping[str, bytes]:
        """Exact projected bytes authenticated against the pinned archive."""

        self._ensure_open()
        return self._authenticated_metadata

    @property
    def authenticated_manifest(self) -> dict[str, Any]:
        """Return an isolated copy of the manifest authenticated by this session."""

        self._ensure_open()
        if self._authenticated_manifest is None:
            raise CalvinArchiveValidationError("this reader was not opened from an authenticated v4 manifest")
        return copy.deepcopy(dict(self._authenticated_manifest))

    def read_authenticated_metadata_bytes(self, relative_path: str) -> bytes:
        self._ensure_open()
        if relative_path not in CALVIN_CRITICAL_FILES:
            raise KeyError(f"not an authenticated CALVIN metadata projection: {relative_path}")
        try:
            return self._authenticated_metadata[relative_path]
        except KeyError as exc:
            raise CalvinArchiveValidationError("this reader was not opened from an authenticated v4 manifest") from exc

    def _record(self, relative_path: str) -> ArchiveMemberRecord:
        _validate_requested_path(relative_path)
        row = self._connection.execute(f"{_MEMBER_SELECT} WHERE path=?", (relative_path,)).fetchone()
        if row is None:
            raise KeyError(f"CALVIN archive member is absent: {relative_path}")
        return _row_to_record(row)

    def member_record(self, relative_path: str) -> ArchiveMemberRecord:
        """Return the authenticated immutable identity for one logical member."""

        self._ensure_open()
        self._archive.assert_unchanged()
        self._index.assert_unchanged()
        record = self._record(relative_path)
        data_offset = _local_header_data_offset(
            self._archive.descriptor,
            record,
            archive_root=self.archive_root,
            central_offset=self.central_offset,
        )
        if data_offset != record.data_offset:
            raise CalvinArchiveValidationError(f"runtime local-header data offset differs: {relative_path!r}")
        self._archive.assert_unchanged()
        self._index.assert_unchanged()
        return record

    def read_member_bytes(self, relative_path: str, *, maximum_bytes: int = _MAX_RUNTIME_MEMBER_BYTES) -> bytes:
        self._ensure_open()
        self._archive.assert_unchanged()
        self._index.assert_unchanged()
        record = self._record(relative_path)
        if not record.is_file:
            raise IsADirectoryError(relative_path)
        data_offset = _local_header_data_offset(
            self._archive.descriptor,
            record,
            archive_root=self.archive_root,
            central_offset=self.central_offset,
        )
        if data_offset != record.data_offset:
            raise CalvinArchiveValidationError(f"runtime local-header data offset differs: {relative_path!r}")
        content, _digest = _inflate_member(
            self._archive.descriptor,
            record,
            data_offset=data_offset,
            expected_logical_sha256=record.logical_sha256,
            collect=True,
            maximum_logical_bytes=maximum_bytes,
        )
        self._archive.assert_unchanged()
        self._index.assert_unchanged()
        self._ensure_open()
        if content is None:
            raise AssertionError("collecting member decoder returned no bytes")
        return content

    def iter_frame_records_physical(self, *, split: str | None = None) -> Iterator[ArchiveMemberRecord]:
        """Yield authenticated frame metadata in data-offset order without loading a ZIP inventory."""

        self._ensure_open()
        self._archive.assert_unchanged()
        self._index.assert_unchanged()
        if split not in (None, "training", "validation"):
            raise ValueError("split must be training, validation, or None")
        if split is None:
            cursor = self._connection.execute(f"{_MEMBER_SELECT} WHERE global_index IS NOT NULL ORDER BY data_offset")
        else:
            cursor = self._connection.execute(
                f"{_MEMBER_SELECT} WHERE split=? AND global_index IS NOT NULL ORDER BY data_offset",
                (split,),
            )
        try:
            for row in cursor:
                self._archive.assert_unchanged()
                self._index.assert_unchanged()
                record = _row_to_record(row)
                data_offset = _local_header_data_offset(
                    self._archive.descriptor,
                    record,
                    archive_root=self.archive_root,
                    central_offset=self.central_offset,
                )
                if data_offset != record.data_offset:
                    raise CalvinArchiveValidationError(
                        f"physical iterator local-header offset differs: {record.path!r}"
                    )
                self._archive.assert_unchanged()
                self._index.assert_unchanged()
                yield record
        finally:
            self._archive.assert_unchanged()
            self._index.assert_unchanged()
            self._ensure_open()

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            try:
                self._connection.close()
            finally:
                try:
                    self._index.close()
                finally:
                    try:
                        self._archive.close()
                    finally:
                        if self._pinned_root is not None:
                            self._pinned_root.close()

    def __enter__(self) -> CalvinArchiveReader:
        self._ensure_open()
        return self

    def __exit__(self, *_exc_info: object) -> None:
        self.close()


def _validate_requested_path(path: str) -> None:
    if not path or path.startswith("/") or path.endswith("/") or "\\" in path or "\0" in path:
        raise ValueError("member path must be a canonical relative file path")
    components = path.split("/")
    if any(component in ("", ".", "..") for component in components):
        raise ValueError("member path must not contain empty, dot, or dot-dot components")


__all__ = [
    "CALVIN_ARCHIVE_BYTES",
    "CALVIN_ARCHIVE_NAME",
    "CALVIN_ARCHIVE_READER_SCHEMA",
    "CALVIN_ARCHIVE_SHA256",
    "CALVIN_ARCHIVE_URL",
    "CALVIN_CRITICAL_FILES",
    "CALVIN_INDEX_NAME",
    "CALVIN_MANIFEST_NAME",
    "CALVIN_MANIFEST_SCHEMA",
    "CALVIN_MEMBER_INDEX_SCHEMA",
    "CALVIN_STATE_ACTION_SIDECAR_SCHEMA",
    "OFFICIAL_CENTRAL_DIRECTORY_CONTRACT",
    "ArchiveMemberRecord",
    "CalvinArchiveError",
    "CalvinArchivePublicationError",
    "CalvinArchiveReader",
    "CalvinArchiveValidationError",
    "CentralDirectoryContract",
    "PinnedRegularFile",
    "PreparedCalvinArchive",
    "load_calvin_archive_manifest",
    "prepare_calvin_archive",
]
