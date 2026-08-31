"""Train-only normalization artifact for the official CALVIN ABC-to-D archive."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import secrets
import sqlite3
import stat
import zipfile
from bisect import bisect_right
from contextlib import closing, suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from duo_vla.data.calvin import CalvinEpisodeSplit, CalvinNpzDataset, make_calvin_episode_split
from duo_vla.data.calvin_archive import (
    CALVIN_ARCHIVE_NAME,
    CALVIN_ARCHIVE_READER_SCHEMA,
    CALVIN_CHECKSUM_URL,
    CALVIN_INDEX_NAME,
    CALVIN_MANIFEST_SCHEMA,
    CALVIN_MEMBER_INDEX_SCHEMA,
    OFFICIAL_CENTRAL_DIRECTORY_CONTRACT,
    CalvinArchiveReader,
    CentralDirectoryContract,
    PinnedDirectoryPath,
    PinnedRegularFile,
    load_calvin_archive_manifest,
)
from duo_vla.normalization import ActionNormalizer, PercentileNormalizer

CALVIN_ABC_D_ARCHIVE_SHA256 = "c2036c67eb4c06966af1d1e1665bdb572c69e1404f5e77ffd46b384ff2b79f74"
CALVIN_ABC_D_ARCHIVE_BYTES = 555_309_812_705
CALVIN_ABC_D_ARCHIVE_URL = "http://calvin.cs.uni-freiburg.de/dataset/task_ABC_D.zip"
CALVIN_DATASET_MANIFEST_SCHEMA_V3 = "duo-vla-calvin-dataset-manifest-v3"
CALVIN_DATASET_MANIFEST_SCHEMA = CALVIN_MANIFEST_SCHEMA
CALVIN_STATS_SCHEMA_V3 = "duo-vla-calvin-abc-to-d-normalization-v3"
CALVIN_STATS_SCHEMA = "duo-vla-calvin-abc-to-d-normalization-v4"
CALVIN_AUTHENTICATED_GENERATION_SCHEMA = "duo-vla-calvin-authenticated-generation-v2"
CALVIN_STORAGE_IDENTITY_SCHEMA = "duo-vla-calvin-storage-identity-v1"
CALVIN_STORAGE_MODE_ARCHIVE_DIRECT = "archive-direct"
CALVIN_STORAGE_MODE_VERIFIED_EXTRACTION = "verified-extraction"
DEFAULT_SPLIT_SEED = 1729
CALVIN_CRITICAL_TRAIN_METADATA = (
    "ep_start_end_ids.npy",
    "lang_annotations/auto_lang_ann.npy",
    "scene_info.npy",
    ".hydra/merged_config.yaml",
)
CALVIN_DATASET_CRITICAL_FILES = (
    *(f"training/{relative}" for relative in CALVIN_CRITICAL_TRAIN_METADATA),
    "validation/ep_start_end_ids.npy",
    "validation/.hydra/merged_config.yaml",
)


def _valid_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(character in "0123456789abcdef" for character in value)


@dataclass(frozen=True, slots=True)
class CalvinStorageIdentity:
    """Exact backend identity shared by data, normalization, and checkpoints."""

    mode: str
    manifest_schema: str
    manifest_content_sha256: str
    manifest_file_sha256: str
    archive_path: str
    archive_bytes: int
    archive_sha256: str
    archive_url: str
    checksum_url: str
    member_inventory: tuple[tuple[str, int | str | None], ...]
    member_index_path: str
    member_index_schema: str
    member_index_bytes: int
    member_index_sha256: str
    reader_schema: str | None
    central_directory: tuple[tuple[str, int | str | bool], ...] | None
    content_sha256: str

    @property
    def member_inventory_sha256(self) -> str:
        value = dict(self.member_inventory).get("sha256")
        if not isinstance(value, str):
            raise AssertionError("validated member inventory has no SHA-256")
        return value

    def to_dict(self) -> dict[str, Any]:
        return {
            "archive": {
                "bytes": self.archive_bytes,
                "path": self.archive_path,
                "sha256": self.archive_sha256,
                "url": self.archive_url,
            },
            "central_directory": dict(self.central_directory) if self.central_directory is not None else None,
            "checksum_url": self.checksum_url,
            "content_sha256": self.content_sha256,
            "manifest": {
                "content_sha256": self.manifest_content_sha256,
                "file_sha256": self.manifest_file_sha256,
                "schema": self.manifest_schema,
            },
            "member_index": {
                "bytes": self.member_index_bytes,
                "path": self.member_index_path,
                "schema": self.member_index_schema,
                "sha256": self.member_index_sha256,
            },
            "member_inventory": dict(self.member_inventory),
            "mode": self.mode,
            "reader_schema": self.reader_schema,
            "schema": CALVIN_STORAGE_IDENTITY_SCHEMA,
        }

    @classmethod
    def from_dict(cls, value: Any) -> CalvinStorageIdentity:
        required = {
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
        if not isinstance(value, dict) or set(value) != required:
            raise ValueError("invalid CALVIN storage identity fields")
        if value.get("schema") != CALVIN_STORAGE_IDENTITY_SCHEMA:
            raise ValueError("unsupported CALVIN storage identity schema")
        recorded = value.get("content_sha256")
        unsigned = {name: item for name, item in value.items() if name != "content_sha256"}
        if not _valid_sha256(recorded) or recorded != hashlib.sha256(_canonical_json(unsigned).encode()).hexdigest():
            raise ValueError("CALVIN storage identity content hash mismatch")
        archive = value.get("archive")
        manifest = value.get("manifest")
        member_index = value.get("member_index")
        inventory = value.get("member_inventory")
        central = value.get("central_directory")
        if not isinstance(archive, dict) or set(archive) != {"bytes", "path", "sha256", "url"}:
            raise ValueError("CALVIN storage archive identity fields differ")
        if not isinstance(manifest, dict) or set(manifest) != {"content_sha256", "file_sha256", "schema"}:
            raise ValueError("CALVIN storage manifest identity fields differ")
        if not isinstance(member_index, dict) or set(member_index) != {"bytes", "path", "schema", "sha256"}:
            raise ValueError("CALVIN storage member-index identity fields differ")
        inventory_fields = {
            "compressed_bytes",
            "directory_member_count",
            "file_member_count",
            "member_count",
            "npz_member_count",
            "sha256",
            "uncompressed_bytes",
        }
        if not isinstance(inventory, dict) or set(inventory) != inventory_fields:
            raise ValueError("CALVIN storage member-inventory fields differ")
        mode = value.get("mode")
        if mode not in (CALVIN_STORAGE_MODE_ARCHIVE_DIRECT, CALVIN_STORAGE_MODE_VERIFIED_EXTRACTION):
            raise ValueError("unsupported CALVIN storage mode")
        for field in (archive.get("sha256"), manifest.get("content_sha256"), manifest.get("file_sha256")):
            if not _valid_sha256(field):
                raise ValueError("CALVIN storage identity contains an invalid SHA-256")
        if not _valid_sha256(member_index.get("sha256")) or not _valid_sha256(inventory.get("sha256")):
            raise ValueError("CALVIN storage member identity contains an invalid SHA-256")
        if (
            type(archive.get("bytes")) is not int
            or archive["bytes"] <= 0
            or type(member_index.get("bytes")) is not int
            or member_index["bytes"] <= 0
            or not all(isinstance(archive.get(name), str) and archive[name] for name in ("path", "url"))
            or not all(isinstance(member_index.get(name), str) and member_index[name] for name in ("path", "schema"))
            or not isinstance(manifest.get("schema"), str)
            or not isinstance(value.get("checksum_url"), str)
        ):
            raise ValueError("CALVIN storage identity scalar fields are invalid")
        numeric_inventory = inventory_fields - {"sha256", "directory_member_count"}
        if any(type(inventory[name]) is not int or inventory[name] <= 0 for name in numeric_inventory):
            raise ValueError("CALVIN storage member-inventory counts are invalid")
        if mode == CALVIN_STORAGE_MODE_ARCHIVE_DIRECT:
            if (
                manifest["schema"] != CALVIN_DATASET_MANIFEST_SCHEMA
                or archive["path"] != CALVIN_ARCHIVE_NAME
                or member_index["path"] != CALVIN_INDEX_NAME
                or member_index["schema"] != CALVIN_MEMBER_INDEX_SCHEMA
                or value.get("reader_schema") != CALVIN_ARCHIVE_READER_SCHEMA
                or not isinstance(central, dict)
                or set(central) != {"bytes", "entries", "offset", "sha256", "zip64"}
                or type(inventory.get("directory_member_count")) is not int
                or inventory["directory_member_count"] < 0
                or type(central.get("bytes")) is not int
                or central["bytes"] <= 0
                or type(central.get("entries")) is not int
                or central["entries"] <= 0
                or type(central.get("offset")) is not int
                or central["offset"] <= 0
                or type(central.get("zip64")) is not bool
                or not _valid_sha256(central.get("sha256"))
                or central["entries"] != inventory["member_count"]
                or inventory["file_member_count"] + inventory["directory_member_count"] != inventory["member_count"]
                or inventory["npz_member_count"] > inventory["file_member_count"]
            ):
                raise ValueError("CALVIN archive-direct storage identity is invalid")
        elif (
            manifest["schema"] != CALVIN_DATASET_MANIFEST_SCHEMA_V3
            or archive["path"] != CALVIN_ARCHIVE_NAME
            or member_index["path"] != "task_ABC_D.members.sqlite3"
            or member_index["schema"] != "duo-vla-calvin-member-index-v1"
            or value.get("reader_schema") is not None
            or central is not None
            or inventory.get("directory_member_count") is not None
        ):
            raise ValueError("CALVIN verified-extraction storage identity is invalid")
        return cls(
            mode=mode,
            manifest_schema=manifest["schema"],
            manifest_content_sha256=manifest["content_sha256"],
            manifest_file_sha256=manifest["file_sha256"],
            archive_path=archive["path"],
            archive_bytes=archive["bytes"],
            archive_sha256=archive["sha256"],
            archive_url=archive["url"],
            checksum_url=value["checksum_url"],
            member_inventory=tuple(sorted(inventory.items())),
            member_index_path=member_index["path"],
            member_index_schema=member_index["schema"],
            member_index_bytes=member_index["bytes"],
            member_index_sha256=member_index["sha256"],
            reader_schema=value["reader_schema"],
            central_directory=tuple(sorted(central.items())) if central is not None else None,
            content_sha256=recorded,
        )


@dataclass(frozen=True, slots=True)
class AuthenticatedCalvinDatasetGeneration:
    """Broadcastable capability minted from one exact v3 or v4 manifest."""

    training_root: str
    storage: CalvinStorageIdentity
    archive_file_identity: tuple[tuple[str, int], ...] | None
    metadata_sha256: str
    critical_files: tuple[tuple[str, str], ...]
    content_sha256: str

    @property
    def dataset_manifest_content_sha256(self) -> str:
        return self.storage.manifest_content_sha256

    @property
    def dataset_manifest_file_sha256(self) -> str:
        return self.storage.manifest_file_sha256

    @property
    def member_index_bytes(self) -> int:
        return self.storage.member_index_bytes

    @property
    def member_index_sha256(self) -> str:
        return self.storage.member_index_sha256

    @property
    def member_inventory_sha256(self) -> str:
        return self.storage.member_inventory_sha256

    def to_dict(self) -> dict[str, Any]:
        return {
            "content_sha256": self.content_sha256,
            "critical_files": dict(self.critical_files),
            "archive_file_identity": (
                dict(self.archive_file_identity) if self.archive_file_identity is not None else None
            ),
            "metadata_sha256": self.metadata_sha256,
            "schema": CALVIN_AUTHENTICATED_GENERATION_SCHEMA,
            "storage": self.storage.to_dict(),
            "training_root": self.training_root,
        }

    @classmethod
    def from_dict(cls, value: Any) -> AuthenticatedCalvinDatasetGeneration:
        if not isinstance(value, dict) or set(value) != {
            "content_sha256",
            "critical_files",
            "archive_file_identity",
            "metadata_sha256",
            "schema",
            "storage",
            "training_root",
        }:
            raise ValueError("invalid authenticated CALVIN generation fields")
        if value.get("schema") != CALVIN_AUTHENTICATED_GENERATION_SCHEMA:
            raise ValueError("unsupported authenticated CALVIN generation schema")
        content_sha256 = value.get("content_sha256")
        without_hash = {name: item for name, item in value.items() if name != "content_sha256"}
        if (
            not _valid_sha256(content_sha256)
            or content_sha256 != hashlib.sha256(_canonical_json(without_hash).encode()).hexdigest()
        ):
            raise ValueError("authenticated CALVIN generation content hash mismatch")
        training_root = value.get("training_root")
        if (
            not isinstance(training_root, str)
            or not Path(training_root).is_absolute()
            or os.path.abspath(training_root) != training_root
            or Path(training_root).name != "training"
            or Path(training_root).parent.name != "task_ABC_D"
        ):
            raise ValueError("authenticated CALVIN generation training root is invalid")
        critical = value.get("critical_files")
        if (
            not isinstance(critical, dict)
            or set(critical) != set(CALVIN_DATASET_CRITICAL_FILES)
            or not all(_valid_sha256(item) for item in critical.values())
        ):
            raise ValueError("authenticated CALVIN generation critical-file identity is invalid")
        if not _valid_sha256(value.get("metadata_sha256")):
            raise ValueError("authenticated CALVIN generation metadata SHA-256 is invalid")
        storage = CalvinStorageIdentity.from_dict(value["storage"])
        archive_file_identity = value.get("archive_file_identity")
        identity_fields = {"ctime_ns", "device", "inode", "link_count", "mode", "mtime_ns", "size"}
        if storage.mode == CALVIN_STORAGE_MODE_ARCHIVE_DIRECT:
            if (
                not isinstance(archive_file_identity, dict)
                or set(archive_file_identity) != identity_fields
                or any(type(item) is not int for item in archive_file_identity.values())
                or archive_file_identity["device"] < 0
                or archive_file_identity["inode"] <= 0
                or archive_file_identity["mode"] <= 0
                or not stat.S_ISREG(archive_file_identity["mode"])
                or archive_file_identity["link_count"] != 1
                or archive_file_identity["size"] != storage.archive_bytes
                or archive_file_identity["mtime_ns"] < 0
                or archive_file_identity["ctime_ns"] < 0
            ):
                raise ValueError("authenticated CALVIN v4 archive file identity is invalid")
        elif archive_file_identity is not None:
            raise ValueError("legacy CALVIN generation must not contain a v4 archive file identity")
        return cls(
            training_root=os.path.abspath(training_root),
            storage=storage,
            archive_file_identity=(
                tuple(sorted(archive_file_identity.items())) if archive_file_identity is not None else None
            ),
            metadata_sha256=value["metadata_sha256"],
            critical_files=tuple(sorted(critical.items())),
            content_sha256=content_sha256,
        )


def _canonical_json(value: Any) -> str:
    return json.dumps(value, allow_nan=False, separators=(",", ":"), sort_keys=True)


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, value in pairs:
        if name in result:
            raise ValueError(f"duplicate JSON field {name!r}")
        result[name] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant {value}")


def _read_strict_json(path: Path) -> tuple[dict[str, Any], str]:
    absolute = Path(os.path.abspath(os.fspath(path)))
    try:
        with (
            PinnedDirectoryPath.open(absolute.parent) as parent,
            PinnedRegularFile.open_at(
                parent.descriptor,
                absolute.name,
                display_path=absolute,
            ) as pinned,
        ):
            if pinned.identity.size > 64 * 1024 * 1024:
                raise ValueError(f"CALVIN artifact exceeds the strict JSON size bound: {absolute}")
            raw = bytearray()
            offset = 0
            while offset < pinned.identity.size:
                block = os.pread(
                    pinned.descriptor,
                    min(8 * 1024 * 1024, pinned.identity.size - offset),
                    offset,
                )
                if not block:
                    raise ValueError(f"CALVIN artifact ended before its pinned byte length: {absolute}")
                raw.extend(block)
                offset += len(block)
            pinned.assert_bound_at(parent.descriptor, absolute.name)
            parent.assert_bound()
        raw_bytes = bytes(raw)
        value = json.loads(
            raw_bytes.decode("utf-8"),
            object_pairs_hook=_unique_json_object,
            parse_constant=_reject_json_constant,
        )
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        raise ValueError(f"CALVIN artifact is not strict finite UTF-8 JSON: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"CALVIN artifact JSON root must be an object: {path}")
    return value, hashlib.sha256(raw_bytes).hexdigest()


def _content_hash(payload: dict[str, Any]) -> str:
    without_hash = {key: value for key, value in payload.items() if key != "content_sha256"}
    return hashlib.sha256(_canonical_json(without_hash).encode()).hexdigest()


def _strict_finite_float64_vector(value: Any, *, length: int, label: str) -> np.ndarray:
    """Parse a JSON numeric vector without accepting bool coercions or overflow."""

    if not isinstance(value, list) or len(value) != length or any(type(item) not in (int, float) for item in value):
        raise ValueError(f"{label} must be a strict numeric vector of length {length}")
    try:
        result = np.asarray(value, dtype=np.float64)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be representable as finite float64") from exc
    if result.shape != (length,) or not bool(np.isfinite(result).all()):
        raise ValueError(f"{label} must be a finite float64 vector of length {length}")
    return result


def _validated_state_percentiles(
    q01_value: Any,
    q99_value: Any,
    *,
    constant_dimensions: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Validate serialized float64 percentiles against their float32 runtime form."""

    raw_q01 = _strict_finite_float64_vector(q01_value, length=7, label="CALVIN normalization q01")
    raw_q99 = _strict_finite_float64_vector(q99_value, length=7, label="CALVIN normalization q99")
    if bool((raw_q99 < raw_q01).any()):
        raise ValueError("CALVIN normalization q99 must be greater than or equal to q01 in serialized float64")
    raw_constant_dimensions = np.flatnonzero(np.abs(raw_q99 - raw_q01) < 1e-6).tolist()
    if constant_dimensions != raw_constant_dimensions:
        raise ValueError("CALVIN normalization constant dimensions differ from q01/q99")

    with np.errstate(over="ignore", invalid="ignore"):
        q01 = torch.from_numpy(raw_q01.astype(np.float32))
        q99 = torch.from_numpy(raw_q99.astype(np.float32))
    if not bool(torch.isfinite(q01).all() and torch.isfinite(q99).all()):
        raise ValueError("CALVIN normalization percentile bounds must remain finite float32 vectors at runtime")
    if bool((q99 < q01).any()):
        raise ValueError("CALVIN normalization q99 must remain greater than or equal to q01 in runtime float32")
    runtime_constant_dimensions = torch.nonzero((q99 - q01).abs() < 1e-6, as_tuple=False).flatten().tolist()
    if runtime_constant_dimensions != raw_constant_dimensions:
        raise ValueError("CALVIN normalization constant dimensions change after runtime float32 conversion")
    return q01, q99


def _indices_hash(indices: tuple[int, ...]) -> str:
    return hashlib.sha256(",".join(map(str, indices)).encode()).hexdigest()


def _files_sha256(paths: tuple[Path, ...]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"required CALVIN metadata is missing: {path}")
        digest.update(path.name.encode())
        with path.open("rb") as handle:
            while block := handle.read(8 * 1024 * 1024):
                digest.update(block)
    return digest.hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _archive_member_inventory(path: Path) -> dict[str, int | str]:
    """Recompute the exact central-directory identity used by the extractor."""

    digest = hashlib.sha256()
    member_count = 0
    file_member_count = 0
    npz_member_count = 0
    compressed_bytes = 0
    uncompressed_bytes = 0
    seen: set[str] = set()
    with zipfile.ZipFile(path) as archive:
        for info in sorted(archive.infolist(), key=lambda item: item.filename):
            if info.filename in seen:
                raise ValueError(f"duplicate CALVIN ZIP member: {info.filename}")
            seen.add(info.filename)
            record = {
                "compress_size": info.compress_size,
                "crc32": f"{info.CRC:08x}",
                "file_size": info.file_size,
                "is_dir": info.is_dir(),
                "name": info.filename,
            }
            digest.update(_canonical_json(record).encode() + b"\n")
            member_count += 1
            if not info.is_dir():
                file_member_count += 1
                compressed_bytes += info.compress_size
                uncompressed_bytes += info.file_size
                if info.filename.endswith(".npz"):
                    npz_member_count += 1
    return {
        "compressed_bytes": compressed_bytes,
        "file_member_count": file_member_count,
        "member_count": member_count,
        "npz_member_count": npz_member_count,
        "sha256": digest.hexdigest(),
        "uncompressed_bytes": uncompressed_bytes,
    }


def _archive_critical_sha256(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    with zipfile.ZipFile(path) as archive:
        for relative in CALVIN_DATASET_CRITICAL_FILES:
            member = f"task_ABC_D/{relative}"
            digest = hashlib.sha256()
            try:
                with archive.open(member) as handle:
                    while block := handle.read(8 * 1024 * 1024):
                        digest.update(block)
            except KeyError as exc:
                raise ValueError(f"pinned CALVIN archive is missing critical member: {member}") from exc
            result[relative] = digest.hexdigest()
    return result


def _verify_member_index_central_directory(
    archive_path: Path,
    connection: sqlite3.Connection,
) -> None:
    """Bind every indexed extracted path/size/CRC to the pinned ZIP central directory."""

    rows = iter(connection.execute("SELECT path, bytes, crc32, sha256 FROM members ORDER BY path"))
    with zipfile.ZipFile(archive_path) as archive:
        file_infos = sorted((info for info in archive.infolist() if not info.is_dir()), key=lambda info: info.filename)
        for info in file_infos:
            prefix = "task_ABC_D/"
            if not info.filename.startswith(prefix):
                raise ValueError(f"CALVIN ZIP member is outside task_ABC_D/: {info.filename}")
            try:
                row = next(rows)
            except StopIteration as exc:
                raise ValueError(f"CALVIN member index is missing archive member: {info.filename}") from exc
            relative = info.filename[len(prefix) :]
            if (
                row[0] != relative
                or row[1] != info.file_size
                or row[2] != info.CRC
                or not isinstance(row[3], str)
                or len(row[3]) != 64
                or any(character not in "0123456789abcdef" for character in row[3])
            ):
                raise ValueError(f"CALVIN member index differs from the ZIP central directory: {relative}")
    try:
        extra = next(rows)
    except StopIteration:
        extra = None
    if extra is not None:
        raise ValueError(f"CALVIN member index contains a path absent from the ZIP: {extra[0]}")


def _verify_member_index_schema(connection: sqlite3.Connection) -> None:
    objects = list(
        connection.execute(
            "SELECT type, name, tbl_name, sql FROM sqlite_master WHERE name NOT LIKE 'sqlite_%' ORDER BY type, name"
        )
    )
    if [(row[0], row[1], row[2]) for row in objects] != [
        ("table", "members", "members"),
        ("table", "metadata", "metadata"),
    ] or any(not isinstance(row[3], str) or "WITHOUT ROWID" not in row[3].upper() for row in objects):
        raise ValueError("CALVIN member-index SQLite objects differ from the exact schema")
    if list(connection.execute("PRAGMA table_info(members)")) != [
        (0, "path", "TEXT", 1, None, 1),
        (1, "bytes", "INTEGER", 1, None, 0),
        (2, "crc32", "INTEGER", 1, None, 0),
        (3, "sha256", "TEXT", 1, None, 0),
    ] or list(connection.execute("PRAGMA table_info(metadata)")) != [
        (0, "name", "TEXT", 1, None, 1),
        (1, "value", "TEXT", 1, None, 0),
    ]:
        raise ValueError("CALVIN member-index SQLite columns differ from the exact schema")


def calvin_metadata_sha256(training_root: str | Path) -> str:
    root = Path(training_root)
    return _files_sha256(tuple(root / relative for relative in CALVIN_CRITICAL_TRAIN_METADATA))


def calvin_dataset_manifest_path(training_root: str | Path) -> Path:
    root = Path(os.path.abspath(os.fspath(training_root)))
    if root.name != "training" or root.parent.name != "task_ABC_D":
        raise ValueError("CALVIN training root must be <data-root>/task_ABC_D/training")
    return root.parent.with_name(f"{root.parent.name}.manifest.json")


def calvin_member_index_path(training_root: str | Path) -> Path:
    root = Path(os.path.abspath(os.fspath(training_root)))
    if root.name != "training" or root.parent.name != "task_ABC_D":
        raise ValueError("CALVIN training root must be <data-root>/task_ABC_D/training")
    return root.parent.with_name(f"{root.parent.name}.members.sqlite3")


def _load_calvin_v3_dataset_manifest(
    training_root: str | Path,
    *,
    verify_archive: bool = True,
) -> dict[str, Any]:
    """Authenticate the verified-extraction manifest and current critical A/B/C metadata."""

    root = Path(os.path.abspath(os.fspath(training_root)))
    path = calvin_dataset_manifest_path(root)
    if not path.is_file():
        raise FileNotFoundError(f"verified CALVIN dataset manifest is missing: {path}")
    payload, _manifest_file_sha256 = _read_strict_json(path)
    if not isinstance(payload, dict) or set(payload) != {
        "archive",
        "checksum_url",
        "content_sha256",
        "critical_files",
        "dataset",
        "extraction",
        "schema",
    }:
        raise ValueError("CALVIN dataset manifest fields differ")
    if payload.get("schema") != CALVIN_DATASET_MANIFEST_SCHEMA_V3:
        raise ValueError("unsupported CALVIN dataset manifest schema")
    if not _valid_sha256(payload.get("content_sha256")) or payload.get("content_sha256") != _content_hash(payload):
        raise ValueError("CALVIN dataset manifest content hash mismatch")
    if payload.get("dataset") != "task_ABC_D":
        raise ValueError("CALVIN dataset manifest dataset identity mismatch")
    archive = payload.get("archive")
    if not isinstance(archive, dict) or set(archive) != {
        "bytes",
        "member_inventory",
        "sha256",
        "uncompressed_bytes",
        "url",
    }:
        raise ValueError("CALVIN dataset manifest has no archive identity")
    if (
        archive.get("bytes") != CALVIN_ABC_D_ARCHIVE_BYTES
        or archive.get("sha256") != CALVIN_ABC_D_ARCHIVE_SHA256
        or archive.get("url") != CALVIN_ABC_D_ARCHIVE_URL
    ):
        raise ValueError("CALVIN dataset manifest archive identity mismatch")
    inventory = archive.get("member_inventory")
    if not isinstance(inventory, dict) or set(inventory) != {
        "compressed_bytes",
        "file_member_count",
        "member_count",
        "npz_member_count",
        "sha256",
        "uncompressed_bytes",
    }:
        raise ValueError("CALVIN dataset manifest member inventory schema mismatch")
    numeric_inventory = (
        "compressed_bytes",
        "file_member_count",
        "member_count",
        "npz_member_count",
        "uncompressed_bytes",
    )
    if any(type(inventory[name]) is not int or inventory[name] <= 0 for name in numeric_inventory):
        raise ValueError("CALVIN dataset manifest member inventory counts are invalid")
    inventory_sha256 = inventory.get("sha256")
    if not _valid_sha256(inventory_sha256):
        raise ValueError("CALVIN dataset manifest member inventory SHA-256 is invalid")
    critical = payload.get("critical_files")
    if (
        not isinstance(critical, dict)
        or set(critical) != set(CALVIN_DATASET_CRITICAL_FILES)
        or not all(_valid_sha256(value) for value in critical.values())
    ):
        raise ValueError("CALVIN dataset manifest critical-file inventory mismatch")
    extraction = payload.get("extraction")
    if not isinstance(extraction, dict) or set(extraction) != {
        "file_members_verified",
        "member_index",
        "verification",
    }:
        raise ValueError("CALVIN dataset manifest extraction-verification contract mismatch")
    if (
        extraction.get("file_members_verified") != inventory["file_member_count"]
        or extraction.get("verification") != "size-and-crc32-against-every-pinned-zip-member"
    ):
        raise ValueError("CALVIN dataset manifest extracted-file count or verification method mismatch")
    member_index = extraction.get("member_index")
    expected_member_index_path = calvin_member_index_path(root)
    if not isinstance(member_index, dict) or set(member_index) != {"bytes", "path", "schema", "sha256"}:
        raise ValueError("CALVIN dataset manifest member-index identity is invalid")
    if (
        member_index.get("path") != expected_member_index_path.name
        or member_index.get("schema") != "duo-vla-calvin-member-index-v1"
        or type(member_index.get("bytes")) is not int
        or member_index["bytes"] <= 0
        or not _valid_sha256(member_index.get("sha256"))
    ):
        raise ValueError("CALVIN dataset manifest member-index contract mismatch")
    dataset_root = root.parent
    for name in CALVIN_DATASET_CRITICAL_FILES:
        current = _file_sha256(dataset_root / name)
        if critical.get(name) != current:
            raise ValueError(f"CALVIN dataset manifest critical file changed: {name}")
    archive_path = dataset_root.with_suffix(".zip")
    if not archive_path.is_file() or archive_path.stat().st_size != CALVIN_ABC_D_ARCHIVE_BYTES:
        raise ValueError("pinned CALVIN archive is missing or has the wrong byte length")
    if verify_archive:
        if _file_sha256(archive_path) != CALVIN_ABC_D_ARCHIVE_SHA256:
            raise ValueError("pinned CALVIN archive SHA-256 mismatch")
        if _archive_member_inventory(archive_path) != inventory:
            raise ValueError("CALVIN archive member inventory differs from the verified manifest")
        archived_critical = _archive_critical_sha256(archive_path)
        if critical != archived_critical:
            raise ValueError("CALVIN critical extracted files do not match the pinned archive members")
        if (
            not expected_member_index_path.is_file()
            or expected_member_index_path.stat().st_size != member_index["bytes"]
            or _file_sha256(expected_member_index_path) != member_index["sha256"]
        ):
            raise ValueError("CALVIN extracted-member index file identity mismatch")
        uri = expected_member_index_path.as_uri() + "?mode=ro&immutable=1"
        with closing(sqlite3.connect(uri, uri=True)) as connection:
            connection.execute("PRAGMA query_only=ON")
            _verify_member_index_schema(connection)
            metadata = dict(connection.execute("SELECT name, value FROM metadata"))
            row_count = connection.execute("SELECT count(*) FROM members").fetchone()
            integrity = connection.execute("PRAGMA integrity_check").fetchone()
            _verify_member_index_central_directory(archive_path, connection)
        if (
            metadata
            != {
                "file_member_count": str(inventory["file_member_count"]),
                "schema": "duo-vla-calvin-member-index-v1",
            }
            or row_count != (inventory["file_member_count"],)
            or integrity != ("ok",)
        ):
            raise ValueError("CALVIN extracted-member index database contract mismatch")
    return payload


def _peek_calvin_manifest(training_root: str | Path) -> tuple[dict[str, Any], str]:
    root = Path(os.path.abspath(os.fspath(training_root)))
    path = calvin_dataset_manifest_path(root)
    if not path.is_file():
        raise FileNotFoundError(f"verified CALVIN dataset manifest is missing: {path}")
    payload, file_sha256 = _read_strict_json(path)
    schema = payload.get("schema")
    if schema not in (CALVIN_DATASET_MANIFEST_SCHEMA_V3, CALVIN_DATASET_MANIFEST_SCHEMA):
        raise ValueError(f"unsupported CALVIN dataset manifest schema: {schema!r}")
    return payload, file_sha256


def load_calvin_dataset_manifest(
    training_root: str | Path,
    *,
    verify_archive: bool = True,
    allow_legacy_v3: bool = False,
) -> dict[str, Any]:
    """Dispatch only on the exact committed manifest schema, never frame presence."""

    root = Path(os.path.abspath(os.fspath(training_root)))
    peeked, _file_sha256_value = _peek_calvin_manifest(root)
    if peeked["schema"] == CALVIN_DATASET_MANIFEST_SCHEMA_V3:
        if not allow_legacy_v3:
            raise ValueError("legacy extracted CALVIN v3 storage is fixture/parity-only")
        loaded = _load_calvin_v3_dataset_manifest(root, verify_archive=verify_archive)
    else:
        loaded = load_calvin_archive_manifest(root.parent.parent)
    if loaded != peeked:
        raise ValueError("CALVIN dataset manifest changed during schema dispatch")
    return loaded


def _authenticated_generation_payload(
    root: Path,
    manifest: dict[str, Any],
    *,
    manifest_file_sha256: str,
    archive_file_identity: dict[str, int] | None = None,
) -> dict[str, Any]:
    storage = _storage_identity_from_manifest(manifest, manifest_file_sha256=manifest_file_sha256)
    critical_files = _critical_sha256_map(manifest)
    value: dict[str, Any] = {
        "archive_file_identity": archive_file_identity,
        "critical_files": critical_files,
        "metadata_sha256": _metadata_identity_sha256(critical_files),
        "schema": CALVIN_AUTHENTICATED_GENERATION_SCHEMA,
        "storage": storage.to_dict(),
        "training_root": str(root),
    }
    value["content_sha256"] = hashlib.sha256(_canonical_json(value).encode()).hexdigest()
    return value


def _critical_sha256_map(manifest: dict[str, Any]) -> dict[str, str]:
    critical = manifest.get("critical_files")
    if not isinstance(critical, dict) or set(critical) != set(CALVIN_DATASET_CRITICAL_FILES):
        raise ValueError("CALVIN manifest critical-file inventory differs")
    if manifest.get("schema") == CALVIN_DATASET_MANIFEST_SCHEMA_V3:
        result = critical
    elif manifest.get("schema") == CALVIN_DATASET_MANIFEST_SCHEMA:
        result = {
            relative: identity.get("sha256") if isinstance(identity, dict) else None
            for relative, identity in critical.items()
        }
    else:
        raise ValueError("unsupported CALVIN manifest schema")
    if not all(_valid_sha256(value) for value in result.values()):
        raise ValueError("CALVIN manifest critical-file SHA-256 inventory is invalid")
    return dict(result)


def _metadata_identity_sha256(critical_files: dict[str, str]) -> str:
    selected = {
        f"training/{relative}": critical_files[f"training/{relative}"] for relative in CALVIN_CRITICAL_TRAIN_METADATA
    }
    return hashlib.sha256(_canonical_json(selected).encode()).hexdigest()


def _storage_identity_from_manifest(
    manifest: dict[str, Any],
    *,
    manifest_file_sha256: str,
) -> CalvinStorageIdentity:
    schema = manifest.get("schema")
    archive = manifest.get("archive")
    if not isinstance(archive, dict):
        raise ValueError("CALVIN manifest archive identity is missing")
    inventory = archive.get("member_inventory")
    if not isinstance(inventory, dict):
        raise ValueError("CALVIN manifest member inventory is missing")
    normalized_inventory = {
        "compressed_bytes": inventory.get("compressed_bytes"),
        "directory_member_count": inventory.get("directory_member_count"),
        "file_member_count": inventory.get("file_member_count"),
        "member_count": inventory.get("member_count"),
        "npz_member_count": inventory.get("npz_member_count"),
        "sha256": inventory.get("sha256"),
        "uncompressed_bytes": inventory.get("uncompressed_bytes"),
    }
    if schema == CALVIN_DATASET_MANIFEST_SCHEMA_V3:
        member_index = manifest.get("extraction", {}).get("member_index")
        mode = CALVIN_STORAGE_MODE_VERIFIED_EXTRACTION
        reader_schema = None
        central = None
        archive_path = CALVIN_ARCHIVE_NAME
    elif schema == CALVIN_DATASET_MANIFEST_SCHEMA:
        member_index = manifest.get("storage", {}).get("member_index")
        mode = manifest.get("storage", {}).get("mode")
        reader_schema = manifest.get("storage", {}).get("reader_schema")
        central = archive.get("central_directory")
        archive_path = archive.get("path")
    else:
        raise ValueError(f"unsupported CALVIN manifest schema: {schema!r}")
    if not isinstance(member_index, dict):
        raise ValueError("CALVIN manifest member-index identity is missing")
    payload: dict[str, Any] = {
        "archive": {
            "bytes": archive.get("bytes"),
            "path": archive_path,
            "sha256": archive.get("sha256"),
            "url": archive.get("url"),
        },
        "central_directory": central,
        "checksum_url": manifest.get("checksum_url"),
        "manifest": {
            "content_sha256": manifest.get("content_sha256"),
            "file_sha256": manifest_file_sha256,
            "schema": schema,
        },
        "member_index": member_index,
        "member_inventory": normalized_inventory,
        "mode": mode,
        "reader_schema": reader_schema,
        "schema": CALVIN_STORAGE_IDENTITY_SCHEMA,
    }
    payload["content_sha256"] = hashlib.sha256(_canonical_json(payload).encode()).hexdigest()
    return CalvinStorageIdentity.from_dict(payload)


def authenticate_calvin_dataset_generation(
    training_root: str | Path,
    *,
    allow_legacy_v3: bool = False,
) -> tuple[dict[str, Any], AuthenticatedCalvinDatasetGeneration]:
    """Authenticate official v4 production storage and mint a broadcast capability."""

    return _authenticate_calvin_dataset_generation(
        training_root,
        expected_archive_bytes=CALVIN_ABC_D_ARCHIVE_BYTES,
        expected_archive_sha256=CALVIN_ABC_D_ARCHIVE_SHA256,
        expected_central_directory=OFFICIAL_CENTRAL_DIRECTORY_CONTRACT,
        expected_archive_url=CALVIN_ABC_D_ARCHIVE_URL,
        expected_checksum_url=CALVIN_CHECKSUM_URL,
        allow_legacy_v3=allow_legacy_v3,
        verify_archive=True,
    )


def _authenticate_calvin_dataset_generation(
    training_root: str | Path,
    *,
    expected_archive_bytes: int,
    expected_archive_sha256: str,
    expected_central_directory: CentralDirectoryContract | None,
    expected_archive_url: str,
    expected_checksum_url: str,
    allow_legacy_v3: bool,
    verify_archive: bool,
) -> tuple[dict[str, Any], AuthenticatedCalvinDatasetGeneration]:
    """Parameterized fixture hook; production callers use the official wrapper."""

    root = Path(os.path.abspath(os.fspath(training_root)))
    peeked, _ = _peek_calvin_manifest(root)
    archive_file_identity: dict[str, int] | None = None
    if peeked["schema"] == CALVIN_DATASET_MANIFEST_SCHEMA:
        with CalvinArchiveReader.from_manifest(
            root.parent.parent,
            expected_archive_bytes=expected_archive_bytes,
            expected_archive_sha256=expected_archive_sha256,
            expected_central_directory=expected_central_directory,
            expected_archive_url=expected_archive_url,
            expected_checksum_url=expected_checksum_url,
        ) as reader:
            manifest = reader.authenticated_manifest
            archive_file_identity = reader.archive_file_identity
    elif allow_legacy_v3:
        manifest = _load_calvin_v3_dataset_manifest(root, verify_archive=verify_archive)
    else:
        raise ValueError("CALVIN production authentication requires archive-direct manifest v4")
    if manifest != peeked:
        raise ValueError("CALVIN dataset manifest changed during authentication")
    manifest_path = calvin_dataset_manifest_path(root)
    current_manifest, manifest_file_sha256 = _read_strict_json(manifest_path)
    if current_manifest != manifest:
        raise ValueError("CALVIN dataset manifest changed after archive authentication")
    capability = AuthenticatedCalvinDatasetGeneration.from_dict(
        _authenticated_generation_payload(
            root,
            manifest,
            manifest_file_sha256=manifest_file_sha256,
            archive_file_identity=archive_file_identity,
        )
    )
    verify_calvin_dataset_generation(root, capability)
    return manifest, capability


def verify_calvin_dataset_generation(
    training_root: str | Path,
    authenticated_generation: AuthenticatedCalvinDatasetGeneration,
) -> dict[str, Any]:
    """Compare live files with a capability minted from the pinned archive.

    This intentionally avoids re-reading the 517 GiB archive.  It is safe only
    when ``authenticated_generation`` came directly from
    :func:`authenticate_calvin_dataset_generation` in this job (for example by
    an authenticated rank-zero process-group broadcast).
    """

    root = Path(os.path.abspath(os.fspath(training_root)))
    if str(root) != authenticated_generation.training_root:
        raise ValueError("authenticated CALVIN generation belongs to a different training root")
    manifest_path = calvin_dataset_manifest_path(root)
    current_manifest, current_file_sha256 = _read_strict_json(manifest_path)
    if current_file_sha256 != authenticated_generation.dataset_manifest_file_sha256:
        raise ValueError("CALVIN dataset manifest changed after archive authentication")
    if current_manifest.get("schema") != authenticated_generation.storage.manifest_schema:
        raise ValueError("CALVIN dataset manifest storage schema changed after authentication")
    if current_manifest.get("content_sha256") != authenticated_generation.dataset_manifest_content_sha256:
        raise ValueError("CALVIN dataset manifest identity differs from the authenticated generation")
    current_storage = _storage_identity_from_manifest(
        current_manifest,
        manifest_file_sha256=current_file_sha256,
    )
    if current_storage != authenticated_generation.storage:
        raise ValueError("CALVIN storage identity differs from the authenticated generation")
    critical = _critical_sha256_map(current_manifest)
    if tuple(sorted(critical.items())) != authenticated_generation.critical_files:
        raise ValueError("CALVIN critical-file identities differ from the authenticated generation")
    if _metadata_identity_sha256(critical) != authenticated_generation.metadata_sha256:
        raise ValueError("CALVIN training metadata differs from the authenticated generation")
    if authenticated_generation.storage.mode == CALVIN_STORAGE_MODE_ARCHIVE_DIRECT:
        if authenticated_generation.archive_file_identity is None:
            raise ValueError("authenticated CALVIN v4 generation has no live archive file identity")
        data_root = root.parent.parent
        with PinnedDirectoryPath.open(data_root) as pinned_root:
            with PinnedRegularFile.open_at(
                pinned_root.descriptor,
                authenticated_generation.storage.archive_path,
                display_path=data_root / authenticated_generation.storage.archive_path,
            ) as archive:
                if archive.identity.as_ephemeral_capability() != dict(authenticated_generation.archive_file_identity):
                    raise ValueError("CALVIN archive differs from the authenticated ephemeral file identity")
                archive.assert_bound_at(
                    pinned_root.descriptor,
                    authenticated_generation.storage.archive_path,
                )
            with PinnedRegularFile.open_at(
                pinned_root.descriptor,
                authenticated_generation.storage.member_index_path,
                display_path=data_root / authenticated_generation.storage.member_index_path,
            ) as member_index:
                if (
                    member_index.identity.size != authenticated_generation.member_index_bytes
                    or member_index.sha256() != authenticated_generation.member_index_sha256
                ):
                    raise ValueError("CALVIN member index differs from the authenticated generation")
                member_index.assert_bound_at(
                    pinned_root.descriptor,
                    authenticated_generation.storage.member_index_path,
                )
            pinned_root.assert_bound()
        return current_manifest
    else:
        # Legacy v3 remains an explicit fixture/parity backend. Its projected
        # metadata and archive stat are still rechecked by the v3 loader.
        _load_calvin_v3_dataset_manifest(root, verify_archive=False)
        member_path = calvin_member_index_path(root)
    if (
        not member_path.is_file()
        or member_path.stat().st_size != authenticated_generation.member_index_bytes
        or _file_sha256(member_path) != authenticated_generation.member_index_sha256
    ):
        raise ValueError("CALVIN member index differs from the authenticated generation")
    return current_manifest


def open_calvin_archive_reader(
    training_root: str | Path,
    authenticated_generation: AuthenticatedCalvinDatasetGeneration,
) -> CalvinArchiveReader:
    """Open the v4 reader owned by one dataset and cross-bind its manifest."""

    root = Path(os.path.abspath(os.fspath(training_root)))
    if str(root) != authenticated_generation.training_root:
        raise ValueError("authenticated CALVIN generation belongs to a different training root")
    storage = authenticated_generation.storage
    if storage.mode != CALVIN_STORAGE_MODE_ARCHIVE_DIRECT or storage.central_directory is None:
        raise ValueError("authenticated CALVIN generation is not archive-direct v4")
    central = dict(storage.central_directory)
    inventory = dict(storage.member_inventory)
    contract = CentralDirectoryContract(
        offset=central["offset"],
        size=central["bytes"],
        sha256=central["sha256"],
        member_count=inventory["member_count"],
        file_member_count=inventory["file_member_count"],
        directory_member_count=inventory["directory_member_count"],
        npz_member_count=inventory["npz_member_count"],
    )
    if authenticated_generation.archive_file_identity is None:
        raise ValueError("authenticated CALVIN v4 generation has no live archive file identity")
    reader = CalvinArchiveReader.from_manifest_fast(
        root.parent.parent,
        expected_archive_file_identity=dict(authenticated_generation.archive_file_identity),
        expected_archive_bytes=storage.archive_bytes,
        expected_archive_sha256=storage.archive_sha256,
        expected_central_directory=contract,
        expected_archive_url=storage.archive_url,
        expected_checksum_url=storage.checksum_url,
    )
    try:
        opened_manifest = reader.authenticated_manifest
        opened_storage = _storage_identity_from_manifest(
            opened_manifest,
            manifest_file_sha256=storage.manifest_file_sha256,
        )
        if opened_storage != storage:
            raise ValueError("opened CALVIN archive reader differs from the authenticated generation")
        return reader
    except BaseException:
        reader.close()
        raise


def _normalization_dataset_identity(
    storage: CalvinStorageIdentity,
    *,
    metadata_sha256: str,
) -> dict[str, Any]:
    return {
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
        "name": "task_ABC_D",
        "reader_schema": storage.reader_schema,
        "split": "training",
        "storage_identity_sha256": storage.content_sha256,
        "storage_mode": storage.mode,
    }


def compute_calvin_normalization_artifact(
    dataset: CalvinNpzDataset,
    *,
    split_seed: int = DEFAULT_SPLIT_SEED,
    validation_fraction: float = 0.1,
    archive_sha256: str = CALVIN_ABC_D_ARCHIVE_SHA256,
    verify_archive: bool = True,
    authenticated_generation: AuthenticatedCalvinDatasetGeneration | None = None,
    allow_legacy_v3: bool = False,
    allow_non_official_archive: bool = False,
) -> dict[str, Any]:
    """Fit q01/q99 with physical v4 reads and canonical ordinal placement."""

    if not allow_non_official_archive and archive_sha256 != CALVIN_ABC_D_ARCHIVE_SHA256:
        raise ValueError("CALVIN ABC_D archive SHA-256 does not match the published checksum")
    if authenticated_generation is not None:
        dataset_manifest = verify_calvin_dataset_generation(dataset.root, authenticated_generation)
        storage = authenticated_generation.storage
        metadata_sha256 = authenticated_generation.metadata_sha256
    else:
        if not allow_legacy_v3:
            raise ValueError("production CALVIN normalization requires an authenticated v4 generation")
        dataset_manifest = load_calvin_dataset_manifest(
            dataset.root,
            verify_archive=verify_archive,
            allow_legacy_v3=True,
        )
        if dataset_manifest.get("schema") != CALVIN_DATASET_MANIFEST_SCHEMA_V3:
            raise ValueError("unauthenticated normalization is supported only for explicit v3 parity fixtures")
        _manifest, manifest_file_sha256 = _peek_calvin_manifest(dataset.root)
        storage = _storage_identity_from_manifest(
            dataset_manifest,
            manifest_file_sha256=manifest_file_sha256,
        )
        metadata_sha256 = _metadata_identity_sha256(_critical_sha256_map(dataset_manifest))
    if storage.archive_sha256 != archive_sha256:
        raise ValueError("normalization archive SHA-256 differs from the authenticated storage identity")
    if storage.mode == CALVIN_STORAGE_MODE_VERIFIED_EXTRACTION and not allow_legacy_v3:
        raise ValueError("legacy extracted CALVIN v3 normalization is fixture/parity-only")
    if getattr(dataset, "storage_mode", storage.mode) != storage.mode:
        raise ValueError("CALVIN dataset backend differs from its authenticated storage identity")
    split = make_calvin_episode_split(
        dataset.episodes,
        dataset.annotations,
        validation_fraction=validation_fraction,
        seed=split_seed,
    )
    training_frames = sum(dataset.episodes[index].length for index in split.train_episode_indices)
    if training_frames <= 0:
        raise ValueError("CALVIN training split contains no frames")
    continuous_states = np.empty((training_frames, 7), dtype=np.float32)
    state_gripper_values: set[float] = set()
    action_gripper_values: set[float] = set()
    action_min = np.full(6, np.inf, dtype=np.float64)
    action_max = np.full(6, -np.inf, dtype=np.float64)
    selected_episodes = tuple(dataset.episodes[index] for index in split.train_episode_indices)
    starts = tuple(episode.global_start for episode in selected_episodes)
    offsets: list[int] = []
    running = 0
    for episode in selected_episodes:
        offsets.append(running)
        running += episode.length
    if running != training_frames:
        raise AssertionError("CALVIN canonical training-frame ordinal count differs")
    seen = np.zeros(training_frames, dtype=np.bool_)

    def canonical_ordinal(global_index: int) -> int | None:
        episode_offset = bisect_right(starts, global_index) - 1
        if episode_offset < 0:
            return None
        episode = selected_episodes[episode_offset]
        if global_index > episode.global_end_inclusive:
            return None
        return offsets[episode_offset] + global_index - episode.global_start

    def place(global_index: int, robot_source: np.ndarray, action_source: np.ndarray) -> None:
        nonlocal action_min, action_max
        ordinal = canonical_ordinal(global_index)
        if ordinal is None:
            return
        if seen[ordinal]:
            raise ValueError(f"CALVIN normalization encountered duplicate selected frame {global_index}")
        robot_obs = np.asarray(robot_source, dtype=np.float32)
        rel_actions = np.asarray(action_source, dtype=np.float32)
        continuous_states[ordinal] = robot_obs[:7]
        state_gripper_values.add(float(robot_obs[14]))
        action_gripper_values.add(float(rel_actions[6]))
        action_min = np.minimum(action_min, rel_actions[:6])
        action_max = np.maximum(action_max, rel_actions[:6])
        seen[ordinal] = True

    if storage.mode == CALVIN_STORAGE_MODE_ARCHIVE_DIRECT:
        for global_index, robot_source, action_source in dataset.iter_state_actions_physical():
            place(global_index, robot_source, action_source)
        scan_order = "archive data_offset ascending; values placed by canonical episode/global ordinal"
    else:
        for episode in selected_episodes:
            for global_index in range(episode.global_start, episode.global_end_inclusive + 1):
                robot_source, action_source = dataset.read_state_action(global_index)
                place(global_index, robot_source, action_source)
        scan_order = "canonical episode index ascending, global timestep ascending (v3 parity only)"
    if not bool(seen.all()):
        missing = int((~seen).sum())
        raise ValueError(f"CALVIN normalization did not visit every selected frame exactly once: missing={missing}")
    if state_gripper_values != {-1.0, 1.0}:
        raise ValueError(f"CALVIN state gripper values must contain exactly {{-1,+1}}, got {state_gripper_values}")
    if action_gripper_values != {-1.0, 1.0}:
        raise ValueError(f"CALVIN action gripper values must contain exactly {{-1,+1}}, got {action_gripper_values}")
    if bool((action_min < -1.0 - 1e-6).any()) or bool((action_max > 1.0 + 1e-6).any()):
        raise ValueError("CALVIN rel_actions continuous channels are outside official [-1,1] scaling")
    state_q01, state_q99 = np.quantile(
        continuous_states,
        (0.01, 0.99),
        axis=0,
        method="linear",
    )
    constant_dimensions = np.flatnonzero(np.abs(state_q99 - state_q01) < 1e-6).tolist()
    # Refuse to publish an artifact whose serialized float64 semantics would
    # silently change when materialized as the runtime float32 normalizer.
    _validated_state_percentiles(
        state_q01.tolist(),
        state_q99.tolist(),
        constant_dimensions=constant_dimensions,
    )
    payload: dict[str, Any] = {
        "schema": CALVIN_STATS_SCHEMA,
        "dataset": _normalization_dataset_identity(storage, metadata_sha256=metadata_sha256),
        "split": _split_payload(split, split_seed=split_seed, validation_fraction=validation_fraction),
        "counts": {
            "total_episodes": len(dataset.episodes),
            "total_annotations": len(dataset.annotations),
            "tasks": len(dataset.tasks),
            "training_episodes": len(split.train_episode_indices),
            "training_frames": training_frames,
            "validation_episodes": len(split.validation_episode_indices),
            "validation_frames": sum(dataset.episodes[index].length for index in split.validation_episode_indices),
        },
        "state": {
            "dimension": 8,
            "continuous_dimensions": [0, 1, 2, 3, 4, 5, 6],
            "gripper_index": 7,
            "observed_gripper_values": [-1.0, 1.0],
            "q01": state_q01.tolist(),
            "q99": state_q99.tolist(),
            "constant_dimensions": constant_dimensions,
        },
        "action": {
            "dimension": 7,
            "continuous_dimensions": [0, 1, 2, 3, 4, 5],
            "gripper_index": 6,
            "observed_gripper_values": [-1.0, 1.0],
            "continuous_min": action_min.tolist(),
            "continuous_max": action_max.tolist(),
            "transform": "identity_official_scaled_rel_actions",
        },
        "algorithm": {
            "quantile": "numpy.quantile(method=linear)",
            "source_dtype": "float32",
            "result_dtype": "float64",
            "frame_order": scan_order,
            "canonical_ordinal_placement": "episode index ascending, then global timestep ascending",
            "selected_frame_exact_once_bitmap": True,
            "arrays_scanned": ["robot_obs", "rel_actions"],
            "state_gripper_excluded_from_percentiles": True,
            "actions_re_normalized": False,
        },
    }
    payload["content_sha256"] = _content_hash(payload)
    return payload


def _split_payload(
    split: CalvinEpisodeSplit,
    *,
    split_seed: int,
    validation_fraction: float,
) -> dict[str, Any]:
    return {
        "algorithm": "scene-grouped stable sha256 whole-episode ordering with all-task coverage assertion",
        "seed": split_seed,
        "validation_fraction": validation_fraction,
        "train_episode_indices": list(split.train_episode_indices),
        "validation_episode_indices": list(split.validation_episode_indices),
        "train_episode_sha256": _indices_hash(split.train_episode_indices),
        "validation_episode_sha256": _indices_hash(split.validation_episode_indices),
    }


def save_calvin_normalization_artifact(path: str | Path, payload: dict[str, Any]) -> None:
    output = Path(os.path.abspath(os.fspath(path)))
    if payload.get("schema") != CALVIN_STATS_SCHEMA or payload.get("content_sha256") != _content_hash(payload):
        raise ValueError("CALVIN normalization payload schema or content hash is invalid")
    data = (json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n").encode("utf-8")
    if not output.parent.is_dir():
        raise FileNotFoundError(f"CALVIN normalization output parent must already exist: {output.parent}")
    temporary_name = f".{output.name}.tmp-{os.getpid()}-{secrets.token_hex(8)}"
    parent_descriptor = os.open(
        output.parent,
        os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW,
    )
    descriptor: int | None = None
    temporary_present = False
    try:
        descriptor = os.open(
            temporary_name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW,
            0o600,
            dir_fd=parent_descriptor,
        )
        temporary_present = True
        view = memoryview(data)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("CALVIN normalization artifact write made no progress")
            view = view[written:]
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = None
        os.link(
            temporary_name,
            output.name,
            src_dir_fd=parent_descriptor,
            dst_dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
        try:
            os.unlink(temporary_name, dir_fd=parent_descriptor)
            temporary_present = False
        except OSError as exc:
            try:
                os.fsync(parent_descriptor)
            except OSError as sync_exc:
                raise OSError(
                    "CALVIN normalization destination may be committed with a stale temporary link; "
                    "directory fsync also failed"
                ) from sync_exc
            raise OSError("CALVIN normalization destination may be committed; stale temporary link remains") from exc
        os.fsync(parent_descriptor)
    except FileExistsError as exc:
        raise FileExistsError(f"CALVIN normalization artifact already exists: {output}") from exc
    finally:
        try:
            if descriptor is not None:
                os.close(descriptor)
        finally:
            try:
                if temporary_present:
                    with suppress(FileNotFoundError):
                        os.unlink(temporary_name, dir_fd=parent_descriptor)
            except OSError:
                # Preserve the primary publication failure. A persistent
                # cleanup error may leave the explicitly named private temp,
                # but must never leak the parent descriptor.
                pass
            finally:
                os.close(parent_descriptor)


def load_calvin_state_normalizer(
    path: str | Path,
    *,
    expected_archive_sha256: str = CALVIN_ABC_D_ARCHIVE_SHA256,
    training_root: str | Path | None = None,
    verify_archive: bool = True,
    authenticated_generation: AuthenticatedCalvinDatasetGeneration | None = None,
    allow_legacy_v3: bool = False,
) -> tuple[ActionNormalizer, dict[str, Any]]:
    if authenticated_generation is not None and training_root is None:
        raise ValueError("authenticated CALVIN generation requires training_root")
    payload, _artifact_file_sha256 = _read_strict_json(Path(path))
    if (
        not isinstance(payload, dict)
        or set(payload) != {"action", "algorithm", "content_sha256", "counts", "dataset", "schema", "split", "state"}
        or payload.get("schema") != CALVIN_STATS_SCHEMA
    ):
        raise ValueError("unsupported CALVIN normalization artifact schema")
    if payload.get("content_sha256") != _content_hash(payload):
        raise ValueError("CALVIN normalization artifact content hash mismatch")
    dataset = payload.get("dataset", {})
    expected_dataset_fields = {
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
    if (
        not isinstance(dataset, dict)
        or set(dataset) != expected_dataset_fields
        or dataset.get("name") != "task_ABC_D"
        or dataset.get("split") != "training"
        or dataset.get("archive_sha256") != expected_archive_sha256
        or type(dataset.get("archive_bytes")) is not int
        or dataset["archive_bytes"] <= 0
    ):
        raise ValueError("CALVIN normalization artifact dataset identity mismatch")
    if dataset.get("metadata_files") != list(CALVIN_CRITICAL_TRAIN_METADATA):
        raise ValueError("CALVIN normalization artifact critical metadata contract mismatch")
    for name in (
        "dataset_manifest_sha256",
        "dataset_manifest_file_sha256",
        "member_inventory_sha256",
        "metadata_sha256",
        "storage_identity_sha256",
    ):
        value = dataset.get(name)
        if not _valid_sha256(value):
            raise ValueError(f"CALVIN normalization artifact has no valid {name}")
    member_index = dataset.get("member_index")
    if (
        not isinstance(member_index, dict)
        or set(member_index) != {"bytes", "path", "schema", "sha256"}
        or type(member_index.get("bytes")) is not int
        or member_index["bytes"] <= 0
        or not isinstance(member_index.get("path"), str)
        or not isinstance(member_index.get("schema"), str)
        or not _valid_sha256(member_index.get("sha256"))
    ):
        raise ValueError("CALVIN normalization artifact member-index identity is invalid")
    storage_mode = dataset.get("storage_mode")
    if storage_mode == CALVIN_STORAGE_MODE_ARCHIVE_DIRECT:
        if (
            dataset.get("dataset_manifest_schema") != CALVIN_DATASET_MANIFEST_SCHEMA
            or dataset.get("reader_schema") != CALVIN_ARCHIVE_READER_SCHEMA
            or not _valid_sha256(dataset.get("central_directory_sha256"))
        ):
            raise ValueError("CALVIN normalization archive-direct storage identity is invalid")
    elif storage_mode == CALVIN_STORAGE_MODE_VERIFIED_EXTRACTION and allow_legacy_v3:
        if (
            dataset.get("dataset_manifest_schema") != CALVIN_DATASET_MANIFEST_SCHEMA_V3
            or dataset.get("reader_schema") is not None
            or dataset.get("central_directory_sha256") is not None
        ):
            raise ValueError("CALVIN normalization legacy storage identity is invalid")
    else:
        raise ValueError("CALVIN normalization requires archive-direct v4 production storage")
    if training_root is not None:
        if authenticated_generation is not None:
            verify_calvin_dataset_generation(training_root, authenticated_generation)
            if authenticated_generation.storage.mode == CALVIN_STORAGE_MODE_ARCHIVE_DIRECT:
                # The strict artifact reload is the publication gate. In
                # addition to the lightweight generation check above, validate
                # central/index semantics and projected bytes through one
                # fully pinned fast reader (the full 517 GiB SHA remains
                # intentionally skipped by the ephemeral inode capability).
                with open_calvin_archive_reader(training_root, authenticated_generation):
                    pass
            expected_dataset = _normalization_dataset_identity(
                authenticated_generation.storage,
                metadata_sha256=authenticated_generation.metadata_sha256,
            )
            if dataset != expected_dataset:
                raise ValueError("CALVIN normalization artifact storage identity mismatch")
        else:
            if not allow_legacy_v3:
                raise ValueError("production normalization loading requires an authenticated v4 generation")
            current_manifest = load_calvin_dataset_manifest(
                training_root,
                verify_archive=verify_archive,
                allow_legacy_v3=True,
            )
            _current_payload, current_file_sha256 = _peek_calvin_manifest(training_root)
            current_storage = _storage_identity_from_manifest(
                current_manifest,
                manifest_file_sha256=current_file_sha256,
            )
            expected_dataset = _normalization_dataset_identity(
                current_storage,
                metadata_sha256=_metadata_identity_sha256(_critical_sha256_map(current_manifest)),
            )
            if dataset != expected_dataset:
                raise ValueError("CALVIN normalization artifact legacy storage identity mismatch")
    split = payload.get("split")
    split_fields = {
        "algorithm",
        "seed",
        "train_episode_indices",
        "train_episode_sha256",
        "validation_episode_indices",
        "validation_episode_sha256",
        "validation_fraction",
    }
    if not isinstance(split, dict) or set(split) != split_fields:
        raise ValueError("CALVIN normalization split contract fields differ")
    train_indices = split.get("train_episode_indices")
    validation_indices = split.get("validation_episode_indices")
    if (
        split.get("algorithm") != "scene-grouped stable sha256 whole-episode ordering with all-task coverage assertion"
        or type(split.get("seed")) is not int
        or not isinstance(split.get("validation_fraction"), float)
        or not 0.0 < split["validation_fraction"] < 1.0
        or not isinstance(train_indices, list)
        or not isinstance(validation_indices, list)
        or not train_indices
        or not validation_indices
        or any(type(index) is not int or index < 0 for index in (*train_indices, *validation_indices))
        or len(set(train_indices)) != len(train_indices)
        or len(set(validation_indices)) != len(validation_indices)
        or set(train_indices) & set(validation_indices)
        or split.get("train_episode_sha256") != _indices_hash(tuple(train_indices))
        or split.get("validation_episode_sha256") != _indices_hash(tuple(validation_indices))
    ):
        raise ValueError("CALVIN normalization split identity is invalid")
    counts = payload.get("counts")
    count_fields = {
        "tasks",
        "total_annotations",
        "total_episodes",
        "training_episodes",
        "training_frames",
        "validation_episodes",
        "validation_frames",
    }
    if (
        not isinstance(counts, dict)
        or set(counts) != count_fields
        or any(type(counts[name]) is not int or counts[name] <= 0 for name in count_fields)
        or counts["training_episodes"] != len(train_indices)
        or counts["validation_episodes"] != len(validation_indices)
        or counts["total_episodes"] != len(train_indices) + len(validation_indices)
    ):
        raise ValueError("CALVIN normalization count contract is invalid")
    state = payload.get("state", {})
    if (
        not isinstance(state, dict)
        or set(state)
        != {
            "constant_dimensions",
            "continuous_dimensions",
            "dimension",
            "gripper_index",
            "observed_gripper_values",
            "q01",
            "q99",
        }
        or state.get("dimension") != 8
        or state.get("continuous_dimensions") != list(range(7))
        or state.get("gripper_index") != 7
        or state.get("observed_gripper_values") != [-1.0, 1.0]
        or not isinstance(state.get("constant_dimensions"), list)
        or any(type(index) is not int or not 0 <= index < 7 for index in state["constant_dimensions"])
    ):
        raise ValueError("CALVIN normalization artifact state schema mismatch")
    q01, q99 = _validated_state_percentiles(
        state.get("q01"),
        state.get("q99"),
        constant_dimensions=state.get("constant_dimensions"),
    )
    action = payload.get("action")
    if (
        not isinstance(action, dict)
        or set(action)
        != {
            "continuous_dimensions",
            "continuous_max",
            "continuous_min",
            "dimension",
            "gripper_index",
            "observed_gripper_values",
            "transform",
        }
        or action.get("dimension") != 7
        or action.get("continuous_dimensions") != list(range(6))
        or action.get("gripper_index") != 6
        or action.get("observed_gripper_values") != [-1.0, 1.0]
        or action.get("transform") != "identity_official_scaled_rel_actions"
    ):
        raise ValueError("CALVIN actions must retain the official identity transform")
    action_min = torch.from_numpy(
        _strict_finite_float64_vector(
            action.get("continuous_min"),
            length=6,
            label="CALVIN normalization action continuous_min",
        )
    )
    action_max = torch.from_numpy(
        _strict_finite_float64_vector(
            action.get("continuous_max"),
            length=6,
            label="CALVIN normalization action continuous_max",
        )
    )
    if (
        action_min.shape != (6,)
        or action_max.shape != (6,)
        or not bool(torch.isfinite(action_min).all() and torch.isfinite(action_max).all())
        or bool((action_max < action_min).any())
        or bool((action_min < -1.0 - 1e-6).any())
        or bool((action_max > 1.0 + 1e-6).any())
    ):
        raise ValueError("CALVIN normalization official action ranges are invalid")
    algorithm = payload.get("algorithm")
    algorithm_fields = {
        "actions_re_normalized",
        "arrays_scanned",
        "canonical_ordinal_placement",
        "frame_order",
        "quantile",
        "result_dtype",
        "selected_frame_exact_once_bitmap",
        "source_dtype",
        "state_gripper_excluded_from_percentiles",
    }
    expected_frame_order = (
        "archive data_offset ascending; values placed by canonical episode/global ordinal"
        if storage_mode == CALVIN_STORAGE_MODE_ARCHIVE_DIRECT
        else "canonical episode index ascending, global timestep ascending (v3 parity only)"
    )
    if (
        not isinstance(algorithm, dict)
        or set(algorithm) != algorithm_fields
        or algorithm.get("actions_re_normalized") is not False
        or algorithm.get("arrays_scanned") != ["robot_obs", "rel_actions"]
        or algorithm.get("canonical_ordinal_placement") != "episode index ascending, then global timestep ascending"
        or algorithm.get("frame_order") != expected_frame_order
        or algorithm.get("quantile") != "numpy.quantile(method=linear)"
        or algorithm.get("result_dtype") != "float64"
        or algorithm.get("selected_frame_exact_once_bitmap") is not True
        or algorithm.get("source_dtype") != "float32"
        or algorithm.get("state_gripper_excluded_from_percentiles") is not True
    ):
        raise ValueError("CALVIN normalization algorithm contract differs")
    normalizer = ActionNormalizer(
        PercentileNormalizer(
            q01,
            q99,
        ),
        action_dim=8,
        gripper_index=7,
    )
    return normalizer, payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("training_root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--split-seed", type=int, default=DEFAULT_SPLIT_SEED)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--archive-sha256", default=CALVIN_ABC_D_ARCHIVE_SHA256)
    parser.add_argument("--max-cached-frames", type=int, default=512)
    args = parser.parse_args()
    # Authenticate pickle-bearing metadata before CalvinNpzDataset loads it.
    _, authenticated_generation = authenticate_calvin_dataset_generation(args.training_root)
    with CalvinNpzDataset(
        args.training_root,
        max_cached_frames=args.max_cached_frames,
        expected_scenes=("calvin_scene_A", "calvin_scene_B", "calvin_scene_C"),
        authenticated_generation=authenticated_generation,
    ) as dataset:
        artifact = compute_calvin_normalization_artifact(
            dataset,
            split_seed=args.split_seed,
            validation_fraction=args.validation_fraction,
            archive_sha256=args.archive_sha256,
            authenticated_generation=authenticated_generation,
        )
    save_calvin_normalization_artifact(args.output, artifact)
    _normalizer, reloaded = load_calvin_state_normalizer(
        args.output,
        expected_archive_sha256=args.archive_sha256,
        training_root=args.training_root,
        authenticated_generation=authenticated_generation,
    )
    if reloaded != artifact:
        raise RuntimeError("published CALVIN normalization artifact differs after strict reload")
    print(json.dumps(artifact, indent=2, sort_keys=True))


if __name__ == "__main__":
    from duo_vla.data.calvin_stats import main as _canonical_main

    _canonical_main()
