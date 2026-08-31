"""Authenticated replay bundles and reset banks for CALVIN A/B/C development.

This module is deliberately Python-3.8 compatible and imports neither the
modern archive reader nor Torch.  The Python-3.11 exporter is the only process
which may decode episode members from the authenticated archive.  Simulator
processes validate a content-addressed replay bundle against the current v4
manifest, v2 member index, projected metadata, normalization artifact, and
pinned simulator source before replaying its in-memory action arrays.
"""

from __future__ import annotations

# The simulator is pinned to Python 3.8.  Keep legacy typing spellings even
# though repository lint runs under a newer interpreter.
# ``zip(strict=...)`` is unavailable on Python 3.8, and percent formatting is
# retained in error-only paths so this file stays directly executable there.
# ruff: noqa: B905, RUF007, UP006, UP031, UP035, UP045
import ctypes
import errno
import hashlib
import io
import json
import os
import shutil
import sqlite3
import stat
import struct
import subprocess
import uuid
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np

ABC_SCENES = ("calvin_scene_A", "calvin_scene_B", "calvin_scene_C")
DATASET_NAME = "task_ABC_D"
DATASET_MANIFEST_SCHEMA = "duo-vla-calvin-dataset-manifest-v4"
MEMBER_INDEX_SCHEMA = "duo-vla-calvin-member-index-v2"
ARCHIVE_READER_SCHEMA = "duo-vla-calvin-archive-reader-v1"
STATE_ACTION_SIDECAR_SCHEMA = "duo-vla-calvin-state-action-sidecar-v1"
NORMALIZATION_SCHEMA = "duo-vla-calvin-abc-to-d-normalization-v4"
DEV_INPUT_SCHEMA = "duo-vla-calvin-heldout-abc-inputs-v2"
REPLAY_BUNDLE_SCHEMA = "duo-vla-calvin-heldout-abc-replay-bundle-v1"
DEV_BANK_SCHEMA = "duo-vla-calvin-heldout-abc-reset-bank-v2"
RESET_ID_DOMAIN = "duo-vla-calvin-heldout-abc-reset-v2"
CANDIDATE_RANK_DOMAIN = "duo-vla-calvin-heldout-abc-candidate-rank-v1"
SMOKE_SELECTION_DOMAIN = "duo-vla-calvin-heldout-abc-smoke-selection-v1"
STATE_ENCODING = "contiguous-little-endian-float64"
SPLIT_ALGORITHM = "scene-grouped stable sha256 whole-episode ordering with all-task coverage assertion"

ARCHIVE_BYTES = 555_309_812_705
ARCHIVE_SHA256 = "c2036c67eb4c06966af1d1e1665bdb572c69e1404f5e77ffd46b384ff2b79f74"
ARCHIVE_URL = "http://calvin.cs.uni-freiburg.de/dataset/task_ABC_D.zip"
CHECKSUM_URL = "http://calvin.cs.uni-freiburg.de/dataset/sha256sum.txt"
ARCHIVE_NAME = "task_ABC_D.zip"
MEMBER_INDEX_NAME = "task_ABC_D.members-v2.sqlite3"
MANIFEST_NAME = "task_ABC_D.manifest.json"
VERIFICATION_CONTRACT = (
    "full-archive-sha256+streamed-central-inventory+zip64-local-header+raw-deflate-eof-length-crc32-logical-sha256"
)
OFFICIAL_CENTRAL_DIRECTORY_OFFSET = 555_080_601_096
OFFICIAL_CENTRAL_DIRECTORY_BYTES = 229_211_511
OFFICIAL_CENTRAL_DIRECTORY_SHA256 = "b4f79bda7f6b966b51aa419badd0f7db7a8972a7b58d6d342af60aceff0ea31b"
OFFICIAL_MEMBER_COUNT = 1_894_126
OFFICIAL_FILE_MEMBER_COUNT = 1_894_106
OFFICIAL_NPZ_MEMBER_COUNT = 1_894_067
OFFICIAL_DIRECTORY_MEMBER_COUNT = 20

CALVIN_REVISION = "fa03f01f19c65920e18cf37398a9ce859274af76"
CALVIN_ENV_REVISION = "1431a46bd36bde5903fb6345e68b5ccc30def666"
CALVIN_TACTO_REVISION = "dd53360d9a8c186f0d6439372ec0be0fa5e21731"
SCENE_CONFIG_SHA256 = {
    "calvin_scene_A": "e91f76b7af0950828ad9bc426768c4600c8badcdbd41a719dcce326e12e7b05d",
    "calvin_scene_B": "7577625ef8dc40918936875697806b3dc1fa53a7ba59c4d0950cd2937a47baa0",
    "calvin_scene_C": "a7cf204d27465ae80ec41aaa3a62ea4fc30346dda8413d6702969fc1e70a7ddc",
}
TASK_ORACLE_SHA256 = "6e905de3ca05118efdd8a51f8a7756ec6e61ffdb2b9b6a2843f0b7e0e9e51dcf"

TRAIN_CRITICAL_FILES = (
    "ep_start_end_ids.npy",
    "lang_annotations/auto_lang_ann.npy",
    "scene_info.npy",
    ".hydra/merged_config.yaml",
)
DATASET_CRITICAL_FILES = (
    *("training/" + relative for relative in TRAIN_CRITICAL_FILES),
    "validation/ep_start_end_ids.npy",
    "validation/.hydra/merged_config.yaml",
)

ROBOT_SHAPE = (15,)
SCENE_SHAPE = (24,)
ACTION_SHAPE = (7,)
RESET_DTYPE = np.dtype("<f8")
ROBOT_ARTIFACT = "robot_obs.npy"
SCENE_ARTIFACT = "scene_obs.npy"
ACTION_ARTIFACT = "rel_actions.npy"

_DIRECTORY_OPEN_FLAGS = os.O_RDONLY | os.O_NONBLOCK | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
_FILE_OPEN_FLAGS = os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW | os.O_CLOEXEC
_IDENTITY_FIELDS = ("st_dev", "st_ino", "st_mode", "st_size", "st_mtime_ns", "st_ctime_ns", "st_nlink")
_AT_FDCWD = -100
_RENAME_NOREPLACE = 1

_SPLIT_KEYS = {
    "algorithm",
    "seed",
    "train_episode_indices",
    "train_episode_sha256",
    "validation_episode_indices",
    "validation_episode_sha256",
    "validation_fraction",
}
_INPUT_IDENTITY_KEYS = {
    "archive_bytes",
    "archive_sha256",
    "calvin_archive_source_sha256",
    "calvin_env_revision",
    "calvin_revision",
    "calvin_tacto_revision",
    "central_directory_sha256",
    "dataset_manifest_file_sha256",
    "dataset_manifest_schema",
    "dataset_manifest_sha256",
    "dev_states_source_sha256",
    "input_schema",
    "member_index_bytes",
    "member_index_path",
    "member_index_schema",
    "member_index_sha256",
    "member_inventory_sha256",
    "normalization_content_sha256",
    "normalization_file_sha256",
    "metadata_files",
    "metadata_sha256",
    "projected_training_metadata_sha256",
    "reader_schema",
    "replay_exporter_source_sha256",
    "replay_generator_source_sha256",
    "scene_config_sha256",
    "split_sha256",
    "storage_mode",
    "storage_identity_sha256",
    "task_oracle_sha256",
}
_MEMBER_IDENTITY_KEYS = {"global_index", "logical_bytes", "logical_sha256", "path"}
_BUNDLE_RECORD_KEYS = {
    "action_count",
    "action_offset",
    "action_sequence_sha256",
    "annotation_end_exclusive",
    "annotation_index",
    "episode_index",
    "global_start",
    "instruction",
    "member_identities",
    "record_sha256",
    "scene",
    "source_frame_sha256",
    "source_state_sha256",
    "task",
}
_ARTIFACT_KEYS = {"bytes", "dtype", "path", "sha256", "shape"}
_BUNDLE_TOP_LEVEL_KEYS = {"artifacts", "identity", "records", "root_sha256", "schema", "source"}
_BUNDLE_SOURCE_KEYS = {
    "calvin_archive_source_sha256",
    "dev_states_source_sha256",
    "replay_exporter_source_sha256",
    "schema",
}
_BANK_RECORD_KEYS = {
    "annotation_end_exclusive",
    "annotation_index",
    "candidate_rank_sha256",
    "episode_index",
    "global_start",
    "instruction",
    "replay_actions",
    "replay_bundle_record_sha256",
    "reset_id_sha256",
    "scene",
    "source_frame_sha256",
    "source_state_sha256",
    "task",
}
_SELECTION_KEYS = {"algorithm", "base_seed", "smoke_reset_indices", "smoke_tasks_per_scene"}
_REJECTION_KEYS = {
    "annotation_index",
    "candidate_rank_sha256",
    "episode_index",
    "global_start",
    "reason",
    "replay_bundle_record_sha256",
    "scene",
    "task",
}
_BANK_TOP_LEVEL_KEYS = {
    "artifacts",
    "identity",
    "records",
    "rejections",
    "replay_bundle",
    "root_sha256",
    "schema",
    "selection",
    "split",
}


class CalvinDevStateError(RuntimeError):
    """A replay-bundle, reset-bank, or authenticated-input invariant failed."""


@dataclass(frozen=True)
class CalvinDevCandidate:
    annotation_index: int
    episode_index: int
    global_start: int
    global_end_exclusive: int
    instruction: str
    task: str
    scene: str


@dataclass(frozen=True)
class CalvinResetFrame:
    robot_obs: np.ndarray
    scene_obs: np.ndarray
    source_frame_sha256: str


@dataclass(frozen=True)
class BundledCalvinReplay:
    candidate: CalvinDevCandidate
    frame: CalvinResetFrame
    actions: np.ndarray
    member_identities: Tuple[Dict[str, Any], ...]
    record_sha256: str


@dataclass(frozen=True)
class MaterializedCalvinReset:
    candidate: CalvinDevCandidate
    frame: CalvinResetFrame
    candidate_rank_sha256: str
    replay_actions: int
    replay_bundle_record_sha256: str


@dataclass(frozen=True)
class AuthenticatedCalvinDevInputs:
    identity: Dict[str, Any]
    member_index: Dict[str, Any]
    split: Dict[str, Any]
    stats: Dict[str, Any]
    metadata: Dict[str, bytes] = field(default_factory=dict)
    source_files: Dict[str, bytes] = field(default_factory=dict)
    training_root: str = ""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CalvinDevStateError(message)


def _exact_keys(value: Mapping[str, Any], expected: Set[str], name: str) -> None:
    _require(isinstance(value, Mapping), name + " must be an object")
    observed = set(value)
    _require(
        observed == expected,
        "%s fields differ: missing=%s extra=%s" % (name, sorted(expected - observed), sorted(observed - expected)),
    )


def _is_integer(value: Any, minimum: int = 0) -> bool:
    return type(value) is int and value >= minimum


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def canonical_json_bytes(value: Any, pretty: bool = False) -> bytes:
    options = {"allow_nan": False, "sort_keys": True}
    if pretty:
        return (json.dumps(value, indent=2, **options) + "\n").encode("utf-8")
    return json.dumps(value, separators=(",", ":"), **options).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _content_sha256(payload: Mapping[str, Any], field_name: str = "content_sha256") -> str:
    detached = dict(payload)
    detached.pop(field_name, None)
    return canonical_sha256(detached)


def _indices_sha256(indices: Sequence[int]) -> str:
    return hashlib.sha256(",".join(str(index) for index in indices).encode("ascii")).hexdigest()


def _same_identity(left: os.stat_result, right: os.stat_result) -> bool:
    return all(getattr(left, name) == getattr(right, name) for name in _IDENTITY_FIELDS)


def stable_regular_file_bytes(path: Path, require_single_link: bool = True) -> bytes:
    """Read one regular file once and retain its pathname/inode binding."""

    source = Path(path)
    try:
        before = os.stat(str(source), follow_symlinks=False)
    except OSError as exc:
        raise CalvinDevStateError("cannot stat authenticated file: %s" % source) from exc
    _require(stat.S_ISREG(before.st_mode), "authenticated input must be a regular non-symlink file: %s" % source)
    if require_single_link:
        _require(before.st_nlink == 1, "authenticated input must have exactly one link: %s" % source)
    try:
        descriptor = os.open(str(source), _FILE_OPEN_FLAGS)
    except OSError as exc:
        raise CalvinDevStateError("cannot no-follow open authenticated file: %s" % source) from exc
    try:
        opened = os.fstat(descriptor)
        _require(_same_identity(before, opened), "authenticated file changed while opening: %s" % source)
        blocks = []  # type: List[bytes]
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            blocks.append(block)
        after_descriptor = os.fstat(descriptor)
        after_path = os.stat(str(source), follow_symlinks=False)
        _require(
            _same_identity(opened, after_descriptor) and _same_identity(opened, after_path),
            "authenticated file changed while reading: %s" % source,
        )
        raw = b"".join(blocks)
        _require(len(raw) == opened.st_size, "authenticated file size changed while reading: %s" % source)
        return raw
    finally:
        os.close(descriptor)


def file_sha256(path: Path) -> str:
    return hashlib.sha256(stable_regular_file_bytes(Path(path))).hexdigest()


def _unique_object(pairs: List[Tuple[str, Any]]) -> Dict[str, Any]:
    value = {}  # type: Dict[str, Any]
    for key, item in pairs:
        if key in value:
            raise CalvinDevStateError("duplicate JSON field: %s" % key)
        value[key] = item
    return value


def _reject_constant(value: str) -> None:
    raise CalvinDevStateError("non-finite JSON number is forbidden: %s" % value)


def load_strict_json_bytes(raw: bytes, label: str) -> Dict[str, Any]:
    try:
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=_unique_object, parse_constant=_reject_constant)
    except (UnicodeDecodeError, json.JSONDecodeError, CalvinDevStateError) as exc:
        raise CalvinDevStateError("cannot parse strict JSON for %s" % label) from exc
    _require(isinstance(value, dict), "%s must contain one JSON object" % label)
    return value


def load_strict_json(path: Path) -> Dict[str, Any]:
    return load_strict_json_bytes(stable_regular_file_bytes(Path(path)), str(path))


def require_full_training_root(training_root: Path) -> Path:
    root = Path(training_root).resolve(strict=True)
    _require(root.name == "training", "CALVIN development source must end in /training")
    _require(root.parent.name == DATASET_NAME, "CALVIN development source must be task_ABC_D/training")
    _require(root.is_dir(), "CALVIN projected training root is missing")
    return root


def _manifest_root_sha256(manifest: Mapping[str, Any]) -> str:
    payload = dict(manifest)
    payload.pop("root_sha256", None)
    return canonical_sha256(payload)


def _validate_v4_manifest(payload: Mapping[str, Any]) -> None:
    _exact_keys(
        payload,
        {"archive", "checksum_url", "content_sha256", "critical_files", "dataset", "schema", "storage"},
        "CALVIN v4 dataset manifest",
    )
    _require(payload.get("schema") == DATASET_MANIFEST_SCHEMA, "CALVIN development requires a v4 dataset manifest")
    _require(payload.get("dataset") == DATASET_NAME, "CALVIN dataset identity differs")
    _require(payload.get("checksum_url") == CHECKSUM_URL, "CALVIN checksum source differs")
    _require(payload.get("content_sha256") == _content_sha256(payload), "CALVIN v4 manifest content hash differs")
    archive = payload.get("archive")
    storage = payload.get("storage")
    critical = payload.get("critical_files")
    _require(
        isinstance(archive, dict) and isinstance(storage, dict) and isinstance(critical, dict),
        "CALVIN v4 sections are missing",
    )
    _exact_keys(
        archive, {"bytes", "central_directory", "member_inventory", "path", "sha256", "url"}, "CALVIN archive identity"
    )
    _require(
        archive.get("bytes") == ARCHIVE_BYTES
        and archive.get("sha256") == ARCHIVE_SHA256
        and archive.get("url") == ARCHIVE_URL
        and archive.get("path") == ARCHIVE_NAME,
        "CALVIN archive identity differs from the official contract",
    )
    central = archive.get("central_directory")
    inventory = archive.get("member_inventory")
    _exact_keys(central, {"bytes", "entries", "offset", "sha256", "zip64"}, "CALVIN central-directory identity")
    _exact_keys(
        inventory,
        {
            "compressed_bytes",
            "directory_member_count",
            "file_member_count",
            "member_count",
            "npz_member_count",
            "sha256",
            "uncompressed_bytes",
        },
        "CALVIN member inventory",
    )
    _require(
        central.get("bytes") == OFFICIAL_CENTRAL_DIRECTORY_BYTES
        and central.get("offset") == OFFICIAL_CENTRAL_DIRECTORY_OFFSET
        and central.get("sha256") == OFFICIAL_CENTRAL_DIRECTORY_SHA256
        and central.get("entries") == OFFICIAL_MEMBER_COUNT
        and central.get("zip64") is True,
        "CALVIN central-directory identity differs from the official contract",
    )
    _require(
        inventory.get("member_count") == OFFICIAL_MEMBER_COUNT
        and inventory.get("file_member_count") == OFFICIAL_FILE_MEMBER_COUNT
        and inventory.get("directory_member_count") == OFFICIAL_DIRECTORY_MEMBER_COUNT
        and inventory.get("npz_member_count") == OFFICIAL_NPZ_MEMBER_COUNT
        and _is_sha256(inventory.get("sha256")),
        "CALVIN member inventory differs from the official contract",
    )
    _exact_keys(
        storage,
        {"derived_artifacts", "materialized_files", "member_index", "mode", "reader_schema", "verification"},
        "CALVIN archive-direct storage",
    )
    _require(
        storage.get("mode") == "archive-direct"
        and storage.get("reader_schema") == ARCHIVE_READER_SCHEMA
        and storage.get("verification") == VERIFICATION_CONTRACT,
        "CALVIN archive-direct reader/storage contract differs",
    )
    _require(
        storage.get("materialized_files") == list(DATASET_CRITICAL_FILES), "CALVIN projected metadata inventory differs"
    )
    _require(
        storage.get("derived_artifacts")
        == {"state_action_sidecar": None, "state_action_sidecar_schema_hook": STATE_ACTION_SIDECAR_SCHEMA},
        "CALVIN archive-direct derived-artifact contract differs",
    )
    index = storage.get("member_index")
    _exact_keys(index, {"bytes", "path", "schema", "sha256"}, "CALVIN v2 member-index identity")
    _require(
        index.get("path") == MEMBER_INDEX_NAME
        and index.get("schema") == MEMBER_INDEX_SCHEMA
        and _is_integer(index.get("bytes"), 1)
        and _is_sha256(index.get("sha256")),
        "CALVIN v2 member-index identity differs",
    )
    _require(set(critical) == set(DATASET_CRITICAL_FILES), "CALVIN critical metadata inventory differs")
    for relative, record in critical.items():
        _exact_keys(record, {"bytes", "crc32", "sha256"}, "CALVIN critical metadata record")
        _require(
            _is_integer(record.get("bytes"))
            and _is_integer(record.get("crc32"))
            and record["crc32"] <= 0xFFFFFFFF
            and _is_sha256(record.get("sha256")),
            "CALVIN critical metadata identity is invalid: %s" % relative,
        )


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
_EXPECTED_METADATA_COLUMNS = [(0, "name", "TEXT", 1, None, 1), (1, "value", "TEXT", 1, None, 0)]
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


def _open_authenticated_index(
    training_root: Path, manifest: Mapping[str, Any]
) -> Tuple[int, sqlite3.Connection, os.stat_result]:
    data_root = training_root.parents[1]
    identity = manifest["storage"]["member_index"]
    path = data_root / identity["path"]
    before = os.stat(str(path), follow_symlinks=False)
    _require(
        stat.S_ISREG(before.st_mode) and before.st_nlink == 1 and before.st_size == identity["bytes"],
        "CALVIN v2 member-index file identity differs",
    )
    descriptor = os.open(str(path), _FILE_OPEN_FLAGS)
    connection = None  # type: Optional[sqlite3.Connection]
    try:
        opened = os.fstat(descriptor)
        _require(_same_identity(before, opened), "CALVIN v2 member index changed while opening")
        digest = hashlib.sha256()
        while True:
            block = os.read(descriptor, 8 * 1024 * 1024)
            if not block:
                break
            digest.update(block)
        _require(digest.hexdigest() == identity["sha256"], "CALVIN v2 member-index SHA-256 differs")
        uri = "file:/proc/self/fd/%d?mode=ro&immutable=1" % descriptor
        connection = sqlite3.connect(uri, uri=True)
        connection.execute("PRAGMA query_only=ON")
        connection.execute("PRAGMA trusted_schema=OFF")
        _require(
            list(connection.execute("PRAGMA table_info(members)")) == _EXPECTED_MEMBER_COLUMNS,
            "CALVIN v2 member columns differ",
        )
        _require(
            list(connection.execute("PRAGMA table_info(metadata)")) == _EXPECTED_METADATA_COLUMNS,
            "CALVIN v2 metadata columns differ",
        )
        index_list = {
            (row[1], row[2], row[3], row[4])
            for row in connection.execute("PRAGMA index_list(members)")
            if not row[1].startswith("sqlite_")
        }
        _require(
            index_list
            == {
                ("members_episode", 1, "c", 1),
                ("members_local_header", 0, "c", 0),
                ("members_physical", 0, "c", 0),
            },
            "CALVIN v2 member-index properties differ",
        )
        _require(
            [row[2] for row in connection.execute("PRAGMA index_info(members_episode)")] == ["split", "global_index"]
            and [row[2] for row in connection.execute("PRAGMA index_info(members_local_header)")]
            == ["local_header_offset"]
            and [row[2] for row in connection.execute("PRAGMA index_info(members_physical)")] == ["data_offset"],
            "CALVIN v2 index columns differ",
        )
        _require(
            connection.execute("PRAGMA application_id").fetchone() == (1145853251,), "CALVIN v2 application id differs"
        )
        _require(connection.execute("PRAGMA user_version").fetchone() == (2,), "CALVIN v2 user version differs")
        _require(connection.execute("PRAGMA integrity_check").fetchone() == ("ok",), "CALVIN v2 integrity check failed")
        metadata = dict(connection.execute("SELECT name,value FROM metadata"))
        expected_names = _INDEX_METADATA_BASE_NAMES | {"critical_sha256:" + item for item in DATASET_CRITICAL_FILES}
        _require(set(metadata) == expected_names, "CALVIN v2 metadata inventory differs")
        archive = manifest["archive"]
        central = archive["central_directory"]
        inventory = archive["member_inventory"]
        expected_metadata = {
            "archive_bytes": str(archive["bytes"]),
            "archive_root": DATASET_NAME,
            "archive_sha256": archive["sha256"],
            "central_directory_bytes": str(central["bytes"]),
            "central_directory_offset": str(central["offset"]),
            "central_directory_sha256": central["sha256"],
            "central_directory_zip64": "1",
            "compressed_bytes": str(inventory["compressed_bytes"]),
            "directory_member_count": str(inventory["directory_member_count"]),
            "file_member_count": str(inventory["file_member_count"]),
            "member_count": str(inventory["member_count"]),
            "member_inventory_sha256": inventory["sha256"],
            "npz_member_count": str(inventory["npz_member_count"]),
            "reader_schema": ARCHIVE_READER_SCHEMA,
            "schema": MEMBER_INDEX_SCHEMA,
            "state_action_sidecar_schema": STATE_ACTION_SIDECAR_SCHEMA,
            "state_action_sidecar_status": "absent",
            "status": "complete",
            "uncompressed_bytes": str(inventory["uncompressed_bytes"]),
        }
        expected_metadata.update(
            {"critical_sha256:" + name: manifest["critical_files"][name]["sha256"] for name in DATASET_CRITICAL_FILES}
        )
        _require(metadata == expected_metadata, "CALVIN v2 index metadata differs from the v4 manifest")
        _require(
            connection.execute("SELECT count(*) FROM members").fetchone() == (inventory["member_count"],),
            "CALVIN v2 row count differs",
        )
        _require(os.fstat(descriptor).st_size == opened.st_size, "CALVIN v2 index changed during validation")
        after_path = os.stat(str(path), follow_symlinks=False)
        _require(
            _same_identity(opened, os.fstat(descriptor)) and _same_identity(opened, after_path),
            "CALVIN v2 index binding changed",
        )
        return descriptor, connection, opened
    except BaseException:
        if connection is not None:
            connection.close()
        os.close(descriptor)
        raise


def _close_authenticated_index(
    descriptor: int,
    connection: sqlite3.Connection,
    identity: os.stat_result,
    training_root: Path,
    manifest: Mapping[str, Any],
) -> None:
    path = training_root.parents[1] / manifest["storage"]["member_index"]["path"]
    try:
        _require(
            _same_identity(identity, os.fstat(descriptor))
            and _same_identity(identity, os.stat(str(path), follow_symlinks=False)),
            "CALVIN v2 member-index binding changed before close",
        )
    finally:
        try:
            connection.close()
        finally:
            os.close(descriptor)


def _read_projected_metadata(training_root: Path, manifest: Mapping[str, Any]) -> Dict[str, bytes]:
    dataset_root = training_root.parent
    result = {}  # type: Dict[str, bytes]
    for relative in DATASET_CRITICAL_FILES:
        raw = stable_regular_file_bytes(dataset_root / relative)
        record = manifest["critical_files"][relative]
        _require(len(raw) == record["bytes"], "projected metadata byte count differs: %s" % relative)
        _require(
            hashlib.sha256(raw).hexdigest() == record["sha256"], "projected metadata SHA-256 differs: %s" % relative
        )
        _require((zlib.crc32(raw) & 0xFFFFFFFF) == record["crc32"], "projected metadata CRC32 differs: %s" % relative)
        result[relative] = raw
    return result


def _training_metadata_sha256(metadata: Mapping[str, bytes]) -> str:
    digest = hashlib.sha256()
    digest.update(b"duo-vla-calvin-projected-training-metadata\x00v1\x00")
    digest.update(struct.pack(">Q", len(TRAIN_CRITICAL_FILES)))
    for short_name in TRAIN_CRITICAL_FILES:
        relative = "training/" + short_name
        raw = metadata[relative]
        encoded = relative.encode("utf-8")
        digest.update(struct.pack(">Q", len(encoded)))
        digest.update(encoded)
        digest.update(struct.pack(">Q", len(raw)))
        digest.update(raw)
    return digest.hexdigest()


def _normalization_metadata_sha256(manifest: Mapping[str, Any]) -> str:
    selected = {
        "training/" + relative: manifest["critical_files"]["training/" + relative]["sha256"]
        for relative in TRAIN_CRITICAL_FILES
    }
    return canonical_sha256(selected)


def _storage_identity(manifest: Mapping[str, Any], manifest_file_sha256: str) -> Dict[str, Any]:
    archive = manifest["archive"]
    storage = manifest["storage"]
    payload = {
        "archive": {
            "bytes": archive["bytes"],
            "path": archive["path"],
            "sha256": archive["sha256"],
            "url": archive["url"],
        },
        "central_directory": dict(archive["central_directory"]),
        "checksum_url": manifest["checksum_url"],
        "manifest": {
            "content_sha256": manifest["content_sha256"],
            "file_sha256": manifest_file_sha256,
            "schema": manifest["schema"],
        },
        "member_index": dict(storage["member_index"]),
        "member_inventory": dict(archive["member_inventory"]),
        "mode": storage["mode"],
        "reader_schema": storage["reader_schema"],
        "schema": "duo-vla-calvin-storage-identity-v1",
    }
    payload["content_sha256"] = canonical_sha256(payload)
    return payload


def _read_revision_file(path: Path) -> Dict[str, str]:
    raw = stable_regular_file_bytes(Path(path))
    values = {}  # type: Dict[str, str]
    try:
        lines = raw.decode("utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise CalvinDevStateError("CALVIN revision file is not UTF-8") from exc
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        key, separator, value = stripped.partition("=")
        _require(bool(separator and key and value) and key not in values, "invalid CALVIN revision line")
        values[key] = value
    return values


def _git_revision(path: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise CalvinDevStateError("cannot authenticate git revision at %s" % path) from exc
    return result.stdout.strip()


def _git_clean(path: Path) -> bool:
    try:
        result = subprocess.run(
            ["git", "-C", str(path), "status", "--porcelain", "--untracked-files=no"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise CalvinDevStateError("cannot authenticate git worktree at %s" % path) from exc
    return not result.stdout.strip()


def _authenticate_source_with_bytes(source_root: Path, revision_file: Path) -> Tuple[Dict[str, Any], Dict[str, bytes]]:
    root = Path(source_root).resolve(strict=True)
    _require(root.is_dir(), "pinned CALVIN source root is missing: %s" % root)
    revisions = _read_revision_file(revision_file)
    expected = {
        "CALVIN_REVISION": CALVIN_REVISION,
        "CALVIN_ENV_REVISION": CALVIN_ENV_REVISION,
        "CALVIN_TACTO_REVISION": CALVIN_TACTO_REVISION,
    }
    for key, value in expected.items():
        _require(revisions.get(key) == value, "CALVIN pinned revision contract changed: %s" % key)
    _require(_git_revision(root) == CALVIN_REVISION, "CALVIN checkout revision mismatch")
    _require(_git_revision(root / "calvin_env") == CALVIN_ENV_REVISION, "calvin_env checkout revision mismatch")
    _require(
        _git_revision(root / "calvin_env" / "tacto") == CALVIN_TACTO_REVISION,
        "calvin_env/tacto checkout revision mismatch",
    )
    for path in (root, root / "calvin_env", root / "calvin_env" / "tacto"):
        _require(_git_clean(path), "pinned CALVIN checkout has tracked modifications: %s" % path)
    source_files = {}  # type: Dict[str, bytes]
    scene_hashes = {}  # type: Dict[str, str]
    for scene in ABC_SCENES:
        path = root / "calvin_env" / "conf" / "scene" / (scene + ".yaml")
        raw = stable_regular_file_bytes(path)
        digest = hashlib.sha256(raw).hexdigest()
        _require(digest == SCENE_CONFIG_SHA256[scene], "pinned scene config changed: %s" % scene)
        scene_hashes[scene] = digest
        source_files["scene/" + scene + ".yaml"] = raw
    task_path = root / "calvin_models" / "conf" / "callbacks" / "rollout" / "tasks" / "new_playtable_tasks.yaml"
    task_raw = stable_regular_file_bytes(task_path)
    task_digest = hashlib.sha256(task_raw).hexdigest()
    _require(task_digest == TASK_ORACLE_SHA256, "pinned CALVIN task oracle config changed")
    source_files["task_oracle.yaml"] = task_raw
    return (
        {
            "calvin_env_revision": CALVIN_ENV_REVISION,
            "calvin_revision": CALVIN_REVISION,
            "calvin_tacto_revision": CALVIN_TACTO_REVISION,
            "scene_config_sha256": scene_hashes,
            "task_oracle_sha256": task_digest,
        },
        source_files,
    )


def authenticate_source(source_root: Path, revision_file: Path) -> Dict[str, Any]:
    identity, _files = _authenticate_source_with_bytes(source_root, revision_file)
    return identity


def development_tool_source_identity() -> Dict[str, str]:
    project_root = Path(__file__).resolve().parents[3]
    paths = {
        "calvin_archive_source_sha256": project_root / "src/duo_vla/data/calvin_archive.py",
        "dev_states_source_sha256": Path(__file__).resolve(),
        "replay_exporter_source_sha256": project_root / "scripts/calvin/export_calvin_dev_replay_bundle.py",
        "replay_generator_source_sha256": project_root / "scripts/calvin/generate_calvin_dev_states.py",
    }
    return {name: file_sha256(path) for name, path in paths.items()}


def _validate_split(split: Mapping[str, Any]) -> Dict[str, Any]:
    _exact_keys(split, _SPLIT_KEYS, "CALVIN normalization split")
    result = dict(split)
    train = result["train_episode_indices"]
    validation = result["validation_episode_indices"]
    for name, values in (("train", train), ("validation", validation)):
        _require(
            isinstance(values, list)
            and bool(values)
            and all(_is_integer(index) for index in values)
            and values == sorted(set(values)),
            "CALVIN %s episode indices are invalid" % name,
        )
        _require(result[name + "_episode_sha256"] == _indices_sha256(values), "CALVIN %s split hash differs" % name)
    _require(set(train).isdisjoint(validation), "CALVIN train/validation episodes overlap")
    _require(result.get("algorithm") == SPLIT_ALGORITHM, "CALVIN held-out split algorithm changed")
    _require(result.get("seed") == 1729, "CALVIN held-out split seed changed")
    _require(result.get("validation_fraction") == 0.1, "CALVIN held-out split fraction changed")
    return result


def _validate_normalization(
    stats_path: Path,
    manifest: Mapping[str, Any],
    projected_metadata_sha256: str,
    manifest_file_sha256: str,
) -> Tuple[Dict[str, Any], Dict[str, Any], str]:
    raw = stable_regular_file_bytes(stats_path)
    stats = load_strict_json_bytes(raw, "CALVIN normalization artifact")
    _require(stats.get("schema") == NORMALIZATION_SCHEMA, "CALVIN development requires normalization schema v4")
    _require(stats.get("content_sha256") == _content_sha256(stats), "CALVIN normalization content hash differs")
    _exact_keys(
        stats,
        {"action", "algorithm", "content_sha256", "counts", "dataset", "schema", "split", "state"},
        "CALVIN normalization artifact",
    )
    dataset = stats.get("dataset")
    _require(isinstance(dataset, dict), "CALVIN normalization dataset identity is missing")
    _require(_is_sha256(projected_metadata_sha256), "projected metadata byte identity is invalid")
    expected_fields = {
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
    _exact_keys(dataset, expected_fields, "CALVIN normalization dataset identity")
    index = manifest["storage"]["member_index"]
    storage_identity = _storage_identity(manifest, manifest_file_sha256)
    expected_dataset = {
        "archive_bytes": manifest["archive"]["bytes"],
        "archive_sha256": ARCHIVE_SHA256,
        "central_directory_sha256": manifest["archive"]["central_directory"]["sha256"],
        "dataset_manifest_file_sha256": manifest_file_sha256,
        "dataset_manifest_schema": DATASET_MANIFEST_SCHEMA,
        "dataset_manifest_sha256": manifest["content_sha256"],
        "member_index": dict(index),
        "member_inventory_sha256": manifest["archive"]["member_inventory"]["sha256"],
        "metadata_files": list(TRAIN_CRITICAL_FILES),
        "metadata_sha256": _normalization_metadata_sha256(manifest),
        "name": DATASET_NAME,
        "reader_schema": ARCHIVE_READER_SCHEMA,
        "split": "training",
        "storage_identity_sha256": storage_identity["content_sha256"],
        "storage_mode": "archive-direct",
    }
    _require(dataset == expected_dataset, "CALVIN normalization storage/dataset identity differs")
    action = stats.get("action")
    state = stats.get("state")
    algorithm = stats.get("algorithm")
    counts = stats.get("counts")
    _require(
        isinstance(action, dict)
        and set(action)
        == {
            "continuous_dimensions",
            "continuous_max",
            "continuous_min",
            "dimension",
            "gripper_index",
            "observed_gripper_values",
            "transform",
        }
        and action.get("dimension") == 7
        and action.get("continuous_dimensions") == list(range(6))
        and action.get("gripper_index") == 6
        and action.get("observed_gripper_values") == [-1.0, 1.0]
        and action.get("transform") == "identity_official_scaled_rel_actions",
        "CALVIN normalization action contract differs",
    )
    action_min = np.asarray(action.get("continuous_min"), dtype=np.float64)
    action_max = np.asarray(action.get("continuous_max"), dtype=np.float64)
    _require(
        action_min.shape == (6,)
        and action_max.shape == (6,)
        and bool(np.isfinite(action_min).all() and np.isfinite(action_max).all())
        and bool((action_min <= action_max).all())
        and bool((action_min >= -1.0 - 1e-6).all())
        and bool((action_max <= 1.0 + 1e-6).all()),
        "CALVIN normalization action ranges differ",
    )
    _require(
        isinstance(state, dict)
        and set(state)
        == {
            "constant_dimensions",
            "continuous_dimensions",
            "dimension",
            "gripper_index",
            "observed_gripper_values",
            "q01",
            "q99",
        }
        and state.get("dimension") == 8
        and state.get("continuous_dimensions") == list(range(7))
        and state.get("gripper_index") == 7
        and state.get("observed_gripper_values") == [-1.0, 1.0],
        "CALVIN normalization state contract differs",
    )
    constant_dimensions = state.get("constant_dimensions")
    q01 = np.asarray(state.get("q01"), dtype=np.float64)
    q99 = np.asarray(state.get("q99"), dtype=np.float64)
    _require(
        isinstance(constant_dimensions, list)
        and all(_is_integer(index) and index < 7 for index in constant_dimensions)
        and len(constant_dimensions) == len(set(constant_dimensions))
        and q01.shape == (7,)
        and q99.shape == (7,)
        and bool(np.isfinite(q01).all() and np.isfinite(q99).all())
        and bool((q01 <= q99).all()),
        "CALVIN normalization state bounds differ",
    )
    expected_algorithm = {
        "actions_re_normalized": False,
        "arrays_scanned": ["robot_obs", "rel_actions"],
        "canonical_ordinal_placement": "episode index ascending, then global timestep ascending",
        "frame_order": "archive data_offset ascending; values placed by canonical episode/global ordinal",
        "quantile": "numpy.quantile(method=linear)",
        "result_dtype": "float64",
        "selected_frame_exact_once_bitmap": True,
        "source_dtype": "float32",
        "state_gripper_excluded_from_percentiles": True,
    }
    _require(algorithm == expected_algorithm, "CALVIN normalization algorithm contract differs")
    expected_count_fields = {
        "tasks",
        "total_annotations",
        "total_episodes",
        "training_episodes",
        "training_frames",
        "validation_episodes",
        "validation_frames",
    }
    _exact_keys(counts, expected_count_fields, "CALVIN normalization counts")
    _require(
        all(_is_integer(counts[name], 1) for name in expected_count_fields),
        "CALVIN normalization counts are invalid",
    )
    split = _validate_split(stats.get("split"))
    _require(
        counts["training_episodes"] == len(split["train_episode_indices"])
        and counts["validation_episodes"] == len(split["validation_episode_indices"])
        and counts["total_episodes"] == counts["training_episodes"] + counts["validation_episodes"],
        "CALVIN normalization episode counts differ from its split",
    )
    return stats, split, hashlib.sha256(raw).hexdigest()


def authenticate_dev_inputs(
    training_root: Path,
    stats_path: Path,
    source_root: Path,
    revision_file: Path,
) -> AuthenticatedCalvinDevInputs:
    """Authenticate the current v4 generation without opening its archive."""

    root = require_full_training_root(training_root)
    manifest_path = root.parents[1] / MANIFEST_NAME
    manifest_raw = stable_regular_file_bytes(manifest_path)
    manifest = load_strict_json_bytes(manifest_raw, "CALVIN v4 dataset manifest")
    _validate_v4_manifest(manifest)
    descriptor, connection, index_identity = _open_authenticated_index(root, manifest)
    _close_authenticated_index(descriptor, connection, index_identity, root, manifest)
    metadata = _read_projected_metadata(root, manifest)
    projected_sha256 = _training_metadata_sha256(metadata)
    manifest_file_sha256 = hashlib.sha256(manifest_raw).hexdigest()
    stats, split, stats_file_sha256 = _validate_normalization(
        stats_path,
        manifest,
        projected_sha256,
        manifest_file_sha256,
    )
    _validate_metadata_counts(metadata, split, stats["counts"])
    source, source_files = _authenticate_source_with_bytes(source_root, revision_file)
    tool_source = development_tool_source_identity()
    storage_identity_sha256 = _storage_identity(manifest, manifest_file_sha256)["content_sha256"]
    index = manifest["storage"]["member_index"]
    identity = {
        "archive_bytes": manifest["archive"]["bytes"],
        "archive_sha256": manifest["archive"]["sha256"],
        "central_directory_sha256": manifest["archive"]["central_directory"]["sha256"],
        "dataset_manifest_file_sha256": manifest_file_sha256,
        "dataset_manifest_schema": manifest["schema"],
        "dataset_manifest_sha256": manifest["content_sha256"],
        "input_schema": DEV_INPUT_SCHEMA,
        "member_index_bytes": index["bytes"],
        "member_index_path": index["path"],
        "member_index_schema": index["schema"],
        "member_index_sha256": index["sha256"],
        "member_inventory_sha256": manifest["archive"]["member_inventory"]["sha256"],
        "metadata_files": list(stats["dataset"]["metadata_files"]),
        "metadata_sha256": stats["dataset"]["metadata_sha256"],
        "normalization_content_sha256": stats["content_sha256"],
        "normalization_file_sha256": stats_file_sha256,
        "projected_training_metadata_sha256": projected_sha256,
        "reader_schema": manifest["storage"]["reader_schema"],
        "split_sha256": canonical_sha256(split),
        "storage_mode": manifest["storage"]["mode"],
        "storage_identity_sha256": storage_identity_sha256,
        **source,
        **tool_source,
    }
    _exact_keys(identity, _INPUT_IDENTITY_KEYS, "CALVIN development input identity")
    return AuthenticatedCalvinDevInputs(
        identity=identity,
        member_index=dict(index),
        split=split,
        stats=stats,
        metadata=metadata,
        source_files=source_files,
        training_root=str(root),
    )


def _npy_from_bytes(raw: bytes, *, allow_pickle: bool, label: str) -> Any:
    try:
        return np.load(io.BytesIO(raw), allow_pickle=allow_pickle)
    except (OSError, ValueError, EOFError) as exc:
        raise CalvinDevStateError("cannot decode authenticated CALVIN metadata: %s" % label) from exc


def _validate_metadata_counts(
    metadata: Mapping[str, bytes],
    split: Mapping[str, Any],
    expected: Mapping[str, Any],
) -> None:
    episodes = np.asarray(
        _npy_from_bytes(
            metadata["training/ep_start_end_ids.npy"],
            allow_pickle=False,
            label="training/ep_start_end_ids.npy",
        )
    )
    _require(
        episodes.ndim == 2 and episodes.shape[1] == 2 and np.issubdtype(episodes.dtype, np.integer),
        "CALVIN metadata episode intervals are invalid",
    )
    try:
        annotations = _npy_from_bytes(
            metadata["training/lang_annotations/auto_lang_ann.npy"],
            allow_pickle=True,
            label="training/lang_annotations/auto_lang_ann.npy",
        ).item()
        intervals = annotations["info"]["indx"]
        tasks = annotations["language"]["task"]
    except (AttributeError, KeyError, TypeError, ValueError) as exc:
        raise CalvinDevStateError("CALVIN language annotation counts are invalid") from exc
    train = split["train_episode_indices"]
    validation = split["validation_episode_indices"]

    def frame_count(indices: Sequence[int]) -> int:
        return sum(int(episodes[index, 1]) - int(episodes[index, 0]) + 1 for index in indices)

    observed = {
        "tasks": len(set(tasks)),
        "total_annotations": len(intervals),
        "total_episodes": len(episodes),
        "training_episodes": len(train),
        "training_frames": frame_count(train),
        "validation_episodes": len(validation),
        "validation_frames": frame_count(validation),
    }
    _require(observed == dict(expected), "CALVIN normalization counts differ from authenticated metadata")


def _load_scene_intervals(raw: bytes) -> Dict[str, Tuple[int, int]]:
    loaded = _npy_from_bytes(raw, allow_pickle=True, label="training/scene_info.npy")
    try:
        payload = loaded.item()
    except (AttributeError, ValueError) as exc:
        raise CalvinDevStateError("CALVIN scene metadata is invalid") from exc
    _require(
        isinstance(payload, dict) and set(payload) == set(ABC_SCENES),
        "CALVIN development source must contain exactly scenes A/B/C",
    )
    result = {}  # type: Dict[str, Tuple[int, int]]
    for scene, interval in payload.items():
        _require(
            isinstance(interval, (list, tuple, np.ndarray))
            and len(interval) == 2
            and all(_is_integer(int(value)) and int(value) == value for value in interval),
            "CALVIN scene interval is invalid",
        )
        start, end = int(interval[0]), int(interval[1])
        _require(start <= end, "CALVIN scene interval is empty")
        result[scene] = (start, end)
    ordered = sorted(result.values())
    _require(all(left[1] < right[0] for left, right in zip(ordered, ordered[1:])), "CALVIN scene intervals overlap")
    return result


def _episode_scene(start: int, end: int, intervals: Mapping[str, Tuple[int, int]]) -> str:
    matches = [scene for scene, (lower, upper) in intervals.items() if lower <= start <= end <= upper]
    _require(len(matches) == 1, "CALVIN episode must belong to exactly one A/B/C scene")
    return matches[0]


def load_validation_candidates(
    authenticated_metadata: Mapping[str, bytes],
    split: Mapping[str, Any],
) -> Tuple[CalvinDevCandidate, ...]:
    """Derive held-out annotations only from already-authenticated metadata bytes."""

    _validate_split(split)
    required = {
        "training/ep_start_end_ids.npy",
        "training/scene_info.npy",
        "training/lang_annotations/auto_lang_ann.npy",
    }
    _require(required.issubset(set(authenticated_metadata)), "authenticated training metadata is incomplete")
    episodes_array = np.asarray(
        _npy_from_bytes(
            authenticated_metadata["training/ep_start_end_ids.npy"],
            allow_pickle=False,
            label="training/ep_start_end_ids.npy",
        )
    )
    _require(
        episodes_array.ndim == 2 and episodes_array.shape[1] == 2 and np.issubdtype(episodes_array.dtype, np.integer),
        "CALVIN episode intervals must have integer shape [episodes,2]",
    )
    scene_intervals = _load_scene_intervals(authenticated_metadata["training/scene_info.npy"])
    episodes = []  # type: List[Tuple[int, int, str]]
    previous_end = -1
    for start_value, end_value in episodes_array.tolist():
        start, end = int(start_value), int(end_value)
        _require(start <= end, "CALVIN episode interval is empty")
        _require(start > previous_end, "CALVIN episode intervals overlap or are unordered")
        episodes.append((start, end, _episode_scene(start, end, scene_intervals)))
        previous_end = end
    train = tuple(split["train_episode_indices"])
    validation = tuple(split["validation_episode_indices"])
    _require(
        set(train) | set(validation) == set(range(len(episodes))), "CALVIN split does not partition every whole episode"
    )
    annotations_array = _npy_from_bytes(
        authenticated_metadata["training/lang_annotations/auto_lang_ann.npy"],
        allow_pickle=True,
        label="training/lang_annotations/auto_lang_ann.npy",
    )
    try:
        annotations = annotations_array.item()
        intervals = annotations["info"]["indx"]
        instructions = annotations["language"]["ann"]
        tasks = annotations["language"]["task"]
    except (AttributeError, KeyError, TypeError, ValueError) as exc:
        raise CalvinDevStateError("CALVIN language annotations are invalid") from exc
    _require(len(intervals) == len(instructions) == len(tasks), "CALVIN annotation fields differ in length")
    all_candidates = []  # type: List[CalvinDevCandidate]
    for annotation_index, interval in enumerate(intervals):
        _require(len(interval) == 2, "CALVIN annotation interval is invalid")
        start, end_exclusive = int(interval[0]), int(interval[1])
        _require(start < end_exclusive, "CALVIN annotation interval is empty")
        matches = [
            episode_index
            for episode_index, (episode_start, episode_end, _scene) in enumerate(episodes)
            if episode_start <= start < end_exclusive <= episode_end + 1
        ]
        _require(len(matches) == 1, "CALVIN annotation crosses or falls outside a whole episode")
        episode_index = matches[0]
        instruction = instructions[annotation_index]
        task = tasks[annotation_index]
        _require(isinstance(instruction, str) and bool(instruction), "CALVIN annotation instruction is invalid")
        _require(isinstance(task, str) and bool(task), "CALVIN annotation task is invalid")
        all_candidates.append(
            CalvinDevCandidate(
                annotation_index=annotation_index,
                episode_index=episode_index,
                global_start=start,
                global_end_exclusive=end_exclusive,
                instruction=instruction,
                task=task,
                scene=episodes[episode_index][2],
            )
        )
    expected_validation = set()  # type: Set[int]
    for scene in ABC_SCENES:
        scene_episodes = [index for index, (_start, _end, name) in enumerate(episodes) if name == scene]
        _require(len(scene_episodes) >= 2, "CALVIN scene %s needs two episodes for a held-out split" % scene)
        ordered = sorted(
            scene_episodes,
            key=lambda index: (
                hashlib.sha256(("calvin:%s:%s:%s" % (split["seed"], scene, index)).encode()).digest(),
                index,
            ),
        )
        heldout_count = min(len(ordered) - 1, max(1, round(len(ordered) * split["validation_fraction"])))
        expected_validation.update(ordered[:heldout_count])
    _require(
        expected_validation == set(validation),
        "CALVIN held-out indices differ from the deterministic whole-episode recipe",
    )
    train_tasks = {candidate.task for candidate in all_candidates if candidate.episode_index in set(train)}
    validation_tasks = {candidate.task for candidate in all_candidates if candidate.episode_index in set(validation)}
    _require(
        train_tasks == validation_tasks and bool(train_tasks),
        "CALVIN held-out split does not preserve every task on both sides",
    )
    result = tuple(candidate for candidate in all_candidates if candidate.episode_index in set(validation))
    _require(bool(result), "CALVIN held-out A/B/C episodes contain no annotations")
    return result


def _canonical_reset_array(value: Any, shape: Tuple[int, ...], name: str) -> np.ndarray:
    source = np.asarray(value)
    _require(source.shape == shape, "%s must have shape %s" % (name, shape))
    _require(np.issubdtype(source.dtype, np.floating), "%s must have floating dtype" % name)
    _require(bool(np.isfinite(source).all()), "%s contains non-finite values" % name)
    return np.ascontiguousarray(source, dtype=RESET_DTYPE).copy(order="C")


def _canonical_action_matrix(value: Any) -> np.ndarray:
    source = np.asarray(value)
    _require(
        source.ndim == 2 and source.shape[0] > 0 and source.shape[1:] == ACTION_SHAPE,
        "replay actions must have shape [T,7]",
    )
    _require(np.issubdtype(source.dtype, np.floating), "replay actions must have floating dtype")
    result = np.ascontiguousarray(source, dtype=RESET_DTYPE).copy(order="C")
    _require(bool(np.isfinite(result).all()), "replay actions contain non-finite values")
    _require(bool(np.isin(result[:, 6], (-1.0, 1.0)).all()), "replay action gripper must be exactly {-1,+1}")
    _require(bool((np.abs(result[:, :6]) <= 1.0 + 1e-6).all()), "replay actions exceed official scaled [-1,1] units")
    return result


def candidate_rank_sha256(candidate: CalvinDevCandidate, base_seed: int) -> str:
    _require(_is_integer(base_seed), "candidate base seed must be nonnegative")
    return canonical_sha256(
        [
            CANDIDATE_RANK_DOMAIN,
            base_seed,
            candidate.scene,
            candidate.task,
            candidate.episode_index,
            candidate.annotation_index,
            candidate.global_start,
            candidate.global_end_exclusive,
            candidate.instruction,
        ]
    )


def rank_candidates(
    candidates: Sequence[CalvinDevCandidate],
    base_seed: int,
) -> Dict[Tuple[str, str], Tuple[CalvinDevCandidate, ...]]:
    grouped = {}  # type: Dict[Tuple[str, str], List[CalvinDevCandidate]]
    for candidate in candidates:
        _require(candidate.scene in ABC_SCENES, "candidate scene is outside A/B/C")
        grouped.setdefault((candidate.scene, candidate.task), []).append(candidate)
    return {
        key: tuple(sorted(values, key=lambda item: (candidate_rank_sha256(item, base_seed), item.annotation_index)))
        for key, values in grouped.items()
    }


def source_state_sha256(robot_obs: Any, scene_obs: Any) -> str:
    robot = _canonical_reset_array(robot_obs, ROBOT_SHAPE, "robot_obs")
    scene = _canonical_reset_array(scene_obs, SCENE_SHAPE, "scene_obs")
    digest = hashlib.sha256()
    digest.update(b"duo-vla-calvin-reset-state-v2\x00")
    digest.update(robot.tobytes(order="C"))
    digest.update(scene.tobytes(order="C"))
    return digest.hexdigest()


def _action_sequence_sha256(actions: Any) -> str:
    canonical = _canonical_action_matrix(actions)
    digest = hashlib.sha256()
    digest.update(b"duo-vla-calvin-replay-actions-v1\x00")
    digest.update(struct.pack(">Q", canonical.shape[0]))
    digest.update(canonical.tobytes(order="C"))
    return digest.hexdigest()


def reset_id_sha256(
    candidate: CalvinDevCandidate,
    frame: CalvinResetFrame,
    replay_bundle_record_sha256: str,
    replay_bundle_root_sha256: str,
) -> str:
    _require(_is_sha256(replay_bundle_record_sha256), "replay-bundle record hash is invalid")
    _require(_is_sha256(replay_bundle_root_sha256), "replay-bundle root hash is invalid")
    return canonical_sha256(
        {
            "annotation_index": candidate.annotation_index,
            "domain": RESET_ID_DOMAIN,
            "episode_index": candidate.episode_index,
            "global_start": candidate.global_start,
            "replay_bundle_record_sha256": replay_bundle_record_sha256,
            "replay_bundle_root_sha256": replay_bundle_root_sha256,
            "scene": candidate.scene,
            "source_state_sha256": source_state_sha256(frame.robot_obs, frame.scene_obs),
            "task": candidate.task,
        }
    )


def deterministic_npy_bytes(values: Any, expected_shape_tail: Tuple[int, ...]) -> bytes:
    source = np.asarray(values)
    _require(source.ndim == 2 and source.shape[0] > 0, "artifact must have a non-empty matrix shape")
    _require(source.shape[1:] == expected_shape_tail, "artifact shape tail changed")
    canonical = np.ascontiguousarray(source, dtype=RESET_DTYPE)
    _require(bool(np.isfinite(canonical).all()), "artifact contains non-finite values")
    output = io.BytesIO()
    np.lib.format.write_array(output, canonical, version=(1, 0), allow_pickle=False)
    return output.getvalue()


def _artifact_record(path: str, data: bytes, shape: Tuple[int, int]) -> Dict[str, Any]:
    return {
        "bytes": len(data),
        "dtype": RESET_DTYPE.str,
        "path": path,
        "sha256": hashlib.sha256(data).hexdigest(),
        "shape": list(shape),
    }


def _candidate_fields(candidate: CalvinDevCandidate) -> Dict[str, Any]:
    return {
        "annotation_end_exclusive": candidate.global_end_exclusive,
        "annotation_index": candidate.annotation_index,
        "episode_index": candidate.episode_index,
        "global_start": candidate.global_start,
        "instruction": candidate.instruction,
        "scene": candidate.scene,
        "task": candidate.task,
    }


def _validate_member_identity(value: Mapping[str, Any], expected_index: Optional[int] = None) -> Dict[str, Any]:
    _exact_keys(value, _MEMBER_IDENTITY_KEYS, "CALVIN replay member identity")
    path = value.get("path")
    global_index = value.get("global_index")
    _require(_is_integer(global_index), "CALVIN replay member global index is invalid")
    if expected_index is not None:
        _require(global_index == expected_index, "CALVIN replay member ordering differs")
    expected_path = "training/episode_%07d.npz" % global_index
    _require(path == expected_path, "CALVIN replay member path differs")
    _require(_is_integer(value.get("logical_bytes"), 1), "CALVIN replay member byte count is invalid")
    _require(_is_sha256(value.get("logical_sha256")), "CALVIN replay member logical SHA-256 is invalid")
    return dict(value)


def build_replay_bundle(
    replays: Sequence[BundledCalvinReplay],
    inputs: AuthenticatedCalvinDevInputs,
    source_identity: Mapping[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, bytes]]:
    """Build deterministic bytes for the Python-3.11 -> Python-3.8 boundary."""

    _exact_keys(inputs.identity, _INPUT_IDENTITY_KEYS, "CALVIN development input identity")
    _exact_keys(source_identity, _BUNDLE_SOURCE_KEYS, "CALVIN replay exporter source identity")
    expected_source = {
        "calvin_archive_source_sha256": inputs.identity["calvin_archive_source_sha256"],
        "dev_states_source_sha256": inputs.identity["dev_states_source_sha256"],
        "replay_exporter_source_sha256": inputs.identity["replay_exporter_source_sha256"],
        "schema": "duo-vla-calvin-dev-replay-export-source-v1",
    }
    _require(dict(source_identity) == expected_source, "CALVIN replay exporter source identity differs")
    expected_candidates = load_validation_candidates(inputs.metadata, inputs.split)
    materialized = list(replays)
    _require(len(materialized) == len(expected_candidates), "replay bundle omitted a held-out candidate")
    robots = []  # type: List[np.ndarray]
    scenes = []  # type: List[np.ndarray]
    action_chunks = []  # type: List[np.ndarray]
    records = []  # type: List[Dict[str, Any]]
    action_offset = 0
    for replay, expected_candidate in zip(materialized, expected_candidates):
        _require(
            replay.candidate == expected_candidate, "replay bundle candidate order differs from authenticated metadata"
        )
        candidate = replay.candidate
        actions = _canonical_action_matrix(replay.actions)
        _require(
            actions.shape[0] == candidate.global_end_exclusive - candidate.global_start,
            "replay action count differs from annotation interval",
        )
        members = [
            _validate_member_identity(value, expected_index=global_index)
            for value, global_index in zip(
                replay.member_identities,
                range(candidate.global_start, candidate.global_end_exclusive),
            )
        ]
        _require(len(members) == actions.shape[0], "replay bundle omitted or added a member identity")
        frame = CalvinResetFrame(
            robot_obs=_canonical_reset_array(replay.frame.robot_obs, ROBOT_SHAPE, "robot_obs"),
            scene_obs=_canonical_reset_array(replay.frame.scene_obs, SCENE_SHAPE, "scene_obs"),
            source_frame_sha256=replay.frame.source_frame_sha256,
        )
        _require(frame.robot_obs[14] in (-1.0, 1.0), "CALVIN reset gripper state must be exactly {-1,+1}")
        _require(
            frame.source_frame_sha256 == members[0]["logical_sha256"],
            "reset frame SHA-256 differs from its source member",
        )
        record = {
            **_candidate_fields(candidate),
            "action_count": actions.shape[0],
            "action_offset": action_offset,
            "action_sequence_sha256": _action_sequence_sha256(actions),
            "member_identities": members,
            "source_frame_sha256": frame.source_frame_sha256,
            "source_state_sha256": source_state_sha256(frame.robot_obs, frame.scene_obs),
        }
        record["record_sha256"] = canonical_sha256(record)
        records.append(record)
        robots.append(frame.robot_obs)
        scenes.append(frame.scene_obs)
        action_chunks.append(actions)
        action_offset += actions.shape[0]
    robot_matrix = np.stack(robots)
    scene_matrix = np.stack(scenes)
    action_matrix = np.concatenate(action_chunks, axis=0)
    artifacts = {
        ROBOT_ARTIFACT: deterministic_npy_bytes(robot_matrix, ROBOT_SHAPE),
        SCENE_ARTIFACT: deterministic_npy_bytes(scene_matrix, SCENE_SHAPE),
        ACTION_ARTIFACT: deterministic_npy_bytes(action_matrix, ACTION_SHAPE),
    }
    manifest = {
        "artifacts": {
            "rel_actions": _artifact_record(ACTION_ARTIFACT, artifacts[ACTION_ARTIFACT], action_matrix.shape),
            "robot_obs": _artifact_record(ROBOT_ARTIFACT, artifacts[ROBOT_ARTIFACT], robot_matrix.shape),
            "scene_obs": _artifact_record(SCENE_ARTIFACT, artifacts[SCENE_ARTIFACT], scene_matrix.shape),
        },
        "identity": dict(inputs.identity),
        "records": records,
        "schema": REPLAY_BUNDLE_SCHEMA,
        "source": dict(source_identity),
    }
    manifest["root_sha256"] = _manifest_root_sha256(manifest)
    validate_replay_bundle(manifest, artifacts, inputs, validate_member_index=False)
    return manifest, artifacts


def _decode_artifact(data: bytes, label: str) -> np.ndarray:
    try:
        result = np.load(io.BytesIO(data), allow_pickle=False)
    except (OSError, ValueError, EOFError) as exc:
        raise CalvinDevStateError("cannot decode CALVIN %s artifact" % label) from exc
    _require(isinstance(result, np.ndarray), "CALVIN %s artifact is not an array" % label)
    return result


def _verify_bundle_members_against_index(
    inputs: AuthenticatedCalvinDevInputs,
    records: Sequence[Mapping[str, Any]],
) -> None:
    _require(bool(inputs.training_root), "authenticated CALVIN input has no training-root binding")
    root = require_full_training_root(Path(inputs.training_root))
    manifest = load_strict_json(root.parents[1] / MANIFEST_NAME)
    _validate_v4_manifest(manifest)
    _require(
        manifest["storage"]["member_index"] == inputs.member_index,
        "current member index differs from authenticated inputs",
    )
    descriptor, connection, identity = _open_authenticated_index(root, manifest)
    try:
        observed = {}  # type: Dict[str, Tuple[int, str, int, str]]
        for record in records:
            for member in record["member_identities"]:
                path = member["path"]
                if path in observed:
                    expected = observed[path]
                else:
                    row = connection.execute(
                        "SELECT logical_bytes,logical_sha256,kind,split,global_index FROM members WHERE path=?",
                        (path,),
                    ).fetchone()
                    _require(row is not None and len(row) == 5, "CALVIN v2 index omitted replay member: %s" % path)
                    logical_bytes, logical_sha256, kind, split, global_index = row
                    _require(
                        isinstance(logical_sha256, bytes) and len(logical_sha256) == 32,
                        "CALVIN v2 logical SHA-256 is invalid",
                    )
                    expected = (logical_bytes, logical_sha256.hex(), global_index, split)
                    _require(kind == 0 and split == "training", "CALVIN replay member is not a training file")
                    observed[path] = expected
                _require(
                    expected == (member["logical_bytes"], member["logical_sha256"], member["global_index"], "training"),
                    "CALVIN replay member identity differs from the current v2 index: %s" % path,
                )
    finally:
        _close_authenticated_index(descriptor, connection, identity, root, manifest)


def validate_replay_bundle(
    manifest: Mapping[str, Any],
    artifacts: Mapping[str, bytes],
    inputs: AuthenticatedCalvinDevInputs,
    *,
    validate_member_index: bool = True,
) -> Tuple[BundledCalvinReplay, ...]:
    _exact_keys(manifest, _BUNDLE_TOP_LEVEL_KEYS, "CALVIN replay bundle")
    _require(manifest.get("schema") == REPLAY_BUNDLE_SCHEMA, "CALVIN replay-bundle schema differs")
    _require(_is_sha256(manifest.get("root_sha256")), "CALVIN replay-bundle root hash is invalid")
    _require(_manifest_root_sha256(manifest) == manifest["root_sha256"], "CALVIN replay-bundle root hash differs")
    _require(manifest.get("identity") == inputs.identity, "CALVIN replay-bundle input identity differs")
    current_tool_source = development_tool_source_identity()
    for name, digest in current_tool_source.items():
        _require(
            inputs.identity.get(name) == digest,
            "CALVIN development source changed after input authentication: %s" % name,
        )
    source = manifest.get("source")
    _exact_keys(source, _BUNDLE_SOURCE_KEYS, "CALVIN replay exporter source identity")
    _require(
        source
        == {
            "calvin_archive_source_sha256": inputs.identity["calvin_archive_source_sha256"],
            "dev_states_source_sha256": inputs.identity["dev_states_source_sha256"],
            "replay_exporter_source_sha256": inputs.identity["replay_exporter_source_sha256"],
            "schema": "duo-vla-calvin-dev-replay-export-source-v1",
        },
        "CALVIN replay exporter source identity differs",
    )
    artifact_manifest = manifest.get("artifacts")
    records = manifest.get("records")
    _require(
        isinstance(artifact_manifest, dict) and isinstance(records, list) and bool(records),
        "CALVIN replay-bundle sections are invalid",
    )
    _require(
        set(artifact_manifest) == {"rel_actions", "robot_obs", "scene_obs"}, "CALVIN replay-bundle artifacts differ"
    )
    _require(
        set(artifacts) == {ACTION_ARTIFACT, ROBOT_ARTIFACT, SCENE_ARTIFACT},
        "CALVIN replay-bundle artifact files differ",
    )
    by_name = {"rel_actions": ACTION_ARTIFACT, "robot_obs": ROBOT_ARTIFACT, "scene_obs": SCENE_ARTIFACT}
    decoded = {}  # type: Dict[str, np.ndarray]
    for name, path in by_name.items():
        record = artifact_manifest[name]
        _exact_keys(record, _ARTIFACT_KEYS, "CALVIN replay artifact")
        _require(
            record.get("path") == path and record.get("dtype") == RESET_DTYPE.str,
            "CALVIN replay artifact contract differs",
        )
        raw = artifacts[path]
        _require(record.get("bytes") == len(raw), "CALVIN replay artifact byte count differs")
        _require(record.get("sha256") == hashlib.sha256(raw).hexdigest(), "CALVIN replay artifact hash differs")
        array = _decode_artifact(raw, name)
        _require(
            array.dtype.str == RESET_DTYPE.str and list(array.shape) == record.get("shape"),
            "CALVIN replay artifact dtype/shape differs",
        )
        _require(array.ndim == 2 and bool(np.isfinite(array).all()), "CALVIN replay artifact contains invalid values")
        decoded[name] = array
    expected_candidates = load_validation_candidates(inputs.metadata, inputs.split)
    _require(len(records) == len(expected_candidates), "CALVIN replay bundle omitted or added held-out candidates")
    _require(decoded["robot_obs"].shape == (len(records), 15), "CALVIN replay robot artifact shape differs")
    _require(decoded["scene_obs"].shape == (len(records), 24), "CALVIN replay scene artifact shape differs")
    replay_values = []  # type: List[BundledCalvinReplay]
    next_offset = 0
    for index, (record, candidate) in enumerate(zip(records, expected_candidates)):
        _exact_keys(record, _BUNDLE_RECORD_KEYS, "CALVIN replay-bundle record")
        for name, expected in _candidate_fields(candidate).items():
            _require(record.get(name) == expected, "CALVIN replay candidate metadata/order differs: %s" % name)
        count = candidate.global_end_exclusive - candidate.global_start
        _require(
            record.get("action_offset") == next_offset and record.get("action_count") == count,
            "CALVIN replay action offsets/counts differ",
        )
        _require(
            next_offset + count <= decoded["rel_actions"].shape[0], "CALVIN replay action range exceeds its artifact"
        )
        actions = _canonical_action_matrix(decoded["rel_actions"][next_offset : next_offset + count])
        _require(
            record.get("action_sequence_sha256") == _action_sequence_sha256(actions),
            "CALVIN replay action sequence hash differs",
        )
        members = record.get("member_identities")
        _require(isinstance(members, list) and len(members) == count, "CALVIN replay member inventory is incomplete")
        canonical_members = tuple(
            _validate_member_identity(member, expected_index=global_index)
            for member, global_index in zip(members, range(candidate.global_start, candidate.global_end_exclusive))
        )
        frame = CalvinResetFrame(
            robot_obs=_canonical_reset_array(decoded["robot_obs"][index], ROBOT_SHAPE, "robot_obs"),
            scene_obs=_canonical_reset_array(decoded["scene_obs"][index], SCENE_SHAPE, "scene_obs"),
            source_frame_sha256=record["source_frame_sha256"],
        )
        _require(frame.robot_obs[14] in (-1.0, 1.0), "CALVIN replay reset gripper is invalid")
        _require(
            frame.source_frame_sha256 == canonical_members[0]["logical_sha256"],
            "CALVIN replay start-member hash differs",
        )
        _require(
            record.get("source_state_sha256") == source_state_sha256(frame.robot_obs, frame.scene_obs),
            "CALVIN replay source-state hash differs",
        )
        detached = dict(record)
        observed_record_sha256 = detached.pop("record_sha256", None)
        _require(
            _is_sha256(observed_record_sha256) and canonical_sha256(detached) == observed_record_sha256,
            "CALVIN replay record hash differs",
        )
        replay_values.append(
            BundledCalvinReplay(
                candidate=candidate,
                frame=frame,
                actions=actions,
                member_identities=canonical_members,
                record_sha256=observed_record_sha256,
            )
        )
        next_offset += count
    _require(next_offset == decoded["rel_actions"].shape[0], "CALVIN replay action artifact has unreferenced rows")
    if validate_member_index:
        _verify_bundle_members_against_index(inputs, records)
    return tuple(replay_values)


def _directory_artifact_bytes(root: Path, manifest: Mapping[str, Any], names: Mapping[str, str]) -> Dict[str, bytes]:
    expected_files = {"manifest.json"} | set(names.values())
    _require(root.is_dir() and not root.is_symlink(), "CALVIN artifact directory is invalid: %s" % root)
    observed = {item.name for item in root.iterdir()}
    _require(observed == expected_files, "CALVIN artifact directory inventory differs")
    artifacts = {}  # type: Dict[str, bytes]
    for path in names.values():
        artifacts[path] = stable_regular_file_bytes(root / path)
    return artifacts


def load_replay_bundle(
    bundle_dir: Path,
    inputs: AuthenticatedCalvinDevInputs,
) -> Tuple[Dict[str, Any], Tuple[BundledCalvinReplay, ...]]:
    root = Path(bundle_dir).resolve(strict=True)
    manifest = load_strict_json(root / "manifest.json")
    artifacts = _directory_artifact_bytes(
        root,
        manifest,
        {"rel_actions": ACTION_ARTIFACT, "robot_obs": ROBOT_ARTIFACT, "scene_obs": SCENE_ARTIFACT},
    )
    replays = validate_replay_bundle(manifest, artifacts, inputs, validate_member_index=True)
    return manifest, replays


def build_bank(
    resets: Sequence[MaterializedCalvinReset],
    identity: Mapping[str, Any],
    split: Mapping[str, Any],
    replay_bundle_manifest: Mapping[str, Any],
    base_seed: int,
    *,
    smoke_tasks_per_scene: int = 4,
    rejections: Sequence[Mapping[str, Any]] = (),
) -> Tuple[Dict[str, Any], Dict[str, bytes]]:
    _exact_keys(identity, _INPUT_IDENTITY_KEYS, "CALVIN development identity")
    _validate_split(split)
    _require(replay_bundle_manifest.get("schema") == REPLAY_BUNDLE_SCHEMA, "reset bank requires a replay bundle")
    bundle_root = replay_bundle_manifest.get("root_sha256")
    _require(_is_sha256(bundle_root), "reset bank replay-bundle root hash is invalid")
    materialized = sorted(
        list(resets),
        key=lambda item: (
            item.candidate.scene,
            item.candidate.task,
            item.candidate_rank_sha256,
            item.candidate.annotation_index,
        ),
    )
    _require(bool(materialized), "cannot build an empty CALVIN reset bank")
    pairs = [(item.candidate.scene, item.candidate.task) for item in materialized]
    _require(len(pairs) == len(set(pairs)), "bank must contain at most one reset per scene/task")
    robots = []  # type: List[np.ndarray]
    scenes = []  # type: List[np.ndarray]
    records = []  # type: List[Dict[str, Any]]
    reset_ids = set()  # type: Set[str]
    validation_indices = set(split["validation_episode_indices"])
    bundle_records = {record["record_sha256"] for record in replay_bundle_manifest["records"]}
    for item in materialized:
        candidate = item.candidate
        _require(candidate.episode_index in validation_indices, "bank record is not from a held-out episode")
        _require(_is_sha256(item.candidate_rank_sha256), "candidate rank hash is invalid")
        _require(
            item.replay_bundle_record_sha256 in bundle_records, "bank reset is not bound to a replay-bundle record"
        )
        frame = CalvinResetFrame(
            robot_obs=_canonical_reset_array(item.frame.robot_obs, ROBOT_SHAPE, "robot_obs"),
            scene_obs=_canonical_reset_array(item.frame.scene_obs, SCENE_SHAPE, "scene_obs"),
            source_frame_sha256=item.frame.source_frame_sha256,
        )
        reset_id = reset_id_sha256(candidate, frame, item.replay_bundle_record_sha256, bundle_root)
        _require(reset_id not in reset_ids, "bank contains duplicate reset identities")
        reset_ids.add(reset_id)
        records.append(
            {
                **_candidate_fields(candidate),
                "candidate_rank_sha256": item.candidate_rank_sha256,
                "replay_actions": item.replay_actions,
                "replay_bundle_record_sha256": item.replay_bundle_record_sha256,
                "reset_id_sha256": reset_id,
                "source_frame_sha256": frame.source_frame_sha256,
                "source_state_sha256": source_state_sha256(frame.robot_obs, frame.scene_obs),
            }
        )
        robots.append(frame.robot_obs)
        scenes.append(frame.scene_obs)
    _require({record["scene"] for record in records} == set(ABC_SCENES), "CALVIN reset bank does not cover A/B/C")
    _require(_is_integer(smoke_tasks_per_scene, 1), "smoke tasks per scene must be positive")
    smoke_indices = []  # type: List[int]
    for scene in ABC_SCENES:
        candidates_for_scene = [(index, record) for index, record in enumerate(records) if record["scene"] == scene]
        _require(
            len(candidates_for_scene) >= smoke_tasks_per_scene, "CALVIN smoke view has too few tasks in %s" % scene
        )
        ordered = sorted(
            candidates_for_scene,
            key=lambda pair: (
                canonical_sha256([SMOKE_SELECTION_DOMAIN, base_seed, scene, pair[1]["task"]]),
                pair[1]["reset_id_sha256"],
            ),
        )
        smoke_indices.extend(index for index, _record in ordered[:smoke_tasks_per_scene])
    rejection_values = []  # type: List[Dict[str, Any]]
    for source in rejections:
        record = dict(source)
        _exact_keys(record, _REJECTION_KEYS, "CALVIN reset rejection")
        _require(
            record["scene"] in ABC_SCENES and _is_sha256(record["replay_bundle_record_sha256"]),
            "CALVIN reset rejection identity is invalid",
        )
        rejection_values.append(record)
    robot_matrix = np.stack(robots)
    scene_matrix = np.stack(scenes)
    artifacts = {
        ROBOT_ARTIFACT: deterministic_npy_bytes(robot_matrix, ROBOT_SHAPE),
        SCENE_ARTIFACT: deterministic_npy_bytes(scene_matrix, SCENE_SHAPE),
    }
    manifest = {
        "artifacts": {
            "robot_obs": _artifact_record(ROBOT_ARTIFACT, artifacts[ROBOT_ARTIFACT], robot_matrix.shape),
            "scene_obs": _artifact_record(SCENE_ARTIFACT, artifacts[SCENE_ARTIFACT], scene_matrix.shape),
        },
        "identity": dict(identity),
        "records": records,
        "rejections": rejection_values,
        "replay_bundle": {
            "records_sha256": canonical_sha256(replay_bundle_manifest["records"]),
            "root_sha256": bundle_root,
            "schema": REPLAY_BUNDLE_SCHEMA,
        },
        "schema": DEV_BANK_SCHEMA,
        "selection": {
            "algorithm": (
                "stable-sha256 four-distinct-tasks-per-scene smoke view over one replay-valid reset per scene/task"
            ),
            "base_seed": base_seed,
            "smoke_reset_indices": smoke_indices,
            "smoke_tasks_per_scene": smoke_tasks_per_scene,
        },
        "split": dict(split),
    }
    manifest["root_sha256"] = _manifest_root_sha256(manifest)
    validate_bank(manifest, artifacts)
    return manifest, artifacts


def validate_bank(manifest: Mapping[str, Any], artifacts: Mapping[str, bytes]) -> Tuple[np.ndarray, np.ndarray]:
    _exact_keys(manifest, _BANK_TOP_LEVEL_KEYS, "CALVIN development bank")
    _require(manifest.get("schema") == DEV_BANK_SCHEMA, "CALVIN development bank schema differs")
    _require(
        _is_sha256(manifest.get("root_sha256")) and _manifest_root_sha256(manifest) == manifest["root_sha256"],
        "CALVIN development bank root hash differs",
    )
    _exact_keys(manifest["identity"], _INPUT_IDENTITY_KEYS, "CALVIN development identity")
    split = _validate_split(manifest["split"])
    replay_bundle = manifest.get("replay_bundle")
    _exact_keys(replay_bundle, {"records_sha256", "root_sha256", "schema"}, "CALVIN bank replay-bundle identity")
    _require(
        replay_bundle.get("schema") == REPLAY_BUNDLE_SCHEMA
        and _is_sha256(replay_bundle.get("root_sha256"))
        and _is_sha256(replay_bundle.get("records_sha256")),
        "CALVIN bank replay-bundle hashes are invalid",
    )
    artifact_manifest = manifest["artifacts"]
    _require(set(artifact_manifest) == {"robot_obs", "scene_obs"}, "CALVIN reset-bank artifacts differ")
    _require(set(artifacts) == {ROBOT_ARTIFACT, SCENE_ARTIFACT}, "CALVIN reset-bank artifact files differ")
    decoded = {}  # type: Dict[str, np.ndarray]
    for name, path, width in (("robot_obs", ROBOT_ARTIFACT, 15), ("scene_obs", SCENE_ARTIFACT, 24)):
        record = artifact_manifest[name]
        _exact_keys(record, _ARTIFACT_KEYS, "CALVIN reset artifact")
        raw = artifacts[path]
        _require(
            record.get("path") == path and record.get("bytes") == len(raw), "CALVIN reset artifact identity differs"
        )
        _require(record.get("sha256") == hashlib.sha256(raw).hexdigest(), "CALVIN reset artifact hash differs")
        array = _decode_artifact(raw, name)
        _require(
            array.dtype.str == RESET_DTYPE.str and array.shape == (len(manifest["records"]), width),
            "CALVIN reset artifact shape/dtype differs",
        )
        _require(
            list(array.shape) == record.get("shape") and bool(np.isfinite(array).all()),
            "CALVIN reset artifact values differ",
        )
        decoded[name] = array
    records = manifest["records"]
    _require(isinstance(records, list) and bool(records), "CALVIN reset bank has no records")
    validation_indices = set(split["validation_episode_indices"])
    seen_ids = set()  # type: Set[str]
    seen_pairs = set()  # type: Set[Tuple[str, str]]
    for index, record in enumerate(records):
        _exact_keys(record, _BANK_RECORD_KEYS, "CALVIN reset record")
        _require(
            record["scene"] in ABC_SCENES and record["episode_index"] in validation_indices,
            "CALVIN reset record is not held out A/B/C",
        )
        for name in (
            "candidate_rank_sha256",
            "replay_bundle_record_sha256",
            "reset_id_sha256",
            "source_frame_sha256",
            "source_state_sha256",
        ):
            _require(_is_sha256(record[name]), "CALVIN reset record has invalid %s" % name)
        candidate = CalvinDevCandidate(
            annotation_index=record["annotation_index"],
            episode_index=record["episode_index"],
            global_start=record["global_start"],
            global_end_exclusive=record["annotation_end_exclusive"],
            instruction=record["instruction"],
            task=record["task"],
            scene=record["scene"],
        )
        frame = CalvinResetFrame(
            robot_obs=decoded["robot_obs"][index],
            scene_obs=decoded["scene_obs"][index],
            source_frame_sha256=record["source_frame_sha256"],
        )
        _require(
            record["source_state_sha256"] == source_state_sha256(frame.robot_obs, frame.scene_obs),
            "CALVIN reset source-state hash differs",
        )
        expected_reset = reset_id_sha256(
            candidate, frame, record["replay_bundle_record_sha256"], replay_bundle["root_sha256"]
        )
        _require(
            record["reset_id_sha256"] == expected_reset and expected_reset not in seen_ids,
            "CALVIN reset identity differs or repeats",
        )
        seen_ids.add(expected_reset)
        pair = (record["scene"], record["task"])
        _require(pair not in seen_pairs, "CALVIN bank has multiple resets for a scene/task")
        seen_pairs.add(pair)
    _require({record["scene"] for record in records} == set(ABC_SCENES), "CALVIN reset bank does not cover A/B/C")
    rejections = manifest["rejections"]
    _require(isinstance(rejections, list), "CALVIN reset rejection log is invalid")
    for rejection in rejections:
        _exact_keys(rejection, _REJECTION_KEYS, "CALVIN reset rejection")
        _require(
            rejection["scene"] in ABC_SCENES and _is_sha256(rejection["replay_bundle_record_sha256"]),
            "CALVIN reset rejection differs",
        )
    selection = manifest["selection"]
    _exact_keys(selection, _SELECTION_KEYS, "CALVIN reset selection")
    smoke = selection["smoke_reset_indices"]
    _require(
        isinstance(smoke, list)
        and len(smoke) == len(set(smoke))
        and all(_is_integer(item) and item < len(records) for item in smoke),
        "CALVIN smoke reset indices differ",
    )
    for scene in ABC_SCENES:
        selected = [records[index] for index in smoke if records[index]["scene"] == scene]
        _require(len(selected) == selection["smoke_tasks_per_scene"], "CALVIN smoke scene balance differs")
        _require(len({record["task"] for record in selected}) == len(selected), "CALVIN smoke tasks are not distinct")
    return decoded["robot_obs"], decoded["scene_obs"]


def _rename_noreplace(source: Path, destination: Path) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise CalvinDevStateError("exclusive artifact publication requires renameat2")
    renameat2.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    renameat2.restype = ctypes.c_int
    result = renameat2(
        _AT_FDCWD,
        os.fsencode(str(source)),
        _AT_FDCWD,
        os.fsencode(str(destination)),
        _RENAME_NOREPLACE,
    )
    if result != 0:
        error = ctypes.get_errno()
        if error == errno.EEXIST:
            raise FileExistsError(error, os.strerror(error), str(destination))
        raise OSError(error, os.strerror(error), str(destination))


def _write_directory_exclusive(output_dir: Path, manifest: Mapping[str, Any], artifacts: Mapping[str, bytes]) -> None:
    output = Path(output_dir).absolute()
    parent = output.parent
    _require(parent.is_dir() and not parent.is_symlink(), "artifact output parent must be a real existing directory")
    _require(not output.exists() and not output.is_symlink(), "refusing to overwrite an existing CALVIN artifact")
    stage = parent / (".%s.stage-%s" % (output.name, uuid.uuid4().hex))
    stage.mkdir(mode=0o700)
    published = False
    try:
        for name, raw in sorted(artifacts.items()):
            path = stage / name
            with path.open("xb") as handle:
                handle.write(raw)
                handle.flush()
                os.fsync(handle.fileno())
        with (stage / "manifest.json").open("xb") as handle:
            handle.write(canonical_json_bytes(dict(manifest), pretty=True))
            handle.flush()
            os.fsync(handle.fileno())
        directory_descriptor = os.open(str(stage), _DIRECTORY_OPEN_FLAGS)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
        _rename_noreplace(stage, output)
        published = True
        parent_descriptor = os.open(str(parent), _DIRECTORY_OPEN_FLAGS)
        try:
            os.fsync(parent_descriptor)
        finally:
            os.close(parent_descriptor)
    finally:
        if not published and stage.exists():
            shutil.rmtree(str(stage))


def write_replay_bundle_exclusive(
    output_dir: Path,
    manifest: Mapping[str, Any],
    artifacts: Mapping[str, bytes],
    inputs: AuthenticatedCalvinDevInputs,
) -> None:
    validate_replay_bundle(manifest, artifacts, inputs, validate_member_index=True)
    _write_directory_exclusive(output_dir, manifest, artifacts)


def write_bank_exclusive(output_dir: Path, manifest: Mapping[str, Any], artifacts: Mapping[str, bytes]) -> None:
    validate_bank(manifest, artifacts)
    _write_directory_exclusive(output_dir, manifest, artifacts)


def load_bank(bank_dir: Path) -> Tuple[Dict[str, Any], np.ndarray, np.ndarray]:
    root = Path(bank_dir).resolve(strict=True)
    manifest = load_strict_json(root / "manifest.json")
    artifacts = _directory_artifact_bytes(root, manifest, {"robot_obs": ROBOT_ARTIFACT, "scene_obs": SCENE_ARTIFACT})
    robot_obs, scene_obs = validate_bank(manifest, artifacts)
    return manifest, robot_obs, scene_obs


def assert_bank_matches_inputs(manifest: Mapping[str, Any], inputs: AuthenticatedCalvinDevInputs) -> None:
    _require(manifest.get("identity") == inputs.identity, "CALVIN reset bank input identity drifted")
    _require(manifest.get("split") == inputs.split, "CALVIN reset bank split drifted")


__all__ = [
    "ABC_SCENES",
    "ACTION_ARTIFACT",
    "ACTION_SHAPE",
    "ARCHIVE_BYTES",
    "ARCHIVE_READER_SCHEMA",
    "ARCHIVE_SHA256",
    "ARCHIVE_URL",
    "CALVIN_ENV_REVISION",
    "CALVIN_REVISION",
    "CALVIN_TACTO_REVISION",
    "CHECKSUM_URL",
    "DATASET_CRITICAL_FILES",
    "DATASET_MANIFEST_SCHEMA",
    "DEV_BANK_SCHEMA",
    "DEV_INPUT_SCHEMA",
    "MANIFEST_NAME",
    "MEMBER_INDEX_NAME",
    "MEMBER_INDEX_SCHEMA",
    "NORMALIZATION_SCHEMA",
    "REPLAY_BUNDLE_SCHEMA",
    "ROBOT_ARTIFACT",
    "SCENE_ARTIFACT",
    "TRAIN_CRITICAL_FILES",
    "AuthenticatedCalvinDevInputs",
    "BundledCalvinReplay",
    "CalvinDevCandidate",
    "CalvinDevStateError",
    "CalvinResetFrame",
    "MaterializedCalvinReset",
    "assert_bank_matches_inputs",
    "authenticate_dev_inputs",
    "authenticate_source",
    "build_bank",
    "build_replay_bundle",
    "candidate_rank_sha256",
    "canonical_json_bytes",
    "canonical_sha256",
    "development_tool_source_identity",
    "file_sha256",
    "load_bank",
    "load_replay_bundle",
    "load_strict_json",
    "load_validation_candidates",
    "rank_candidates",
    "require_full_training_root",
    "reset_id_sha256",
    "source_state_sha256",
    "stable_regular_file_bytes",
    "validate_bank",
    "validate_replay_bundle",
    "write_bank_exclusive",
    "write_replay_bundle_exclusive",
]
