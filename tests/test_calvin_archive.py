from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import stat
import struct
import subprocess
import warnings
import zipfile
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import pytest

import duo_vla.data.calvin_archive as archive_module
from duo_vla.data.calvin_archive import (
    CALVIN_ARCHIVE_BYTES,
    CALVIN_ARCHIVE_NAME,
    CALVIN_ARCHIVE_SHA256,
    CALVIN_ARCHIVE_URL,
    CALVIN_CHECKSUM_URL,
    CALVIN_CRITICAL_FILES,
    CALVIN_DATASET_NAME,
    CALVIN_INDEX_NAME,
    CALVIN_MANIFEST_NAME,
    CALVIN_MANIFEST_SCHEMA,
    CALVIN_MEMBER_INDEX_SCHEMA,
    CalvinArchivePublicationError,
    CalvinArchiveReader,
    CalvinArchiveValidationError,
    PinnedRegularFile,
    load_calvin_archive_manifest,
    prepare_calvin_archive,
)

ROOT = Path(__file__).resolve().parents[1]
PREPARE_SCRIPT = ROOT / "scripts" / "calvin" / "prepare_archive_direct.py"
FIXTURE_ARCHIVE_URL = "fixture://task_ABC_D.zip"
FIXTURE_CHECKSUM_URL = "fixture://sha256sum.txt"

_EOCD = struct.Struct("<IHHHHIIH")
_ZIP64_EOCD = struct.Struct("<IQHHIIQQQQ")
_ZIP64_LOCATOR = struct.Struct("<IIQI")
_CENTRAL = struct.Struct("<I6H3I5H2I")


@dataclass(frozen=True)
class FixtureGeneration:
    data_root: Path
    archive: Path
    archive_bytes: int
    archive_sha256: str
    frame_payloads: dict[str, bytes]


def _zip_info(
    name: str,
    *,
    kind: str = "file",
    method: int | None = None,
    extra: bytes = b"",
) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name)
    info.create_system = 3
    info.extra = extra
    if kind == "directory":
        info.external_attr = ((stat.S_IFDIR | 0o755) << 16) | 0x10
        info.compress_type = zipfile.ZIP_STORED
    elif kind == "symlink":
        info.external_attr = (stat.S_IFLNK | 0o777) << 16
        info.compress_type = zipfile.ZIP_DEFLATED
    elif kind == "fifo":
        info.external_attr = (stat.S_IFIFO | 0o600) << 16
        info.compress_type = zipfile.ZIP_DEFLATED
    else:
        info.external_attr = (stat.S_IFREG | 0o644) << 16
        info.compress_type = zipfile.ZIP_DEFLATED if method is None else method
    return info


def _base_entries(*, include_root_directory: bool = True) -> list[tuple[zipfile.ZipInfo, bytes]]:
    entries = [
        (_zip_info("task_ABC_D/training/", kind="directory"), b""),
        (_zip_info("task_ABC_D/training/lang_annotations/", kind="directory"), b""),
        (_zip_info("task_ABC_D/training/.hydra/", kind="directory"), b""),
        (_zip_info("task_ABC_D/validation/", kind="directory"), b""),
        (_zip_info("task_ABC_D/validation/.hydra/", kind="directory"), b""),
    ]
    if include_root_directory:
        entries.insert(0, (_zip_info("task_ABC_D/", kind="directory"), b""))
    entries.extend(
        (
            _zip_info(f"task_ABC_D/{relative}"),
            f"authenticated fixture:{relative}\n".encode(),
        )
        for relative in CALVIN_CRITICAL_FILES
    )
    return entries


def _write_fixture_archive(
    path: Path,
    *,
    extra_entries: Iterable[tuple[zipfile.ZipInfo, bytes]] = (),
    include_frames: bool = True,
    force_zip64_frame: bool = False,
    include_root_directory: bool = True,
) -> dict[str, bytes]:
    frame_payloads = {
        "training/episode_0000002.npz": b"npz-frame-two" * 19,
        "training/episode_0000000.npz": b"npz-frame-zero" * 23,
        "validation/episode_0000001.npz": b"npz-frame-validation" * 17,
    }
    entries = _base_entries(include_root_directory=include_root_directory)
    entries.extend(extra_entries)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        with zipfile.ZipFile(path, "w") as archive:
            for info, payload in entries:
                archive.writestr(info, payload)
            if include_frames:
                for relative, payload in frame_payloads.items():
                    info = _zip_info(f"task_ABC_D/{relative}")
                    if force_zip64_frame and relative == "training/episode_0000002.npz":
                        with archive.open(info, "w", force_zip64=True) as sink:
                            sink.write(payload)
                    else:
                        archive.writestr(info, payload)
    return frame_payloads if include_frames else {}


def _fixture_generation(tmp_path: Path, **archive_options: object) -> FixtureGeneration:
    data_root = tmp_path / "calvin"
    data_root.mkdir(parents=True)
    archive = data_root / CALVIN_ARCHIVE_NAME
    frame_payloads = _write_fixture_archive(archive, **archive_options)
    raw = archive.read_bytes()
    return FixtureGeneration(
        data_root=data_root,
        archive=archive,
        archive_bytes=len(raw),
        archive_sha256=hashlib.sha256(raw).hexdigest(),
        frame_payloads=frame_payloads,
    )


def _prepare(fixture: FixtureGeneration):
    return prepare_calvin_archive(
        fixture.archive,
        fixture.data_root,
        expected_archive_bytes=fixture.archive_bytes,
        expected_archive_sha256=fixture.archive_sha256,
        expected_central_directory=None,
        archive_url=FIXTURE_ARCHIVE_URL,
        checksum_url=FIXTURE_CHECKSUM_URL,
    )


def _reader_from_fixture(fixture: FixtureGeneration) -> CalvinArchiveReader:
    return CalvinArchiveReader.from_manifest(
        fixture.data_root,
        expected_archive_bytes=fixture.archive_bytes,
        expected_archive_sha256=fixture.archive_sha256,
        expected_central_directory=None,
        expected_archive_url=FIXTURE_ARCHIVE_URL,
        expected_checksum_url=FIXTURE_CHECKSUM_URL,
    )


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _index_contract(path: Path) -> tuple[int, str]:
    return path.stat().st_size, _file_sha256(path)


def _rebind_manifest_to_index(data_root: Path, mutate=None) -> None:
    manifest_path = data_root / CALVIN_MANIFEST_NAME
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    index_path = data_root / CALVIN_INDEX_NAME
    payload["storage"]["member_index"]["bytes"] = index_path.stat().st_size
    payload["storage"]["member_index"]["sha256"] = _file_sha256(index_path)
    if mutate is not None:
        mutate(payload)
    payload["content_sha256"] = archive_module._content_sha256(payload)
    os.chmod(manifest_path, 0o644)
    manifest_path.write_text(
        json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.chmod(manifest_path, 0o444)


def _central_entries(path: Path) -> dict[str, tuple[int, int]]:
    raw = path.read_bytes()
    eocd_offset = raw.rfind(struct.pack("<I", 0x06054B50))
    assert eocd_offset >= 0
    fields = _EOCD.unpack_from(raw, eocd_offset)
    count = fields[4]
    central_offset = fields[6]
    result: dict[str, tuple[int, int]] = {}
    cursor = central_offset
    for _ in range(count):
        central = _CENTRAL.unpack_from(raw, cursor)
        assert central[0] == 0x02014B50
        name_length, extra_length, comment_length = central[10:13]
        name = raw[cursor + _CENTRAL.size : cursor + _CENTRAL.size + name_length].decode("ascii")
        local_offset = central[-1]
        result[name] = (cursor, local_offset)
        cursor += _CENTRAL.size + name_length + extra_length + comment_length
    return result


def _rewrite_archive(path: Path, mutate) -> tuple[int, str]:
    raw = bytearray(path.read_bytes())
    mutate(raw, _central_entries(path))
    path.write_bytes(raw)
    return len(raw), hashlib.sha256(raw).hexdigest()


def _as_zip64_trailer(path: Path) -> tuple[int, str]:
    raw = path.read_bytes()
    eocd_offset = raw.rfind(struct.pack("<I", 0x06054B50))
    fields = _EOCD.unpack_from(raw, eocd_offset)
    entry_count = fields[4]
    central_size = fields[5]
    central_offset = fields[6]
    zip64_offset = eocd_offset
    zip64_eocd = _ZIP64_EOCD.pack(
        0x06064B50,
        44,
        45,
        45,
        0,
        0,
        entry_count,
        entry_count,
        central_size,
        central_offset,
    )
    locator = _ZIP64_LOCATOR.pack(0x07064B50, 0, zip64_offset, 1)
    terminal = _EOCD.pack(
        0x06054B50,
        0,
        0,
        0xFFFF,
        0xFFFF,
        0xFFFFFFFF,
        0xFFFFFFFF,
        0,
    )
    rewritten = raw[:eocd_offset] + zip64_eocd + locator + terminal
    path.write_bytes(rewritten)
    return len(rewritten), hashlib.sha256(rewritten).hexdigest()


def test_prepare_script_exposes_only_the_pinned_contract() -> None:
    completed = subprocess.run(
        [str(ROOT / ".venv" / "bin" / "python"), str(PREPARE_SCRIPT), "print-contract"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(completed.stdout) == {
        "archive": CALVIN_ARCHIVE_NAME,
        "bytes": CALVIN_ARCHIVE_BYTES,
        "checksum_url": CALVIN_CHECKSUM_URL,
        "sha256": CALVIN_ARCHIVE_SHA256,
        "url": CALVIN_ARCHIVE_URL,
    }


def test_prepare_projects_only_metadata_and_reader_returns_exact_members(tmp_path: Path) -> None:
    fixture = _fixture_generation(tmp_path)
    prepared = _prepare(fixture)

    materialized = {
        str(path.relative_to(prepared.dataset_root)) for path in prepared.dataset_root.rglob("*") if path.is_file()
    }
    assert materialized == set(CALVIN_CRITICAL_FILES)
    assert not list(prepared.dataset_root.rglob("episode_*.npz"))
    assert prepared.manifest["schema"] == CALVIN_MANIFEST_SCHEMA
    assert prepared.manifest["storage"]["member_index"]["schema"] == CALVIN_MEMBER_INDEX_SCHEMA
    assert prepared.manifest["storage"]["derived_artifacts"]["state_action_sidecar"] is None
    assert prepared.manifest_path.is_file()

    loaded = load_calvin_archive_manifest(
        fixture.data_root,
        expected_archive_bytes=fixture.archive_bytes,
        expected_archive_sha256=fixture.archive_sha256,
        expected_central_directory=None,
        expected_archive_url=FIXTURE_ARCHIVE_URL,
        expected_checksum_url=FIXTURE_CHECKSUM_URL,
    )
    assert loaded == prepared.manifest
    for relative in CALVIN_CRITICAL_FILES:
        assert (prepared.dataset_root / relative).read_bytes() == f"authenticated fixture:{relative}\n".encode()

    with _reader_from_fixture(fixture) as reader:
        manifest_copy = reader.authenticated_manifest
        archive_file_identity = reader.archive_file_identity
        assert set(archive_file_identity) == {
            "device",
            "inode",
            "mode",
            "link_count",
            "size",
            "mtime_ns",
            "ctime_ns",
        }
        archive_file_identity["inode"] += 1
        assert reader.archive_file_identity["inode"] != archive_file_identity["inode"]
        assert manifest_copy == prepared.manifest
        manifest_copy["storage"]["mode"] = "caller-mutated-copy"
        assert reader.authenticated_manifest["storage"]["mode"] == "archive-direct"
        for relative in CALVIN_CRITICAL_FILES:
            assert reader.read_authenticated_metadata_bytes(relative) == (prepared.dataset_root / relative).read_bytes()
        for relative, expected in fixture.frame_payloads.items():
            record = reader.member_record(relative)
            assert record.path == relative
            assert record.logical_bytes == len(expected)
            assert record.logical_sha256.hex() == hashlib.sha256(expected).hexdigest()
            assert reader.read_member_bytes(relative) == expected
        physical = list(reader.iter_frame_records_physical())
        assert [(record.split, record.global_index) for record in physical] == [
            ("training", 2),
            ("training", 0),
            ("validation", 1),
        ]
        assert all(record.state_action_row is None for record in physical)
        with pytest.raises(CalvinArchiveValidationError, match="bounded logical-size"):
            reader.read_member_bytes("training/episode_0000002.npz", maximum_bytes=1)
    with pytest.raises(CalvinArchiveValidationError, match="closed"):
        _ = reader.authenticated_manifest
    with pytest.raises(CalvinArchiveValidationError, match="closed"):
        _ = reader.archive_file_identity


def test_fast_manifest_reader_uses_only_fresh_ephemeral_archive_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture_generation(tmp_path)
    _prepare(fixture)
    with _reader_from_fixture(fixture) as authenticated:
        capability = authenticated.archive_file_identity

    real_sha256 = PinnedRegularFile.sha256

    def forbid_second_archive_hash(pinned: PinnedRegularFile) -> str:
        if pinned.path.name == CALVIN_ARCHIVE_NAME:
            raise AssertionError("fast reader performed a second full archive hash")
        return real_sha256(pinned)

    monkeypatch.setattr(PinnedRegularFile, "sha256", forbid_second_archive_hash)
    with CalvinArchiveReader.from_manifest_fast(
        fixture.data_root,
        expected_archive_file_identity=capability,
        expected_archive_bytes=fixture.archive_bytes,
        expected_archive_sha256=fixture.archive_sha256,
        expected_central_directory=None,
        expected_archive_url=FIXTURE_ARCHIVE_URL,
        expected_checksum_url=FIXTURE_CHECKSUM_URL,
    ) as reader:
        assert reader.archive_file_identity == capability
        assert (
            reader.read_member_bytes("training/episode_0000000.npz")
            == fixture.frame_payloads["training/episode_0000000.npz"]
        )

    forged = dict(capability)
    forged["inode"] += 1
    with pytest.raises(CalvinArchiveValidationError, match="ephemeral file identity"):
        CalvinArchiveReader.from_manifest_fast(
            fixture.data_root,
            expected_archive_file_identity=forged,
            expected_archive_bytes=fixture.archive_bytes,
            expected_archive_sha256=fixture.archive_sha256,
            expected_central_directory=None,
            expected_archive_url=FIXTURE_ARCHIVE_URL,
            expected_checksum_url=FIXTURE_CHECKSUM_URL,
        )

    malformed = dict(capability)
    malformed["unexpected"] = 0
    with pytest.raises(ValueError, match="field inventory"):
        CalvinArchiveReader.from_manifest_fast(
            fixture.data_root,
            expected_archive_file_identity=malformed,
            expected_archive_bytes=fixture.archive_bytes,
            expected_archive_sha256=fixture.archive_sha256,
            expected_central_directory=None,
            expected_archive_url=FIXTURE_ARCHIVE_URL,
            expected_checksum_url=FIXTURE_CHECKSUM_URL,
        )


def test_v2_index_uses_complete_physical_and_sidecar_absent_contract(tmp_path: Path) -> None:
    fixture = _fixture_generation(tmp_path)
    prepared = _prepare(fixture)
    with sqlite3.connect(prepared.index_path) as connection:
        objects = {
            (kind, name)
            for kind, name in connection.execute("SELECT type,name FROM sqlite_schema WHERE name NOT LIKE 'sqlite_%'")
        }
        assert objects == {
            ("table", "members"),
            ("table", "metadata"),
            ("index", "members_episode"),
            ("index", "members_local_header"),
            ("index", "members_physical"),
        }
        metadata = dict(connection.execute("SELECT name,value FROM metadata"))
        assert metadata["schema"] == CALVIN_MEMBER_INDEX_SCHEMA
        assert metadata["status"] == "complete"
        assert metadata["state_action_sidecar_status"] == "absent"
        assert connection.execute("SELECT count(*) FROM members WHERE data_offset < 0").fetchone() == (0,)
        assert connection.execute("SELECT count(*) FROM members WHERE state_action_row IS NOT NULL").fetchone() == (0,)


def test_standard_archive_with_forced_local_zip64_sizes_is_supported(tmp_path: Path) -> None:
    fixture = _fixture_generation(tmp_path, force_zip64_frame=True)
    _prepare(fixture)
    with _reader_from_fixture(fixture) as reader:
        assert (
            reader.read_member_bytes("training/episode_0000002.npz")
            == fixture.frame_payloads["training/episode_0000002.npz"]
        )


def test_central_zip64_offset_extension_keeps_deflate_extraction_version() -> None:
    archive_module._validate_central_zip64_version(
        version_needed=20,
        logical_32=123,
        compressed_32=100,
    )
    with pytest.raises(CalvinArchiveValidationError, match="member size"):
        archive_module._validate_central_zip64_version(
            version_needed=20,
            logical_32=0xFFFFFFFF,
            compressed_32=100,
        )


def test_zip64_eocd_locator_and_sentinel_contract_is_supported(tmp_path: Path) -> None:
    fixture = _fixture_generation(tmp_path)
    archive_bytes, archive_sha256 = _as_zip64_trailer(fixture.archive)
    fixture = FixtureGeneration(
        data_root=fixture.data_root,
        archive=fixture.archive,
        archive_bytes=archive_bytes,
        archive_sha256=archive_sha256,
        frame_payloads=fixture.frame_payloads,
    )
    prepared = _prepare(fixture)
    assert prepared.manifest["archive"]["central_directory"]["zip64"] is True
    with _reader_from_fixture(fixture) as reader:
        assert (
            reader.read_member_bytes("training/episode_0000000.npz")
            == fixture.frame_payloads["training/episode_0000000.npz"]
        )


@pytest.mark.parametrize(
    ("info", "message"),
    [
        (_zip_info("task_ABC_D/training//evil.npz"), "path"),
        (_zip_info(r"task_ABC_D/training\evil.npz"), "path"),
        (_zip_info("task_ABC_D/training/../evil.npz"), "path"),
        (_zip_info("task_ABC_D/training/./evil.npz"), "path"),
        (_zip_info("outside/evil.npz"), "outside"),
        (_zip_info("task_ABC_D/training/link.npz", kind="symlink"), "non-regular"),
        (_zip_info("task_ABC_D/training/fifo.npz", kind="fifo"), "non-regular"),
        (_zip_info("task_ABC_D/training/stored.npz", method=zipfile.ZIP_STORED), "DEFLATE"),
    ],
    ids=("empty", "backslash", "dot-dot", "dot", "wrong-root", "symlink", "fifo", "stored-file"),
)
def test_prepare_rejects_unsafe_paths_types_and_methods(
    tmp_path: Path,
    info: zipfile.ZipInfo,
    message: str,
) -> None:
    fixture = _fixture_generation(tmp_path, extra_entries=[(info, b"malicious")])
    with pytest.raises(CalvinArchiveValidationError, match=message):
        _prepare(fixture)


def test_prepare_rejects_duplicate_namespace_and_missing_parent(tmp_path: Path) -> None:
    duplicate = _zip_info(f"task_ABC_D/{CALVIN_CRITICAL_FILES[0]}")
    fixture = _fixture_generation(tmp_path, extra_entries=[(duplicate, b"duplicate")])
    with pytest.raises(CalvinArchiveValidationError, match=r"duplicate|colliding"):
        _prepare(fixture)

    second = tmp_path / "second"
    child = _zip_info("task_ABC_D/not_yet_declared/child.npz")
    fixture = _fixture_generation(second, extra_entries=[(child, b"child")])
    with pytest.raises(CalvinArchiveValidationError, match="parent directory"):
        _prepare(fixture)


def test_prepare_accepts_optional_root_and_parent_directory_declared_later(tmp_path: Path) -> None:
    child = _zip_info("task_ABC_D/declared_later/child.npz")
    parent = _zip_info("task_ABC_D/declared_later/", kind="directory")
    fixture = _fixture_generation(
        tmp_path,
        include_root_directory=False,
        extra_entries=[(child, b"child"), (parent, b"")],
    )
    _prepare(fixture)
    with _reader_from_fixture(fixture) as reader:
        assert reader.read_member_bytes("declared_later/child.npz") == b"child"


def test_prepare_rejects_unknown_extra_field(tmp_path: Path) -> None:
    unknown = struct.pack("<HH4s", 0x9999, 4, b"evil")
    fixture = _fixture_generation(
        tmp_path,
        extra_entries=[(_zip_info("task_ABC_D/training/unknown-extra.npz", extra=unknown), b"payload")],
    )
    with pytest.raises(CalvinArchiveValidationError, match="extra-field"):
        _prepare(fixture)


def test_prepare_rejects_nonzero_general_purpose_flags(tmp_path: Path) -> None:
    fixture = _fixture_generation(tmp_path)
    target = "task_ABC_D/training/episode_0000002.npz"

    def mutate(raw: bytearray, entries: dict[str, tuple[int, int]]) -> None:
        central, local = entries[target]
        struct.pack_into("<H", raw, central + 8, 1)
        struct.pack_into("<H", raw, local + 6, 1)

    size, digest = _rewrite_archive(fixture.archive, mutate)
    fixture = FixtureGeneration(fixture.data_root, fixture.archive, size, digest, fixture.frame_payloads)
    with pytest.raises(CalvinArchiveValidationError, match="flags"):
        _prepare(fixture)


@pytest.mark.parametrize(
    "field",
    ["name", "method", "version"],
    ids=("name", "method", "version"),
)
def test_prepare_rejects_local_central_mismatch(tmp_path: Path, field: str) -> None:
    fixture = _fixture_generation(tmp_path)
    target = "task_ABC_D/training/episode_0000002.npz"

    def mutate(raw: bytearray, entries: dict[str, tuple[int, int]]) -> None:
        _central, local = entries[target]
        if field == "name":
            raw[local + 30] ^= 1
        elif field == "method":
            struct.pack_into("<H", raw, local + 8, zipfile.ZIP_STORED)
        else:
            current = struct.unpack_from("<H", raw, local + 4)[0]
            struct.pack_into("<H", raw, local + 4, current + 1)

    size, digest = _rewrite_archive(fixture.archive, mutate)
    fixture = FixtureGeneration(fixture.data_root, fixture.archive, size, digest, fixture.frame_payloads)
    with pytest.raises(CalvinArchiveValidationError, match="local/central"):
        _prepare(fixture)


def test_prepare_rejects_crc_even_when_local_and_central_claim_match(tmp_path: Path) -> None:
    fixture = _fixture_generation(tmp_path)
    target = "task_ABC_D/training/episode_0000002.npz"

    def mutate(raw: bytearray, entries: dict[str, tuple[int, int]]) -> None:
        central, local = entries[target]
        crc = struct.unpack_from("<I", raw, central + 16)[0] ^ 1
        struct.pack_into("<I", raw, central + 16, crc)
        struct.pack_into("<I", raw, local + 14, crc)

    size, digest = _rewrite_archive(fixture.archive, mutate)
    fixture = FixtureGeneration(fixture.data_root, fixture.archive, size, digest, fixture.frame_payloads)
    with pytest.raises(CalvinArchiveValidationError, match="CRC32"):
        _prepare(fixture)


def test_prepare_rejects_raw_deflate_non_eof(tmp_path: Path) -> None:
    fixture = _fixture_generation(tmp_path)
    target = "task_ABC_D/validation/episode_0000001.npz"

    def mutate(raw: bytearray, entries: dict[str, tuple[int, int]]) -> None:
        central, local = entries[target]
        compressed = struct.unpack_from("<I", raw, central + 20)[0]
        assert compressed > 1
        struct.pack_into("<I", raw, central + 20, compressed - 1)
        struct.pack_into("<I", raw, local + 18, compressed - 1)

    size, digest = _rewrite_archive(fixture.archive, mutate)
    fixture = FixtureGeneration(fixture.data_root, fixture.archive, size, digest, fixture.frame_payloads)
    with pytest.raises(CalvinArchiveValidationError, match=r"terminate exactly|CRC32"):
        _prepare(fixture)


def test_terminal_junk_and_malformed_zip64_locator_are_rejected(tmp_path: Path) -> None:
    fixture = _fixture_generation(tmp_path)
    fixture.archive.write_bytes(fixture.archive.read_bytes() + b"trailing-junk")
    raw = fixture.archive.read_bytes()
    fixture = FixtureGeneration(
        fixture.data_root,
        fixture.archive,
        len(raw),
        hashlib.sha256(raw).hexdigest(),
        fixture.frame_payloads,
    )
    with pytest.raises(CalvinArchiveValidationError, match="EOCD"):
        _prepare(fixture)

    second = _fixture_generation(tmp_path / "second")
    size, _digest = _as_zip64_trailer(second.archive)
    raw = bytearray(second.archive.read_bytes())
    locator_offset = len(raw) - _EOCD.size - _ZIP64_LOCATOR.size
    struct.pack_into("<I", raw, locator_offset + 16, 2)
    second.archive.write_bytes(raw)
    second = FixtureGeneration(
        second.data_root,
        second.archive,
        size,
        hashlib.sha256(raw).hexdigest(),
        second.frame_payloads,
    )
    with pytest.raises(CalvinArchiveValidationError, match="ZIP64 locator"):
        _prepare(second)


def test_archive_and_index_nofollow_single_link_pins(tmp_path: Path) -> None:
    fixture = _fixture_generation(tmp_path)
    link = fixture.data_root / "archive-link.zip"
    link.symlink_to(fixture.archive)
    with pytest.raises(CalvinArchiveValidationError, match="no-follow"):
        PinnedRegularFile.open(link)

    hardlink = fixture.data_root / "archive-hardlink.zip"
    os.link(fixture.archive, hardlink)
    with pytest.raises(CalvinArchiveValidationError, match="hard link"):
        PinnedRegularFile.open(fixture.archive)


def test_reader_rejects_index_symlink_hardlink_and_extra_sqlite_object(tmp_path: Path) -> None:
    fixture = _fixture_generation(tmp_path)
    prepared = _prepare(fixture)
    index = prepared.index_path
    original = index.with_suffix(".real")
    index.rename(original)
    index.symlink_to(original.name)
    size, digest = _index_contract(original)
    with pytest.raises(CalvinArchiveValidationError, match="no-follow"):
        CalvinArchiveReader.open(
            fixture.archive,
            index,
            expected_archive_bytes=fixture.archive_bytes,
            expected_archive_sha256=fixture.archive_sha256,
            expected_index_bytes=size,
            expected_index_sha256=digest,
        )
    index.unlink()
    original.rename(index)
    hardlink = index.with_suffix(".hardlink")
    os.link(index, hardlink)
    with pytest.raises(CalvinArchiveValidationError, match="hard link"):
        PinnedRegularFile.open(index)
    hardlink.unlink()

    os.chmod(index, 0o644)
    with sqlite3.connect(index) as connection:
        connection.execute("CREATE VIEW covert_members AS SELECT * FROM members")
    os.chmod(index, 0o444)
    size, digest = _index_contract(index)
    with pytest.raises(CalvinArchiveValidationError, match="object inventory"):
        CalvinArchiveReader.open(
            fixture.archive,
            index,
            expected_archive_bytes=fixture.archive_bytes,
            expected_archive_sha256=fixture.archive_sha256,
            expected_index_bytes=size,
            expected_index_sha256=digest,
        )


@pytest.mark.parametrize("attack", ["logical-sha", "data-offset"], ids=("logical-sha", "data-offset"))
def test_reader_revalidates_logical_sha_and_local_data_offset(tmp_path: Path, attack: str) -> None:
    fixture = _fixture_generation(tmp_path)
    prepared = _prepare(fixture)
    index = prepared.index_path
    os.chmod(index, 0o644)
    with sqlite3.connect(index) as connection:
        if attack == "logical-sha":
            connection.execute(
                "UPDATE members SET logical_sha256=? WHERE path=?",
                (b"x" * 32, "training/episode_0000000.npz"),
            )
        else:
            connection.execute(
                "UPDATE members SET data_offset=data_offset+1 WHERE path=?",
                ("training/episode_0000000.npz",),
            )
    os.chmod(index, 0o444)
    size, digest = _index_contract(index)
    if attack == "data-offset":
        with pytest.raises(CalvinArchiveValidationError, match=r"physical ranges|payload range"):
            CalvinArchiveReader.open(
                fixture.archive,
                index,
                expected_archive_bytes=fixture.archive_bytes,
                expected_archive_sha256=fixture.archive_sha256,
                expected_index_bytes=size,
                expected_index_sha256=digest,
            )
    else:
        with (
            CalvinArchiveReader.open(
                fixture.archive,
                index,
                expected_archive_bytes=fixture.archive_bytes,
                expected_archive_sha256=fixture.archive_sha256,
                expected_index_bytes=size,
                expected_index_sha256=digest,
            ) as reader,
            pytest.raises(CalvinArchiveValidationError, match="logical SHA-256"),
        ):
            reader.read_member_bytes("training/episode_0000000.npz")


def test_reader_detects_in_place_archive_mutation_after_session_pin(tmp_path: Path) -> None:
    fixture = _fixture_generation(tmp_path)
    prepared = _prepare(fixture)
    index_contract = prepared.manifest["storage"]["member_index"]
    with CalvinArchiveReader.open(
        fixture.archive,
        prepared.index_path,
        expected_archive_bytes=fixture.archive_bytes,
        expected_archive_sha256=fixture.archive_sha256,
        expected_index_bytes=index_contract["bytes"],
        expected_index_sha256=index_contract["sha256"],
    ) as reader:
        with sqlite3.connect(prepared.index_path) as connection:
            data_offset = connection.execute(
                "SELECT data_offset FROM members WHERE path=?",
                ("training/episode_0000000.npz",),
            ).fetchone()[0]
        descriptor = os.open(fixture.archive, os.O_RDWR)
        try:
            original = os.pread(descriptor, 1, data_offset)
            os.pwrite(descriptor, bytes([original[0] ^ 1]), data_offset)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        with pytest.raises(CalvinArchiveValidationError, match="identity changed"):
            reader.read_member_bytes("training/episode_0000000.npz")


def test_manifest_loader_authenticates_materialized_metadata(tmp_path: Path) -> None:
    fixture = _fixture_generation(tmp_path)
    prepared = _prepare(fixture)
    target = prepared.dataset_root / CALVIN_CRITICAL_FILES[0]
    os.chmod(target, 0o644)
    target.write_bytes(b"changed after authenticated projection")
    os.chmod(target, 0o444)
    with pytest.raises(CalvinArchiveValidationError, match="metadata"):
        load_calvin_archive_manifest(
            fixture.data_root,
            expected_archive_bytes=fixture.archive_bytes,
            expected_archive_sha256=fixture.archive_sha256,
            expected_central_directory=None,
            expected_archive_url=FIXTURE_ARCHIVE_URL,
            expected_checksum_url=FIXTURE_CHECKSUM_URL,
        )


def test_manifest_loader_rejects_symlinked_projection_parent(tmp_path: Path) -> None:
    fixture = _fixture_generation(tmp_path)
    prepared = _prepare(fixture)
    training = prepared.dataset_root / "training"
    held = prepared.dataset_root / "training-held"
    os.rename(training, held)
    training.symlink_to(held.name, target_is_directory=True)
    with pytest.raises(CalvinArchiveValidationError, match="projection contains a symlink"):
        load_calvin_archive_manifest(
            fixture.data_root,
            expected_archive_bytes=fixture.archive_bytes,
            expected_archive_sha256=fixture.archive_sha256,
            expected_central_directory=None,
            expected_archive_url=FIXTURE_ARCHIVE_URL,
            expected_checksum_url=FIXTURE_CHECKSUM_URL,
        )


def test_prepare_refuses_active_download_and_every_existing_publication(tmp_path: Path) -> None:
    fixture = _fixture_generation(tmp_path)
    control = fixture.archive.with_name(f"{fixture.archive.name}.aria2")
    control.write_bytes(b"active")
    with pytest.raises(CalvinArchivePublicationError, match="control file"):
        _prepare(fixture)
    control.unlink()
    prepared = _prepare(fixture)
    assert prepared.manifest_path.is_file()
    with pytest.raises(CalvinArchivePublicationError, match="replace existing"):
        _prepare(fixture)


def test_prepare_refuses_stale_private_stage_without_blessing_it(tmp_path: Path) -> None:
    fixture = _fixture_generation(tmp_path)
    stale = fixture.data_root / ".task_ABC_D.archive-direct.interrupted"
    stale.mkdir()
    (stale / "untrusted").write_bytes(b"partial index")
    with pytest.raises(CalvinArchivePublicationError, match="explicit recovery"):
        _prepare(fixture)
    assert not os.path.lexists(fixture.data_root / CALVIN_MANIFEST_NAME)
    assert (stale / "untrusted").read_bytes() == b"partial index"


def test_manifest_is_last_commit_marker_on_injected_publication_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture_generation(tmp_path)
    real_publish = archive_module._publish_file_noreplace

    def fail_manifest(source, destination_descriptor: int, destination_name: str):
        if destination_name == CALVIN_MANIFEST_NAME:
            raise CalvinArchivePublicationError("injected manifest publication failure")
        return real_publish(source, destination_descriptor, destination_name)

    monkeypatch.setattr(archive_module, "_publish_file_noreplace", fail_manifest)
    with pytest.raises(CalvinArchivePublicationError, match="injected"):
        _prepare(fixture)
    assert (fixture.data_root / "task_ABC_D").is_dir()
    assert (fixture.data_root / CALVIN_INDEX_NAME).is_file()
    assert not os.path.lexists(fixture.data_root / CALVIN_MANIFEST_NAME)
    stages = list(fixture.data_root.glob(".task_ABC_D.archive-direct.*"))
    assert len(stages) == 1
    assert (stages[0] / CALVIN_MANIFEST_NAME).is_file()


def test_reader_rejects_exact_schema_partial_predicate_substitution(tmp_path: Path) -> None:
    fixture = _fixture_generation(tmp_path)
    prepared = _prepare(fixture)
    os.chmod(prepared.index_path, 0o644)
    with sqlite3.connect(prepared.index_path) as connection:
        connection.execute("DROP INDEX members_episode")
        connection.execute("CREATE UNIQUE INDEX members_episode ON members(split,global_index) WHERE global_index >= 0")
    os.chmod(prepared.index_path, 0o444)
    size, digest = _index_contract(prepared.index_path)
    with pytest.raises(CalvinArchiveValidationError, match="object inventory"):
        CalvinArchiveReader.open(
            fixture.archive,
            prepared.index_path,
            expected_archive_bytes=fixture.archive_bytes,
            expected_archive_sha256=fixture.archive_sha256,
            expected_index_bytes=size,
            expected_index_sha256=digest,
        )


def test_self_rehashed_forged_episode_identity_is_not_a_trust_root(tmp_path: Path) -> None:
    fixture = _fixture_generation(tmp_path)
    prepared = _prepare(fixture)
    os.chmod(prepared.index_path, 0o644)
    with sqlite3.connect(prepared.index_path) as connection:
        connection.execute(
            "UPDATE members SET split='validation',global_index=999 WHERE path=?",
            ("training/episode_0000000.npz",),
        )
    os.chmod(prepared.index_path, 0o444)
    _rebind_manifest_to_index(fixture.data_root)
    with pytest.raises(CalvinArchiveValidationError, match="split/global identity"):
        _reader_from_fixture(fixture)


def test_reader_rejects_noncanonical_decimal_metadata(tmp_path: Path) -> None:
    fixture = _fixture_generation(tmp_path)
    prepared = _prepare(fixture)
    os.chmod(prepared.index_path, 0o644)
    with sqlite3.connect(prepared.index_path) as connection:
        value = connection.execute("SELECT value FROM metadata WHERE name='archive_bytes'").fetchone()[0]
        connection.execute(
            "UPDATE metadata SET value=? WHERE name='archive_bytes'",
            (f"0{value}",),
        )
    os.chmod(prepared.index_path, 0o444)
    size, digest = _index_contract(prepared.index_path)
    with pytest.raises(CalvinArchiveValidationError, match="canonical ASCII decimal"):
        CalvinArchiveReader.open(
            fixture.archive,
            prepared.index_path,
            expected_archive_bytes=fixture.archive_bytes,
            expected_archive_sha256=fixture.archive_sha256,
            expected_index_bytes=size,
            expected_index_sha256=digest,
        )


def test_manifest_verification_contract_requires_exact_equality(tmp_path: Path) -> None:
    fixture = _fixture_generation(tmp_path)
    _prepare(fixture)

    def mutate(payload: dict[str, object]) -> None:
        payload["storage"]["verification"] += "+weaker-self-claim"

    _rebind_manifest_to_index(fixture.data_root, mutate)
    with pytest.raises(CalvinArchiveValidationError, match="storage contract"):
        load_calvin_archive_manifest(
            fixture.data_root,
            expected_archive_bytes=fixture.archive_bytes,
            expected_archive_sha256=fixture.archive_sha256,
            expected_central_directory=None,
            expected_archive_url=FIXTURE_ARCHIVE_URL,
            expected_checksum_url=FIXTURE_CHECKSUM_URL,
        )


def test_manifest_with_two_links_is_not_a_visible_commit_marker(tmp_path: Path) -> None:
    fixture = _fixture_generation(tmp_path)
    prepared = _prepare(fixture)
    extra_link = fixture.data_root / "manifest-transient-link"
    os.link(prepared.manifest_path, extra_link)
    with pytest.raises(CalvinArchiveValidationError, match="exactly one hard link"):
        load_calvin_archive_manifest(
            fixture.data_root,
            expected_archive_bytes=fixture.archive_bytes,
            expected_archive_sha256=fixture.archive_sha256,
            expected_central_directory=None,
            expected_archive_url=FIXTURE_ARCHIVE_URL,
            expected_checksum_url=FIXTURE_CHECKSUM_URL,
        )


def test_physical_iterator_revalidates_each_local_header(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture_generation(tmp_path)
    _prepare(fixture)
    with _reader_from_fixture(fixture) as reader:
        real_local_header = archive_module._local_header_data_offset
        checked: list[str] = []

        def counted_local_header(*args, **kwargs):
            checked.append(args[1].path)
            return real_local_header(*args, **kwargs)

        monkeypatch.setattr(archive_module, "_local_header_data_offset", counted_local_header)
        records = list(reader.iter_frame_records_physical())
    assert checked == [record.path for record in records]


def test_source_substitution_is_retained_without_foreign_unlink(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture_generation(tmp_path)
    real_publish = archive_module._publish_file_noreplace

    def substitute_source(source, destination_descriptor: int, destination_name: str):
        if destination_name == CALVIN_INDEX_NAME:
            assert source.source_parent_descriptor is not None
            assert source.source_name is not None
            os.rename(
                source.source_name,
                f"{source.source_name}.held",
                src_dir_fd=source.source_parent_descriptor,
                dst_dir_fd=source.source_parent_descriptor,
            )
            descriptor = os.open(
                source.source_name,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW,
                0o600,
                dir_fd=source.source_parent_descriptor,
            )
            try:
                os.write(descriptor, b"foreign-index-source")
            finally:
                os.close(descriptor)
        return real_publish(source, destination_descriptor, destination_name)

    monkeypatch.setattr(archive_module, "_publish_file_noreplace", substitute_source)
    with pytest.raises(CalvinArchivePublicationError, match=r"source binding changed|staged publication inode changed"):
        _prepare(fixture)
    stages = list(fixture.data_root.glob(".task_ABC_D.archive-direct.*"))
    assert len(stages) == 1
    assert (stages[0] / CALVIN_INDEX_NAME).read_bytes() == b"foreign-index-source"
    assert (stages[0] / f"{CALVIN_INDEX_NAME}.held").is_file()
    assert not (fixture.data_root / CALVIN_INDEX_NAME).exists()
    assert not (fixture.data_root / CALVIN_MANIFEST_NAME).exists()


def test_destination_substitution_is_not_deleted_on_noreplace_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture_generation(tmp_path)
    real_publish = archive_module._publish_file_noreplace

    def substitute_destination(source, destination_descriptor: int, destination_name: str):
        if destination_name == CALVIN_INDEX_NAME:
            descriptor = os.open(
                destination_name,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW,
                0o600,
                dir_fd=destination_descriptor,
            )
            try:
                os.write(descriptor, b"foreign-index-destination")
            finally:
                os.close(descriptor)
        return real_publish(source, destination_descriptor, destination_name)

    monkeypatch.setattr(archive_module, "_publish_file_noreplace", substitute_destination)
    with pytest.raises(CalvinArchivePublicationError, match="replace existing"):
        _prepare(fixture)
    assert (fixture.data_root / CALVIN_INDEX_NAME).read_bytes() == b"foreign-index-destination"
    assert not (fixture.data_root / CALVIN_MANIFEST_NAME).exists()


def test_data_root_path_swap_fails_before_manifest_commit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture_generation(tmp_path)
    held_root = tmp_path / "held-calvin-root"
    real_publish = archive_module._publish_directory_noreplace

    def swap_root(*args, **kwargs):
        result = real_publish(*args, **kwargs)
        os.rename(fixture.data_root, held_root)
        fixture.data_root.mkdir()
        return result

    monkeypatch.setattr(archive_module, "_publish_directory_noreplace", swap_root)
    with pytest.raises(CalvinArchiveValidationError, match="path binding changed"):
        _prepare(fixture)
    assert not (fixture.data_root / CALVIN_MANIFEST_NAME).exists()
    assert not (held_root / CALVIN_MANIFEST_NAME).exists()
    assert (held_root / CALVIN_DATASET_NAME).is_dir()


def test_prepare_lock_unlink_recreate_fails_without_deleting_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture_generation(tmp_path)
    real_publish = archive_module._publish_directory_noreplace
    lock_name = ".task_ABC_D.prepare.lock"

    def replace_lock(*args, **kwargs):
        result = real_publish(*args, **kwargs)
        destination_descriptor = args[3]
        os.unlink(lock_name, dir_fd=destination_descriptor)
        descriptor = os.open(
            lock_name,
            os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW,
            0o600,
            dir_fd=destination_descriptor,
        )
        os.close(descriptor)
        return result

    monkeypatch.setattr(archive_module, "_publish_directory_noreplace", replace_lock)
    with pytest.raises(CalvinArchivePublicationError, match="lock path identity changed"):
        _prepare(fixture)
    assert (fixture.data_root / lock_name).is_file()
    assert not (fixture.data_root / CALVIN_MANIFEST_NAME).exists()


def test_post_commit_fsync_failure_returns_explicit_committed_outcome(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture_generation(tmp_path)
    real_fsync = archive_module._fsync_directory_descriptor
    injected = False

    def fail_once_after_manifest(descriptor: int, *, label: str) -> None:
        nonlocal injected
        if not injected and label == "data root":
            try:
                os.stat(CALVIN_MANIFEST_NAME, dir_fd=descriptor, follow_symlinks=False)
            except FileNotFoundError:
                pass
            else:
                injected = True
                raise CalvinArchivePublicationError("injected post-commit fsync failure")
        real_fsync(descriptor, label=label)

    monkeypatch.setattr(archive_module, "_fsync_directory_descriptor", fail_once_after_manifest)
    prepared = _prepare(fixture)
    assert injected
    assert prepared.manifest_path.is_file()
    assert any("manifest is committed" in warning for warning in prepared.publication_warnings)
    with _reader_from_fixture(fixture):
        pass


def test_post_publish_manifest_error_is_reconciled_as_committed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture_generation(tmp_path)
    real_publish = archive_module._publish_file_noreplace

    def publish_then_raise(source, destination_descriptor: int, destination_name: str):
        result = real_publish(source, destination_descriptor, destination_name)
        if destination_name == CALVIN_MANIFEST_NAME:
            raise CalvinArchivePublicationError("injected error after manifest publication")
        return result

    monkeypatch.setattr(archive_module, "_publish_file_noreplace", publish_then_raise)
    prepared = _prepare(fixture)
    assert prepared.manifest_path.is_file()
    assert any("post-publication error" in warning for warning in prepared.publication_warnings)
    with _reader_from_fixture(fixture):
        pass


def test_sparse_zip64_central_offset_above_four_gib(tmp_path: Path) -> None:
    archive = tmp_path / "sparse.zip"
    central_offset = (1 << 32) + 4096
    central_payload = b"central-directory-sparse-fixture"
    zip64_offset = central_offset + len(central_payload)
    with archive.open("wb") as stream:
        stream.seek(central_offset)
        stream.write(central_payload)
        stream.write(
            _ZIP64_EOCD.pack(
                0x06064B50,
                44,
                45,
                45,
                0,
                0,
                1,
                1,
                len(central_payload),
                central_offset,
            )
        )
        stream.write(_ZIP64_LOCATOR.pack(0x07064B50, 0, zip64_offset, 1))
        stream.write(
            _EOCD.pack(
                0x06054B50,
                0,
                0,
                0xFFFF,
                0xFFFF,
                0xFFFFFFFF,
                0xFFFFFFFF,
                0,
            )
        )
    descriptor = os.open(archive, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    try:
        central = archive_module._read_central_directory_info(
            descriptor,
            archive.stat().st_size,
        )
    finally:
        os.close(descriptor)
    assert central.offset == central_offset
    assert central.zip64 is True
    assert central.sha256 == hashlib.sha256(central_payload).hexdigest()


def test_zip64_classic_nonsentinel_value_must_cross_match(tmp_path: Path) -> None:
    fixture = _fixture_generation(tmp_path)
    size, _digest = _as_zip64_trailer(fixture.archive)
    raw = bytearray(fixture.archive.read_bytes())
    eocd_offset = len(raw) - _EOCD.size
    struct.pack_into("<I", raw, eocd_offset + 12, 1)
    fixture.archive.write_bytes(raw)
    fixture = FixtureGeneration(
        fixture.data_root,
        fixture.archive,
        size,
        hashlib.sha256(raw).hexdigest(),
        fixture.frame_payloads,
    )
    with pytest.raises(CalvinArchiveValidationError, match="classic/ZIP64 EOCD"):
        _prepare(fixture)


def test_phase_a_cli_has_no_misleading_archive_override() -> None:
    completed = subprocess.run(
        [str(ROOT / ".venv" / "bin" / "python"), str(PREPARE_SCRIPT), "--help"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "--archive" not in completed.stdout


def test_prepare_source_does_not_touch_legacy_downloader_or_integrations() -> None:
    source = PREPARE_SCRIPT.read_text(encoding="utf-8")
    assert "download_dataset.sh" not in source
    assert "train_calvin.py" not in source
    assert "serve_policy.py" not in source
    # The CLI intentionally delegates the nullable schema hook to the archive
    # module and has no sidecar creation switch.
    assert "sidecar" not in source.lower()


@pytest.mark.skipif(
    os.environ.get("DUO_VLA_RUN_CALVIN_DEBUG_ARCHIVE_TEST") != "1",
    reason="set DUO_VLA_RUN_CALVIN_DEBUG_ARCHIVE_TEST=1 for the local 1.3 GB archive parity test",
)
def test_local_debug_archive_matches_extracted_member_and_projection(tmp_path: Path) -> None:
    cache_root = Path(os.environ.get("DUO_VLA_CACHE_ROOT", "/root/.cache/duo-vla"))
    source_root = cache_root / "data" / "calvin"
    source_archive = source_root / "calvin_debug_dataset.zip"
    extracted = source_root / "calvin_debug_dataset"
    if not source_archive.is_file() or not extracted.is_dir():
        pytest.skip("local CALVIN debug archive and extracted reference are both required")
    archive_bytes = source_archive.stat().st_size
    archive_sha256 = _file_sha256(source_archive)
    prepared = prepare_calvin_archive(
        source_archive,
        tmp_path,
        expected_archive_bytes=archive_bytes,
        expected_archive_sha256=archive_sha256,
        expected_central_directory=None,
        archive_root="calvin_debug_dataset",
        archive_url="fixture://calvin-debug",
        checksum_url="fixture://calvin-debug-sha256",
    )
    index = prepared.manifest["storage"]["member_index"]
    with CalvinArchiveReader.open(
        source_archive,
        prepared.index_path,
        expected_archive_bytes=archive_bytes,
        expected_archive_sha256=archive_sha256,
        expected_index_bytes=index["bytes"],
        expected_index_sha256=index["sha256"],
    ) as reader:
        first = next(reader.iter_frame_records_physical(split="training"))
        assert reader.read_member_bytes(first.path) == (extracted / first.path).read_bytes()
    for relative in CALVIN_CRITICAL_FILES:
        assert (prepared.dataset_root / relative).read_bytes() == (extracted / relative).read_bytes()
