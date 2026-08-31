from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sqlite3
import stat
import subprocess
import types
import zipfile
import zlib
from pathlib import Path

import pytest

from duo_vla.data import calvin_archive as CALVIN_ARCHIVE
from duo_vla.data.calvin_archive import prepare_calvin_archive
from duo_vla.data.calvin_stats import (
    CalvinStorageIdentity,
    _normalization_dataset_identity,
)

ROOT = Path(__file__).parents[1]
PREFLIGHT_PATH = ROOT / "scripts" / "calvin" / "preflight.py"
SPEC = importlib.util.spec_from_file_location("calvin_preflight", PREFLIGHT_PATH)
assert SPEC is not None and SPEC.loader is not None
PREFLIGHT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PREFLIGHT)


def test_official_projected_metadata_contract_matches_real_archive_inventory() -> None:
    expected = (
        "training/ep_start_end_ids.npy",
        "training/lang_annotations/auto_lang_ann.npy",
        "training/scene_info.npy",
        "training/.hydra/merged_config.yaml",
        "validation/ep_start_end_ids.npy",
        "validation/.hydra/merged_config.yaml",
    )
    assert expected == CALVIN_ARCHIVE.CALVIN_CRITICAL_FILES
    assert expected == PREFLIGHT.DATASET_CRITICAL_FILES
    assert expected[-2:] == PREFLIGHT.VALIDATION_CRITICAL_FILES


def _write_dataset_contract(tmp_path: Path) -> tuple[Path, Path, dict[str, object]]:
    root = tmp_path / "task_ABC_D"
    validation_file_count = len(PREFLIGHT.VALIDATION_CRITICAL_FILES)
    critical: dict[str, str] = {}
    for index, relative in enumerate(PREFLIGHT.DATASET_CRITICAL_FILES):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"critical-{index}-{relative}".encode())
        critical[relative] = hashlib.sha256(path.read_bytes()).hexdigest()

    index_path = tmp_path / "task_ABC_D.members.sqlite3"
    with sqlite3.connect(index_path) as connection:
        connection.execute(
            "CREATE TABLE members(path TEXT PRIMARY KEY, bytes INTEGER NOT NULL, "
            "crc32 INTEGER NOT NULL, sha256 TEXT NOT NULL) WITHOUT ROWID"
        )
        connection.execute("CREATE TABLE metadata(name TEXT PRIMARY KEY, value TEXT NOT NULL) WITHOUT ROWID")
        for relative in PREFLIGHT.VALIDATION_CRITICAL_FILES:
            raw = (root / relative).read_bytes()
            connection.execute(
                "INSERT INTO members(path, bytes, crc32, sha256) VALUES (?, ?, ?, ?)",
                (relative, len(raw), zlib.crc32(raw) & 0xFFFFFFFF, hashlib.sha256(raw).hexdigest()),
            )
        connection.executemany(
            "INSERT INTO metadata(name, value) VALUES (?, ?)",
            (
                ("file_member_count", str(validation_file_count)),
                ("schema", PREFLIGHT.LEGACY_MEMBER_INDEX_SCHEMA),
            ),
        )

    manifest: dict[str, object] = {
        "archive": {
            "bytes": PREFLIGHT.ARCHIVE_BYTES,
            "member_inventory": {
                "compressed_bytes": 1,
                "file_member_count": validation_file_count,
                "member_count": validation_file_count,
                "npz_member_count": 1,
                "sha256": "1" * 64,
                "uncompressed_bytes": 100,
            },
            "sha256": PREFLIGHT.ARCHIVE_SHA256,
            "uncompressed_bytes": 100,
            "url": PREFLIGHT.ARCHIVE_URL,
        },
        "checksum_url": PREFLIGHT.CHECKSUM_URL,
        "critical_files": critical,
        "dataset": "task_ABC_D",
        "extraction": {
            "file_members_verified": validation_file_count,
            "member_index": {
                "bytes": index_path.stat().st_size,
                "path": index_path.name,
                "schema": PREFLIGHT.LEGACY_MEMBER_INDEX_SCHEMA,
                "sha256": hashlib.sha256(index_path.read_bytes()).hexdigest(),
            },
            "verification": "size-and-crc32-against-every-pinned-zip-member",
        },
        "schema": PREFLIGHT.LEGACY_DATASET_MANIFEST_SCHEMA,
    }
    manifest["content_sha256"] = hashlib.sha256(PREFLIGHT.canonical_json_bytes(manifest)).hexdigest()
    manifest_path = tmp_path / "task_ABC_D.manifest.json"
    manifest_path.write_text(json.dumps(manifest, allow_nan=False, sort_keys=True), encoding="utf-8")
    return root, index_path, manifest


def _fixture_zip_info(name: str, *, directory: bool = False) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name)
    info.create_system = 3
    if directory:
        info.external_attr = ((stat.S_IFDIR | 0o755) << 16) | 0x10
        info.compress_type = zipfile.ZIP_STORED
    else:
        info.external_attr = (stat.S_IFREG | 0o644) << 16
        info.compress_type = zipfile.ZIP_DEFLATED
    return info


def _write_v4_dataset_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    include_root: bool = True,
    directories_first: bool = True,
) -> tuple[Path, Path, dict[str, object]]:
    monkeypatch.setattr(CALVIN_ARCHIVE, "CALVIN_CRITICAL_FILES", PREFLIGHT.DATASET_CRITICAL_FILES)
    data_root = tmp_path / "calvin"
    data_root.mkdir()
    archive_path = data_root / PREFLIGHT.ARCHIVE_NAME
    directories = (
        "task_ABC_D/",
        "task_ABC_D/training/",
        "task_ABC_D/training/lang_annotations/",
        "task_ABC_D/training/.hydra/",
        "task_ABC_D/validation/",
        "task_ABC_D/validation/.hydra/",
    )
    selected_directories = directories if include_root else directories[1:]
    with zipfile.ZipFile(archive_path, "w") as archive:

        def write_directories() -> None:
            for name in selected_directories:
                archive.writestr(_fixture_zip_info(name, directory=True), b"")

        def write_files() -> None:
            for index, relative in enumerate(PREFLIGHT.DATASET_CRITICAL_FILES):
                archive.writestr(
                    _fixture_zip_info(f"task_ABC_D/{relative}"),
                    f"authenticated-v4-{index}:{relative}\n".encode(),
                )
            archive.writestr(_fixture_zip_info("task_ABC_D/training/episode_0000000.npz"), b"training-frame")
            archive.writestr(_fixture_zip_info("task_ABC_D/validation/episode_0000001.npz"), b"validation-frame")

        if directories_first:
            write_directories()
            write_files()
        else:
            write_files()
            write_directories()
    archive_raw = archive_path.read_bytes()
    prepared = prepare_calvin_archive(
        archive_path,
        data_root,
        expected_archive_bytes=len(archive_raw),
        expected_archive_sha256=hashlib.sha256(archive_raw).hexdigest(),
        expected_central_directory=None,
        archive_url=PREFLIGHT.ARCHIVE_URL,
        checksum_url=PREFLIGHT.CHECKSUM_URL,
    )
    manifest_path = prepared.manifest_path
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    archive = manifest["archive"]
    central = archive["central_directory"]
    inventory = archive["member_inventory"]
    monkeypatch.setattr(PREFLIGHT, "ARCHIVE_BYTES", archive["bytes"])
    monkeypatch.setattr(PREFLIGHT, "ARCHIVE_SHA256", archive["sha256"])
    monkeypatch.setattr(PREFLIGHT, "CENTRAL_DIRECTORY_OFFSET", central["offset"])
    monkeypatch.setattr(PREFLIGHT, "CENTRAL_DIRECTORY_BYTES", central["bytes"])
    monkeypatch.setattr(PREFLIGHT, "CENTRAL_DIRECTORY_SHA256", central["sha256"])
    monkeypatch.setattr(PREFLIGHT, "MEMBER_COUNT", inventory["member_count"])
    monkeypatch.setattr(PREFLIGHT, "FILE_MEMBER_COUNT", inventory["file_member_count"])
    monkeypatch.setattr(PREFLIGHT, "DIRECTORY_MEMBER_COUNT", inventory["directory_member_count"])
    monkeypatch.setattr(PREFLIGHT, "NPZ_MEMBER_COUNT", inventory["npz_member_count"])
    monkeypatch.setattr(
        PREFLIGHT,
        "_validate_zip64_tail",
        lambda _archive: {
            "classic_eocd_sentinels": True,
            "record_bytes": PREFLIGHT.ZIP64_TRAILER_BYTES,
            "sha256": PREFLIGHT.ZIP64_TRAILER_SHA256,
            "version_made": PREFLIGHT.ZIP64_VERSION_MADE,
            "version_needed": 45,
            "zip64_eocd_offset": central["offset"] + central["bytes"],
        },
    )

    index_path = prepared.index_path
    os.chmod(index_path, 0o644)
    with sqlite3.connect(index_path) as connection:
        connection.execute("UPDATE metadata SET value='1' WHERE name='central_directory_zip64'")
    os.chmod(index_path, 0o444)
    manifest["archive"]["central_directory"]["zip64"] = True
    manifest["storage"]["member_index"]["bytes"] = index_path.stat().st_size
    manifest["storage"]["member_index"]["sha256"] = hashlib.sha256(index_path.read_bytes()).hexdigest()
    manifest["content_sha256"] = PREFLIGHT._content_sha256(manifest, "content_sha256")
    os.chmod(manifest_path, 0o644)
    manifest_path.write_text(json.dumps(manifest, allow_nan=False, sort_keys=True), encoding="utf-8")
    os.chmod(manifest_path, 0o444)
    return prepared.dataset_root, index_path, manifest


def _rebind_v4_manifest(index_path: Path) -> None:
    manifest_path = index_path.parent / PREFLIGHT.MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["storage"]["member_index"]["bytes"] = index_path.stat().st_size
    manifest["storage"]["member_index"]["sha256"] = hashlib.sha256(index_path.read_bytes()).hexdigest()
    manifest["content_sha256"] = PREFLIGHT._content_sha256(manifest, "content_sha256")
    os.chmod(manifest_path, 0o644)
    manifest_path.write_text(json.dumps(manifest, allow_nan=False, sort_keys=True), encoding="utf-8")
    os.chmod(manifest_path, 0o444)


def _fixture_zip64_tail(*, central_offset: int, central_bytes: int, member_count: int) -> bytes:
    central_end = central_offset + central_bytes
    return b"".join(
        (
            PREFLIGHT._ZIP64_EOCD_PREFIX.pack(PREFLIGHT._ZIP64_EOCD_SIGNATURE, 44),
            PREFLIGHT._ZIP64_EOCD_BODY.pack(
                PREFLIGHT.ZIP64_VERSION_MADE,
                45,
                0,
                0,
                member_count,
                member_count,
                central_bytes,
                central_offset,
            ),
            PREFLIGHT._ZIP64_LOCATOR.pack(PREFLIGHT._ZIP64_LOCATOR_SIGNATURE, 0, central_end, 1),
            PREFLIGHT._EOCD.pack(
                PREFLIGHT._EOCD_SIGNATURE,
                0,
                0,
                0xFFFF,
                0xFFFF,
                central_bytes,
                0xFFFFFFFF,
                0,
            ),
        )
    )


def test_zip64_tail_pins_actual_offset_only_classic_eocd_layout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    central_offset = 101
    central_bytes = 67
    member_count = 70_000
    tail = _fixture_zip64_tail(
        central_offset=central_offset,
        central_bytes=central_bytes,
        member_count=member_count,
    )
    assert len(tail) == PREFLIGHT.ZIP64_TRAILER_BYTES

    class Archive:
        def pread(self, count: int, offset: int) -> bytes:
            assert count == len(tail)
            assert offset == central_offset + central_bytes
            return tail

    monkeypatch.setattr(PREFLIGHT, "ARCHIVE_BYTES", central_offset + central_bytes + len(tail))
    monkeypatch.setattr(PREFLIGHT, "CENTRAL_DIRECTORY_OFFSET", central_offset)
    monkeypatch.setattr(PREFLIGHT, "CENTRAL_DIRECTORY_BYTES", central_bytes)
    monkeypatch.setattr(PREFLIGHT, "MEMBER_COUNT", member_count)
    monkeypatch.setattr(PREFLIGHT, "ZIP64_TRAILER_SHA256", hashlib.sha256(tail).hexdigest())

    report = PREFLIGHT._validate_zip64_tail(Archive())

    assert report == {
        "classic_eocd_sentinels": True,
        "record_bytes": len(tail),
        "sha256": hashlib.sha256(tail).hexdigest(),
        "version_made": PREFLIGHT.ZIP64_VERSION_MADE,
        "version_needed": 45,
        "zip64_eocd_offset": central_offset + central_bytes,
    }


def test_calvin_sequence_digest_is_canonical() -> None:
    sequences = [({"b": 2, "a": 1}, ["task"])]
    expected = hashlib.sha256(
        json.dumps(sequences, allow_nan=False, separators=(",", ":"), sort_keys=True).encode()
    ).hexdigest()
    assert PREFLIGHT.canonical_sequence_sha256(sequences) == expected


def test_calvin_dataset_probe_never_creates_or_downloads(tmp_path: Path) -> None:
    absent = tmp_path / "task_ABC_D"
    result = PREFLIGHT.inspect_dataset(absent, require_dataset=False)
    assert result["present"] is False
    assert not absent.exists()
    with pytest.raises(RuntimeError, match="incomplete"):
        PREFLIGHT.inspect_dataset(absent, require_dataset=True)


def test_calvin_dataset_probe_accepts_minimum_layout(tmp_path: Path) -> None:
    root = tmp_path / "task_ABC_D"
    (root / "training" / "lang_annotations").mkdir(parents=True)
    (root / "validation").mkdir()
    (root / "training" / "lang_annotations" / "auto_lang_ann.npy").touch()
    assert PREFLIGHT.inspect_dataset(root, require_dataset=True)["present"] is True


def test_full_constraints_file_is_the_package_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    constraints = ROOT / "scripts" / "calvin" / "constraints-py38.txt"
    expected, expected_sha256 = PREFLIGHT._parse_constraints(constraints)
    explicit_lines = [
        line
        for line in constraints.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    assert len(expected) == len(explicit_lines)

    from importlib import metadata

    monkeypatch.setattr(metadata, "version", lambda name: expected[name])
    report = PREFLIGHT.verify_packages(constraints)

    assert report["packages"] == expected
    assert report["constraints_sha256"] == expected_sha256


def test_checkout_requires_every_nested_repo_clean_including_untracked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "calvin"
    tacto = source / "calvin_env" / "tacto"
    tacto.mkdir(parents=True)
    revisions = {
        source.resolve(): PREFLIGHT.CALVIN_REVISION,
        (source / "calvin_env").resolve(): PREFLIGHT.CALVIN_ENV_REVISION,
        tacto.resolve(): PREFLIGHT.CALVIN_TACTO_REVISION,
    }
    calls: list[tuple[Path, tuple[str, ...]]] = []

    def git_output(path: Path, *args: str) -> str:
        resolved = path.resolve()
        calls.append((resolved, args))
        if args[0] == "rev-parse":
            return revisions[resolved]
        return ""

    monkeypatch.setattr(PREFLIGHT, "_git_output", git_output)
    result = PREFLIGHT.verify_checkout(source)

    assert set(result) == {"calvin", "calvin_env", "tacto"}
    status_calls = [args for _path, args in calls if args[0] == "status"]
    assert status_calls == [("status", "--porcelain=v1", "--untracked-files=all")] * 3

    def dirty_git_output(path: Path, *args: str) -> str:
        if args[0] == "rev-parse":
            return revisions[path.resolve()]
        return "?? injected.py" if path.resolve() == tacto.resolve() else ""

    monkeypatch.setattr(PREFLIGHT, "_git_output", dirty_git_output)
    with pytest.raises(RuntimeError, match="tacto checkout is not clean"):
        PREFLIGHT.verify_checkout(source)


def test_module_origins_must_be_exact_checkout_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = tmp_path / "calvin"
    agent_path = source / "calvin_models" / "calvin_agent" / "__init__.py"
    env_path = source / "calvin_env" / "calvin_env" / "__init__.py"
    for path in (agent_path, env_path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")
    modules = {
        "calvin_agent": types.SimpleNamespace(__file__=str(agent_path)),
        "calvin_env": types.SimpleNamespace(__file__=str(env_path)),
    }
    monkeypatch.setattr(PREFLIGHT.importlib, "import_module", lambda name: modules[name])
    origins = PREFLIGHT.verify_module_origins(source)
    assert origins["calvin_agent"]["origin"] == str(agent_path.resolve())

    modules["calvin_env"] = types.SimpleNamespace(__file__=str(tmp_path / "shadow" / "__init__.py"))
    with pytest.raises(RuntimeError, match="unexpected path"):
        PREFLIGHT.verify_module_origins(source)


def test_official_yaml_attestation_uses_fixed_raw_byte_hashes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = tmp_path / "calvin"
    annotation = source / "calvin_models" / "conf" / "annotations" / "new_playtable_validation.yaml"
    oracle = source / "calvin_models" / "conf" / "callbacks" / "rollout" / "tasks" / "new_playtable_tasks.yaml"
    annotation.parent.mkdir(parents=True)
    oracle.parent.mkdir(parents=True)
    annotation.write_bytes(b"annotation: raw bytes\n")
    oracle.write_bytes(b"oracle: raw bytes\n")
    monkeypatch.setattr(PREFLIGHT, "VALIDATION_ANNOTATIONS_SHA256", hashlib.sha256(annotation.read_bytes()).hexdigest())
    monkeypatch.setattr(PREFLIGHT, "TASK_ORACLE_SHA256", hashlib.sha256(oracle.read_bytes()).hexdigest())

    identities = PREFLIGHT.verify_official_yaml_files(source)

    assert identities["validation_annotations"]["path"] == str(annotation.resolve())
    oracle.write_bytes(b"oracle: parsed the same, raw identity changed\n")
    with pytest.raises(RuntimeError, match="task_oracle YAML hash mismatch"):
        PREFLIGHT.verify_official_yaml_files(source)


def test_runtime_attestation_binds_all_standalone_evaluator_sources(tmp_path: Path) -> None:
    expected: dict[str, str] = {}
    for index, name in enumerate(PREFLIGHT._SCRIPT_SOURCE_NAMES):
        path = tmp_path / name
        path.write_bytes(f"source-{index}".encode())
        expected[name] = hashlib.sha256(path.read_bytes()).hexdigest()

    identities = PREFLIGHT.evaluator_source_identities(tmp_path)

    assert set(identities) == set(PREFLIGHT._SCRIPT_SOURCE_NAMES)
    assert {name: identity["sha256"] for name, identity in identities.items()} == expected


@pytest.mark.parametrize("mutated_name", PREFLIGHT._SCRIPT_SOURCE_NAMES)
def test_preflight_import_snapshot_rejects_each_evaluator_source_mutation(
    tmp_path: Path,
    mutated_name: str,
) -> None:
    for name in PREFLIGHT._SCRIPT_SOURCE_NAMES:
        (tmp_path / name).write_text(f"source:{name}\n", encoding="utf-8")
    snapshot = PREFLIGHT.evaluator_source_identities(tmp_path)
    (tmp_path / mutated_name).write_text("mutated source\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="changed after import-time snapshot"):
        PREFLIGHT.require_evaluator_sources_unchanged(snapshot)


def test_attestation_file_identity_rejects_symlink(tmp_path: Path) -> None:
    target = tmp_path / "target.yaml"
    target.write_text("value: 1\n", encoding="utf-8")
    link = tmp_path / "linked.yaml"
    link.symlink_to(target)

    with pytest.raises(RuntimeError, match="unavailable"):
        PREFLIGHT._file_identity(link)


@pytest.mark.parametrize(
    ("mutated_name", "mutated_value"),
    [
        *((name, "mutated") for name in sorted(PREFLIGHT._CANONICAL_STATIC_ENVIRONMENT)),
        *((name, "/injected") for name in PREFLIGHT._FORBIDDEN_EVALUATOR_ENVIRONMENT),
        ("PATH", "/injected/bin"),
    ],
)
def test_canonical_runtime_contract_rejects_every_environment_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutated_name: str,
    mutated_value: str,
) -> None:
    source_root = tmp_path / "source"
    cache_root = tmp_path / "cache"
    source_root.mkdir()
    cache_root.mkdir()
    for name in list(PREFLIGHT.os.environ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("CALVIN_SOURCE_ROOT", str(source_root.resolve()))
    monkeypatch.setenv("DUO_VLA_CACHE_ROOT", str(cache_root.resolve()))
    expected = PREFLIGHT.canonical_evaluator_environment()
    for name, value in expected.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setenv(mutated_name, mutated_value)

    with pytest.raises(RuntimeError, match="canonical env-i contract"):
        PREFLIGHT.require_canonical_evaluator_runtime()


def test_legacy_v3_dataset_attestation_is_explicit_parity_only(tmp_path: Path) -> None:
    root, _index_path, manifest = _write_dataset_contract(tmp_path)

    with pytest.raises(RuntimeError, match="parity-only"):
        PREFLIGHT.verify_dataset_identity(root)
    identity = PREFLIGHT.verify_dataset_identity(root, allow_legacy_v3=True)

    assert identity["manifest"]["content_sha256"] == manifest["content_sha256"]
    assert identity["storage_mode"] == "legacy-extracted-parity-only"
    assert set(identity["validation_critical_files"]) == set(PREFLIGHT.VALIDATION_CRITICAL_FILES)
    assert len(identity["member_index"]["validation_rows_sha256"]) == 64

    (root / "validation" / ".hydra" / "merged_config.yaml").write_bytes(b"changed after extraction")
    with pytest.raises(RuntimeError, match="validation critical file differs"):
        PREFLIGHT.verify_dataset_identity(root, allow_legacy_v3=True)


def test_legacy_v3_rejects_a_rehashed_index_with_wrong_validation_row(tmp_path: Path) -> None:
    root, index_path, manifest = _write_dataset_contract(tmp_path)
    relative = "validation/.hydra/merged_config.yaml"
    with sqlite3.connect(index_path) as connection:
        connection.execute("UPDATE members SET sha256 = ? WHERE path = ?", ("f" * 64, relative))

    extraction = manifest["extraction"]
    assert isinstance(extraction, dict)
    member_index = extraction["member_index"]
    assert isinstance(member_index, dict)
    member_index["bytes"] = index_path.stat().st_size
    member_index["sha256"] = hashlib.sha256(index_path.read_bytes()).hexdigest()
    manifest["content_sha256"] = PREFLIGHT._content_sha256(manifest, "content_sha256")
    (tmp_path / "task_ABC_D.manifest.json").write_text(
        json.dumps(manifest, allow_nan=False, sort_keys=True),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="validation critical file differs"):
        PREFLIGHT.verify_dataset_identity(root, allow_legacy_v3=True)


def test_v4_attestation_binds_archive_index_and_projected_validation_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, _index_path, manifest = _write_v4_dataset_contract(tmp_path, monkeypatch)

    identity = PREFLIGHT.verify_dataset_identity(root)

    assert identity["manifest"]["content_sha256"] == manifest["content_sha256"]
    assert identity["storage"]["mode"] == "archive-direct"
    assert identity["storage"]["reader_schema"] == PREFLIGHT.ARCHIVE_READER_SCHEMA
    assert identity["member_index"]["schema"] == PREFLIGHT.MEMBER_INDEX_SCHEMA
    assert identity["critical_files"] == manifest["critical_files"]
    assert set(identity["validation_critical_files"]) == set(PREFLIGHT.VALIDATION_CRITICAL_FILES)
    assert len(identity["member_index_attestation"]["sqlite_schema_content_sha256"]) == 64
    assert identity["calvin_identity"]["member_index"] == identity["member_index"]
    assert identity["calvin_identity"]["metadata_files"] == list(PREFLIGHT.TRAINING_METADATA_FILES)
    storage_identity = CalvinStorageIdentity.from_dict(identity["storage_identity"])
    assert storage_identity.content_sha256 == identity["storage_identity_sha256"]
    assert (
        _normalization_dataset_identity(
            storage_identity,
            metadata_sha256=identity["metadata_sha256"],
        )
        == identity["calvin_identity"]
    )

    changed = json.loads(json.dumps(identity))
    changed["calvin_identity"]["dataset_manifest_sha256"] = "f" * 64
    with pytest.raises(RuntimeError, match="aliases differ"):
        PREFLIGHT.validate_official_dataset_identity(changed)

    changed = json.loads(json.dumps(identity))
    changed["validation_critical_files"]["validation/.hydra/merged_config.yaml"]["sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="projected/archive metadata identities differ"):
        PREFLIGHT.validate_official_dataset_identity(changed)

    changed = json.loads(json.dumps(identity))
    changed["critical_files"]["training/scene_info.npy"]["bytes"] += 1
    with pytest.raises(RuntimeError, match="manifest content hash differs"):
        PREFLIGHT.validate_official_dataset_identity(changed)


def test_v4_attestation_accepts_virtual_root_and_late_directory_declarations(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, _index_path, _manifest = _write_v4_dataset_contract(
        tmp_path,
        monkeypatch,
        include_root=False,
        directories_first=False,
    )

    identity = PREFLIGHT.verify_dataset_identity(root)

    assert identity["storage_mode"] == "archive-direct"
    assert identity["member_inventory"]["directory_member_count"] == PREFLIGHT.DIRECTORY_MEMBER_COUNT


def test_v4_official_attestation_reads_only_critical_archive_payloads(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, _index_path, _manifest = _write_v4_dataset_contract(tmp_path, monkeypatch)
    authenticated: list[str] = []
    implementation = PREFLIGHT._authenticate_indexed_archive_member

    def record(archive: object, row: tuple[object, ...], relative: str) -> dict[str, object]:
        authenticated.append(relative)
        return implementation(archive, row, relative)

    monkeypatch.setattr(PREFLIGHT, "_authenticate_indexed_archive_member", record)

    PREFLIGHT.verify_dataset_identity(root)

    assert authenticated == list(PREFLIGHT.DATASET_CRITICAL_FILES)
    assert all(not relative.endswith(".npz") for relative in authenticated)


def test_v4_rehashed_index_cannot_forge_episode_path_split_global_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, index_path, _manifest = _write_v4_dataset_contract(tmp_path, monkeypatch)
    os.chmod(index_path, 0o644)
    with sqlite3.connect(index_path) as connection:
        connection.execute(
            "UPDATE members SET split='validation',global_index=999 WHERE path='training/episode_0000000.npz'"
        )
    os.chmod(index_path, 0o444)
    _rebind_v4_manifest(index_path)

    with pytest.raises(RuntimeError, match="split/global identity"):
        PREFLIGHT.verify_dataset_identity(root)


def test_v4_rehashed_index_cannot_weaken_critical_deflate_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, index_path, _manifest = _write_v4_dataset_contract(tmp_path, monkeypatch)
    os.chmod(index_path, 0o644)
    with sqlite3.connect(index_path) as connection:
        connection.execute(
            "UPDATE members SET method=0,version_needed=10 WHERE path=?",
            (PREFLIGHT.DATASET_CRITICAL_FILES[0],),
        )
    os.chmod(index_path, 0o444)
    _rebind_v4_manifest(index_path)

    with pytest.raises(RuntimeError, match="DEFLATE contract"):
        PREFLIGHT.verify_dataset_identity(root)


def test_v4_rehashed_critical_crc_must_match_the_local_header(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, index_path, _manifest = _write_v4_dataset_contract(tmp_path, monkeypatch)
    os.chmod(index_path, 0o644)
    with sqlite3.connect(index_path) as connection:
        connection.execute(
            "UPDATE members SET crc32=(crc32 + 1) % 4294967296 WHERE path=?",
            (PREFLIGHT.DATASET_CRITICAL_FILES[0],),
        )
    os.chmod(index_path, 0o444)
    _rebind_v4_manifest(index_path)

    with pytest.raises(RuntimeError, match="local-header CRC/size"):
        PREFLIGHT.verify_dataset_identity(root)


def test_v4_manifest_schema_and_root_inventory_cannot_cross_select_branches(tmp_path: Path) -> None:
    root, _index, manifest = _write_dataset_contract(tmp_path)
    manifest["schema"] = PREFLIGHT.DATASET_MANIFEST_SCHEMA
    manifest["content_sha256"] = PREFLIGHT._content_sha256(manifest, "content_sha256")
    (tmp_path / PREFLIGHT.MANIFEST_NAME).write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(RuntimeError, match="unsupported or ambiguous"):
        PREFLIGHT.verify_dataset_identity(root, allow_legacy_v3=True)


def test_v4_attestation_rejects_a_symlinked_data_root_ancestor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real = tmp_path / "real"
    real.mkdir()
    root, _index_path, _manifest = _write_v4_dataset_contract(real, monkeypatch)
    alias = tmp_path / "alias"
    alias.symlink_to(root.parent, target_is_directory=True)

    with pytest.raises(RuntimeError, match="no-follow open attestation directory"):
        PREFLIGHT.verify_dataset_identity(alias / "task_ABC_D")


def test_runtime_attestation_finishes_raw_checks_before_importing_calvin(monkeypatch: pytest.MonkeyPatch) -> None:
    events: list[str] = []
    monkeypatch.setenv("CALVIN_SOURCE_ROOT", "/source")

    monkeypatch.setattr(PREFLIGHT, "runtime_identity", lambda: events.append("runtime") or {})
    monkeypatch.setattr(PREFLIGHT, "verify_checkout", lambda _root: events.append("checkout") or {})
    monkeypatch.setattr(PREFLIGHT, "verify_packages", lambda _path: events.append("packages") or {})
    monkeypatch.setattr(PREFLIGHT, "verify_official_yaml_files", lambda _root: events.append("yaml-bytes") or {})
    monkeypatch.setattr(
        PREFLIGHT,
        "evaluator_source_identities",
        lambda _path: events.append("sources") or PREFLIGHT._IMPORT_EVALUATOR_SOURCE_IDENTITIES,
    )
    monkeypatch.setattr(PREFLIGHT, "verify_module_origins", lambda _root: events.append("module-imports") or {})

    attestation = PREFLIGHT.build_runtime_attestation(Path("/source"), script_dir=PREFLIGHT._SCRIPT_DIR)

    assert events == ["runtime", "checkout", "packages", "yaml-bytes", "sources", "module-imports"]
    assert attestation["content_sha256"] == PREFLIGHT._content_sha256(attestation, "content_sha256")


def test_official_attestation_authenticates_dataset_before_calvin_module_imports(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    runtime: dict[str, object] = {
        "schema": PREFLIGHT.RUNTIME_ATTESTATION_SCHEMA,
        "sources": PREFLIGHT._IMPORT_EVALUATOR_SOURCE_IDENTITIES,
    }
    runtime["content_sha256"] = PREFLIGHT._content_sha256(runtime, "content_sha256")

    dataset = {
        "storage": {
            "materialized_files": list(PREFLIGHT.DATASET_CRITICAL_FILES),
            "mode": "archive-direct",
            "reader_schema": PREFLIGHT.ARCHIVE_READER_SCHEMA,
        }
    }
    monkeypatch.setattr(
        PREFLIGHT,
        "verify_dataset_identity",
        lambda _root: events.append("dataset-bytes") or dataset,
    )
    monkeypatch.setattr(
        PREFLIGHT,
        "build_runtime_attestation",
        lambda _root, script_dir=None: events.append("runtime-and-module-origin") or runtime,
    )

    PREFLIGHT.build_official_attestation(Path("/source"), Path("/dataset"))

    assert events == ["dataset-bytes", "runtime-and-module-origin"]


@pytest.mark.parametrize(
    "script",
    ["checkout.sh", "bootstrap_env.sh", "run_official_evaluator.sh", "run_preflight.sh"],
)
def test_calvin_shell_scripts_parse(script: str) -> None:
    subprocess.run(["bash", "-n", str(ROOT / "scripts" / "calvin" / script)], check=True)


def test_unified_evaluator_launcher_owns_closed_runtime_and_egl_contract() -> None:
    scripts = ROOT / "scripts" / "calvin"
    launcher = (scripts / "run_official_evaluator.sh").read_text(encoding="utf-8")
    for fragment in (
        "/usr/bin/env -i",
        "CUDA_VISIBLE_DEVICES=0",
        "EGL_PLATFORM=surfaceless",
        "EGL_VISIBLE_DEVICES=0",
        "PYOPENGL_PLATFORM=egl",
        "PYTHONHASHSEED=0",
        "PYTHONNOUSERSITE=1",
        '"${venv_root}/bin/python" -B',
        "preflight)",
        "infrastructure)",
        "official-score)",
    ):
        assert fragment in launcher
    assert "fallback" in launcher
    assert 'exec "$script_dir/run_official_evaluator.sh" preflight' in (scripts / "run_preflight.sh").read_text(
        encoding="utf-8"
    )
    assert '"$script_dir/run_official_evaluator.sh" preflight' in (scripts / "bootstrap_env.sh").read_text(
        encoding="utf-8"
    )


def test_calvin_revisions_are_consistent() -> None:
    revisions = dict(
        line.split("=", 1) for line in (ROOT / "scripts" / "calvin" / "revisions.env").read_text().splitlines() if line
    )
    assert revisions["CALVIN_REVISION"] == PREFLIGHT.CALVIN_REVISION
    assert revisions["CALVIN_ENV_REVISION"] == PREFLIGHT.CALVIN_ENV_REVISION
    assert revisions["CALVIN_TACTO_REVISION"] == PREFLIGHT.CALVIN_TACTO_REVISION
    assert revisions["CALVIN_SEQUENCE_SHA256"] == PREFLIGHT.SEQUENCE_SHA256
