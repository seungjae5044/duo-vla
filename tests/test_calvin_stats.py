from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import subprocess
import sys
import textwrap
import zipfile
from pathlib import Path

import numpy as np
import pytest
import torch

import duo_vla.data.calvin_stats as calvin_stats_module
from duo_vla.data.calvin import CalvinAnnotation, CalvinEpisode
from duo_vla.data.calvin_archive import CalvinArchiveValidationError
from duo_vla.data.calvin_stats import (
    CALVIN_ABC_D_ARCHIVE_BYTES,
    CALVIN_ABC_D_ARCHIVE_SHA256,
    CALVIN_ABC_D_ARCHIVE_URL,
    CALVIN_DATASET_CRITICAL_FILES,
    CALVIN_DATASET_MANIFEST_SCHEMA,
    CALVIN_DATASET_MANIFEST_SCHEMA_V3,
    CALVIN_STATS_SCHEMA,
    CALVIN_STORAGE_IDENTITY_SCHEMA,
    CALVIN_STORAGE_MODE_ARCHIVE_DIRECT,
    AuthenticatedCalvinDatasetGeneration,
    CalvinStorageIdentity,
    _authenticated_generation_payload,
    _verify_member_index_central_directory,
    _verify_member_index_schema,
    compute_calvin_normalization_artifact,
    load_calvin_dataset_manifest,
    load_calvin_state_normalizer,
    save_calvin_normalization_artifact,
    verify_calvin_dataset_generation,
)


def test_module_cli_preserves_authenticated_generation_type_identity() -> None:
    project_root = Path(__file__).resolve().parents[1]
    probe = textwrap.dedent(
        """
        import json
        import runpy
        import sys

        observed = {}

        def trace(frame, event, _argument):
            if (
                event == "call"
                and frame.f_code.co_name == "main"
                and frame.f_code.co_filename.endswith("calvin_stats.py")
            ):
                values = frame.f_globals
                returned_type = values["AuthenticatedCalvinDatasetGeneration"]
                generation = returned_type(
                    training_root="/synthetic/task_ABC_D/training",
                    storage=object(),
                    archive_file_identity=None,
                    metadata_sha256="0" * 64,
                    critical_files=(),
                    content_sha256="1" * 64,
                )

                def authenticate(_root):
                    return {}, generation

                class ProbeDataset:
                    def __init__(self, *_args, authenticated_generation=None, **_kwargs):
                        from duo_vla.data.calvin_stats import (
                            AuthenticatedCalvinDatasetGeneration as canonical_type,
                        )

                        observed.update(
                            {
                                "cli_main_module": values["__name__"],
                                "generation_isinstance": isinstance(authenticated_generation, canonical_type),
                                "returned_type_is_canonical": returned_type is canonical_type,
                            }
                        )
                        if not observed["generation_isinstance"]:
                            raise TypeError(
                                "authenticated_generation must be an AuthenticatedCalvinDatasetGeneration"
                            )

                    def __enter__(self):
                        return self

                    def __exit__(self, *_args):
                        return None

                values["authenticate_calvin_dataset_generation"] = authenticate
                values["CalvinNpzDataset"] = ProbeDataset
                values["compute_calvin_normalization_artifact"] = lambda *_args, **_kwargs: observed
                values["save_calvin_normalization_artifact"] = lambda *_args, **_kwargs: None
                values["load_calvin_state_normalizer"] = lambda *_args, **_kwargs: (None, observed)
            return trace

        sys.settrace(trace)
        sys.argv = ["calvin_stats", "/synthetic/task_ABC_D/training", "/synthetic/normalization.json"]
        runpy.run_module("duo_vla.data.calvin_stats", run_name="__main__", alter_sys=True)
        """
    )
    completed = subprocess.run(
        [sys.executable, "-B", "-c", probe],
        cwd=project_root,
        env={**os.environ, "PYTHONPATH": str(project_root / "src")},
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout) == {
        "cli_main_module": "duo_vla.data.calvin_stats",
        "generation_isinstance": True,
        "returned_type_is_canonical": True,
    }


class _FakeDataset:
    def __init__(self, root: Path) -> None:
        self.root = root
        (root / "lang_annotations").mkdir(parents=True)
        (root / "ep_start_end_ids.npy").write_bytes(b"episode-metadata")
        (root / "lang_annotations" / "auto_lang_ann.npy").write_bytes(b"language-metadata")
        (root / "scene_info.npy").write_bytes(b"scene-metadata")
        (root / ".hydra").mkdir()
        (root / ".hydra" / "merged_config.yaml").write_bytes(b"config-metadata")
        validation = root.parent / "validation"
        (validation / ".hydra").mkdir(parents=True)
        (validation / "ep_start_end_ids.npy").write_bytes(b"validation-episode-metadata")
        (validation / ".hydra" / "merged_config.yaml").write_bytes(b"validation-config-metadata")
        archive = root.parent.with_suffix(".zip")
        with archive.open("wb") as handle:
            handle.truncate(CALVIN_ABC_D_ARCHIVE_BYTES)
        critical = {}
        for relative in CALVIN_DATASET_CRITICAL_FILES:
            critical[relative] = hashlib.sha256((root.parent / relative).read_bytes()).hexdigest()
        manifest = {
            "archive": {
                "bytes": CALVIN_ABC_D_ARCHIVE_BYTES,
                "member_inventory": {
                    "compressed_bytes": 1,
                    "file_member_count": 1,
                    "member_count": 1,
                    "npz_member_count": 1,
                    "sha256": "3" * 64,
                    "uncompressed_bytes": 1,
                },
                "sha256": CALVIN_ABC_D_ARCHIVE_SHA256,
                "uncompressed_bytes": 1,
                "url": CALVIN_ABC_D_ARCHIVE_URL,
            },
            "checksum_url": "http://calvin.cs.uni-freiburg.de/dataset/sha256sum.txt",
            "critical_files": critical,
            "dataset": "task_ABC_D",
            "extraction": {
                "file_members_verified": 1,
                "member_index": {
                    "bytes": 1,
                    "path": "task_ABC_D.members.sqlite3",
                    "schema": "duo-vla-calvin-member-index-v1",
                    "sha256": "4" * 64,
                },
                "verification": "size-and-crc32-against-every-pinned-zip-member",
            },
            "schema": CALVIN_DATASET_MANIFEST_SCHEMA_V3,
        }
        canonical = json.dumps(manifest, allow_nan=False, separators=(",", ":"), sort_keys=True).encode()
        manifest["content_sha256"] = hashlib.sha256(canonical).hexdigest()
        root.parent.with_name(f"{root.parent.name}.manifest.json").write_text(
            json.dumps(manifest, allow_nan=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        self.episodes = tuple(CalvinEpisode(index, 2 * index, 2 * index + 1) for index in range(10))
        self.annotations = tuple(
            CalvinAnnotation(
                annotation_index=2 * episode + task_offset,
                episode_index=episode,
                global_start=2 * episode,
                global_end_exclusive=2 * episode + 2,
                instruction=f"instruction {episode} {task_offset}",
                task=f"task_{task_offset}",
            )
            for episode in range(10)
            for task_offset in range(2)
        )
        self.tasks = ("task_0", "task_1")

    @staticmethod
    def _read_frame(index: int) -> dict[str, np.ndarray]:
        robot = np.zeros(15, dtype=np.float32)
        robot[:7] = np.arange(7, dtype=np.float32) + index
        robot[14] = -1.0 if index % 2 else 1.0
        action = np.zeros(7, dtype=np.float32)
        action[:6] = (index - 10) / 20
        action[6] = robot[14]
        return {"robot_obs": robot, "rel_actions": action}

    @classmethod
    def read_state_action(cls, index: int) -> tuple[np.ndarray, np.ndarray]:
        frame = cls._read_frame(index)
        return frame["robot_obs"], frame["rel_actions"]


class _PhysicalDataset(_FakeDataset):
    storage_mode = CALVIN_STORAGE_MODE_ARCHIVE_DIRECT

    def __init__(self, root: Path, *, order: tuple[int, ...] | None = None) -> None:
        super().__init__(root)
        self.order = order or (17, 2, 19, 0, 8, 5, 14, 1, 12, 7, 18, 3, 10, 4, 16, 6, 15, 9, 13, 11)

    def iter_state_actions_physical(self):
        for index in self.order:
            frame = self._read_frame(index)
            yield index, frame["robot_obs"], frame["rel_actions"]


def _v4_generation(root: Path) -> AuthenticatedCalvinDatasetGeneration:
    storage_payload = {
        "archive": {
            "bytes": CALVIN_ABC_D_ARCHIVE_BYTES,
            "path": "task_ABC_D.zip",
            "sha256": CALVIN_ABC_D_ARCHIVE_SHA256,
            "url": CALVIN_ABC_D_ARCHIVE_URL,
        },
        "central_directory": {
            "bytes": 100,
            "entries": 30,
            "offset": 200,
            "sha256": "5" * 64,
            "zip64": True,
        },
        "checksum_url": "http://calvin.cs.uni-freiburg.de/dataset/sha256sum.txt",
        "manifest": {
            "content_sha256": "1" * 64,
            "file_sha256": "2" * 64,
            "schema": CALVIN_DATASET_MANIFEST_SCHEMA,
        },
        "member_index": {
            "bytes": 123,
            "path": "task_ABC_D.members-v2.sqlite3",
            "schema": "duo-vla-calvin-member-index-v2",
            "sha256": "3" * 64,
        },
        "member_inventory": {
            "compressed_bytes": 1,
            "directory_member_count": 1,
            "file_member_count": 29,
            "member_count": 30,
            "npz_member_count": 20,
            "sha256": "4" * 64,
            "uncompressed_bytes": 1,
        },
        "mode": CALVIN_STORAGE_MODE_ARCHIVE_DIRECT,
        "reader_schema": "duo-vla-calvin-archive-reader-v1",
        "schema": CALVIN_STORAGE_IDENTITY_SCHEMA,
    }
    storage_payload["content_sha256"] = hashlib.sha256(
        json.dumps(storage_payload, allow_nan=False, separators=(",", ":"), sort_keys=True).encode()
    ).hexdigest()
    storage = CalvinStorageIdentity.from_dict(storage_payload)
    critical = tuple((relative, "6" * 64) for relative in sorted(CALVIN_DATASET_CRITICAL_FILES))
    return AuthenticatedCalvinDatasetGeneration(
        training_root=str(root.resolve()),
        storage=storage,
        archive_file_identity=(
            ("ctime_ns", 1),
            ("device", 1),
            ("inode", 1),
            ("link_count", 1),
            ("mode", 0o100444),
            ("mtime_ns", 1),
            ("size", CALVIN_ABC_D_ARCHIVE_BYTES),
        ),
        metadata_sha256="7" * 64,
        critical_files=critical,
        content_sha256="8" * 64,
    )


def test_member_index_rows_are_bound_to_zip_paths_sizes_and_crc(tmp_path: Path) -> None:
    archive_path = tmp_path / "tiny.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("task_ABC_D/training/episode_0000000.npz", b"frame-zero")
        archive.writestr("task_ABC_D/validation/episode_0000000.npz", b"frame-validation")
    infos = {}
    with zipfile.ZipFile(archive_path) as archive:
        infos = {info.filename.removeprefix("task_ABC_D/"): info for info in archive.infolist()}
    database = tmp_path / "members.sqlite3"
    with sqlite3.connect(database) as connection:
        connection.execute(
            "CREATE TABLE members(path TEXT PRIMARY KEY, bytes INTEGER NOT NULL, "
            "crc32 INTEGER NOT NULL, sha256 TEXT NOT NULL) WITHOUT ROWID"
        )
        for relative, info in infos.items():
            connection.execute(
                "INSERT INTO members(path, bytes, crc32, sha256) VALUES (?, ?, ?, ?)",
                (relative, info.file_size, info.CRC, "a" * 64),
            )
        _verify_member_index_central_directory(archive_path, connection)
        connection.execute(
            "UPDATE members SET crc32 = crc32 + 1 WHERE path = ?",
            ("training/episode_0000000.npz",),
        )
        with pytest.raises(ValueError, match="central directory"):
            _verify_member_index_central_directory(archive_path, connection)


def test_member_index_rejects_extra_sqlite_objects() -> None:
    with sqlite3.connect(":memory:") as connection:
        connection.execute(
            "CREATE TABLE members(path TEXT PRIMARY KEY, bytes INTEGER NOT NULL, "
            "crc32 INTEGER NOT NULL, sha256 TEXT NOT NULL) WITHOUT ROWID"
        )
        connection.execute("CREATE TABLE metadata(name TEXT PRIMARY KEY, value TEXT NOT NULL) WITHOUT ROWID")
        _verify_member_index_schema(connection)
        connection.execute("CREATE VIEW covert_members AS SELECT * FROM members")
        with pytest.raises(ValueError, match="SQLite objects"):
            _verify_member_index_schema(connection)


def test_authenticated_generation_prevents_fast_path_manifest_laundering(tmp_path: Path) -> None:
    dataset = _FakeDataset(tmp_path / "task_ABC_D" / "training")
    member_index = tmp_path / "task_ABC_D.members.sqlite3"
    member_index.write_bytes(b"authenticated-index")
    manifest_path = tmp_path / "task_ABC_D.manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["extraction"]["member_index"]["bytes"] = member_index.stat().st_size
    manifest["extraction"]["member_index"]["sha256"] = hashlib.sha256(member_index.read_bytes()).hexdigest()
    manifest.pop("content_sha256")
    manifest["content_sha256"] = hashlib.sha256(
        json.dumps(manifest, allow_nan=False, separators=(",", ":"), sort_keys=True).encode()
    ).hexdigest()
    manifest_path.write_text(json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="fixture/parity-only"):
        load_calvin_dataset_manifest(dataset.root, verify_archive=False)
    loaded = load_calvin_dataset_manifest(dataset.root, verify_archive=False, allow_legacy_v3=True)
    capability = AuthenticatedCalvinDatasetGeneration.from_dict(
        _authenticated_generation_payload(
            dataset.root.resolve(),
            loaded,
            manifest_file_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        )
    )

    assert verify_calvin_dataset_generation(dataset.root, capability) == loaded
    member_index.write_bytes(b"attacker-rewritten!")
    assert member_index.stat().st_size == capability.member_index_bytes
    with pytest.raises(ValueError, match="member index differs"):
        verify_calvin_dataset_generation(dataset.root, capability)


def test_manifest_dispatch_never_falls_back_from_declared_v4_to_present_frames(tmp_path: Path) -> None:
    dataset = _FakeDataset(tmp_path / "task_ABC_D" / "training")
    manifest_path = tmp_path / "task_ABC_D.manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["schema"] = CALVIN_DATASET_MANIFEST_SCHEMA
    manifest.pop("content_sha256")
    manifest["content_sha256"] = hashlib.sha256(
        json.dumps(manifest, allow_nan=False, separators=(",", ":"), sort_keys=True).encode()
    ).hexdigest()
    manifest_path.write_text(json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8")
    assert dataset.root.joinpath("episode_0000000.npz").exists() is False
    dataset.root.joinpath("episode_0000000.npz").write_bytes(b"misleading extracted frame")

    with pytest.raises(CalvinArchiveValidationError, match="archive-direct manifest root fields differ"):
        load_calvin_dataset_manifest(dataset.root, verify_archive=False)


def test_calvin_stats_use_train_episodes_once_and_leave_actions_in_official_units(tmp_path: Path) -> None:
    dataset = _FakeDataset(tmp_path / "task_ABC_D" / "training")
    artifact = compute_calvin_normalization_artifact(  # type: ignore[arg-type]
        dataset,
        validation_fraction=0.2,
        split_seed=7,
        verify_archive=False,
        allow_legacy_v3=True,
    )

    train_episodes = artifact["split"]["train_episode_indices"]
    train_indices = [index for episode in train_episodes for index in (2 * episode, 2 * episode + 1)]
    source = np.asarray([np.arange(7) + index for index in train_indices], dtype=np.float32)
    expected = np.quantile(source, (0.01, 0.99), axis=0, method="linear")
    np.testing.assert_allclose(artifact["state"]["q01"], expected[0])
    np.testing.assert_allclose(artifact["state"]["q99"], expected[1])
    assert artifact["counts"]["training_frames"] == len(train_indices)
    assert artifact["action"]["transform"] == "identity_official_scaled_rel_actions"
    assert artifact["algorithm"]["actions_re_normalized"] is False


def test_v4_physical_scan_places_canonical_ordinals_and_matches_v3_quantiles(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    v3 = _FakeDataset(tmp_path / "v3" / "task_ABC_D" / "training")
    v4 = _PhysicalDataset(tmp_path / "v4" / "task_ABC_D" / "training")
    generation = _v4_generation(v4.root)
    monkeypatch.setattr(
        calvin_stats_module,
        "verify_calvin_dataset_generation",
        lambda _root, received: {"content_sha256": received.storage.manifest_content_sha256},
    )

    v3_artifact = compute_calvin_normalization_artifact(  # type: ignore[arg-type]
        v3,
        validation_fraction=0.2,
        split_seed=7,
        verify_archive=False,
        allow_legacy_v3=True,
    )
    v4_artifact = compute_calvin_normalization_artifact(  # type: ignore[arg-type]
        v4,
        validation_fraction=0.2,
        split_seed=7,
        authenticated_generation=generation,
    )

    assert v4_artifact["schema"] == CALVIN_STATS_SCHEMA
    assert v4_artifact["dataset"]["storage_mode"] == CALVIN_STORAGE_MODE_ARCHIVE_DIRECT
    assert v4_artifact["dataset"]["storage_identity_sha256"] == generation.storage.content_sha256
    assert v4_artifact["algorithm"]["selected_frame_exact_once_bitmap"] is True
    assert "data_offset ascending" in v4_artifact["algorithm"]["frame_order"]
    assert v4_artifact["split"] == v3_artifact["split"]
    assert v4_artifact["counts"] == v3_artifact["counts"]
    assert v4_artifact["state"] == v3_artifact["state"]
    assert v4_artifact["action"] == v3_artifact["action"]


@pytest.mark.parametrize(
    ("order", "message"),
    [
        ((*range(20), *range(20)), "duplicate selected frame"),
        (tuple(range(0, 20, 2)), "did not visit every selected frame exactly once"),
    ],
)
def test_v4_physical_scan_requires_exactly_once_selected_frames(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    order: tuple[int, ...],
    message: str,
) -> None:
    dataset = _PhysicalDataset(tmp_path / "task_ABC_D" / "training", order=order)
    generation = _v4_generation(dataset.root)
    monkeypatch.setattr(
        calvin_stats_module,
        "verify_calvin_dataset_generation",
        lambda _root, received: {"content_sha256": received.storage.manifest_content_sha256},
    )

    with pytest.raises(ValueError, match=message):
        compute_calvin_normalization_artifact(  # type: ignore[arg-type]
            dataset,
            validation_fraction=0.2,
            split_seed=7,
            authenticated_generation=generation,
        )


def test_calvin_stats_round_trip_and_binary_state_channel(tmp_path: Path) -> None:
    artifact = compute_calvin_normalization_artifact(
        _FakeDataset(tmp_path / "task_ABC_D" / "training"),  # type: ignore[arg-type]
        validation_fraction=0.2,
        verify_archive=False,
        allow_legacy_v3=True,
    )
    output = tmp_path / "normalization.json"
    save_calvin_normalization_artifact(output, artifact)

    normalizer, loaded = load_calvin_state_normalizer(
        output,
        training_root=tmp_path / "task_ABC_D" / "training",
        verify_archive=False,
        allow_legacy_v3=True,
    )

    assert loaded == artifact
    assert normalizer.action_dim == 8
    assert normalizer.resolved_gripper_index == 7
    values = torch.tensor([[5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, -1.0]])
    assert normalizer.normalize(values)[0, -1].item() == -1.0

    with pytest.raises(ValueError, match="archive-direct v4 production storage"):
        load_calvin_state_normalizer(output)


def test_calvin_stats_publication_never_replaces_an_existing_artifact(tmp_path: Path) -> None:
    artifact = compute_calvin_normalization_artifact(
        _FakeDataset(tmp_path / "task_ABC_D" / "training"),  # type: ignore[arg-type]
        validation_fraction=0.2,
        verify_archive=False,
        allow_legacy_v3=True,
    )
    output = tmp_path / "normalization.json"
    output.write_bytes(b"existing immutable artifact\n")

    with pytest.raises(FileExistsError, match="already exists"):
        save_calvin_normalization_artifact(output, artifact)

    assert output.read_bytes() == b"existing immutable artifact\n"
    assert not tuple(tmp_path.glob(".normalization.json.tmp-*"))


def test_calvin_stats_token_generation_failure_opens_no_parent_fd(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact = compute_calvin_normalization_artifact(
        _FakeDataset(tmp_path / "task_ABC_D" / "training"),  # type: ignore[arg-type]
        validation_fraction=0.2,
        verify_archive=False,
        allow_legacy_v3=True,
    )
    opened: list[object] = []
    original_open = calvin_stats_module.os.open

    def tracked_open(*args: object, **kwargs: object) -> int:
        opened.append(args[0])
        return original_open(*args, **kwargs)  # type: ignore[arg-type]

    def fail_entropy(_length: int) -> str:
        raise OSError("injected entropy failure")

    monkeypatch.setattr(calvin_stats_module.os, "open", tracked_open)
    monkeypatch.setattr(calvin_stats_module.secrets, "token_hex", fail_entropy)

    with pytest.raises(OSError, match="entropy failure"):
        save_calvin_normalization_artifact(tmp_path / "normalization.json", artifact)

    assert opened == []


def test_calvin_stats_unlink_failure_closes_parent_and_reports_ambiguous_commit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact = compute_calvin_normalization_artifact(
        _FakeDataset(tmp_path / "task_ABC_D" / "training"),  # type: ignore[arg-type]
        validation_fraction=0.2,
        verify_archive=False,
        allow_legacy_v3=True,
    )
    output = tmp_path / "normalization.json"
    parent_descriptors: list[int] = []
    original_open = calvin_stats_module.os.open

    def tracked_open(path: object, *args: object, **kwargs: object) -> int:
        descriptor = original_open(path, *args, **kwargs)  # type: ignore[arg-type]
        if path == output.parent:
            parent_descriptors.append(descriptor)
        return descriptor

    def fail_unlink(*_args: object, **_kwargs: object) -> None:
        raise OSError("injected persistent unlink failure")

    monkeypatch.setattr(calvin_stats_module.os, "open", tracked_open)
    monkeypatch.setattr(calvin_stats_module.os, "unlink", fail_unlink)

    with pytest.raises(OSError, match="destination may be committed; stale temporary link remains"):
        save_calvin_normalization_artifact(output, artifact)

    assert output.is_file()
    assert len(tuple(tmp_path.glob(".normalization.json.tmp-*"))) == 1
    assert len(parent_descriptors) == 1
    with pytest.raises(OSError):
        os.fstat(parent_descriptors[0])


@pytest.mark.parametrize(
    ("section", "field", "replacement", "message"),
    [
        ("algorithm", "frame_order", "unordered", "algorithm contract"),
        ("action", "transform", "percentile", "official identity transform"),
        ("dataset", "storage_identity_sha256", "0" * 64, "legacy storage identity mismatch"),
        ("state", "constant_dimensions", [0, 0], "constant dimensions"),
    ],
)
def test_normalization_v4_rejects_rehashed_semantic_contract_drift(
    tmp_path: Path,
    section: str,
    field: str,
    replacement: object,
    message: str,
) -> None:
    dataset = _FakeDataset(tmp_path / "task_ABC_D" / "training")
    artifact = compute_calvin_normalization_artifact(  # type: ignore[arg-type]
        dataset,
        validation_fraction=0.2,
        verify_archive=False,
        allow_legacy_v3=True,
    )
    artifact[section][field] = replacement
    artifact["content_sha256"] = hashlib.sha256(
        json.dumps(
            {name: value for name, value in artifact.items() if name != "content_sha256"},
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode()
    ).hexdigest()
    output = tmp_path / f"drift-{section}-{field}.json"
    save_calvin_normalization_artifact(output, artifact)

    with pytest.raises(ValueError, match=message):
        load_calvin_state_normalizer(
            output,
            training_root=dataset.root,
            verify_archive=False,
            allow_legacy_v3=True,
        )


def test_calvin_stats_reject_wrong_archive_and_tampering(tmp_path: Path) -> None:
    dataset = _FakeDataset(tmp_path / "task_ABC_D" / "training")
    with pytest.raises(ValueError, match="published checksum"):
        compute_calvin_normalization_artifact(dataset, archive_sha256="0" * 64)  # type: ignore[arg-type]

    artifact = compute_calvin_normalization_artifact(  # type: ignore[arg-type]
        dataset,
        validation_fraction=0.2,
        archive_sha256=CALVIN_ABC_D_ARCHIVE_SHA256,
        verify_archive=False,
        allow_legacy_v3=True,
    )
    output = tmp_path / "normalization.json"
    save_calvin_normalization_artifact(output, artifact)
    payload = json.loads(output.read_text())
    payload["state"]["q01"][0] += 1
    output.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="content hash"):
        load_calvin_state_normalizer(output)

    output.write_text('{"schema":"first","schema":"second"}', encoding="utf-8")
    with pytest.raises(ValueError, match="strict finite"):
        load_calvin_state_normalizer(output)


def test_calvin_stats_detect_current_metadata_drift(tmp_path: Path) -> None:
    dataset = _FakeDataset(tmp_path / "task_ABC_D" / "training")
    artifact = compute_calvin_normalization_artifact(  # type: ignore[arg-type]
        dataset,
        validation_fraction=0.2,
        verify_archive=False,
        allow_legacy_v3=True,
    )
    output = tmp_path / "normalization.json"
    save_calvin_normalization_artifact(output, artifact)
    (dataset.root / "scene_info.npy").write_bytes(b"changed-scene-metadata")

    with pytest.raises(ValueError, match="critical file changed"):
        load_calvin_state_normalizer(
            output,
            training_root=dataset.root,
            verify_archive=False,
            allow_legacy_v3=True,
        )


def test_calvin_stats_detect_validation_metadata_drift(tmp_path: Path) -> None:
    dataset = _FakeDataset(tmp_path / "task_ABC_D" / "training")
    artifact = compute_calvin_normalization_artifact(  # type: ignore[arg-type]
        dataset,
        validation_fraction=0.2,
        verify_archive=False,
        allow_legacy_v3=True,
    )
    output = tmp_path / "normalization.json"
    save_calvin_normalization_artifact(output, artifact)
    (dataset.root.parent / "validation/.hydra/merged_config.yaml").write_bytes(b"changed-validation-config")

    with pytest.raises(ValueError, match="critical file changed"):
        load_calvin_state_normalizer(
            output,
            training_root=dataset.root,
            verify_archive=False,
            allow_legacy_v3=True,
        )


def test_calvin_stats_reject_percentiles_that_overflow_float32(tmp_path: Path) -> None:
    artifact = compute_calvin_normalization_artifact(  # type: ignore[arg-type]
        _FakeDataset(tmp_path / "task_ABC_D" / "training"),
        validation_fraction=0.2,
        verify_archive=False,
        allow_legacy_v3=True,
    )
    artifact["state"]["q99"][0] = 1e39
    without_hash = {key: value for key, value in artifact.items() if key != "content_sha256"}
    artifact["content_sha256"] = hashlib.sha256(
        json.dumps(without_hash, allow_nan=False, separators=(",", ":"), sort_keys=True).encode()
    ).hexdigest()
    output = tmp_path / "overflow-normalization.json"
    save_calvin_normalization_artifact(output, artifact)

    with pytest.raises(ValueError, match="finite float32"):
        load_calvin_state_normalizer(output, allow_legacy_v3=True)


@pytest.mark.parametrize(
    ("q01", "q99", "constant_dimensions", "message"),
    [
        (1.0000000001, 1.0, [0], "serialized float64"),
        (1.0, 1.00000101, [], "change after runtime float32"),
    ],
)
def test_calvin_stats_reject_ambiguous_serialized_and_runtime_percentiles(
    tmp_path: Path,
    q01: float,
    q99: float,
    constant_dimensions: list[int],
    message: str,
) -> None:
    artifact = compute_calvin_normalization_artifact(
        _FakeDataset(tmp_path / "task_ABC_D" / "training"),  # type: ignore[arg-type]
        validation_fraction=0.2,
        verify_archive=False,
        allow_legacy_v3=True,
    )
    artifact["state"]["q01"][0] = q01
    artifact["state"]["q99"][0] = q99
    artifact["state"]["constant_dimensions"] = constant_dimensions
    artifact["content_sha256"] = hashlib.sha256(
        json.dumps(
            {name: value for name, value in artifact.items() if name != "content_sha256"},
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode()
    ).hexdigest()
    output = tmp_path / "ambiguous-normalization.json"
    save_calvin_normalization_artifact(output, artifact)

    with pytest.raises(ValueError, match=message):
        load_calvin_state_normalizer(output, allow_legacy_v3=True)


@pytest.mark.parametrize(
    ("section", "field", "message"),
    [
        ("state", "q01", "strict numeric vector"),
        ("action", "continuous_min", "strict numeric vector"),
    ],
)
def test_calvin_stats_reject_boolean_numeric_vector_members(
    tmp_path: Path,
    section: str,
    field: str,
    message: str,
) -> None:
    artifact = compute_calvin_normalization_artifact(
        _FakeDataset(tmp_path / "task_ABC_D" / "training"),  # type: ignore[arg-type]
        validation_fraction=0.2,
        verify_archive=False,
        allow_legacy_v3=True,
    )
    artifact[section][field][0] = False
    artifact["content_sha256"] = hashlib.sha256(
        json.dumps(
            {name: value for name, value in artifact.items() if name != "content_sha256"},
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode()
    ).hexdigest()
    output = tmp_path / f"boolean-{section}-{field}.json"
    save_calvin_normalization_artifact(output, artifact)

    with pytest.raises(ValueError, match=message):
        load_calvin_state_normalizer(output, allow_legacy_v3=True)
