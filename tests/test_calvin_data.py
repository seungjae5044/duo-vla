from __future__ import annotations

import hashlib
import io
import json
import os
import shutil
import sqlite3
import stat
import warnings
import zipfile
import zlib
from pathlib import Path

import numpy as np
import pytest
import torch

from duo_vla.data.calvin import (
    CalvinAnchor,
    CalvinAnnotation,
    CalvinEpisode,
    CalvinNpzDataset,
    CalvinTaskUniformAnchorSampler,
    make_calvin_episode_split,
)
from duo_vla.data.calvin_archive import (
    CALVIN_ARCHIVE_NAME,
    CALVIN_CRITICAL_FILES,
    CALVIN_INDEX_NAME,
    CalvinArchiveValidationError,
    prepare_calvin_archive,
)
from duo_vla.data.calvin_stats import (
    CALVIN_ABC_D_ARCHIVE_BYTES,
    CALVIN_ABC_D_ARCHIVE_SHA256,
    CALVIN_ABC_D_ARCHIVE_URL,
    CALVIN_AUTHENTICATED_GENERATION_SCHEMA,
    CALVIN_DATASET_CRITICAL_FILES,
    CALVIN_DATASET_MANIFEST_SCHEMA_V3,
    CALVIN_STORAGE_IDENTITY_SCHEMA,
    CALVIN_STORAGE_MODE_ARCHIVE_DIRECT,
    CALVIN_STORAGE_MODE_VERIFIED_EXTRACTION,
    AuthenticatedCalvinDatasetGeneration,
    _authenticate_calvin_dataset_generation,
    calvin_member_index_path,
    compute_calvin_normalization_artifact,
    load_calvin_state_normalizer,
    save_calvin_normalization_artifact,
)

FIXTURE_ARCHIVE_URL = "fixture://task_ABC_D.zip"
FIXTURE_CHECKSUM_URL = "fixture://sha256sum.txt"


def _frame(path: Path, index: int) -> None:
    robot = np.zeros(15, dtype=np.float32)
    robot[:7] = index
    robot[14] = -1.0 if index % 2 else 1.0
    action = np.zeros(7, dtype=np.float32)
    action[0] = index / 20
    action[6] = -1.0 if index % 2 else 1.0
    np.savez_compressed(
        path / f"episode_{index:07d}.npz",
        rgb_static=np.full((200, 200, 3), index, dtype=np.uint8),
        rgb_gripper=np.full((84, 84, 3), index, dtype=np.uint8),
        robot_obs=robot,
        rel_actions=action,
    )


def _npy_bytes(value: object, *, allow_pickle: bool) -> bytes:
    sink = io.BytesIO()
    np.save(sink, value, allow_pickle=allow_pickle)
    return sink.getvalue()


def _frame_bytes(index: int) -> bytes:
    robot = np.zeros(15, dtype=np.float32)
    robot[:7] = np.arange(7, dtype=np.float32) + index
    robot[14] = -1.0 if index % 2 else 1.0
    action = np.zeros(7, dtype=np.float32)
    action[:6] = (index - 10) / 20
    action[6] = robot[14]
    sink = io.BytesIO()
    np.savez_compressed(
        sink,
        rgb_static=np.full((200, 200, 3), index, dtype=np.uint8),
        rgb_gripper=np.full((84, 84, 3), index, dtype=np.uint8),
        robot_obs=robot,
        rel_actions=action,
    )
    return sink.getvalue()


def _zip_regular(name: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name)
    info.create_system = 3
    info.external_attr = (stat.S_IFREG | 0o644) << 16
    info.compress_type = zipfile.ZIP_DEFLATED
    return info


def _zip_directory(name: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name)
    info.create_system = 3
    info.external_attr = ((stat.S_IFDIR | 0o755) << 16) | 0x10
    info.compress_type = zipfile.ZIP_STORED
    return info


def _v4_dataset(
    tmp_path: Path,
) -> tuple[Path, AuthenticatedCalvinDatasetGeneration, dict[int, bytes]]:
    data_root = tmp_path / "v4-data"
    data_root.mkdir()
    archive_path = data_root / CALVIN_ARCHIVE_NAME
    episodes = np.asarray([[2 * index, 2 * index + 1] for index in range(10)], dtype=np.int64)
    annotations = {
        "info": {"indx": np.asarray([[2 * episode, 2 * episode + 2] for episode in range(10) for _ in range(2)])},
        "language": {
            "ann": [f"instruction {episode} {task}" for episode in range(10) for task in range(2)],
            "task": [f"task_{task}" for _episode in range(10) for task in range(2)],
            "emb": np.zeros((20, 1, 3), dtype=np.float32),
        },
    }
    metadata = {
        "training/ep_start_end_ids.npy": _npy_bytes(episodes, allow_pickle=False),
        "training/lang_annotations/auto_lang_ann.npy": _npy_bytes(annotations, allow_pickle=True),
        "training/scene_info.npy": _npy_bytes({"calvin_scene_A": [0, 19]}, allow_pickle=True),
        "training/.hydra/merged_config.yaml": b"fixture: training\n",
        "validation/ep_start_end_ids.npy": _npy_bytes(np.asarray([[0, 1]], dtype=np.int64), allow_pickle=False),
        "validation/.hydra/merged_config.yaml": b"fixture: validation\n",
    }
    assert set(metadata) == set(CALVIN_CRITICAL_FILES)
    frames = {index: _frame_bytes(index) for index in range(20)}
    directories = (
        "task_ABC_D/",
        "task_ABC_D/training/",
        "task_ABC_D/training/lang_annotations/",
        "task_ABC_D/training/.hydra/",
        "task_ABC_D/validation/",
        "task_ABC_D/validation/.hydra/",
    )
    physical_order = (17, 2, 19, 0, 8, 5, 14, 1, 12, 7, 18, 3, 10, 4, 16, 6, 15, 9, 13, 11)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        with zipfile.ZipFile(archive_path, "w") as archive:
            for name in directories:
                archive.writestr(_zip_directory(name), b"")
            for relative, raw in metadata.items():
                archive.writestr(_zip_regular(f"task_ABC_D/{relative}"), raw)
            for index in physical_order:
                archive.writestr(_zip_regular(f"task_ABC_D/training/episode_{index:07d}.npz"), frames[index])
    archive_raw = archive_path.read_bytes()
    archive_sha256 = hashlib.sha256(archive_raw).hexdigest()
    prepare_calvin_archive(
        archive_path,
        data_root,
        expected_archive_bytes=len(archive_raw),
        expected_archive_sha256=archive_sha256,
        expected_central_directory=None,
        archive_url=FIXTURE_ARCHIVE_URL,
        checksum_url=FIXTURE_CHECKSUM_URL,
    )
    training_root = data_root / "task_ABC_D" / "training"
    _, generation = _authenticate_calvin_dataset_generation(
        training_root,
        expected_archive_bytes=len(archive_raw),
        expected_archive_sha256=archive_sha256,
        expected_central_directory=None,
        expected_archive_url=FIXTURE_ARCHIVE_URL,
        expected_checksum_url=FIXTURE_CHECKSUM_URL,
        allow_legacy_v3=False,
        verify_archive=True,
    )
    return training_root, generation, frames


def _dataset_root(tmp_path: Path) -> Path:
    root = tmp_path / "training"
    (root / "lang_annotations").mkdir(parents=True)
    np.save(root / "ep_start_end_ids.npy", np.asarray([[0, 4], [5, 9]], dtype=np.int64))
    payload = {
        "info": {"indx": np.asarray([[1, 5], [5, 8]], dtype=np.int64)},
        "language": {
            "ann": ["move the block", "open the drawer"],
            "task": ["move_block", "open_drawer"],
            "emb": np.zeros((2, 1, 3), dtype=np.float32),
        },
    }
    np.save(root / "lang_annotations" / "auto_lang_ann.npy", payload, allow_pickle=True)
    for index in range(10):
        _frame(root, index)
    return root


def _install_frame_authentication(
    root: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Path:
    import duo_vla.data.calvin_stats as calvin_stats

    member_index = tmp_path / "members.sqlite3"
    with sqlite3.connect(member_index) as connection:
        connection.execute(
            "CREATE TABLE members(path TEXT PRIMARY KEY, bytes INTEGER NOT NULL, "
            "crc32 INTEGER NOT NULL, sha256 TEXT NOT NULL) WITHOUT ROWID"
        )
        for path in sorted(root.glob("episode_*.npz")):
            content = path.read_bytes()
            connection.execute(
                "INSERT INTO members(path, bytes, crc32, sha256) VALUES (?, ?, ?, ?)",
                (
                    path.relative_to(root.parent).as_posix(),
                    len(content),
                    zlib.crc32(content) & 0xFFFFFFFF,
                    hashlib.sha256(content).hexdigest(),
                ),
            )
    member_bytes = member_index.read_bytes()
    identity = {
        "bytes": len(member_bytes),
        "path": member_index.name,
        "schema": "duo-vla-calvin-member-index-v1",
        "sha256": hashlib.sha256(member_bytes).hexdigest(),
    }
    monkeypatch.setattr(calvin_stats, "calvin_member_index_path", lambda _root: member_index)
    monkeypatch.setattr(
        calvin_stats,
        "load_calvin_dataset_manifest",
        lambda *_args, **_kwargs: {"extraction": {"member_index": identity}},
    )
    return member_index


def _capability_dataset(
    tmp_path: Path,
) -> tuple[Path, Path, AuthenticatedCalvinDatasetGeneration]:
    root = _dataset_root(tmp_path / "task_ABC_D")
    np.save(root / "scene_info.npy", {"calvin_scene_A": [0, 9]}, allow_pickle=True)
    member_index = calvin_member_index_path(root)
    with sqlite3.connect(member_index) as connection:
        connection.execute(
            "CREATE TABLE members(path TEXT PRIMARY KEY, bytes INTEGER NOT NULL, "
            "crc32 INTEGER NOT NULL, sha256 TEXT NOT NULL) WITHOUT ROWID"
        )
        for path in sorted(root.glob("episode_*.npz")):
            content = path.read_bytes()
            connection.execute(
                "INSERT INTO members(path, bytes, crc32, sha256) VALUES (?, ?, ?, ?)",
                (
                    path.relative_to(root.parent).as_posix(),
                    len(content),
                    zlib.crc32(content) & 0xFFFFFFFF,
                    hashlib.sha256(content).hexdigest(),
                ),
            )
    member_bytes = member_index.read_bytes()
    critical_files = {
        relative: (
            hashlib.sha256((root.parent / relative).read_bytes()).hexdigest()
            if (root.parent / relative).is_file()
            else "0" * 64
        )
        for relative in CALVIN_DATASET_CRITICAL_FILES
    }
    storage = {
        "archive": {
            "bytes": CALVIN_ABC_D_ARCHIVE_BYTES,
            "path": "task_ABC_D.zip",
            "sha256": CALVIN_ABC_D_ARCHIVE_SHA256,
            "url": CALVIN_ABC_D_ARCHIVE_URL,
        },
        "central_directory": None,
        "checksum_url": "http://calvin.cs.uni-freiburg.de/dataset/sha256sum.txt",
        "manifest": {
            "content_sha256": "1" * 64,
            "file_sha256": "2" * 64,
            "schema": CALVIN_DATASET_MANIFEST_SCHEMA_V3,
        },
        "member_index": {
            "bytes": len(member_bytes),
            "path": member_index.name,
            "schema": "duo-vla-calvin-member-index-v1",
            "sha256": hashlib.sha256(member_bytes).hexdigest(),
        },
        "member_inventory": {
            "compressed_bytes": 1,
            "directory_member_count": None,
            "file_member_count": 1,
            "member_count": 1,
            "npz_member_count": 1,
            "sha256": "3" * 64,
            "uncompressed_bytes": 1,
        },
        "mode": CALVIN_STORAGE_MODE_VERIFIED_EXTRACTION,
        "reader_schema": None,
        "schema": CALVIN_STORAGE_IDENTITY_SCHEMA,
    }
    storage["content_sha256"] = hashlib.sha256(
        json.dumps(storage, allow_nan=False, separators=(",", ":"), sort_keys=True).encode()
    ).hexdigest()
    payload = {
        "archive_file_identity": None,
        "critical_files": critical_files,
        "metadata_sha256": "4" * 64,
        "schema": CALVIN_AUTHENTICATED_GENERATION_SCHEMA,
        "storage": storage,
        "training_root": str(root.resolve()),
    }
    payload["content_sha256"] = hashlib.sha256(
        json.dumps(payload, allow_nan=False, separators=(",", ":"), sort_keys=True).encode()
    ).hexdigest()
    return root, member_index, AuthenticatedCalvinDatasetGeneration.from_dict(payload)


def test_calvin_reader_preserves_raw_instruction_and_mixed_boundary_conventions(tmp_path: Path) -> None:
    dataset = CalvinNpzDataset(
        _dataset_root(tmp_path),
        max_cached_frames=4,
        storage_mode=CALVIN_STORAGE_MODE_VERIFIED_EXTRACTION,
    )

    sample = dataset.sample(CalvinAnchor(0, 3, "move_block"), horizon=8)

    assert sample.instruction == "move the block"
    assert sample.episode_index == 0
    assert sample.global_index == 3
    assert sample.observation.third_person.shape == (200, 200, 3)
    assert sample.observation.wrist.shape == (84, 84, 3)
    torch.testing.assert_close(sample.observation.state[:7], torch.full((7,), 3.0))
    assert sample.observation.state[-1].item() == -1.0
    assert sample.action_chunk.valid_mask.tolist() == [True, True, False, False, False, False, False, False]
    torch.testing.assert_close(sample.action_chunk.actions[:2, 0], torch.tensor([0.15, 0.20]))
    torch.testing.assert_close(sample.action_chunk.actions[2:, :6], torch.zeros(6, 6))
    torch.testing.assert_close(sample.action_chunk.actions[2:, 6], torch.ones(6))


def test_calvin_reader_rejects_bad_anchor_before_frame_io(tmp_path: Path) -> None:
    dataset = CalvinNpzDataset(
        _dataset_root(tmp_path),
        storage_mode=CALVIN_STORAGE_MODE_VERIFIED_EXTRACTION,
    )

    with pytest.raises(IndexError, match="language interval"):
        dataset.sample(CalvinAnchor(0, 0, "move_block"))
    with pytest.raises(ValueError, match="task"):
        dataset.sample(CalvinAnchor(0, 1, "wrong"))
    with pytest.raises(ValueError, match="duplicate"):
        dataset.sample_many((CalvinAnchor(0, 1, "move_block"), CalvinAnchor(0, 1, "move_block")))


def test_extracted_fixture_backend_must_be_selected_explicitly(tmp_path: Path) -> None:
    root = _dataset_root(tmp_path)

    with pytest.raises(ValueError, match="explicitly select"):
        CalvinNpzDataset(root)


def test_calvin_reader_authenticates_each_frame_before_npz_loading(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _dataset_root(tmp_path)
    _install_frame_authentication(root, tmp_path, monkeypatch)
    dataset = CalvinNpzDataset(root, verify_frame_files=True, verify_archive=False)
    target = root / "episode_0000003.npz"
    content = bytearray(target.read_bytes())
    content[len(content) // 2] ^= 1
    target.write_bytes(content)

    with pytest.raises(ValueError, match="frame content changed"):
        dataset.sample(CalvinAnchor(0, 3, "move_block"), horizon=1)
    dataset.close()


def test_authenticated_reader_rejects_symlink_even_when_target_bytes_are_valid(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _dataset_root(tmp_path)
    _install_frame_authentication(root, tmp_path, monkeypatch)
    dataset = CalvinNpzDataset(root, verify_frame_files=True, verify_archive=False)
    target = root / "episode_0000003.npz"
    backing = tmp_path / "valid-frame.npz"
    target.replace(backing)
    target.symlink_to(backing)

    with pytest.raises(ValueError, match="regular non-symlink"):
        dataset.sample(CalvinAnchor(0, 3, "move_block"), horizon=1)
    dataset.close()


def test_authenticated_reader_loads_the_same_bytes_it_verified_and_rechecks_replacements(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import duo_vla.data.calvin as calvin_data

    root = _dataset_root(tmp_path)
    _install_frame_authentication(root, tmp_path, monkeypatch)
    dataset = CalvinNpzDataset(root, max_cached_frames=1, verify_frame_files=True, verify_archive=False)
    target = root / "episode_0000003.npz"
    _frame(root, 99)
    replacement = root / "episode_0000099.npz"
    original_load = calvin_data.np.load
    original_open = calvin_data.os.open
    target_open_flags: list[int] = []
    replaced = False

    def tracked_open(path: str | os.PathLike[str], flags: int, *args: object, **kwargs: object) -> int:
        if Path(path) == target:
            target_open_flags.append(flags)
        return original_open(path, flags, *args, **kwargs)

    def replacing_load(source: object, *args: object, **kwargs: object):
        nonlocal replaced
        if isinstance(source, io.BytesIO) and not replaced:
            replacement.replace(target)
            replaced = True
        return original_load(source, *args, **kwargs)

    monkeypatch.setattr(calvin_data.os, "open", tracked_open)
    monkeypatch.setattr(calvin_data.np, "load", replacing_load)

    sample = dataset.sample(CalvinAnchor(0, 3, "move_block"), horizon=1)

    assert replaced
    assert len(target_open_flags) == 1
    assert target_open_flags[0] & os.O_CLOEXEC
    assert target_open_flags[0] & os.O_NOFOLLOW
    torch.testing.assert_close(sample.observation.state[:7], torch.full((7,), 3.0))

    dataset.sample(CalvinAnchor(0, 4, "move_block"), horizon=1)
    with pytest.raises(ValueError, match=r"frame (?:size|content) changed"):
        dataset.sample(CalvinAnchor(0, 3, "move_block"), horizon=1)
    dataset.close()


def test_authenticated_reader_fails_closed_after_close_even_for_a_cached_frame(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _dataset_root(tmp_path)
    _install_frame_authentication(root, tmp_path, monkeypatch)
    dataset = CalvinNpzDataset(root, verify_frame_files=True, verify_archive=False)
    dataset.sample(CalvinAnchor(0, 3, "move_block"), horizon=1)

    dataset.close()

    with pytest.raises(RuntimeError, match="dataset is closed"):
        dataset.sample(CalvinAnchor(0, 3, "move_block"), horizon=1)
    with pytest.raises(RuntimeError, match="dataset is closed"):
        dataset.read_state_action(3)


def test_authenticated_reader_closes_member_index_when_constructor_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import duo_vla.data.calvin as calvin_data

    root = _dataset_root(tmp_path)
    member_index = _install_frame_authentication(root, tmp_path, monkeypatch)
    connection = sqlite3.connect(member_index)
    closed = False

    class ConnectionSpy:
        def close(self) -> None:
            nonlocal closed
            closed = True
            connection.close()

    monkeypatch.setattr(calvin_data.sqlite3, "connect", lambda *_args, **_kwargs: ConnectionSpy())

    with pytest.raises(ValueError, match="expected CALVIN scenes"):
        CalvinNpzDataset(
            root,
            expected_scenes=("calvin_scene_A",),
            verify_frame_files=True,
            verify_archive=False,
        )

    assert closed


def test_generation_capability_rejects_metadata_replaced_after_verification(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import duo_vla.data.calvin_stats as calvin_stats

    root, _member_index, capability = _capability_dataset(tmp_path)
    target = root / "lang_annotations" / "auto_lang_ann.npy"
    replacement = tmp_path / "attacker-auto-lang-ann.npy"
    np.save(replacement, np.asarray(["attacker-controlled"], dtype=object), allow_pickle=True)

    def replace_after_verification(_root: Path, generation: AuthenticatedCalvinDatasetGeneration) -> dict[str, object]:
        assert generation is capability
        replacement.replace(target)
        return {}

    monkeypatch.setattr(calvin_stats, "verify_calvin_dataset_generation", replace_after_verification)

    with pytest.raises(ValueError, match="metadata content differs from the authenticated generation"):
        CalvinNpzDataset(root, authenticated_generation=capability)


def test_generation_capability_loads_metadata_from_the_same_bytes_it_hashed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import duo_vla.data.calvin as calvin_data
    import duo_vla.data.calvin_stats as calvin_stats

    root, _member_index, capability = _capability_dataset(tmp_path)
    target = root / "scene_info.npy"
    replacement = tmp_path / "attacker-scene-info.npy"
    np.save(replacement, {"attacker_scene": [0, 9]}, allow_pickle=True)
    original_load = calvin_data.np.load
    replaced = False
    byte_loads = 0

    def replacing_load(source: object, *args: object, **kwargs: object):
        nonlocal byte_loads, replaced
        if isinstance(source, io.BytesIO):
            byte_loads += 1
            if not replaced:
                replacement.replace(target)
                replaced = True
        return original_load(source, *args, **kwargs)

    monkeypatch.setattr(calvin_stats, "verify_calvin_dataset_generation", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(calvin_data.np, "load", replacing_load)

    dataset = CalvinNpzDataset(
        root,
        expected_scenes=("calvin_scene_A",),
        authenticated_generation=capability,
    )

    assert replaced
    assert byte_loads == 3
    assert dataset.scene_intervals == {"calvin_scene_A": (0, 9)}
    dataset.close()


def test_generation_capability_rejects_metadata_symlink_and_closes_pinned_index_fd(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import duo_vla.data.calvin as calvin_data
    import duo_vla.data.calvin_stats as calvin_stats

    root, _member_index, capability = _capability_dataset(tmp_path)
    target = root / "lang_annotations" / "auto_lang_ann.npy"
    backing = tmp_path / "valid-auto-lang-ann.npy"
    target.replace(backing)
    target.symlink_to(backing)
    original_connect = calvin_data.sqlite3.connect
    pinned_descriptors: list[int] = []

    def capture_pinned_descriptor(database: str, *args: object, **kwargs: object):
        marker = "/proc/self/fd/"
        if marker in database:
            pinned_descriptors.append(int(database.split(marker, 1)[1].split("?", 1)[0]))
        return original_connect(database, *args, **kwargs)

    monkeypatch.setattr(calvin_stats, "verify_calvin_dataset_generation", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(calvin_data.sqlite3, "connect", capture_pinned_descriptor)

    with pytest.raises(ValueError, match="metadata must be a regular non-symlink file"):
        CalvinNpzDataset(root, authenticated_generation=capability)

    assert len(pinned_descriptors) == 1
    with pytest.raises(OSError):
        os.fstat(pinned_descriptors[0])


def test_generation_capability_rejects_member_index_symlink(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import duo_vla.data.calvin_stats as calvin_stats

    root, member_index, capability = _capability_dataset(tmp_path)
    backing = tmp_path / "valid-members.sqlite3"
    member_index.replace(backing)
    member_index.symlink_to(backing)
    monkeypatch.setattr(calvin_stats, "verify_calvin_dataset_generation", lambda *_args, **_kwargs: {})

    with pytest.raises(ValueError, match="member index must be a regular non-symlink file"):
        CalvinNpzDataset(root, authenticated_generation=capability)


def test_generation_capability_fails_closed_if_member_index_is_replaced_during_pinned_open(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import duo_vla.data.calvin as calvin_data
    import duo_vla.data.calvin_stats as calvin_stats

    root, member_index, capability = _capability_dataset(tmp_path)
    attacker_index = tmp_path / "attacker-members.sqlite3"
    with sqlite3.connect(attacker_index) as connection:
        connection.execute(
            "CREATE TABLE members(path TEXT PRIMARY KEY, bytes INTEGER NOT NULL, "
            "crc32 INTEGER NOT NULL, sha256 TEXT NOT NULL) WITHOUT ROWID"
        )
    original_connect = calvin_data.sqlite3.connect
    pinned_descriptors: list[int] = []
    replaced = False

    def replace_path_before_sqlite_open(database: str, *args: object, **kwargs: object):
        nonlocal replaced
        marker = "/proc/self/fd/"
        if marker in database:
            pinned_descriptors.append(int(database.split(marker, 1)[1].split("?", 1)[0]))
            if not replaced:
                attacker_index.replace(member_index)
                replaced = True
        return original_connect(database, *args, **kwargs)

    monkeypatch.setattr(calvin_stats, "verify_calvin_dataset_generation", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(calvin_data.sqlite3, "connect", replace_path_before_sqlite_open)

    with pytest.raises(ValueError, match="could not open the pinned authenticated CALVIN member index"):
        CalvinNpzDataset(root, authenticated_generation=capability)
    assert replaced
    assert len(pinned_descriptors) == 1
    with pytest.raises(OSError):
        os.fstat(pinned_descriptors[0])


def test_generation_capability_keeps_open_member_index_inode_after_path_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import duo_vla.data.calvin_stats as calvin_stats

    root, member_index, capability = _capability_dataset(tmp_path)
    attacker_index = tmp_path / "attacker-members.sqlite3"
    with sqlite3.connect(attacker_index) as connection:
        connection.execute(
            "CREATE TABLE members(path TEXT PRIMARY KEY, bytes INTEGER NOT NULL, "
            "crc32 INTEGER NOT NULL, sha256 TEXT NOT NULL) WITHOUT ROWID"
        )
    monkeypatch.setattr(calvin_stats, "verify_calvin_dataset_generation", lambda *_args, **_kwargs: {})
    dataset = CalvinNpzDataset(root, authenticated_generation=capability)
    pinned_descriptor = dataset._member_index_descriptor
    assert pinned_descriptor is not None

    attacker_index.replace(member_index)
    sample = dataset.sample(CalvinAnchor(0, 3, "move_block"), horizon=1)

    torch.testing.assert_close(sample.observation.state[:7], torch.full((7,), 3.0))
    with sqlite3.connect(member_index) as connection:
        assert connection.execute("SELECT count(*) FROM members").fetchone() == (0,)
    dataset.close()
    with pytest.raises(OSError):
        os.fstat(pinned_descriptor)


def test_calvin_annotation_may_not_cross_an_underlying_episode(tmp_path: Path) -> None:
    root = _dataset_root(tmp_path)
    payload = np.load(root / "lang_annotations" / "auto_lang_ann.npy", allow_pickle=True).item()
    payload["info"]["indx"][0] = [4, 6]
    np.save(root / "lang_annotations" / "auto_lang_ann.npy", payload, allow_pickle=True)

    with pytest.raises(ValueError, match="crosses"):
        CalvinNpzDataset(root, storage_mode=CALVIN_STORAGE_MODE_VERIFIED_EXTRACTION)


def test_calvin_reader_can_fail_closed_on_scene_identity(tmp_path: Path) -> None:
    root = _dataset_root(tmp_path)
    np.save(root / "scene_info.npy", {"calvin_scene_D": [0, 9]}, allow_pickle=True)

    dataset = CalvinNpzDataset(
        root,
        expected_scenes=("calvin_scene_D",),
        storage_mode=CALVIN_STORAGE_MODE_VERIFIED_EXTRACTION,
    )
    assert dataset.scene_intervals == {"calvin_scene_D": (0, 9)}
    with pytest.raises(ValueError, match="scene identity"):
        CalvinNpzDataset(
            root,
            expected_scenes=("calvin_scene_A", "calvin_scene_B", "calvin_scene_C"),
            storage_mode=CALVIN_STORAGE_MODE_VERIFIED_EXTRACTION,
        )


def test_v4_reader_uses_archive_only_decodes_anchors_fully_and_restores_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    training_root, generation, _frames = _v4_dataset(tmp_path)
    assert not list(training_root.glob("episode_*.npz"))
    dataset = CalvinNpzDataset(
        training_root,
        expected_scenes=("calvin_scene_A",),
        authenticated_generation=generation,
    )
    original = dataset._load_archive_member_arrays
    calls: list[tuple[str, tuple[str, ...]]] = []

    def tracked(relative: str, required: tuple[str, ...]) -> dict[str, np.ndarray]:
        calls.append((relative, required))
        return original(relative, required)

    monkeypatch.setattr(dataset, "_load_archive_member_arrays", tracked)
    anchors = (
        CalvinAnchor(7, 6, "task_1"),
        CalvinAnchor(0, 0, "task_0"),
    )

    samples = dataset.sample_many(anchors, horizon=2)

    assert [sample.global_index for sample in samples] == [6, 0]
    full = ("rgb_static", "rgb_gripper", "robot_obs", "rel_actions")
    assert calls == [
        ("training/episode_0000000.npz", full),
        ("training/episode_0000006.npz", full),
        ("training/episode_0000001.npz", ("rel_actions",)),
        ("training/episode_0000007.npz", ("rel_actions",)),
    ]
    torch.testing.assert_close(samples[0].action_chunk.actions[:2, 0], torch.tensor([-0.2, -0.15]))
    torch.testing.assert_close(samples[1].observation.state[:7], torch.arange(7, dtype=torch.float32))
    dataset.close()


def test_v4_pickle_metadata_loads_only_from_reader_authenticated_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import duo_vla.data.calvin as calvin_data

    training_root, generation, _frames = _v4_dataset(tmp_path)
    original_load = calvin_data.np.load
    sources: list[str] = []

    def tracked(source: object, *args: object, **kwargs: object):
        sources.append("bytes" if isinstance(source, io.BytesIO) else "path")
        return original_load(source, *args, **kwargs)

    monkeypatch.setattr(calvin_data.np, "load", tracked)

    dataset = CalvinNpzDataset(training_root, authenticated_generation=generation)

    assert sources == ["bytes", "bytes", "bytes"]
    dataset.close()


def test_v4_reader_validates_every_anchor_before_archive_io(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    training_root, generation, _frames = _v4_dataset(tmp_path)
    dataset = CalvinNpzDataset(training_root, authenticated_generation=generation)
    reads = 0
    reader = dataset._archive_reader
    assert reader is not None
    original = reader.read_member_bytes

    def tracked(*args: object, **kwargs: object) -> bytes:
        nonlocal reads
        reads += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(reader, "read_member_bytes", tracked)

    with pytest.raises(ValueError, match="task"):
        dataset.sample_many((CalvinAnchor(0, 0, "task_0"), CalvinAnchor(2, 2, "wrong")))
    assert reads == 0
    dataset.close()


def test_v4_physical_state_action_iterator_follows_authenticated_data_offsets(tmp_path: Path) -> None:
    training_root, generation, _frames = _v4_dataset(tmp_path)
    dataset = CalvinNpzDataset(training_root, authenticated_generation=generation)

    records = list(dataset.iter_state_actions_physical())

    assert [index for index, _state, _action in records] == [
        17,
        2,
        19,
        0,
        8,
        5,
        14,
        1,
        12,
        7,
        18,
        3,
        10,
        4,
        16,
        6,
        15,
        9,
        13,
        11,
    ]
    np.testing.assert_array_equal(records[3][1][:7], np.arange(7, dtype=np.float32))
    np.testing.assert_array_equal(records[3][2][:6], np.full(6, -0.5, dtype=np.float32))
    dataset.close()


def test_actual_v4_normalization_round_trip_uses_physical_scan_and_storage_identity(tmp_path: Path) -> None:
    training_root, generation, _frames = _v4_dataset(tmp_path)
    with CalvinNpzDataset(training_root, authenticated_generation=generation) as dataset:
        artifact = compute_calvin_normalization_artifact(
            dataset,
            validation_fraction=0.2,
            split_seed=7,
            archive_sha256=generation.storage.archive_sha256,
            authenticated_generation=generation,
            allow_non_official_archive=True,
        )
    output = tmp_path / "normalization-v4.json"
    save_calvin_normalization_artifact(output, artifact)

    normalizer, loaded = load_calvin_state_normalizer(
        output,
        expected_archive_sha256=generation.storage.archive_sha256,
        training_root=training_root,
        authenticated_generation=generation,
    )

    assert loaded == artifact
    assert loaded["counts"]["training_frames"] == 16
    assert loaded["dataset"]["storage_identity_sha256"] == generation.storage.content_sha256
    assert normalizer.action_dim == 8

    data_root = training_root.parent.parent
    member_index = data_root / CALVIN_INDEX_NAME
    backing_index = tmp_path / "byte-identical-member-index.sqlite3"
    member_index.replace(backing_index)
    member_index.symlink_to(backing_index)
    with pytest.raises(CalvinArchiveValidationError, match="no-follow open regular file"):
        load_calvin_state_normalizer(
            output,
            expected_archive_sha256=generation.storage.archive_sha256,
            training_root=training_root,
            authenticated_generation=generation,
        )
    member_index.unlink()
    backing_index.replace(member_index)

    archive = data_root / CALVIN_ARCHIVE_NAME
    rebound_archive = tmp_path / "byte-identical-rebound.zip"
    shutil.copy2(archive, rebound_archive)
    rebound_archive.replace(archive)
    with pytest.raises(ValueError, match="ephemeral file identity"):
        load_calvin_state_normalizer(
            output,
            expected_archive_sha256=generation.storage.archive_sha256,
            training_root=training_root,
            authenticated_generation=generation,
        )


def test_v4_reader_lifetime_and_archive_mutation_fail_closed(tmp_path: Path) -> None:
    training_root, generation, _frames = _v4_dataset(tmp_path)
    dataset = CalvinNpzDataset(training_root, authenticated_generation=generation)
    reader = dataset._archive_reader
    assert reader is not None
    archive_path = training_root.parent.parent / CALVIN_ARCHIVE_NAME
    dataset.sample(CalvinAnchor(0, 0, "task_0"), horizon=1)
    os.chmod(archive_path, 0o644)
    raw = bytearray(archive_path.read_bytes())
    raw[0] ^= 1
    archive_path.write_bytes(raw)

    with pytest.raises(CalvinArchiveValidationError, match="changed"):
        dataset.sample(CalvinAnchor(0, 0, "task_0"), horizon=1)
    dataset.close()
    with pytest.raises(CalvinArchiveValidationError, match="closed"):
        _ = reader.authenticated_manifest
    with pytest.raises(RuntimeError, match="dataset is closed"):
        dataset.sample(CalvinAnchor(0, 0, "task_0"), horizon=1)


def test_v4_generation_round_trip_binds_storage_and_live_archive_identity(tmp_path: Path) -> None:
    _training_root, generation, _frames = _v4_dataset(tmp_path)

    restored = AuthenticatedCalvinDatasetGeneration.from_dict(generation.to_dict())

    assert restored == generation
    assert restored.storage.mode == CALVIN_STORAGE_MODE_ARCHIVE_DIRECT
    assert restored.storage.manifest_schema != CALVIN_DATASET_MANIFEST_SCHEMA_V3
    assert restored.archive_file_identity is not None
    assert dict(restored.archive_file_identity)["size"] == restored.storage.archive_bytes

    tampered = generation.to_dict()
    tampered["archive_file_identity"]["ctime_ns"] += 1
    with pytest.raises(ValueError, match="content hash"):
        AuthenticatedCalvinDatasetGeneration.from_dict(tampered)


def test_v4_reader_rejects_projection_root_inventory_drift(tmp_path: Path) -> None:
    training_root, generation, _frames = _v4_dataset(tmp_path)
    unexpected = training_root / "episode_0000000.npz"
    unexpected.write_bytes(b"must not select the backend from frame presence")

    with pytest.raises(CalvinArchiveValidationError, match="inventory differs"):
        CalvinNpzDataset(training_root, authenticated_generation=generation)


def test_calvin_task_uniform_sampler_is_seeded_and_episode_scoped() -> None:
    annotations = (
        CalvinAnnotation(0, 0, 0, 3, "a", "task_a"),
        CalvinAnnotation(1, 1, 3, 6, "b", "task_b"),
        CalvinAnnotation(2, 2, 6, 9, "c", "task_a"),
    )
    sampler = CalvinTaskUniformAnchorSampler(annotations, (0, 1))
    first = sampler.draw(torch.Generator().manual_seed(77))
    second = sampler.draw(torch.Generator().manual_seed(77))

    assert first == second
    assert first.annotation_index in {0, 1}
    assert sampler.tasks == ("task_a", "task_b")
    assert sampler.population_size == 6


def test_calvin_episode_split_is_whole_episode_and_requires_task_coverage() -> None:
    episodes = tuple(CalvinEpisode(index, index * 10, index * 10 + 9) for index in range(10))
    annotations = tuple(
        CalvinAnnotation(
            annotation_index=index,
            episode_index=index // 2,
            global_start=(index // 2) * 10,
            global_end_exclusive=(index // 2) * 10 + 2,
            instruction=f"instruction {index}",
            task="task_a" if index % 2 == 0 else "task_b",
        )
        for index in range(20)
    )
    # Correct the episode mapping so every episode has both tasks.
    annotations = tuple(
        CalvinAnnotation(
            annotation_index=index,
            episode_index=index // 2,
            global_start=(index // 2) * 10,
            global_end_exclusive=(index // 2) * 10 + 2,
            instruction=annotation.instruction,
            task=annotation.task,
        )
        for index, annotation in enumerate(annotations)
    )

    split = make_calvin_episode_split(episodes, annotations, validation_fraction=0.2, seed=7)

    assert len(split.validation_episode_indices) == 2
    assert len(split.train_episode_indices) == 8
    assert not set(split.train_episode_indices) & set(split.validation_episode_indices)
