from __future__ import annotations

import hashlib
import io
import sqlite3
import zlib
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import duo_vla.data.calvin_dev_states as dev


def npy_bytes(value: object, *, allow_pickle: bool) -> bytes:
    output = io.BytesIO()
    np.save(output, value, allow_pickle=allow_pickle)
    return output.getvalue()


def npz_bytes(**values: object) -> bytes:
    output = io.BytesIO()
    np.savez(output, **values)
    return output.getvalue()


def indices_hash(values: list[int]) -> str:
    return hashlib.sha256(",".join(map(str, values)).encode()).hexdigest()


def content_hash(payload: dict[str, Any]) -> str:
    return dev.canonical_sha256({name: value for name, value in payload.items() if name != "content_sha256"})


def fixture_split(episodes: list[tuple[int, int]], scenes: list[str]) -> dict[str, Any]:
    validation: set[int] = set()
    for scene in dev.ABC_SCENES:
        values = [index for index, observed in enumerate(scenes) if observed == scene]
        ordered = sorted(
            values,
            key=lambda index: (
                hashlib.sha256(f"calvin:1729:{scene}:{index}".encode()).digest(),
                index,
            ),
        )
        validation.add(ordered[0])
    train = sorted(set(range(len(episodes))) - validation)
    heldout = sorted(validation)
    return {
        "algorithm": dev.SPLIT_ALGORITHM,
        "seed": 1729,
        "train_episode_indices": train,
        "train_episode_sha256": indices_hash(train),
        "validation_episode_indices": heldout,
        "validation_episode_sha256": indices_hash(heldout),
        "validation_fraction": 0.1,
    }


def _create_v2_index(
    path: Path,
    *,
    manifest_parts: dict[str, Any],
    critical: dict[str, dict[str, Any]],
    members: dict[str, bytes],
) -> None:
    members_sql = (
        "CREATE TABLE members("
        "path TEXT PRIMARY KEY,kind INTEGER NOT NULL CHECK(kind IN (0,1)),"
        "split TEXT CHECK(split IN ('training','validation') OR split IS NULL),"
        "global_index INTEGER CHECK(global_index >= 0 OR global_index IS NULL),"
        "local_header_offset INTEGER NOT NULL CHECK(local_header_offset >= 0),"
        "data_offset INTEGER NOT NULL CHECK(data_offset >= 0),"
        "compressed_bytes INTEGER NOT NULL CHECK(compressed_bytes >= 0),"
        "logical_bytes INTEGER NOT NULL CHECK(logical_bytes >= 0),"
        "method INTEGER NOT NULL,version_needed INTEGER NOT NULL,flags INTEGER NOT NULL,"
        "crc32 INTEGER NOT NULL CHECK(crc32 >= 0 AND crc32 <= 4294967295),"
        "logical_sha256 BLOB NOT NULL CHECK(length(logical_sha256) = 32),"
        "state_action_row INTEGER CHECK(state_action_row >= 0 OR state_action_row IS NULL)) WITHOUT ROWID"
    )
    with sqlite3.connect(path) as connection:
        connection.execute(members_sql)
        connection.execute("CREATE TABLE metadata(name TEXT PRIMARY KEY, value TEXT NOT NULL) WITHOUT ROWID")
        offset = 100
        for relative, raw in sorted(members.items()):
            split = None
            global_index = None
            if relative.startswith("training/episode_"):
                split = "training"
                global_index = int(relative[-11:-4])
            connection.execute(
                "INSERT INTO members VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    relative,
                    0,
                    split,
                    global_index,
                    offset,
                    offset + 30,
                    len(raw),
                    len(raw),
                    8,
                    20,
                    0,
                    zlib.crc32(raw) & 0xFFFFFFFF,
                    hashlib.sha256(raw).digest(),
                    None,
                ),
            )
            offset += len(raw) + 100
        connection.execute("CREATE INDEX members_local_header ON members(local_header_offset)")
        connection.execute(
            "CREATE UNIQUE INDEX members_episode ON members(split,global_index) WHERE global_index IS NOT NULL"
        )
        connection.execute("CREATE INDEX members_physical ON members(data_offset)")
        archive = manifest_parts["archive"]
        central = archive["central_directory"]
        inventory = archive["member_inventory"]
        metadata = {
            "archive_bytes": str(archive["bytes"]),
            "archive_root": dev.DATASET_NAME,
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
            "reader_schema": dev.ARCHIVE_READER_SCHEMA,
            "schema": dev.MEMBER_INDEX_SCHEMA,
            "state_action_sidecar_schema": dev.STATE_ACTION_SIDECAR_SCHEMA,
            "state_action_sidecar_status": "absent",
            "status": "complete",
            "uncompressed_bytes": str(inventory["uncompressed_bytes"]),
            **{f"critical_sha256:{name}": record["sha256"] for name, record in critical.items()},
        }
        connection.executemany("INSERT INTO metadata VALUES(?,?)", sorted(metadata.items()))
        connection.execute("PRAGMA application_id=1145853251")
        connection.execute("PRAGMA user_version=2")


def make_v4_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, Path, Path, Path, dict[str, bytes]]:
    data_root = tmp_path / "data/calvin"
    dataset_root = data_root / dev.DATASET_NAME
    training = dataset_root / "training"
    training.mkdir(parents=True)
    (dataset_root / "validation").mkdir()

    episodes = [(index * 4, index * 4 + 3) for index in range(6)]
    episode_scenes = [scene for scene in dev.ABC_SCENES for _ in range(2)]
    scene_info = {
        scene: [episodes[indexes[0]][0], episodes[indexes[-1]][1]]
        for scene in dev.ABC_SCENES
        for indexes in [[index for index, value in enumerate(episode_scenes) if value == scene]]
    }
    annotations: list[tuple[int, int]] = []
    instructions: list[str] = []
    tasks: list[str] = []
    episode_members: dict[str, bytes] = {}
    for global_index in range(24):
        annotations.append((global_index, global_index + 1))
        instructions.append(f"fixture instruction {global_index}")
        tasks.append(f"fixture_task_{global_index % 4}")
        robot = np.linspace(0.01, 0.15, 15, dtype=np.float64) + global_index
        robot[14] = -1.0 if global_index % 2 else 1.0
        scene = np.linspace(0.0, 0.23, 24, dtype=np.float64) + global_index
        action = np.zeros(7, dtype=np.float64)
        action[global_index % 6] = 0.25
        action[6] = robot[14]
        episode_members[f"training/episode_{global_index:07d}.npz"] = npz_bytes(
            robot_obs=robot,
            scene_obs=scene,
            rel_actions=action,
        )
    projected = {
        "training/ep_start_end_ids.npy": npy_bytes(np.asarray(episodes, dtype=np.int64), allow_pickle=False),
        "training/lang_annotations/auto_lang_ann.npy": npy_bytes(
            {"info": {"indx": annotations}, "language": {"ann": instructions, "task": tasks}},
            allow_pickle=True,
        ),
        "training/scene_info.npy": npy_bytes(scene_info, allow_pickle=True),
        "training/.hydra/merged_config.yaml": b"env:\n  _target_: fixture.Env\ncameras: {}\nscene: {}\n",
        "validation/ep_start_end_ids.npy": npy_bytes(np.asarray([[0, 0]], dtype=np.int64), allow_pickle=False),
        "validation/.hydra/merged_config.yaml": b"opaque: validation\n",
    }
    for relative, raw in projected.items():
        path = dataset_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
    critical = {
        name: {"bytes": len(raw), "crc32": zlib.crc32(raw) & 0xFFFFFFFF, "sha256": hashlib.sha256(raw).hexdigest()}
        for name, raw in projected.items()
    }
    all_members = {**projected, **episode_members}
    archive_sha256 = "a" * 64
    inventory_sha256 = "b" * 64
    central_sha256 = "c" * 64
    inventory = {
        "compressed_bytes": sum(map(len, all_members.values())),
        "directory_member_count": 0,
        "file_member_count": len(all_members),
        "member_count": len(all_members),
        "npz_member_count": len(episode_members),
        "sha256": inventory_sha256,
        "uncompressed_bytes": sum(map(len, all_members.values())),
    }
    archive = {
        "bytes": 123_456,
        "central_directory": {
            "bytes": 456,
            "entries": len(all_members),
            "offset": 123_000,
            "sha256": central_sha256,
            "zip64": True,
        },
        "member_inventory": inventory,
        "path": dev.ARCHIVE_NAME,
        "sha256": archive_sha256,
        "url": dev.ARCHIVE_URL,
    }
    index_path = data_root / dev.MEMBER_INDEX_NAME
    index_path.parent.mkdir(parents=True, exist_ok=True)
    _create_v2_index(index_path, manifest_parts={"archive": archive}, critical=critical, members=all_members)
    index_identity = {
        "bytes": index_path.stat().st_size,
        "path": dev.MEMBER_INDEX_NAME,
        "schema": dev.MEMBER_INDEX_SCHEMA,
        "sha256": hashlib.sha256(index_path.read_bytes()).hexdigest(),
    }
    manifest: dict[str, Any] = {
        "archive": archive,
        "checksum_url": dev.CHECKSUM_URL,
        "critical_files": critical,
        "dataset": dev.DATASET_NAME,
        "schema": dev.DATASET_MANIFEST_SCHEMA,
        "storage": {
            "derived_artifacts": {
                "state_action_sidecar": None,
                "state_action_sidecar_schema_hook": dev.STATE_ACTION_SIDECAR_SCHEMA,
            },
            "materialized_files": list(dev.DATASET_CRITICAL_FILES),
            "member_index": index_identity,
            "mode": "archive-direct",
            "reader_schema": dev.ARCHIVE_READER_SCHEMA,
            "verification": dev.VERIFICATION_CONTRACT,
        },
    }
    manifest["content_sha256"] = content_hash(manifest)
    manifest_path = data_root / dev.MANIFEST_NAME
    manifest_path.write_bytes(dev.canonical_json_bytes(manifest, pretty=True))
    manifest_file_sha256 = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    metadata_sha256 = dev.canonical_sha256(
        {f"training/{name}": critical[f"training/{name}"]["sha256"] for name in dev.TRAIN_CRITICAL_FILES}
    )
    storage_identity = dev._storage_identity(manifest, manifest_file_sha256)
    split = fixture_split(episodes, episode_scenes)
    stats: dict[str, Any] = {
        "action": {
            "continuous_dimensions": list(range(6)),
            "continuous_max": [1.0] * 6,
            "continuous_min": [-1.0] * 6,
            "dimension": 7,
            "gripper_index": 6,
            "observed_gripper_values": [-1.0, 1.0],
            "transform": "identity_official_scaled_rel_actions",
        },
        "algorithm": {
            "actions_re_normalized": False,
            "arrays_scanned": ["robot_obs", "rel_actions"],
            "canonical_ordinal_placement": "episode index ascending, then global timestep ascending",
            "frame_order": "archive data_offset ascending; values placed by canonical episode/global ordinal",
            "quantile": "numpy.quantile(method=linear)",
            "result_dtype": "float64",
            "selected_frame_exact_once_bitmap": True,
            "source_dtype": "float32",
            "state_gripper_excluded_from_percentiles": True,
        },
        "counts": {
            "tasks": 4,
            "total_annotations": 24,
            "total_episodes": 6,
            "training_episodes": 3,
            "training_frames": 12,
            "validation_episodes": 3,
            "validation_frames": 12,
        },
        "dataset": {
            "archive_bytes": archive["bytes"],
            "archive_sha256": archive_sha256,
            "central_directory_sha256": central_sha256,
            "dataset_manifest_file_sha256": manifest_file_sha256,
            "dataset_manifest_schema": dev.DATASET_MANIFEST_SCHEMA,
            "dataset_manifest_sha256": manifest["content_sha256"],
            "member_index": index_identity,
            "member_inventory_sha256": inventory_sha256,
            "metadata_files": list(dev.TRAIN_CRITICAL_FILES),
            "metadata_sha256": metadata_sha256,
            "name": dev.DATASET_NAME,
            "reader_schema": dev.ARCHIVE_READER_SCHEMA,
            "split": "training",
            "storage_identity_sha256": storage_identity["content_sha256"],
            "storage_mode": "archive-direct",
        },
        "schema": dev.NORMALIZATION_SCHEMA,
        "split": split,
        "state": {
            "constant_dimensions": [],
            "continuous_dimensions": list(range(7)),
            "dimension": 8,
            "gripper_index": 7,
            "observed_gripper_values": [-1.0, 1.0],
            "q01": [0.0] * 7,
            "q99": [1.0] * 7,
        },
    }
    stats["content_sha256"] = content_hash(stats)
    stats_path = tmp_path / "normalization-v4.json"
    stats_path.write_bytes(dev.canonical_json_bytes(stats, pretty=True))

    source_root = tmp_path / "simulators/calvin"
    scene_hashes: dict[str, str] = {}
    for scene in dev.ABC_SCENES:
        raw = f"name: {scene}\n".encode()
        path = source_root / "calvin_env/conf/scene" / f"{scene}.yaml"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
        scene_hashes[scene] = hashlib.sha256(raw).hexdigest()
    task_path = source_root / "calvin_models/conf/callbacks/rollout/tasks/new_playtable_tasks.yaml"
    task_path.parent.mkdir(parents=True, exist_ok=True)
    task_path.write_text("tasks: fixture\n", encoding="utf-8")
    (source_root / "calvin_env/tacto").mkdir(parents=True)
    revision_file = tmp_path / "revisions.env"
    revision_file.write_text(
        f"CALVIN_REVISION={dev.CALVIN_REVISION}\n"
        f"CALVIN_ENV_REVISION={dev.CALVIN_ENV_REVISION}\n"
        f"CALVIN_TACTO_REVISION={dev.CALVIN_TACTO_REVISION}\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(dev, "ARCHIVE_BYTES", archive["bytes"])
    monkeypatch.setattr(dev, "ARCHIVE_SHA256", archive_sha256)
    monkeypatch.setattr(dev, "OFFICIAL_CENTRAL_DIRECTORY_OFFSET", archive["central_directory"]["offset"])
    monkeypatch.setattr(dev, "OFFICIAL_CENTRAL_DIRECTORY_BYTES", archive["central_directory"]["bytes"])
    monkeypatch.setattr(dev, "OFFICIAL_CENTRAL_DIRECTORY_SHA256", central_sha256)
    monkeypatch.setattr(dev, "OFFICIAL_MEMBER_COUNT", inventory["member_count"])
    monkeypatch.setattr(dev, "OFFICIAL_FILE_MEMBER_COUNT", inventory["file_member_count"])
    monkeypatch.setattr(dev, "OFFICIAL_DIRECTORY_MEMBER_COUNT", 0)
    monkeypatch.setattr(dev, "OFFICIAL_NPZ_MEMBER_COUNT", inventory["npz_member_count"])
    monkeypatch.setattr(dev, "SCENE_CONFIG_SHA256", scene_hashes)
    monkeypatch.setattr(dev, "TASK_ORACLE_SHA256", hashlib.sha256(task_path.read_bytes()).hexdigest())

    def fake_revision(path: Path) -> str:
        if path.name == "tacto":
            return dev.CALVIN_TACTO_REVISION
        return dev.CALVIN_ENV_REVISION if path.name == "calvin_env" else dev.CALVIN_REVISION

    monkeypatch.setattr(dev, "_git_revision", fake_revision)
    monkeypatch.setattr(dev, "_git_clean", lambda _path: True)
    return training, stats_path, source_root, revision_file, episode_members


def make_replay_objects(
    inputs: dev.AuthenticatedCalvinDevInputs,
    episode_members: dict[str, bytes],
) -> tuple[dev.BundledCalvinReplay, ...]:
    values: list[dev.BundledCalvinReplay] = []
    for candidate in dev.load_validation_candidates(inputs.metadata, inputs.split):
        members: list[dict[str, Any]] = []
        actions: list[np.ndarray] = []
        robot = None
        scene = None
        for global_index in range(candidate.global_start, candidate.global_end_exclusive):
            path = f"training/episode_{global_index:07d}.npz"
            raw = episode_members[path]
            with np.load(io.BytesIO(raw), allow_pickle=False) as archive:
                if robot is None:
                    robot = np.asarray(archive["robot_obs"])
                    scene = np.asarray(archive["scene_obs"])
                actions.append(np.asarray(archive["rel_actions"]))
            members.append(
                {
                    "global_index": global_index,
                    "logical_bytes": len(raw),
                    "logical_sha256": hashlib.sha256(raw).hexdigest(),
                    "path": path,
                }
            )
        assert robot is not None and scene is not None
        values.append(
            dev.BundledCalvinReplay(
                candidate=candidate,
                frame=dev.CalvinResetFrame(robot, scene, members[0]["logical_sha256"]),
                actions=np.stack(actions),
                member_identities=tuple(members),
                record_sha256="",
            )
        )
    return tuple(values)


def build_bundle_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[dev.AuthenticatedCalvinDevInputs, dict[str, Any], dict[str, bytes], tuple[dev.BundledCalvinReplay, ...]]:
    training, stats, source, revisions, members = make_v4_fixture(tmp_path, monkeypatch)
    inputs = dev.authenticate_dev_inputs(training, stats, source, revisions)
    replays = make_replay_objects(inputs, members)
    source_identity = {
        "calvin_archive_source_sha256": inputs.identity["calvin_archive_source_sha256"],
        "dev_states_source_sha256": inputs.identity["dev_states_source_sha256"],
        "replay_exporter_source_sha256": inputs.identity["replay_exporter_source_sha256"],
        "schema": "duo-vla-calvin-dev-replay-export-source-v1",
    }
    manifest, artifacts = dev.build_replay_bundle(replays, inputs, source_identity)
    return inputs, manifest, artifacts, replays
