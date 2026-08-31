from __future__ import annotations

import ast
import base64
import copy
import hashlib
import importlib
import importlib.util
import json
import os
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
CALVIN_SCRIPTS = ROOT / "scripts" / "calvin"
sys.path.insert(0, str(CALVIN_SCRIPTS))
SCRIPT = CALVIN_SCRIPTS / "serve_policy.py"
SPEC = importlib.util.spec_from_file_location("calvin_policy_server", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
SERVER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SERVER)

from calvin_bridge import SEQUENCE_SHA256, PolicyClient, wait_for_socket  # noqa: E402

from duo_vla.backbones.loading import expected_decoder_attention_lora_adapter_config  # noqa: E402
from duo_vla.data.calvin_archive import (  # noqa: E402
    CALVIN_ARCHIVE_READER_SCHEMA,
    CALVIN_INDEX_NAME,
    CALVIN_MEMBER_INDEX_SCHEMA,
    OFFICIAL_CENTRAL_DIRECTORY_SHA256,
)
from duo_vla.data.calvin_stats import (  # noqa: E402
    CALVIN_ABC_D_ARCHIVE_BYTES,
    CALVIN_ABC_D_ARCHIVE_URL,
    CALVIN_CRITICAL_TRAIN_METADATA,
    CALVIN_DATASET_CRITICAL_FILES,
    CALVIN_DATASET_MANIFEST_SCHEMA,
    CALVIN_DATASET_MANIFEST_SCHEMA_V3,
    CALVIN_STATS_SCHEMA,
    CALVIN_STORAGE_MODE_ARCHIVE_DIRECT,
    calvin_metadata_sha256,
)
from duo_vla.normalization import ActionNormalizer, PercentileNormalizer  # noqa: E402
from duo_vla.policy_contract import policy_contract_from_config  # noqa: E402
from duo_vla.prefix_geometry import (  # noqa: E402
    CameraGeometry,
    SnapshotTreeIdentity,
    build_prefix_geometry_contract,
    save_prefix_geometry_contract,
)
from duo_vla.run_config import (  # noqa: E402
    canonical_config_sha256,
    load_resolved_toml,
    save_resolved_config,
)
from duo_vla.run_journal import create_run_journal, record_latest_checkpoint  # noqa: E402
from duo_vla.runtime_integrity import (  # noqa: E402
    BASE_PYTHON_RUNTIME_IDENTITY_SCHEMA,
    TRAIN_VENV_IDENTITY_SCHEMA,
    canonical_sha256,
    static_environment_identity,
)


def _train_venv_identity() -> dict[str, object]:
    venv_root = "/root/.cache/duo-vla/venvs/train"
    base_python_runtime: dict[str, object] = {
        "base_prefix": "/usr/local",
        "configured_home": "/usr/local/bin",
        "configured_home_resolved": "/usr/local/bin",
        "content_inventory_sha256": "5" * 64,
        "files_verified": 100,
        "pyvenv_cfg_bytes": 128,
        "pyvenv_cfg_sha256": "6" * 64,
        "resolved_executable": "/usr/local/bin/python3.11",
        "resolved_executable_bytes": 20_000,
        "resolved_executable_sha256": "7" * 64,
        "schema": BASE_PYTHON_RUNTIME_IDENTITY_SCHEMA,
        "startup_hooks": [],
        "startup_hooks_sha256": canonical_sha256([]),
        "symlinks_verified": 4,
        "total_bytes": 1_000_000,
        "tree_metadata_sha256": "8" * 64,
        "venv_python": f"{venv_root}/bin/python",
        "venv_python_link_target": "/usr/local/bin/python3.11",
        "venv_root": venv_root,
    }
    base_python_runtime["root_sha256"] = canonical_sha256(base_python_runtime)
    identity: dict[str, object] = {
        "base_python_runtime": base_python_runtime,
        "content_inventory_sha256": "8" * 64,
        "files_verified": 10,
        "root": venv_root,
        "schema": TRAIN_VENV_IDENTITY_SCHEMA,
        "startup_hooks": ["lib/python3.11/site-packages/known.pth"],
        "startup_hooks_sha256": "a" * 64,
        "symlinks_verified": 3,
        "total_bytes": 100,
        "tree_metadata_sha256": "b" * 64,
    }
    identity["root_sha256"] = canonical_sha256(
        {
            "base_python_runtime_root_sha256": base_python_runtime["root_sha256"],
            "content_inventory_sha256": identity["content_inventory_sha256"],
            "files_verified": identity["files_verified"],
            "schema": identity["schema"],
            "startup_hooks_sha256": identity["startup_hooks_sha256"],
            "symlinks_verified": identity["symlinks_verified"],
            "total_bytes": identity["total_bytes"],
            "tree_metadata_sha256": identity["tree_metadata_sha256"],
        }
    )
    return identity


@pytest.fixture(autouse=True)
def _fake_live_train_venv(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DUO_VLA_CACHE_ROOT", "/root/.cache/duo-vla")
    monkeypatch.setenv("HF_HOME", "/root/.cache/huggingface")
    monkeypatch.setattr(SERVER, "content_address_train_venv", lambda _root: _train_venv_identity())


def _split() -> dict[str, object]:
    train = [0, 2]
    validation = [1, 3]
    return {
        "algorithm": "scene-grouped stable sha256 whole-episode ordering with all-task coverage assertion",
        "seed": 1729,
        "train_episode_indices": train,
        "train_episode_sha256": hashlib.sha256(b"0,2").hexdigest(),
        "validation_episode_indices": validation,
        "validation_episode_sha256": hashlib.sha256(b"1,3").hexdigest(),
        "validation_fraction": 0.1,
    }


def _normalization_dataset_identity() -> dict[str, object]:
    return {
        "archive_bytes": CALVIN_ABC_D_ARCHIVE_BYTES,
        "archive_sha256": SERVER.ARCHIVE_SHA256,
        "central_directory_sha256": OFFICIAL_CENTRAL_DIRECTORY_SHA256,
        "dataset_manifest_file_sha256": "5" * 64,
        "dataset_manifest_schema": CALVIN_DATASET_MANIFEST_SCHEMA,
        "dataset_manifest_sha256": "3" * 64,
        "member_index": {
            "bytes": 238_215_168,
            "path": CALVIN_INDEX_NAME,
            "schema": CALVIN_MEMBER_INDEX_SCHEMA,
            "sha256": "6" * 64,
        },
        "member_inventory_sha256": "4" * 64,
        "metadata_files": list(CALVIN_CRITICAL_TRAIN_METADATA),
        "metadata_sha256": "2" * 64,
        "name": "task_ABC_D",
        "reader_schema": CALVIN_ARCHIVE_READER_SCHEMA,
        "split": "training",
        "storage_identity_sha256": "7" * 64,
        "storage_mode": CALVIN_STORAGE_MODE_ARCHIVE_DIRECT,
    }


def _stats() -> dict[str, object]:
    split = _split()
    payload: dict[str, object] = {
        "algorithm": {"actions_re_normalized": False},
        "content_sha256": "1" * 64,
        "counts": {},
        "dataset": _normalization_dataset_identity(),
        "schema": CALVIN_STATS_SCHEMA,
        "split": split,
        "state": {
            "continuous_dimensions": list(range(7)),
            "dimension": 8,
            "gripper_index": 7,
            "observed_gripper_values": [-1.0, 1.0],
            "q01": [-1.0] * 7,
            "q99": [1.0] * 7,
        },
        "action": {
            "continuous_dimensions": list(range(6)),
            "continuous_max": [1.0] * 6,
            "continuous_min": [-1.0] * 6,
            "dimension": 7,
            "gripper_index": 6,
            "observed_gripper_values": [-1.0, 1.0],
            "transform": SERVER.CALVIN_ACTION_ADAPTER,
        },
    }
    return payload


def test_normalization_v4_permanent_identity_rejects_missing_and_forged_fields() -> None:
    stats = _stats()
    expected = _normalization_dataset_identity()
    permanent = SERVER._validate_normalization_contract(stats)
    assert permanent == {name: expected[name] for name in SERVER._CALVIN_PERMANENT_STORAGE_FIELDS}

    missing = copy.deepcopy(stats)
    del missing["dataset"]["reader_schema"]  # type: ignore[index]
    with pytest.raises(RuntimeError, match="field inventory differs"):
        SERVER._validate_normalization_contract(missing)

    forged = copy.deepcopy(stats)
    forged["dataset"]["central_directory_sha256"] = "f" * 64  # type: ignore[index]
    with pytest.raises(RuntimeError, match="central_directory_sha256"):
        SERVER._validate_normalization_contract(forged)

    widened_index = copy.deepcopy(stats)
    widened_index["dataset"]["member_index"]["forged"] = True  # type: ignore[index]
    with pytest.raises(RuntimeError, match="member-index identity field inventory differs"):
        SERVER._validate_normalization_contract(widened_index)


def _training_execution_environment(*, seed: int = 1) -> dict[str, object]:
    train_prefix = Path("/root/.cache/duo-vla/venvs/train")
    site_packages = train_prefix / "lib/python3.11/site-packages"
    module_origins = {
        "duo_vla": str((ROOT / "src/duo_vla/__init__.py").resolve()),
        **{
            module: str((site_packages / module / "__init__.py").resolve())
            for module in SERVER._DISTRIBUTION_IMPORT_NAMES.values()
        },
    }
    train_environment = {
        **SERVER.REQUIRED_TRAIN_ENVIRONMENT,
        "DUO_VLA_CACHE_ROOT": "/root/.cache/duo-vla",
        "DUO_VLA_PROJECT_ROOT": str(ROOT),
        "DUO_VLA_TRAIN_VENV": str(train_prefix),
        "HF_HOME": "/root/.cache/huggingface",
        "PYTHONHASHSEED": str(seed),
    }
    version = f"python{SERVER.sys.version_info.major}.{SERVER.sys.version_info.minor}"
    compact_version = f"python{SERVER.sys.version_info.major}{SERVER.sys.version_info.minor}"
    authenticated_runtime = {
        "environment": train_environment,
        "lock_sha256": SERVER.TRAIN_LOCK_SHA256,
        "module_origins": module_origins,
        "packages": SERVER.EXPECTED_TRAIN_PACKAGES,
        "python": SERVER.EXPECTED_TRAIN_PYTHON,
        "static_environment_sha256": static_environment_identity(train_environment)["sha256"],
        "sys_path": [
            str((ROOT / "src").resolve()),
            str(Path(SERVER.sys.base_prefix) / "lib" / f"{compact_version}.zip"),
            str(Path(SERVER.sys.base_prefix) / "lib" / version),
            str(Path(SERVER.sys.base_exec_prefix) / "lib" / version / "lib-dynload"),
            str(site_packages.resolve()),
        ],
        "torchrun": {
            "group_world_size": 1,
            "local_rank_equals_rank": True,
            "local_world_size": 2,
            "role_world_size": 2,
            "world_size": 2,
        },
        "train_venv": _train_venv_identity(),
    }
    return {
        "authenticated_runtime": authenticated_runtime,
        "cublas_workspace_config": ":4096:8",
        "cuda_runtime": "12.6",
        "cudnn": 91002,
        "cudnn_benchmark": False,
        "cudnn_deterministic": True,
        "cudnn_tf32": False,
        "deterministic_algorithms": True,
        "deterministic_warn_only": False,
        "float32_matmul_precision": "highest",
        "gpu_capability": [[8, 6], [8, 6]],
        "gpu_names": ["Test GPU", "Test GPU"],
        "matmul_tf32": False,
        "peft": SERVER.EXPECTED_TRAIN_PACKAGES["peft"],
        "python": SERVER.EXPECTED_TRAIN_PYTHON,
        "python_hash_seed": str(seed),
        "torch": SERVER.EXPECTED_TRAIN_PACKAGES["torch"],
        "transformers": SERVER.EXPECTED_TRAIN_PACKAGES["transformers"],
        "world_size": 2,
    }


def _model_snapshot_report() -> dict[str, object]:
    return {
        "content_inventory_sha256": "5" * 64,
        "files_verified": 1,
        "id": SERVER.MODEL_ID,
        "revision": SERVER.MODEL_REVISION,
        "shards": 11,
        "snapshot": "/authenticated/model/snapshot",
        "total_snapshot_bytes": 1,
        "tree_metadata_sha256": "3" * 64,
    }


def _execution_geometry(prefix_sha256: str = "6" * 64, width: int = 600) -> dict[str, object]:
    return {
        "expert_batch_isolation": SERVER.EXPERT_BATCH_ISOLATION,
        "experts_implementation": SERVER.EXPERTS_IMPLEMENTATION,
        "fixed_physical_prefix_width": width,
        "physical_batch_size": SERVER.PHYSICAL_BATCH_SIZE,
        "prefix_geometry_content_sha256": prefix_sha256,
    }


def _checkpoint_fixture(tmp_path: Path) -> tuple[Path, dict[str, object], dict[str, object], dict[str, object]]:
    checkpoint = tmp_path / "update-030000"
    artifacts = checkpoint / "artifacts"
    artifacts.mkdir(parents=True)
    lora_dir = checkpoint / "lora"
    lora_dir.mkdir()
    (lora_dir / "adapter_config.json").write_text(
        json.dumps(expected_decoder_attention_lora_adapter_config()),
        encoding="utf-8",
    )
    stats = _stats()
    split = stats["split"]
    normalization_dataset = stats["dataset"]
    assert isinstance(normalization_dataset, dict)
    member_index = normalization_dataset["member_index"]
    assert isinstance(member_index, dict)
    source_revisions = SERVER.PINNED_CALVIN_SOURCE_REVISIONS
    identity = {
        "action_adapter": SERVER.CALVIN_ACTION_ADAPTER,
        **{name: copy.deepcopy(normalization_dataset[name]) for name in SERVER._CALVIN_PERMANENT_STORAGE_FIELDS},
        "calvin_source_revisions": source_revisions,
        "calvin_source_revisions_sha256": canonical_config_sha256(source_revisions),
        "camera_shapes": SERVER.CALVIN_CAMERA_SHAPES,
        "camera_shapes_sha256": canonical_config_sha256(SERVER.CALVIN_CAMERA_SHAPES),
        "normalization_sha256": "1" * 64,
        "protocol": SERVER.PROTOCOL,
        "split": split,
        "split_sha256": canonical_config_sha256(split),
        "state_adapter": SERVER.CALVIN_STATE_ADAPTER,
    }
    config = load_resolved_toml(ROOT / "configs/calvin_abc_to_d.toml")
    snapshot = SnapshotTreeIdentity(
        repository_id=SERVER.MODEL_ID,
        revision=SERVER.MODEL_REVISION,
        tree_metadata_sha256="3" * 64,
        content_inventory_sha256="5" * 64,
        files_verified=1,
        total_bytes=1,
    )
    prefix = build_prefix_geometry_contract(
        model_identity=snapshot,
        processor_identity=snapshot,
        ordered_cameras=(
            CameraGeometry("rgb_static", 200, 200),
            CameraGeometry("rgb_gripper", 84, 84),
        ),
        instruction_lengths={"fixture instruction": 532},
        fixed_physical_prefix_width=600,
        padding_side="left",
    )
    save_prefix_geometry_contract(artifacts / "prefix_geometry.json", prefix)
    config["model"]["experts_implementation"] = SERVER.EXPERTS_IMPLEMENTATION
    config["model"]["expert_batch_isolation"] = SERVER.EXPERT_BATCH_ISOLATION
    config["optimization"]["global_batch_size"] = 64
    config["optimization"]["microbatch_size"] = SERVER.PHYSICAL_BATCH_SIZE
    config["optimization"]["gradient_accumulation_steps"] = 8
    config["optimization"]["physical_batch_size"] = SERVER.PHYSICAL_BATCH_SIZE
    config["benchmark"]["prefix_geometry_content_sha256"] = prefix["content_sha256"]
    config["benchmark"]["fixed_physical_prefix_width"] = 600
    execution_geometry = _execution_geometry(prefix["content_sha256"], 600)
    config["execution_geometry"] = execution_geometry
    config["training_instruction_inventory_sha256"] = "7" * 64
    config["artifact_trees"] = {"model_tree_sha256": "3" * 64}
    config["calvin_identity"] = copy.deepcopy(identity)
    execution_environment = _training_execution_environment(seed=1)
    config["execution_environment"] = execution_environment
    config["run"] = {"max_cached_frames": 512, "seed": 1, "task": None}
    config["source_tree_sha256"] = SERVER._source_tree_sha256(ROOT)
    config_sha256 = save_resolved_config(artifacts / "resolved_config.json", config)
    (artifacts / "normalization.json").write_text("{}\n", encoding="utf-8")
    (checkpoint / "manifest.json").write_text("{}\n", encoding="utf-8")
    policy_contract = policy_contract_from_config(config).to_dict()
    manifest: dict[str, object] = {
        "action_adapter": SERVER.CALVIN_ACTION_ADAPTER,
        "archive_bytes": normalization_dataset["archive_bytes"],
        "archive_sha256": SERVER.ARCHIVE_SHA256,
        "artifacts": {
            "interface": {"path": "interface.safetensors"},
            "lora_config": {"path": "lora/adapter_config.json"},
            "lora_weights": {"path": "lora/adapter_model.safetensors"},
            "normalization": {"path": "artifacts/normalization.json"},
            "prefix_geometry": {"path": "artifacts/prefix_geometry.json"},
            "resolved_config": {"path": "artifacts/resolved_config.json"},
        },
        "calvin_identity": identity,
        "calvin_env_revision": source_revisions["calvin_env"],
        "calvin_revision": source_revisions["calvin"],
        "calvin_source_revisions": source_revisions,
        "calvin_source_revisions_sha256": canonical_config_sha256(source_revisions),
        "calvin_tacto_revision": source_revisions["tacto"],
        "camera_shapes": SERVER.CALVIN_CAMERA_SHAPES,
        "camera_shapes_sha256": canonical_config_sha256(SERVER.CALVIN_CAMERA_SHAPES),
        "complete": True,
        "config_sha256": config_sha256,
        "configured_total_updates": 30_000,
        "dataset": "task_ABC_D",
        "dataset_manifest_file_sha256": normalization_dataset["dataset_manifest_file_sha256"],
        "dataset_manifest_schema": normalization_dataset["dataset_manifest_schema"],
        "dataset_manifest_sha256": "3" * 64,
        "dataset_split": "training",
        "execution_environment": execution_environment,
        "execution_environment_sha256": canonical_config_sha256(execution_environment),
        "execution_geometry": execution_geometry,
        "expert_batch_isolation": SERVER.EXPERT_BATCH_ISOLATION,
        "experts_implementation": SERVER.EXPERTS_IMPLEMENTATION,
        "fixed_physical_prefix_width": 600,
        "kind": "resumable-calvin-abc-to-d-training",
        "central_directory_sha256": normalization_dataset["central_directory_sha256"],
        "member_index_bytes": member_index["bytes"],
        "member_index_path": member_index["path"],
        "member_index_schema": member_index["schema"],
        "member_index_sha256": member_index["sha256"],
        "metadata_sha256": "2" * 64,
        "model_id": SERVER.MODEL_ID,
        "model_revision": SERVER.MODEL_REVISION,
        "model_tree_sha256": "3" * 64,
        "member_inventory_sha256": "4" * 64,
        "normalization_sha256": "1" * 64,
        "policy_contract": policy_contract,
        "policy_contract_sha256": canonical_config_sha256(policy_contract),
        "physical_batch_size": SERVER.PHYSICAL_BATCH_SIZE,
        "prefix_geometry": {
            "instruction_inventory_sha256": prefix["instruction_inventory"]["sha256"],
            "maximum_valid_prefix_length": 532,
            "padding_side": "left",
        },
        "prefix_geometry_content_sha256": prefix["content_sha256"],
        "protocol": SERVER.PROTOCOL,
        "run_seed": 1,
        "source_tree_sha256": config["source_tree_sha256"],
        "split": split,
        "split_sha256": canonical_config_sha256(split),
        "state_adapter": SERVER.CALVIN_STATE_ADAPTER,
        "reader_schema": normalization_dataset["reader_schema"],
        "storage_identity_sha256": normalization_dataset["storage_identity_sha256"],
        "storage_mode": normalization_dataset["storage_mode"],
        "train_episode_sha256": split["train_episode_sha256"],
        "training_instruction_inventory_sha256": "7" * 64,
        "trainer_state": {
            "examples_seen": 30_000 * 64,
            "next_update": 30_000,
            "schema": "duo-vla-trainer-state-v1",
        },
        "validation_episode_sha256": split["validation_episode_sha256"],
        "last_metrics": {"examples_seen": 30_000 * 64, "update": 30_000},
    }
    return checkpoint, manifest, config, stats


def test_checkpoint_permanent_identity_rejects_flat_mismatch_and_missing_nested_field(tmp_path: Path) -> None:
    _checkpoint, manifest, config, stats = _checkpoint_fixture(tmp_path)
    expected = SERVER._validate_normalization_contract(stats)
    SERVER._validate_checkpoint_permanent_storage_identity(manifest, expected)
    assert (
        SERVER._validate_checkpoint_calvin_identity(manifest["calvin_identity"], expected) == config["calvin_identity"]
    )

    mismatched = copy.deepcopy(manifest)
    mismatched["member_index_sha256"] = "f" * 64
    with pytest.raises(RuntimeError, match="member_index"):
        SERVER._validate_checkpoint_permanent_storage_identity(mismatched, expected)

    missing = copy.deepcopy(manifest["calvin_identity"])
    del missing["storage_identity_sha256"]  # type: ignore[index]
    with pytest.raises(RuntimeError, match="field inventory differs"):
        SERVER._validate_checkpoint_calvin_identity(missing, expected)


def test_server_resolves_full_calvin_checkpoint_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import duo_vla.backbones.loading
    import duo_vla.checkpointing
    import duo_vla.data.calvin_stats
    import duo_vla.run_config

    checkpoint, manifest, config, stats = _checkpoint_fixture(tmp_path)
    monkeypatch.setattr(duo_vla.checkpointing, "load_checkpoint_manifest", lambda *args, **kwargs: manifest)
    monkeypatch.setattr(
        duo_vla.backbones.loading,
        "validate_decoder_attention_lora_weights",
        lambda *args, **kwargs: {},
    )
    normalization_loads: list[dict[str, object]] = []

    def load_normalization(*_args: object, **kwargs: object) -> tuple[object, dict[str, object]]:
        normalization_loads.append(dict(kwargs))
        return object(), stats

    monkeypatch.setattr(duo_vla.data.calvin_stats, "load_calvin_state_normalizer", load_normalization)
    original_load_resolved_toml = duo_vla.run_config.load_resolved_toml
    monkeypatch.setattr(
        duo_vla.run_config,
        "load_resolved_toml",
        lambda path: (
            copy.deepcopy(config)
            if Path(path).name in {"calvin_abc_to_d.toml", "calvin_abc_to_d_direct.toml"}
            else original_load_resolved_toml(path)
        ),
    )
    committed_record = SimpleNamespace(
        update=30_000,
        manifest_sha256=SERVER.sha256_file(checkpoint / "manifest.json"),
    )
    committed_calls: list[tuple[Path, str]] = []

    def committed(path: Path, config_sha256: str) -> SimpleNamespace:
        committed_calls.append((path, config_sha256))
        return committed_record

    monkeypatch.setattr(SERVER, "_committed_checkpoint_record", committed)

    authenticated_generation = object()
    _, normalization, seed, report, resolved, contract, identities = SERVER.resolve_checkpoint(
        checkpoint,
        training_root=tmp_path / "task_ABC_D/training",
        project_root=ROOT,
        train_seed_override=None,
        model_snapshot_report=_model_snapshot_report(),
        authenticated_generation=authenticated_generation,
    )

    assert normalization == checkpoint / "artifacts/normalization.json"
    assert seed == 1
    assert resolved == config
    assert contract["objective"] == "rectified_flow" and contract["nfe"] == 10
    assert report["normalization_content_sha256"] == "1" * 64
    assert identities["normalization_metadata_sha256"] == "2" * 64
    assert report["calvin_identity"] == identities["calvin_identity"] == _normalization_dataset_identity()
    assert normalization_loads[0]["authenticated_generation"] is authenticated_generation
    assert normalization_loads[0]["training_root"] == tmp_path / "task_ABC_D/training"
    assert committed_calls == [(checkpoint.resolve(), manifest["config_sha256"])]

    committed_record.update = 29_999
    with pytest.raises(RuntimeError, match="run journal update differs"):
        SERVER.resolve_checkpoint(
            checkpoint,
            training_root=tmp_path / "task_ABC_D/training",
            project_root=ROOT,
            train_seed_override=None,
            model_snapshot_report=_model_snapshot_report(),
        )
    committed_record.update = 30_000
    committed_record.manifest_sha256 = "f" * 64
    with pytest.raises(RuntimeError, match="run journal manifest SHA-256 differs"):
        SERVER.resolve_checkpoint(
            checkpoint,
            training_root=tmp_path / "task_ABC_D/training",
            project_root=ROOT,
            train_seed_override=None,
            model_snapshot_report=_model_snapshot_report(),
        )
    committed_record.manifest_sha256 = SERVER.sha256_file(checkpoint / "manifest.json")

    tampered = copy.deepcopy(stats)
    tampered["action"]["transform"] = "percentile"  # type: ignore[index]
    monkeypatch.setattr(
        duo_vla.data.calvin_stats,
        "load_calvin_state_normalizer",
        lambda *args, **kwargs: (object(), tampered),
    )
    with pytest.raises(RuntimeError, match="official scaled rel_actions"):
        SERVER.resolve_checkpoint(
            checkpoint,
            training_root=tmp_path / "task_ABC_D/training",
            project_root=ROOT,
            train_seed_override=None,
            model_snapshot_report=_model_snapshot_report(),
        )

    alternate = copy.deepcopy(manifest)
    alternate["artifacts"]["interface"]["path"] = "artifacts/benign-interface.safetensors"  # type: ignore[index]
    monkeypatch.setattr(duo_vla.checkpointing, "load_checkpoint_manifest", lambda *args, **kwargs: alternate)
    with pytest.raises(RuntimeError, match="not canonical"):
        SERVER.resolve_checkpoint(
            checkpoint,
            training_root=tmp_path / "task_ABC_D/training",
            project_root=ROOT,
            train_seed_override=None,
            model_snapshot_report=_model_snapshot_report(),
        )


def test_official_checkpoint_must_be_the_transactionally_committed_journal_tip(tmp_path: Path) -> None:
    output = tmp_path / "run"
    output.mkdir()
    config_sha256 = "a" * 64
    journal = create_run_journal(output, config_sha256=config_sha256)
    checkpoint = output / "checkpoints/update-030000"
    checkpoint.mkdir(parents=True)
    metrics = {"examples_seen": 30_000 * 64, "update": 30_000}
    manifest = {
        "config_sha256": config_sha256,
        "last_metrics": metrics,
        "parent_manifest_sha256": None,
        "run_uuid": journal.run_uuid,
        "trainer_state": {"next_update": 30_000},
    }
    (checkpoint / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    manifest_sha256 = SERVER.sha256_file(checkpoint / "manifest.json")
    record_latest_checkpoint(
        output,
        checkpoint=checkpoint,
        update=30_000,
        manifest_sha256=manifest_sha256,
        parent_manifest_sha256=None,
        last_metrics=metrics,
    )

    record = SERVER._committed_checkpoint_record(checkpoint, config_sha256)
    assert record.update == 30_000 and record.manifest_sha256 == manifest_sha256

    stale = output / "checkpoints/update-029999"
    stale.mkdir()
    (stale / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="not the journal latest"):
        SERVER._committed_checkpoint_record(stale, config_sha256)

    outside = tmp_path / "update-030000"
    outside.mkdir()
    with pytest.raises(RuntimeError, match="below a checkpoints directory"):
        SERVER._committed_checkpoint_record(outside, config_sha256)


def test_checkpoint_prefix_geometry_is_external_config_bound_and_old_checkpoints_fail(tmp_path: Path) -> None:
    checkpoint, manifest, config, _stats_payload = _checkpoint_fixture(tmp_path)

    payload, geometry = SERVER._load_checkpoint_prefix_geometry(
        checkpoint,
        manifest,
        config,
        model_snapshot_report=_model_snapshot_report(),
    )
    assert payload["content_sha256"] == geometry["prefix_geometry_content_sha256"]

    legacy = copy.deepcopy(manifest)
    del legacy["artifacts"]["prefix_geometry"]
    with pytest.raises(RuntimeError, match="no canonical copied"):
        SERVER._load_checkpoint_prefix_geometry(
            checkpoint,
            legacy,
            config,
            model_snapshot_report=_model_snapshot_report(),
        )

    substituted = copy.deepcopy(config)
    substituted["benchmark"]["prefix_geometry_content_sha256"] = "f" * 64
    substituted["execution_geometry"]["prefix_geometry_content_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="externally pinned"):
        SERVER._load_checkpoint_prefix_geometry(
            checkpoint,
            manifest,
            substituted,
            model_snapshot_report=_model_snapshot_report(),
        )


def test_server_rejects_checkpoint_from_unpinned_training_runtime() -> None:
    execution = _training_execution_environment(seed=1)
    config = {"execution_environment": copy.deepcopy(execution)}
    manifest = {
        "execution_environment": copy.deepcopy(execution),
        "execution_environment_sha256": canonical_config_sha256(execution),
    }
    manifest["execution_environment"]["python"] = "3.11.14"  # type: ignore[index]
    manifest["execution_environment_sha256"] = canonical_config_sha256(manifest["execution_environment"])
    config["execution_environment"] = copy.deepcopy(manifest["execution_environment"])

    with pytest.raises(RuntimeError, match="canonical training runtime"):
        SERVER._validate_checkpoint_training_environment(
            manifest,
            config,
            project_root=ROOT,
            run_seed=1,
        )


def test_server_rejects_live_train_venv_drift(monkeypatch: pytest.MonkeyPatch) -> None:
    execution = _training_execution_environment(seed=1)
    config = {"execution_environment": copy.deepcopy(execution)}
    manifest = {
        "execution_environment": copy.deepcopy(execution),
        "execution_environment_sha256": canonical_config_sha256(execution),
    }
    changed = _train_venv_identity()
    changed["content_inventory_sha256"] = "f" * 64
    monkeypatch.setattr(SERVER, "content_address_train_venv", lambda _root: changed)

    with pytest.raises(RuntimeError, match="live train venv differs"):
        SERVER._validate_checkpoint_training_environment(
            manifest,
            config,
            project_root=ROOT,
            run_seed=1,
        )


def test_dataset_manifest_authenticates_archive_and_critical_files(tmp_path: Path) -> None:
    root = tmp_path / "task_ABC_D"
    critical: dict[str, str] = {}
    for index, relative in enumerate(CALVIN_DATASET_CRITICAL_FILES):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"critical-{index}".encode())
        critical[relative] = SERVER.sha256_file(path)
    inventory = {
        "compressed_bytes": 100,
        "file_member_count": 10,
        "member_count": 12,
        "npz_member_count": 4,
        "sha256": "5" * 64,
        "uncompressed_bytes": 200,
    }
    payload: dict[str, object] = {
        "archive": {
            "bytes": CALVIN_ABC_D_ARCHIVE_BYTES,
            "member_inventory": inventory,
            "sha256": SERVER.ARCHIVE_SHA256,
            "uncompressed_bytes": 123,
            "url": CALVIN_ABC_D_ARCHIVE_URL,
        },
        "checksum_url": "http://calvin.cs.uni-freiburg.de/dataset/sha256sum.txt",
        "critical_files": critical,
        "dataset": "task_ABC_D",
        "extraction": {
            "file_members_verified": inventory["file_member_count"],
            "member_index": {
                "bytes": 1,
                "path": "task_ABC_D.members.sqlite3",
                "schema": "duo-vla-calvin-member-index-v1",
                "sha256": "6" * 64,
            },
            "verification": "size-and-crc32-against-every-pinned-zip-member",
        },
        "schema": CALVIN_DATASET_MANIFEST_SCHEMA_V3,
    }
    payload["content_sha256"] = SERVER._canonical_sha256(payload)
    manifest = root.with_name("task_ABC_D.manifest.json")
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    with root.with_suffix(".zip").open("wb") as archive:
        archive.truncate(CALVIN_ABC_D_ARCHIVE_BYTES)

    report = SERVER.verify_dataset(root, verify_archive=False, allow_legacy_v3=True)

    assert report["archive_sha256"] == SERVER.ARCHIVE_SHA256
    assert report["dataset_manifest_content_sha256"] == payload["content_sha256"]
    assert report["member_inventory_sha256"] == "5" * 64
    assert report["metadata_sha256"] == calvin_metadata_sha256(root / "training")
    (root / CALVIN_DATASET_CRITICAL_FILES[0]).write_bytes(b"tampered")
    with pytest.raises(ValueError, match="critical file changed"):
        SERVER.verify_dataset(root, verify_archive=False, allow_legacy_v3=True)


def test_calvin_output_is_identity_scaled_with_only_clip_and_gripper_sign() -> None:
    raw = torch.tensor(
        [[[2.0, -2.0, 0.25, -0.5, 1.0, -1.0, 0.0]] * SERVER.ACTION_HORIZON],
        dtype=torch.float32,
    )

    output = SERVER.finalize_calvin_actions(raw)

    assert output.shape == (1, 8, 7)
    assert output[0, 0].tolist() == [1.0, -1.0, 0.25, -0.5, 1.0, -1.0, -1.0]
    assert raw[0, 0, 0].item() == 2.0


def test_singleton_serving_is_exactly_replicated_and_batch_finalization_is_safe() -> None:
    singleton = torch.arange(56, dtype=torch.float32).reshape(1, 8, 7)
    replicated = SERVER._replicate_singleton_tensor(singleton)

    assert replicated.shape == (8, 8, 7)
    SERVER._require_bitwise_exact_replicas(replicated)
    finalized = SERVER.finalize_calvin_actions(replicated)
    assert finalized.shape == replicated.shape
    SERVER._require_bitwise_exact_replicas(finalized)

    changed = replicated.clone()
    changed[-1, -1, -1] += 1
    with pytest.raises(RuntimeError, match="bitwise identical"):
        SERVER._require_bitwise_exact_replicas(changed)
    with pytest.raises(RuntimeError, match="singleton"):
        SERVER._replicate_singleton_tensor(replicated)


def test_only_calvin_state_percentiles_are_moved_and_applied() -> None:
    source = ActionNormalizer(
        PercentileNormalizer(torch.full((7,), -2.0), torch.full((7,), 2.0)),
        action_dim=8,
        gripper_index=7,
    )

    placed = SERVER.place_calvin_state_normalizer(source, device=torch.device("cpu"), dtype=torch.float32)
    state = torch.tensor([[2.0] * 7 + [-1.0]])

    assert placed.normalize(state).tolist() == [[1.0] * 7 + [-1.0]]
    assert placed.action_dim == 8 and placed.resolved_gripper_index == 7


def test_processor_receives_static_then_gripper_then_raw_instruction() -> None:
    class Inputs(dict):
        def to(self, device: torch.device) -> Inputs:
            self["device"] = device
            return self

    class Processor:
        captured: object | None = None

    static = np.full((200, 200, 3), 11, dtype=np.uint8)
    gripper = np.full((84, 84, 3), 29, dtype=np.uint8)
    policy = object.__new__(SERVER.RealPolicy)
    policy.Image = SimpleNamespace(fromarray=lambda value: value)
    policy.processor = Processor()
    policy.device = torch.device("cpu")
    policy.allowed_instructions = frozenset({"  raw CALVIN instruction?!  "})
    policy.execution_geometry = _execution_geometry()
    policy.prefix_geometry = {
        "instruction_inventory": {
            "records": [{"instruction": "  raw CALVIN instruction?!  ", "valid_prefix_length": 532}]
        },
        "ordered_cameras": [{}, {}],
        "tokenization": {"padding_side": "left"},
    }

    def apply_fixed(processor, conversations, **kwargs):
        processor.captured = conversations
        assert kwargs == {
            "expected_batch_size": 8,
            "fixed_physical_prefix_width": 600,
            "images_per_prefix": 2,
            "padding_side": "left",
        }
        return Inputs(attention_mask=torch.ones((8, 532), dtype=torch.long))

    policy.apply_fixed_prefix_chat_template = apply_fixed

    result = policy._processor_inputs(
        {
            "instruction": "  raw CALVIN instruction?!  ",
            "observation": {"rgb_static": static, "rgb_gripper": gripper},
        }
    )

    assert isinstance(policy.processor.captured, list)
    assert len(policy.processor.captured) == 8
    content = policy.processor.captured[0][0]["content"]
    assert [entry["type"] for entry in content] == ["image", "image", "text"]
    np.testing.assert_array_equal(content[0]["image"], static)
    np.testing.assert_array_equal(content[1]["image"], gripper)
    assert content[2]["text"] == "  raw CALVIN instruction?!  "
    assert result["device"] == torch.device("cpu")


def test_calvin_official_and_development_servers_use_the_strict_interface_loader(tmp_path: Path) -> None:
    from safetensors.torch import save_file

    from duo_vla.checkpointing import interface_state_dict

    development_server = importlib.import_module("serve_policy_dev")
    assert development_server.RealPolicy is development_server.official_policy.RealPolicy
    assert "_load_calvin_interface_checkpoint" in SERVER.RealPolicy.__init__.__code__.co_names
    assert "_load_calvin_interface_checkpoint" in development_server.RealPolicy.__init__.__code__.co_names

    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    source = {"action_projector": torch.nn.Linear(3, 4), "velocity_head": torch.nn.Linear(4, 2)}
    state = interface_state_dict(source)
    state["velocity_head.bias"] = state["velocity_head.bias"].to(torch.bfloat16)
    save_file(state, checkpoint / "interface.safetensors")

    for policy_module in (SERVER, development_server.official_policy):
        target = {"action_projector": torch.nn.Linear(3, 4), "velocity_head": torch.nn.Linear(4, 2)}
        with pytest.raises(ValueError, match=r"dtype torch\.float32"):
            policy_module._load_calvin_interface_checkpoint(
                checkpoint,
                target["action_projector"],
                target["velocity_head"],
            )


def test_flow_direct_nfe_selection_and_serving_identity() -> None:
    flow = policy_contract_from_config(load_resolved_toml(ROOT / "configs/calvin_abc_to_d.toml")).to_dict()
    selected = SERVER.select_serving_policy_contract(flow, flow_steps_override=5)
    assert selected["objective"] == "rectified_flow" and selected["nfe"] == 5
    assert flow["nfe"] == 10
    health = SERVER._health_identity(
        mode="real",
        train_seed=7,
        policy_contract=selected,
        artifact_identities={
            "calvin_identity": _normalization_dataset_identity(),
            "checkpoint_manifest_sha256": "1" * 64,
            "execution_geometry": _execution_geometry(),
            "normalization_content_sha256": "2" * 64,
            "normalization_metadata_sha256": "2" * 64,
        },
        serving_runtime_sha256="9" * 64,
    )
    assert health["policy_contract_sha256"] == canonical_config_sha256(selected)
    assert health["nfe"] == 5 and health["mode"] == "real"
    assert health["calvin_identity"] == _normalization_dataset_identity()
    assert health["serving_runtime_sha256"] == "9" * 64
    direct = dict(flow, objective="direct_regression", sampler="single_forward", nfe=1)
    assert SERVER.select_serving_policy_contract(direct, flow_steps_override=None)["nfe"] == 1
    with pytest.raises(RuntimeError, match="forbidden"):
        SERVER.select_serving_policy_contract(direct, flow_steps_override=1)


def _set_canonical_serving_environment(monkeypatch: pytest.MonkeyPatch, *, distributed: bool) -> None:
    for name in tuple(SERVER.os.environ):
        if name.startswith(SERVER._RELEVANT_ENVIRONMENT_PREFIXES):
            monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(SERVER.platform, "python_version", lambda: SERVER.EXPECTED_TRAIN_PYTHON)
    monkeypatch.setattr(SERVER.site, "ENABLE_USER_SITE", False)
    monkeypatch.setattr(
        SERVER.sys,
        "flags",
        SimpleNamespace(dont_write_bytecode=1, no_user_site=1, safe_path=True),
    )
    monkeypatch.setattr(SERVER.sys, "prefix", "/root/.cache/duo-vla/venvs/train")
    monkeypatch.setattr(SERVER.sys, "pycache_prefix", "/dev/null")
    version = f"python{SERVER.sys.version_info.major}.{SERVER.sys.version_info.minor}"
    compact_version = f"python{SERVER.sys.version_info.major}{SERVER.sys.version_info.minor}"
    monkeypatch.setattr(
        SERVER.sys,
        "path",
        [
            str((ROOT / "src").resolve()),
            str(Path(SERVER.sys.base_prefix) / "lib" / f"{compact_version}.zip"),
            str(Path(SERVER.sys.base_prefix) / "lib" / version),
            str(Path(SERVER.sys.base_exec_prefix) / "lib" / version / "lib-dynload"),
            f"/root/.cache/duo-vla/venvs/train/lib/{version}/site-packages",
        ],
    )
    for name, value in SERVER.REQUIRED_SERVING_ENVIRONMENT.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setenv("DUO_VLA_CACHE_ROOT", "/root/.cache/duo-vla")
    monkeypatch.setenv("DUO_VLA_PROJECT_ROOT", str(ROOT))
    monkeypatch.setenv("DUO_VLA_TRAIN_VENV", "/root/.cache/duo-vla/venvs/train")
    monkeypatch.setenv("HF_HOME", "/root/.cache/huggingface")
    monkeypatch.delenv("PYTHONPATH", raising=False)
    if distributed:
        monkeypatch.setenv("GROUP_RANK", "0")
        monkeypatch.setenv("GROUP_WORLD_SIZE", "1")
        monkeypatch.setenv("LOCAL_RANK", "0")
        monkeypatch.setenv("LOCAL_WORLD_SIZE", "2")
        monkeypatch.setenv("MASTER_ADDR", "localhost")
        monkeypatch.setenv("MASTER_PORT", "29400")
        monkeypatch.setenv("RANK", "0")
        monkeypatch.setenv("ROLE_NAME", "default")
        monkeypatch.setenv("ROLE_RANK", "0")
        monkeypatch.setenv("ROLE_WORLD_SIZE", "2")
        monkeypatch.setenv("TORCHELASTIC_ERROR_FILE", "/tmp/torchelastic/error.json")
        monkeypatch.setenv("TORCHELASTIC_MAX_RESTARTS", "0")
        monkeypatch.setenv("TORCHELASTIC_RESTART_COUNT", "0")
        monkeypatch.setenv("TORCHELASTIC_RUN_ID", "test-run")
        monkeypatch.setenv("TORCHELASTIC_SIGNALS_TO_HANDLE", "SIGTERM,SIGINT,SIGHUP,SIGQUIT")
        monkeypatch.setenv("TORCHELASTIC_USE_AGENT_STORE", "True")
        monkeypatch.setenv("WORLD_SIZE", "2")


def test_serving_runtime_is_deterministic_and_content_addressed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = {"deterministic": False, "warn_only": True, "precision": "high"}
    cudnn = SimpleNamespace(benchmark=True, deterministic=False, allow_tf32=True, version=lambda: 91002)
    matmul = SimpleNamespace(allow_tf32=True)
    cuda = SimpleNamespace(
        cudnn_sdp_enabled=lambda: True,
        device_count=lambda: 2,
        flash_sdp_enabled=lambda: True,
        get_device_capability=lambda _index: (8, 6),
        get_device_properties=lambda index: SimpleNamespace(
            name="Test GPU",
            uuid=("0000", "1111")[index],
        ),
        math_sdp_enabled=lambda: True,
        matmul=matmul,
        mem_efficient_sdp_enabled=lambda: True,
        nccl=SimpleNamespace(version=lambda: (2, 29, 3)),
    )

    def use_deterministic_algorithms(enabled: bool, *, warn_only: bool) -> None:
        state["deterministic"] = enabled
        state["warn_only"] = warn_only

    fake_torch = SimpleNamespace(
        __version__="2.13.0+cu126",
        are_deterministic_algorithms_enabled=lambda: state["deterministic"],
        backends=SimpleNamespace(cuda=cuda, cudnn=cudnn),
        cuda=cuda,
        get_float32_matmul_precision=lambda: state["precision"],
        is_deterministic_algorithms_warn_only_enabled=lambda: state["warn_only"],
        set_float32_matmul_precision=lambda value: state.__setitem__("precision", value),
        use_deterministic_algorithms=use_deterministic_algorithms,
        version=SimpleNamespace(cuda="12.6"),
    )
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    monkeypatch.setenv("PYTHONHASHSEED", "0")
    process_identity = {
        "identity": "authenticated-process",
        "module_origins": _training_execution_environment()["authenticated_runtime"]["module_origins"],  # type: ignore[index]
    }
    monkeypatch.setattr(SERVER, "_serving_process_identity", lambda *_args, **_kwargs: process_identity)
    monkeypatch.setattr(
        SERVER.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            stdout=("0, Test GPU, GPU-0000, 570.133.20, 8.6\n1, Test GPU, GPU-1111, 570.133.20, 8.6\n")
        ),
    )
    report = {
        "checkpoint": {
            "execution_geometry": _execution_geometry(),
            "source_tree_sha256": "1" * 64,
            "train_venv": _train_venv_identity(),
            "training_execution_environment": _training_execution_environment(),
            "training_execution_environment_sha256": "5" * 64,
        },
        "lock_sha256": "2" * 64,
        "execution_geometry": _execution_geometry(),
        "model": {
            "content_inventory_sha256": "3" * 64,
            "tree_metadata_sha256": "4" * 64,
        },
        "packages": {"torch": "2.13.0+cu126"},
        "source_revisions": SERVER.PINNED_CALVIN_SOURCE_REVISIONS,
    }

    payload, identity = SERVER.configure_and_identify_serving_runtime(fake_torch, report)

    assert payload["schema"] == SERVER.SERVING_RUNTIME_SCHEMA
    assert payload["determinism"] == {
        "cublas_workspace_config": ":4096:8",
        "cudnn_benchmark": False,
        "cudnn_deterministic": True,
        "cudnn_tf32": False,
        "deterministic_algorithms": True,
        "deterministic_warn_only": False,
        "float32_matmul_precision": "highest",
        "matmul_tf32": False,
        "python_hash_seed": "0",
    }
    assert [gpu["uuid"] for gpu in payload["hardware"]["nvidia_smi_devices"]] == ["GPU-0000", "GPU-1111"]
    assert [gpu["physical_index"] for gpu in payload["hardware"]["logical_cuda_devices"]] == [0, 1]
    assert payload["process"] == process_identity
    assert payload["platform"]["nccl"] == [2, 29, 3]
    assert "training_execution_environment_sha256" not in payload["authenticated_software"]
    assert identity == SERVER._canonical_sha256(payload)

    different_training_identity = copy.deepcopy(report)
    different_training_identity["checkpoint"]["training_execution_environment_sha256"] = "6" * 64
    second_payload, second_identity = SERVER.configure_and_identify_serving_runtime(
        fake_torch,
        different_training_identity,
    )
    assert second_payload == payload
    assert second_identity == identity


def test_serving_runtime_rejects_unpinned_process_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_canonical_serving_environment(monkeypatch, distributed=False)
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG")
    with pytest.raises(RuntimeError, match="CUBLAS_WORKSPACE_CONFIG"):
        SERVER._validated_serving_process_environment(ROOT, require_tp_launch=False)


@pytest.mark.parametrize(
    ("name", "value", "message"),
    (
        ("PYTHONPATH", f"{ROOT / 'src'}:{CALVIN_SCRIPTS}:/tmp/shadow", "unpinned overrides"),
        ("CUDA_VISIBLE_DEVICES", "1,0", "canonical launcher"),
        ("NCCL_ALGO", "Ring", "unpinned overrides"),
        ("LD_PRELOAD", "/tmp/inject.so", "unpinned overrides"),
    ),
)
def test_serving_process_rejects_injection_and_device_remapping(
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    value: str,
    message: str,
) -> None:
    _set_canonical_serving_environment(monkeypatch, distributed=True)
    monkeypatch.setenv(name, value)

    with pytest.raises(RuntimeError, match=message):
        SERVER._validated_serving_process_environment(ROOT)


def test_distribution_content_attestation_rejects_same_origin_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Entry:
        def __init__(self, path: str, content: bytes, *, recorded: bool) -> None:
            self.path = path
            self.size = len(content) if recorded else None
            digest = base64.urlsafe_b64encode(hashlib.sha256(content).digest()).rstrip(b"=").decode()
            self.hash = SimpleNamespace(mode="sha256", value=digest) if recorded else None

        def __str__(self) -> str:
            return self.path

    package = tmp_path / "demo/__init__.py"
    record = tmp_path / "demo-1.0.dist-info/RECORD"
    package.parent.mkdir()
    record.parent.mkdir()
    package.write_bytes(b"trusted package bytes\n")
    record.write_bytes(b"demo/__init__.py,sha256=fixture,22\n")
    entries = [
        Entry("demo/__init__.py", package.read_bytes(), recorded=True),
        Entry("demo-1.0.dist-info/RECORD", record.read_bytes(), recorded=False),
    ]
    distribution = SimpleNamespace(
        files=entries,
        locate_file=lambda entry: tmp_path / str(entry),
        version="1.0",
    )
    monkeypatch.setattr(SERVER.importlib.metadata, "distribution", lambda _name: distribution)

    identity = SERVER._verified_distribution_identity("demo", "1.0", tmp_path)
    assert identity["files_verified"] == 2

    package.write_bytes(b"mutated package bytes\n")
    with pytest.raises(RuntimeError, match="differs from RECORD"):
        SERVER._verified_distribution_identity("demo", "1.0", tmp_path)


def test_fake_server_uses_strict_v2_health_and_deterministic_actions(tmp_path: Path) -> None:
    socket_path = tmp_path / "calvin-policy.sock"
    errors: list[Exception] = []

    def serve() -> None:
        try:
            SERVER.run_fake_server(socket_path, train_seed=7)
        except Exception as exc:  # pragma: no cover - reported below
            errors.append(exc)

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()
    wait_for_socket(socket_path, timeout_seconds=2.0)
    static = np.zeros((200, 200, 3), dtype=np.uint8)
    gripper = np.ones((84, 84, 3), dtype=np.uint8)
    state = np.zeros(8, dtype=np.float32)
    state[-1] = 1.0
    fields = {
        "evaluation_seed": 0,
        "execution_horizon": 4,
        "instruction": "move the slider to the right",
        "replan_idx": 0,
        "rgb_gripper": gripper,
        "rgb_static": static,
        "sequence_idx": 0,
        "sequence_sha256": SEQUENCE_SHA256,
        "state": state,
        "subtask_idx": 0,
        "subtask_name": "move_slider_right",
        "train_seed": 7,
    }
    with PolicyClient(socket_path, timeout_seconds=2.0) as client:
        health = client.health()
        assert health["mode"] == "fake"
        assert health["objective"] == "test_fake" and health["nfe"] == 0
        assert health["calvin_identity"] is None
        assert health["checkpoint_manifest_sha256"] is None
        first, _ = client.predict(**fields)
        second, _ = client.predict(**fields)
        assert np.array_equal(first, second)
        assert set(first[:, 6]) <= {-1.0, 1.0}
        client.shutdown()
    thread.join(timeout=2.0)
    assert not thread.is_alive() and not errors


def test_launcher_enforces_train_env_and_tp2() -> None:
    source = (CALVIN_SCRIPTS / "run_policy_server.sh").read_text(encoding="utf-8")
    assert "venvs/train" in source
    assert "--nproc-per-node=2" in source
    assert "scripts/calvin/serve_policy.py" in source
    assert "--preflight-only" in source and "--fake-policy" in source
    assert "exec /usr/bin/env -i" in source
    assert "PYTHONPATH" not in source
    assert '"PATH=/usr/bin:/bin"' in source
    assert '"CUDA_DEVICE_ORDER=PCI_BUS_ID"' in source
    assert '"CUDA_VISIBLE_DEVICES=0,1"' in source
    assert '"PYTHONSAFEPATH=1"' in source
    assert '"PYTHONDONTWRITEBYTECODE=1"' in source
    assert '"PYTHONPYCACHEPREFIX=/dev/null"' in source
    assert "-P -B -X pycache_prefix=/dev/null" in source
    assert '"TORCH_NCCL_ASYNC_ERROR_HANDLING=1"' in source


def test_policy_server_recomputes_the_exact_trainer_source_tree_identity() -> None:
    train_spec = importlib.util.spec_from_file_location(
        "calvin_train_source_identity",
        ROOT / "scripts/train_calvin.py",
    )
    assert train_spec is not None and train_spec.loader is not None
    train_module = importlib.util.module_from_spec(train_spec)
    train_spec.loader.exec_module(train_module)
    compare_tree = ast.parse((ROOT / "scripts/compare_calvin_training_reproducibility.py").read_text(encoding="utf-8"))
    compare_constants = {
        target.id: ast.literal_eval(node.value)
        for node in compare_tree.body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance((target := node.targets[0]), ast.Name)
        and target.id in {"_CALVIN_SOURCE_TREE_HASH_MAGIC", "_TRAINING_SOURCE_EXPLICIT_RELATIVE_PATHS"}
    }

    assert (
        SERVER._CALVIN_SOURCE_EXPLICIT_RELATIVE_PATHS
        == train_module._CALVIN_SOURCE_EXPLICIT_RELATIVE_PATHS
        == compare_constants["_TRAINING_SOURCE_EXPLICIT_RELATIVE_PATHS"]
    )
    assert (
        SERVER._CALVIN_SOURCE_TREE_HASH_MAGIC
        == train_module._CALVIN_SOURCE_TREE_HASH_MAGIC
        == compare_constants["_CALVIN_SOURCE_TREE_HASH_MAGIC"]
        == b"duo-vla-calvin-training-source-tree\x00v3\x00"
    )
    assert SERVER._source_tree_sha256(ROOT) == train_module._source_tree_sha256(ROOT)


def _write_minimal_server_source_tree(root: Path, *, leaf: str, content: str) -> None:
    for relative in SERVER._CALVIN_SOURCE_EXPLICIT_RELATIVE_PATHS:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"explicit:{relative}", encoding="utf-8")
    source = root / "src/duo_vla" / leaf
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text(content, encoding="utf-8")


def test_policy_server_source_hash_is_injective_and_rejects_symlinks(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    _write_minimal_server_source_tree(first, leaf="a", content="bc")
    _write_minimal_server_source_tree(second, leaf="ab", content="c")
    assert SERVER._source_tree_sha256(first) != SERVER._source_tree_sha256(second)

    target = first / "outside.py"
    target.write_text("outside", encoding="utf-8")
    (first / "src/duo_vla/linked.py").symlink_to(target)
    with pytest.raises(RuntimeError, match="regular file"):
        SERVER._source_tree_sha256(first)

    (first / "src/duo_vla/linked.py").unlink()
    fifo = first / "src/duo_vla/nonregular"
    os.mkfifo(fifo)
    with pytest.raises(RuntimeError, match="regular file"):
        SERVER._source_tree_sha256(first)

    fifo.unlink()
    (first / SERVER._CALVIN_SOURCE_EXPLICIT_RELATIVE_PATHS[0]).unlink()
    with pytest.raises(RuntimeError, match="missing"):
        SERVER._source_tree_sha256(first)


def test_policy_server_source_hash_rejects_symlink_above_project_root(tmp_path: Path) -> None:
    real_parent = tmp_path / "real-parent"
    project = real_parent / "project"
    _write_minimal_server_source_tree(project, leaf="module.py", content="source")
    alias = tmp_path / "alias-parent"
    alias.symlink_to(real_parent.name, target_is_directory=True)

    with pytest.raises(RuntimeError, match="ancestor must be a real directory"):
        SERVER._source_tree_sha256(alias / "project")
