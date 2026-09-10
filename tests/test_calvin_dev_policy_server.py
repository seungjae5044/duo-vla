from __future__ import annotations

import ast
import copy
import hashlib
import importlib.util
import json
import sys
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
CALVIN_SCRIPTS = ROOT / "scripts/calvin"
sys.path.insert(0, str(CALVIN_SCRIPTS))

SCRIPT = CALVIN_SCRIPTS / "serve_policy_dev.py"


def test_development_server_bootstraps_only_its_resolved_sibling_directory() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    insertion = "_SCRIPT_ROOT = Path(__file__).resolve().parent"
    assert insertion in source
    assert source.index(insertion) < source.index("import serve_policy as official_policy")
    assert "sys.path.insert(0, _SCRIPT_ROOT_TEXT)" in source
    removal = "sys.path.remove(_SCRIPT_ROOT_TEXT)"
    assert removal in source
    assert source.index(removal) < source.index("from duo_vla.data.calvin_dev_states import")


@pytest.mark.parametrize(
    ("objective", "tensor_parallel_size", "expected"),
    [
        ("rectified_flow", 2, "calvin_abc_to_d.toml"),
        ("direct_regression", 2, "calvin_abc_to_d_direct.toml"),
        ("rectified_flow", 1, "calvin_abc_to_d_single_gpu.toml"),
        ("direct_regression", 1, "calvin_abc_to_d_direct_single_gpu.toml"),
    ],
)
def test_development_recipe_name_is_bound_to_objective_and_topology(
    objective: str,
    tensor_parallel_size: int,
    expected: str,
) -> None:
    assert SERVER._canonical_development_recipe_name(objective, tensor_parallel_size) == expected


def test_development_resolver_selects_source_inventory_from_checkpoint_topology() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert "single_gpu=tensor_parallel_size == 1" in source


SPEC = importlib.util.spec_from_file_location("calvin_dev_policy_server", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
SERVER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = SERVER
SPEC.loader.exec_module(SERVER)

import calvin_bridge  # noqa: E402
import calvin_dev_bridge  # noqa: E402
from calvin_dev_bridge import (  # noqa: E402
    ACTION_DIM,
    ACTION_HORIZON,
    GRIPPER_IMAGE_SHAPE,
    STATE_DIM,
    STATIC_IMAGE_SHAPE,
    DevBridgeError,
    DevPolicyClient,
    wait_for_socket,
)

from duo_vla.data.calvin_dev_states import ARCHIVE_BYTES, AuthenticatedCalvinDevInputs  # noqa: E402
from duo_vla.run_config import load_resolved_toml, save_resolved_config  # noqa: E402
from duo_vla.run_journal import create_run_journal, record_latest_checkpoint  # noqa: E402

_BANK_SHA256 = "a" * 64
_DATASET_SHA256 = "b" * 64
_NORMALIZATION_SHA256 = "c" * 64
_METADATA_SHA256 = "d" * 64
_MANIFEST_FILE_SHA256 = "e" * 64
_MEMBER_INDEX_SHA256 = "f" * 64
_REPLAY_BUNDLE_SHA256 = "9" * 64
_STORAGE_IDENTITY_SHA256 = "8" * 64


def _calvin_identity() -> dict[str, Any]:
    return {
        "archive_bytes": 555_309_812_705,
        "archive_sha256": "c2036c67eb4c06966af1d1e1665bdb572c69e1404f5e77ffd46b384ff2b79f74",
        "central_directory_sha256": "b4f79bda7f6b966b51aa419badd0f7db7a8972a7b58d6d342af60aceff0ea31b",
        "dataset_manifest_file_sha256": _MANIFEST_FILE_SHA256,
        "dataset_manifest_schema": "duo-vla-calvin-dataset-manifest-v4",
        "dataset_manifest_sha256": _DATASET_SHA256,
        "member_index": {
            "bytes": 456,
            "path": "task_ABC_D.members-v2.sqlite3",
            "schema": "duo-vla-calvin-member-index-v2",
            "sha256": _MEMBER_INDEX_SHA256,
        },
        "member_inventory_sha256": "4" * 64,
        "metadata_files": [
            "ep_start_end_ids.npy",
            "lang_annotations/auto_lang_ann.npy",
            "scene_info.npy",
            ".hydra/merged_config.yaml",
        ],
        "metadata_sha256": _METADATA_SHA256,
        "name": "task_ABC_D",
        "reader_schema": "duo-vla-calvin-archive-reader-v1",
        "split": "training",
        "storage_identity_sha256": _STORAGE_IDENTITY_SHA256,
        "storage_mode": "archive-direct",
    }


def _split() -> dict[str, Any]:
    return {
        "algorithm": "fixture split",
        "seed": 1729,
        "train_episode_indices": [0],
        "train_episode_sha256": hashlib.sha256(b"0").hexdigest(),
        "validation_episode_indices": [1],
        "validation_episode_sha256": hashlib.sha256(b"1").hexdigest(),
        "validation_fraction": 0.1,
    }


def _record() -> dict[str, Any]:
    return {
        "annotation_index": 19,
        "episode_index": 1,
        "global_start": 1234,
        "instruction": "open the drawer",
        "reset_id_sha256": "e" * 64,
        "scene": "calvin_scene_A",
        "task": "open_drawer",
    }


def _binding() -> SERVER.DevelopmentBankBinding:
    split = _split()
    dataset = _calvin_identity()
    identity = {
        "archive_bytes": dataset["archive_bytes"],
        "archive_sha256": dataset["archive_sha256"],
        "calvin_env_revision": "1" * 40,
        "calvin_revision": "2" * 40,
        "calvin_tacto_revision": "3" * 40,
        "central_directory_sha256": dataset["central_directory_sha256"],
        "dataset_manifest_file_sha256": dataset["dataset_manifest_file_sha256"],
        "dataset_manifest_schema": dataset["dataset_manifest_schema"],
        "dataset_manifest_sha256": dataset["dataset_manifest_sha256"],
        "member_index_bytes": dataset["member_index"]["bytes"],
        "member_index_path": dataset["member_index"]["path"],
        "member_index_schema": dataset["member_index"]["schema"],
        "member_index_sha256": dataset["member_index"]["sha256"],
        "member_inventory_sha256": dataset["member_inventory_sha256"],
        "metadata_files": dataset["metadata_files"],
        "metadata_sha256": dataset["metadata_sha256"],
        "normalization_content_sha256": _NORMALIZATION_SHA256,
        "reader_schema": dataset["reader_schema"],
        "scene_config_sha256": {
            "calvin_scene_A": "5" * 64,
            "calvin_scene_B": "6" * 64,
            "calvin_scene_C": "7" * 64,
        },
        "task_oracle_sha256": "8" * 64,
        "storage_identity_sha256": dataset["storage_identity_sha256"],
        "storage_mode": dataset["storage_mode"],
    }
    inputs = AuthenticatedCalvinDevInputs(identity=identity, member_index={}, split=split, stats={"dataset": dataset})
    records = [
        _record(),
        {**_record(), "scene": "calvin_scene_B", "reset_id_sha256": "f" * 64},
        {**_record(), "scene": "calvin_scene_C", "reset_id_sha256": "9" * 64},
    ]
    return SERVER.bind_development_bank(
        {
            "identity": identity,
            "records": records,
            "replay_bundle": {"root_sha256": _REPLAY_BUNDLE_SHA256},
            "root_sha256": _BANK_SHA256,
            "split": split,
        },
        inputs,
    )


def _observation() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    static = np.zeros(STATIC_IMAGE_SHAPE, dtype=np.uint8)
    gripper = np.zeros(GRIPPER_IMAGE_SHAPE, dtype=np.uint8)
    state = np.zeros(STATE_DIM, dtype=np.float32)
    state[-1] = -1.0
    return static, gripper, state


def _predict_fields(**changes: Any) -> dict[str, Any]:
    static, gripper, state = _observation()
    record = _record()
    fields = {
        "annotation_index": record["annotation_index"],
        "episode_index": record["episode_index"],
        "evaluation_seed": 91,
        "execution_horizon": 4,
        "global_start": record["global_start"],
        "instruction": record["instruction"],
        "replan_idx": 2,
        "reset_bank_sha256": _BANK_SHA256,
        "reset_id_sha256": record["reset_id_sha256"],
        "reset_index": 0,
        "rgb_gripper": gripper,
        "rgb_static": static,
        "scene": record["scene"],
        "state": state,
        "task": record["task"],
        "train_seed": 1,
    }
    fields.update(changes)
    return fields


def test_endpoint_is_dev_only_and_reuses_the_authenticated_inference_core() -> None:
    assert SERVER.PROTOCOL == calvin_dev_bridge.PROTOCOL == "duovla-calvin-heldout-abc-v1"
    assert SERVER.TRAINING_PROTOCOL == calvin_bridge.PROTOCOL
    assert calvin_dev_bridge.PROTOCOL != calvin_bridge.PROTOCOL
    assert calvin_dev_bridge.SCHEMA == "duovla-calvin-dev-policy-ipc-v4"
    assert SERVER.RealPolicy is SERVER.official_policy.RealPolicy
    assert SERVER.FakePolicy is SERVER.official_policy.FakePolicy
    assert "not_official" in SERVER.DEVELOPMENT_STATUS


def test_simulator_bridge_boundary_parses_as_python38() -> None:
    source = (CALVIN_SCRIPTS / "calvin_dev_bridge.py").read_text(encoding="utf-8")
    ast.parse(source, filename="calvin_dev_bridge.py", feature_version=(3, 8))


def test_health_binds_bank_split_dataset_normalization_and_checkpoint() -> None:
    binding = _binding()
    contract = {"nfe": 5, "objective": "rectified_flow", "sampler": "euler_uniform"}
    health = SERVER._health_identity(
        mode="real",
        train_seed=1,
        binding=binding,
        policy_contract=contract,
        artifact_identities={
            "checkpoint_manifest_sha256": "1" * 64,
            "execution_geometry": {
                "expert_batch_isolation": "sample_isolated_grouped_mm_v1",
                "experts_implementation": "grouped_mm",
                "fixed_physical_prefix_width": 600,
                "physical_batch_size": 8,
                "prefix_geometry_content_sha256": "6" * 64,
            },
            "normalization_content_sha256": _NORMALIZATION_SHA256,
            "normalization_metadata_sha256": _METADATA_SHA256,
        },
    )

    assert health["reset_bank_sha256"] == _BANK_SHA256
    assert health["reset_count"] == 3
    assert health["calvin_identity"] == _calvin_identity()
    assert health["calvin_identity"]["dataset_manifest_sha256"] == _DATASET_SHA256
    assert health["calvin_identity"]["member_index"]["sha256"] == _MEMBER_INDEX_SHA256
    assert health["replay_bundle_sha256"] == _REPLAY_BUNDLE_SHA256
    assert health["calvin_identity"]["storage_identity_sha256"] == _STORAGE_IDENTITY_SHA256
    assert health["split_sha256"] == SERVER.canonical_sha256(_split())
    assert health["checkpoint_manifest_sha256"] == "1" * 64
    assert health["normalization_content_sha256"] == _NORMALIZATION_SHA256
    assert health["policy_contract_sha256"] == SERVER.official_policy._canonical_sha256(contract)


def test_fake_endpoint_is_deterministic_and_rejects_unbound_reset(tmp_path: Path) -> None:
    socket_path = tmp_path / "calvin-heldout-abc.sock"
    errors: list[Exception] = []

    def serve() -> None:
        try:
            SERVER.run_fake_server(socket_path, train_seed=1, binding=_binding())
        except Exception as exc:  # pragma: no cover - asserted in parent thread
            errors.append(exc)

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()
    wait_for_socket(socket_path, timeout_seconds=2.0)
    with (
        calvin_bridge.PolicyClient(socket_path, timeout_seconds=2.0) as official_client,
        pytest.raises(calvin_bridge.BridgeProtocolError, match="schema"),
    ):
        official_client.health()
    with DevPolicyClient(socket_path, timeout_seconds=2.0) as client:
        health = client.health()
        assert health["mode"] == "fake"
        assert health["reset_bank_sha256"] == _BANK_SHA256
        first, first_metadata = client.predict(**_predict_fields())
        second, second_metadata = client.predict(**_predict_fields())
        assert np.array_equal(first, second)
        assert first.shape == (ACTION_HORIZON, ACTION_DIM)
        assert first_metadata["inference_seed"] == second_metadata["inference_seed"]
        with pytest.raises(DevBridgeError, match="reset_id_sha256 differs"):
            client.predict(**_predict_fields(reset_id_sha256="0" * 64))
        with pytest.raises(DevBridgeError, match="instruction differs"):
            client.predict(**_predict_fields(instruction="close the drawer"))
        client.shutdown()
    thread.join(timeout=2.0)
    assert not thread.is_alive()
    assert not errors
    assert not socket_path.exists()


def _progress_manifest(*, total: int, update: int, complete: bool) -> dict[str, Any]:
    return {
        "complete": complete,
        "configured_total_updates": total,
        "last_metrics": {"examples_seen": update * 64, "update": update},
        "trainer_state": {
            "examples_seen": update * 64,
            "next_update": update,
            "schema": "duo-vla-trainer-state-v1",
        },
    }


def test_progress_accepts_only_consistent_committed_intermediate_or_pilot_final(tmp_path: Path) -> None:
    intermediate = tmp_path / "update-000120"
    intermediate.mkdir()
    (intermediate / "manifest.json").write_text("intermediate\n", encoding="utf-8")
    intermediate_sha = SERVER.official_policy.sha256_file(intermediate / "manifest.json")
    progress = SERVER._validate_development_progress(
        _progress_manifest(total=500, update=120, complete=False),
        {"global_batch_size": 64, "total_updates": 500},
        intermediate,
        committed_update=120,
        committed_manifest_sha256=intermediate_sha,
    )
    assert progress == {
        "complete": False,
        "configured_total_updates": 500,
        "examples_seen": 120 * 64,
        "selected_update": 120,
    }

    final = tmp_path / "update-000500"
    final.mkdir()
    (final / "manifest.json").write_text("pilot-final\n", encoding="utf-8")
    final_sha = SERVER.official_policy.sha256_file(final / "manifest.json")
    assert (
        SERVER._validate_development_progress(
            _progress_manifest(total=500, update=500, complete=True),
            {"global_batch_size": 64, "total_updates": 500},
            final,
            committed_update=500,
            committed_manifest_sha256=final_sha,
        )["complete"]
        is True
    )

    with pytest.raises(RuntimeError, match="complete flag"):
        SERVER._validate_development_progress(
            _progress_manifest(total=500, update=120, complete=True),
            {"global_batch_size": 64, "total_updates": 500},
            intermediate,
            committed_update=120,
            committed_manifest_sha256=intermediate_sha,
        )
    with pytest.raises(RuntimeError, match=r"\[1,30000\]"):
        SERVER._validate_development_progress(
            _progress_manifest(total=30_001, update=120, complete=False),
            {"global_batch_size": 64, "total_updates": 30_001},
            intermediate,
            committed_update=120,
            committed_manifest_sha256=intermediate_sha,
        )


def test_recipe_relaxes_only_total_updates_and_its_trainer_derived_warmup() -> None:
    canonical = load_resolved_toml(ROOT / "configs/calvin_abc_to_d.toml")
    assert canonical["optimization"]["total_updates"] == 30_000
    assert canonical["optimization"]["warmup_updates"] == 1_000
    assert SERVER._canonical_recipe_mismatches(canonical, canonical) == []
    wrong_default_warmup = copy.deepcopy(canonical)
    wrong_default_warmup["optimization"]["warmup_updates"] = 999
    assert SERVER._canonical_recipe_mismatches(wrong_default_warmup, canonical) == ["optimization"]

    pilot = copy.deepcopy(canonical)
    pilot["optimization"]["total_updates"] = 500
    pilot["optimization"]["warmup_updates"] = 50
    assert SERVER._canonical_recipe_mismatches(pilot, canonical) == []

    wrong_warmup = copy.deepcopy(pilot)
    wrong_warmup["optimization"]["warmup_updates"] = 49
    assert SERVER._canonical_recipe_mismatches(wrong_warmup, canonical) == ["optimization"]
    wrong_warmup["optimization"]["warmup_updates"] = 51
    assert SERVER._canonical_recipe_mismatches(wrong_warmup, canonical) == ["optimization"]
    wrong_warmup["optimization"]["warmup_updates"] = 1_000
    assert SERVER._canonical_recipe_mismatches(wrong_warmup, canonical) == ["optimization"]

    pilot["optimization"]["global_batch_size"] = 32
    assert SERVER._canonical_recipe_mismatches(pilot, canonical) == ["optimization"]
    pilot = copy.deepcopy(canonical)
    pilot["lora"]["rank"] = 8
    assert SERVER._canonical_recipe_mismatches(pilot, canonical) == ["lora"]


def test_checkpoint_must_be_the_transactionally_committed_journal_tip(tmp_path: Path) -> None:
    output = tmp_path / "run"
    output.mkdir()
    config_sha256 = "a" * 64
    journal = create_run_journal(output, config_sha256=config_sha256)
    checkpoint = output / "checkpoints/update-000120"
    checkpoint.mkdir(parents=True)
    metrics = {"examples_seen": 120 * 64, "update": 120}
    manifest = {
        "config_sha256": config_sha256,
        "last_metrics": metrics,
        "parent_manifest_sha256": None,
        "run_uuid": journal.run_uuid,
        "trainer_state": {"next_update": 120},
    }
    (checkpoint / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    manifest_sha256 = SERVER.official_policy.sha256_file(checkpoint / "manifest.json")
    record_latest_checkpoint(
        output,
        checkpoint=checkpoint,
        update=120,
        manifest_sha256=manifest_sha256,
        parent_manifest_sha256=None,
        last_metrics=metrics,
    )

    record = SERVER._committed_checkpoint_record(checkpoint, config_sha256)
    assert record.update == 120 and record.manifest_sha256 == manifest_sha256

    uncommitted = output / "checkpoints/update-000121"
    uncommitted.mkdir()
    (uncommitted / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="not the journal latest"):
        SERVER._committed_checkpoint_record(uncommitted, config_sha256)


def _load_official_test_helpers() -> Any:
    path = ROOT / "tests/test_calvin_policy_server.py"
    spec = importlib.util.spec_from_file_location("_official_policy_test_helpers", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _install_v4_storage_identity(
    manifest: dict[str, Any],
    config: dict[str, Any],
    stats: dict[str, Any],
) -> None:
    dataset = stats["dataset"]
    dataset.update(
        {
            "archive_bytes": ARCHIVE_BYTES,
            "central_directory_sha256": "b4f79bda7f6b966b51aa419badd0f7db7a8972a7b58d6d342af60aceff0ea31b",
            "dataset_manifest_file_sha256": "b" * 64,
            "dataset_manifest_schema": "duo-vla-calvin-dataset-manifest-v4",
            "member_index": {
                "bytes": 123,
                "path": "task_ABC_D.members-v2.sqlite3",
                "schema": "duo-vla-calvin-member-index-v2",
                "sha256": "c" * 64,
            },
            "reader_schema": "duo-vla-calvin-archive-reader-v1",
            "storage_identity_sha256": "d" * 64,
            "storage_mode": "archive-direct",
        }
    )
    storage_names = SERVER._CALVIN_STORAGE_IDENTITY_FIELDS
    manifest["calvin_identity"].update({name: copy.deepcopy(dataset[name]) for name in storage_names})
    config["calvin_identity"] = copy.deepcopy(manifest["calvin_identity"])
    member_index = dataset["member_index"]
    flattened = {
        "archive_bytes": dataset["archive_bytes"],
        "archive_sha256": dataset["archive_sha256"],
        "central_directory_sha256": dataset["central_directory_sha256"],
        "dataset_manifest_file_sha256": dataset["dataset_manifest_file_sha256"],
        "dataset_manifest_schema": dataset["dataset_manifest_schema"],
        "dataset_manifest_sha256": dataset["dataset_manifest_sha256"],
        "member_index_bytes": member_index["bytes"],
        "member_index_path": member_index["path"],
        "member_index_schema": member_index["schema"],
        "member_index_sha256": member_index["sha256"],
        "member_inventory_sha256": dataset["member_inventory_sha256"],
        "metadata_sha256": dataset["metadata_sha256"],
        "reader_schema": dataset["reader_schema"],
        "storage_identity_sha256": dataset["storage_identity_sha256"],
        "storage_mode": dataset["storage_mode"],
    }
    manifest.update(flattened)


def test_checkpoint_storage_projection_rejects_every_nested_flattened_or_inventory_drift() -> None:
    dataset: dict[str, Any] = {
        "archive_bytes": 123,
        "archive_sha256": "0" * 64,
        "central_directory_sha256": "1" * 64,
        "dataset_manifest_file_sha256": "2" * 64,
        "dataset_manifest_schema": "duo-vla-calvin-dataset-manifest-v4",
        "dataset_manifest_sha256": "3" * 64,
        "member_index": {
            "bytes": 456,
            "path": "task_ABC_D.members-v2.sqlite3",
            "schema": "duo-vla-calvin-member-index-v2",
            "sha256": "4" * 64,
        },
        "member_inventory_sha256": "5" * 64,
        "metadata_files": ["ep_start_end_ids.npy"],
        "metadata_sha256": "6" * 64,
        "name": "task_ABC_D",
        "reader_schema": "duo-vla-calvin-archive-reader-v1",
        "split": "training",
        "storage_identity_sha256": "7" * 64,
        "storage_mode": "archive-direct",
    }
    calvin_identity = {name: copy.deepcopy(dataset[name]) for name in SERVER._CALVIN_STORAGE_IDENTITY_FIELDS}
    member_index = dataset["member_index"]
    manifest = {
        "archive_bytes": dataset["archive_bytes"],
        "archive_sha256": dataset["archive_sha256"],
        "central_directory_sha256": dataset["central_directory_sha256"],
        "dataset_manifest_file_sha256": dataset["dataset_manifest_file_sha256"],
        "dataset_manifest_schema": dataset["dataset_manifest_schema"],
        "dataset_manifest_sha256": dataset["dataset_manifest_sha256"],
        "member_index_bytes": member_index["bytes"],
        "member_index_path": member_index["path"],
        "member_index_schema": member_index["schema"],
        "member_index_sha256": member_index["sha256"],
        "member_inventory_sha256": dataset["member_inventory_sha256"],
        "metadata_sha256": dataset["metadata_sha256"],
        "reader_schema": dataset["reader_schema"],
        "storage_identity_sha256": dataset["storage_identity_sha256"],
        "storage_mode": dataset["storage_mode"],
    }
    SERVER._require_checkpoint_storage_identity(manifest, calvin_identity, dataset)

    for name in SERVER._CALVIN_STORAGE_IDENTITY_FIELDS:
        changed = copy.deepcopy(calvin_identity)
        changed[name] = None
        with pytest.raises(RuntimeError, match=f"CALVIN identity storage field mismatch: {name}"):
            SERVER._require_checkpoint_storage_identity(manifest, changed, dataset)
    for name in manifest:
        changed = copy.deepcopy(manifest)
        changed[name] = None
        with pytest.raises(RuntimeError, match=f"checkpoint flattened storage field mismatch: {name}"):
            SERVER._require_checkpoint_storage_identity(changed, calvin_identity, dataset)
    changed_dataset = copy.deepcopy(dataset)
    changed_dataset["extra"] = True
    with pytest.raises(RuntimeError, match="dataset storage identity fields differ"):
        SERVER._require_checkpoint_storage_identity(manifest, calvin_identity, changed_dataset)


def test_full_resolver_accepts_only_total_and_progress_relaxations(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import duo_vla.backbones.loading
    import duo_vla.checkpointing
    import duo_vla.data.calvin_stats
    import duo_vla.run_config

    helpers = _load_official_test_helpers()
    checkpoint, manifest, config, stats = helpers._checkpoint_fixture(tmp_path)
    _install_v4_storage_identity(manifest, config, stats)
    pilot_checkpoint = checkpoint.with_name("update-000120")
    checkpoint.rename(pilot_checkpoint)
    config["optimization"]["total_updates"] = 500
    config["optimization"]["warmup_updates"] = 50
    config_sha256 = save_resolved_config(pilot_checkpoint / "artifacts/resolved_config.json", config)
    manifest.update(_progress_manifest(total=500, update=120, complete=False))
    manifest["config_sha256"] = config_sha256

    monkeypatch.setattr(duo_vla.checkpointing, "load_checkpoint_manifest", lambda *args, **kwargs: manifest)
    monkeypatch.setattr(
        duo_vla.backbones.loading,
        "validate_decoder_attention_lora_weights",
        lambda *args, **kwargs: {},
    )
    authenticated_inputs = AuthenticatedCalvinDevInputs(
        identity={},
        member_index=dict(stats["dataset"]["member_index"]),
        split=dict(stats["split"]),
        stats=stats,
        training_root=str((tmp_path / "task_ABC_D/training").resolve()),
    )
    monkeypatch.setattr(
        duo_vla.data.calvin_stats,
        "load_calvin_state_normalizer",
        lambda *args, **kwargs: (object(), stats),
    )
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
    manifest_sha256 = SERVER.official_policy.sha256_file(pilot_checkpoint / "manifest.json")
    monkeypatch.setattr(
        SERVER,
        "_committed_checkpoint_record",
        lambda *args, **kwargs: SimpleNamespace(update=120, manifest_sha256=manifest_sha256),
    )
    monkeypatch.setattr(
        SERVER.official_policy,
        "content_address_train_venv",
        lambda _root: helpers._train_venv_identity(),
    )

    _, _, seed, report, resolved, contract, identities = SERVER.resolve_development_checkpoint(
        pilot_checkpoint,
        training_root=tmp_path / "task_ABC_D/training",
        project_root=ROOT,
        authenticated_inputs=authenticated_inputs,
        train_seed_override=1,
        model_snapshot_report=helpers._model_snapshot_report(),
    )
    assert seed == 1
    assert resolved["optimization"]["total_updates"] == 500
    assert report["checkpoint_progress"]["selected_update"] == 120
    assert report["benchmark_status"] == SERVER.DEVELOPMENT_STATUS
    assert report["train_venv"] == report["training_execution_environment"]["authenticated_runtime"]["train_venv"]
    assert contract["objective"] == "rectified_flow"
    assert identities["checkpoint_manifest_sha256"] == manifest_sha256

    changed = copy.deepcopy(config)
    changed["optimization"]["global_batch_size"] = 32
    changed_sha256 = save_resolved_config(pilot_checkpoint / "artifacts/resolved_config.json", changed)
    manifest["config_sha256"] = changed_sha256
    with pytest.raises(RuntimeError, match=r"optimization\.global_batch_size"):
        SERVER.resolve_development_checkpoint(
            pilot_checkpoint,
            training_root=tmp_path / "task_ABC_D/training",
            project_root=ROOT,
            authenticated_inputs=authenticated_inputs,
            train_seed_override=1,
            model_snapshot_report=helpers._model_snapshot_report(),
        )


def test_launcher_is_closed_and_uses_tp2_dev_executable() -> None:
    launcher = (CALVIN_SCRIPTS / "run_policy_server_dev.sh").read_text(encoding="utf-8")
    assert "unset BASH_ENV CDPATH ENV GLOBIGNORE" in launcher
    assert '"PYTHONSAFEPATH=1"' in launcher
    assert '"PYTHONPYCACHEPREFIX=/dev/null"' in launcher
    assert '"CUDA_VISIBLE_DEVICES=0,1"' in launcher
    assert "exec /usr/bin/env -i" in launcher
    assert "-P -B -X pycache_prefix=/dev/null" in launcher
    assert "--nproc-per-node=2" in launcher
    assert 'scripts/calvin/serve_policy_dev.py"' in launcher
    assert "scripts/calvin/evaluate_calvin.py" not in launcher
    assert "calvin_scene_D" not in launcher
