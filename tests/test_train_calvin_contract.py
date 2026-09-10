from __future__ import annotations

import copy
import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import duo_vla.calvin_source_identity as calvin_source_identity
from duo_vla.benchmarks.common import CanonicalObservation
from duo_vla.data.calvin import CalvinAnchor, CalvinAnnotation, CalvinEpisode, make_calvin_episode_split
from duo_vla.run_config import load_resolved_toml

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "train_calvin.py"
SPEC = importlib.util.spec_from_file_location("train_calvin", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
TRAIN = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TRAIN)


class _FakeAuthenticatedGeneration:
    @staticmethod
    def from_dict(payload: object) -> object:
        return payload


def test_distributed_dataset_authentication_uses_only_a_long_lived_gloo_subgroup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    group = object()
    generation = SimpleNamespace(to_dict=lambda: {"capability": "authenticated"})
    new_group_calls: list[dict[str, object]] = []
    broadcasts: list[tuple[object, object]] = []
    destroyed: list[object] = []

    def new_group(**kwargs: object) -> object:
        new_group_calls.append(kwargs)
        return group

    def broadcast(holder: list[object], *, src: int, group: object) -> None:
        assert src == 0
        broadcasts.append((holder[0], group))

    monkeypatch.setattr(TRAIN.dist, "new_group", new_group)
    monkeypatch.setattr(TRAIN.dist, "get_rank", lambda: 0)
    monkeypatch.setattr(TRAIN.dist, "get_world_size", lambda: 2)
    monkeypatch.setattr(TRAIN.dist, "broadcast_object_list", broadcast)
    monkeypatch.setattr(TRAIN.dist, "all_gather_object", lambda output, value, *, group: None)
    monkeypatch.setattr(TRAIN.dist, "destroy_process_group", lambda value: destroyed.append(value))
    monkeypatch.setattr(TRAIN, "authenticate_calvin_dataset_generation", lambda root: ({}, generation))
    monkeypatch.setattr(TRAIN, "AuthenticatedCalvinDatasetGeneration", _FakeAuthenticatedGeneration)
    monkeypatch.setattr(TRAIN, "verify_calvin_dataset_generation", lambda root, capability: None)

    result = TRAIN._authenticate_calvin_dataset_generation_distributed(tmp_path)

    assert result == {"capability": "authenticated"}
    assert new_group_calls == [
        {
            "backend": "gloo",
            "timeout": TRAIN.CALVIN_DATASET_AUTHENTICATION_TIMEOUT,
        }
    ]
    assert TRAIN.CALVIN_DATASET_AUTHENTICATION_TIMEOUT.total_seconds() == 2 * 60 * 60
    assert broadcasts == [(None, group), ({"capability": "authenticated"}, group)]
    assert destroyed == [group]


def test_distributed_dataset_authentication_rank_one_only_receives_capability(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    group = object()
    broadcast_count = 0

    def broadcast(holder: list[object], *, src: int, group: object) -> None:
        nonlocal broadcast_count
        assert src == 0
        broadcast_count += 1
        if broadcast_count == 2:
            holder[0] = {"capability": "from-rank-zero"}

    def unexpected_authentication(_root: Path) -> None:
        raise AssertionError("nonzero ranks must not authenticate the full archive")

    monkeypatch.setattr(TRAIN.dist, "new_group", lambda **_kwargs: group)
    monkeypatch.setattr(TRAIN.dist, "get_rank", lambda: 1)
    monkeypatch.setattr(TRAIN.dist, "get_world_size", lambda: 2)
    monkeypatch.setattr(TRAIN.dist, "broadcast_object_list", broadcast)
    monkeypatch.setattr(TRAIN.dist, "all_gather_object", lambda output, value, *, group: None)
    monkeypatch.setattr(TRAIN.dist, "destroy_process_group", lambda value: None)
    monkeypatch.setattr(TRAIN, "authenticate_calvin_dataset_generation", unexpected_authentication)
    monkeypatch.setattr(TRAIN, "AuthenticatedCalvinDatasetGeneration", _FakeAuthenticatedGeneration)
    monkeypatch.setattr(TRAIN, "verify_calvin_dataset_generation", lambda root, capability: None)

    result = TRAIN._authenticate_calvin_dataset_generation_distributed(tmp_path)

    assert result == {"capability": "from-rank-zero"}
    assert broadcast_count == 2


def test_distributed_dataset_authentication_rejects_malformed_received_capability_on_all_ranks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    group = object()
    broadcast_count = 0
    destroyed: list[object] = []

    def broadcast(holder: list[object], *, src: int, group: object) -> None:
        nonlocal broadcast_count
        assert src == 0
        broadcast_count += 1
        if broadcast_count == 2:
            holder[0] = {"malformed": True}

    def gather(output: list[str | None], value: str | None, *, group: object) -> None:
        output[:] = [None, value]

    def unexpected_authentication(_root: Path) -> None:
        raise AssertionError("nonzero ranks must not authenticate the full archive")

    monkeypatch.setattr(TRAIN.dist, "new_group", lambda **_kwargs: group)
    monkeypatch.setattr(TRAIN.dist, "get_rank", lambda: 1)
    monkeypatch.setattr(TRAIN.dist, "get_world_size", lambda: 2)
    monkeypatch.setattr(TRAIN.dist, "broadcast_object_list", broadcast)
    monkeypatch.setattr(TRAIN.dist, "all_gather_object", gather)
    monkeypatch.setattr(TRAIN.dist, "destroy_process_group", lambda value: destroyed.append(value))
    monkeypatch.setattr(TRAIN, "authenticate_calvin_dataset_generation", unexpected_authentication)

    with pytest.raises(RuntimeError, match="authenticated generation failed across TP ranks"):
        TRAIN._authenticate_calvin_dataset_generation_distributed(tmp_path)

    assert broadcast_count == 2
    assert destroyed == [group]


def test_distributed_dataset_authentication_broadcasts_failure_and_destroys_subgroup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    group = object()
    destroyed: list[object] = []
    broadcasts = 0

    def fail_authentication(_root: Path) -> None:
        raise ValueError("archive mismatch")

    def broadcast(_holder: list[object], *, src: int, group: object) -> None:
        nonlocal broadcasts
        assert src == 0
        broadcasts += 1

    monkeypatch.setattr(TRAIN.dist, "new_group", lambda **_kwargs: group)
    monkeypatch.setattr(TRAIN.dist, "get_rank", lambda: 0)
    monkeypatch.setattr(TRAIN.dist, "broadcast_object_list", broadcast)
    monkeypatch.setattr(TRAIN.dist, "destroy_process_group", lambda value: destroyed.append(value))
    monkeypatch.setattr(TRAIN, "authenticate_calvin_dataset_generation", fail_authentication)

    with pytest.raises(RuntimeError, match="CALVIN dataset authentication failed: ValueError: archive mismatch"):
        TRAIN._authenticate_calvin_dataset_generation_distributed(tmp_path)

    assert broadcasts == 1
    assert destroyed == [group]


def _split(train: tuple[int, ...], validation: tuple[int, ...]) -> dict[str, object]:
    return {
        "train_episode_indices": list(train),
        "validation_episode_indices": list(validation),
        "train_episode_sha256": TRAIN._indices_sha256(train),
        "validation_episode_sha256": TRAIN._indices_sha256(validation),
    }


def _v4_stats() -> dict[str, object]:
    return {
        "schema": TRAIN.CALVIN_STATS_SCHEMA,
        "content_sha256": "1" * 64,
        "dataset": {
            "archive_bytes": TRAIN.CALVIN_ABC_D_ARCHIVE_BYTES,
            "archive_sha256": TRAIN.CALVIN_ABC_D_ARCHIVE_SHA256,
            "central_directory_sha256": TRAIN.OFFICIAL_CENTRAL_DIRECTORY_SHA256,
            "dataset_manifest_file_sha256": "5" * 64,
            "dataset_manifest_schema": TRAIN.CALVIN_DATASET_MANIFEST_SCHEMA,
            "dataset_manifest_sha256": "3" * 64,
            "member_index": {
                "bytes": 123_456,
                "path": TRAIN.CALVIN_INDEX_NAME,
                "schema": TRAIN.CALVIN_MEMBER_INDEX_SCHEMA,
                "sha256": "6" * 64,
            },
            "member_inventory_sha256": "4" * 64,
            "metadata_files": list(TRAIN.CALVIN_CRITICAL_TRAIN_METADATA),
            "metadata_sha256": "2" * 64,
            "name": "task_ABC_D",
            "reader_schema": TRAIN.CALVIN_ARCHIVE_READER_SCHEMA,
            "split": "training",
            "storage_identity_sha256": "7" * 64,
            "storage_mode": TRAIN.CALVIN_STORAGE_MODE_ARCHIVE_DIRECT,
        },
        "split": _split((0,), (1,)),
        "state": {
            "dimension": 8,
            "continuous_dimensions": list(range(7)),
            "gripper_index": 7,
            "observed_gripper_values": [-1.0, 1.0],
        },
        "action": {
            "dimension": 7,
            "continuous_dimensions": list(range(6)),
            "gripper_index": 6,
            "observed_gripper_values": [-1.0, 1.0],
            "transform": "identity_official_scaled_rel_actions",
        },
        "algorithm": {"actions_re_normalized": False},
    }


def test_calvin_episode_indices_are_taken_from_hashed_stats_partition() -> None:
    stats = {"split": _split((0, 2), (1, 3))}

    train, validation = TRAIN._episode_indices_from_stats(stats, episode_count=4)

    assert train == (0, 2)
    assert validation == (1, 3)
    stats["split"]["train_episode_sha256"] = "0" * 64  # type: ignore[index]
    with pytest.raises(ValueError, match="split hash"):
        TRAIN._episode_indices_from_stats(stats, episode_count=4)


@pytest.mark.parametrize(
    ("field", "wrong_value"),
    (
        ("archive_bytes", TRAIN.CALVIN_ABC_D_ARCHIVE_BYTES - 1),
        ("fixed_physical_prefix_width", 601),
        ("member_index_bytes", 123_455),
        ("physical_batch_size", 7),
    ),
)
def test_resume_manifest_run_contract_accepts_typed_execution_integers_only(
    field: str,
    wrong_value: int,
) -> None:
    run_contract = {
        "archive_bytes": str(TRAIN.CALVIN_ABC_D_ARCHIVE_BYTES),
        "fixed_physical_prefix_width": "600",
        "member_index_bytes": "123456",
        "physical_batch_size": "8",
        "source_tree_sha256": "a" * 64,
    }
    manifest: dict[str, object] = {
        **run_contract,
        "archive_bytes": TRAIN.CALVIN_ABC_D_ARCHIVE_BYTES,
        "fixed_physical_prefix_width": 600,
        "member_index_bytes": 123_456,
        "physical_batch_size": 8,
    }
    expected_run_contract = copy.deepcopy(run_contract)

    TRAIN._validate_resume_manifest_run_contract(manifest, run_contract)

    assert run_contract == expected_run_contract
    changed = {**manifest, field: wrong_value}
    with pytest.raises(ValueError, match=field):
        TRAIN._validate_resume_manifest_run_contract(changed, run_contract)
    wrong_type = {**manifest, field: run_contract[field]}
    with pytest.raises(ValueError, match=field):
        TRAIN._validate_resume_manifest_run_contract(wrong_type, run_contract)


@pytest.mark.parametrize(
    ("section", "field", "wrong_value"),
    (
        ("lora", "rank", 8),
        ("lora", "alpha", 7),
        ("lora", "dropout", 0.25),
        ("action", "timestep_embedding_dimension", 128),
        ("action", "timestep_scale", 1.0),
        ("action", "timestep_max_period", 1000.0),
        ("action", "output_head_initialization_std", 0.01),
    ),
)
def test_calvin_trainer_rejects_off_contract_interface_and_lora_hyperparameters(
    monkeypatch: pytest.MonkeyPatch,
    section: str,
    field: str,
    wrong_value: object,
) -> None:
    config = load_resolved_toml(ROOT / "configs/calvin_abc_to_d.toml")
    config[section][field] = wrong_value
    monkeypatch.setattr(TRAIN.dist, "get_world_size", lambda: 2)

    with pytest.raises(ValueError, match=f"{section}.{field}"):
        TRAIN._validate_and_build_interface_config(config)


def test_calvin_checkpoint_retention_interval_must_be_a_positive_checkpoint_multiple(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_resolved_toml(ROOT / "configs/calvin_abc_to_d.toml")
    monkeypatch.setattr(TRAIN.dist, "get_world_size", lambda: 2)

    assert config["training"]["checkpoint_interval"] == 1000
    assert config["training"]["permanent_checkpoint_interval"] == 5000
    TRAIN._validate_and_build_interface_config(config)
    config["training"]["permanent_checkpoint_interval"] = 1500
    with pytest.raises(ValueError, match="positive multiple"):
        TRAIN._validate_and_build_interface_config(config)


def test_calvin_split_recipe_must_match_config_and_recompute_exactly() -> None:
    episodes = tuple(CalvinEpisode(index, 2 * index, 2 * index + 1, "calvin_scene_A") for index in range(10))
    annotations = tuple(
        CalvinAnnotation(
            2 * episode + task,
            episode,
            2 * episode,
            2 * episode + 2,
            f"instruction {episode} {task}",
            f"task_{task}",
        )
        for episode in range(10)
        for task in range(2)
    )
    expected = make_calvin_episode_split(episodes, annotations, validation_fraction=0.2, seed=7)
    stats = {
        "split": {
            **_split(expected.train_episode_indices, expected.validation_episode_indices),
            "seed": 7,
            "validation_fraction": 0.2,
        }
    }
    dataset = SimpleNamespace(episodes=episodes, annotations=annotations)
    training = {"validation_split_seed": 7, "validation_episode_fraction": 0.2}

    assert TRAIN._validate_calvin_split_recipe(stats, dataset, training) == (
        expected.train_episode_indices,
        expected.validation_episode_indices,
    )
    with pytest.raises(ValueError, match="split recipe"):
        TRAIN._validate_calvin_split_recipe(stats, dataset, {**training, "validation_split_seed": 8})

    tampered = copy.deepcopy(stats)
    tampered["split"]["train_episode_indices"], tampered["split"]["validation_episode_indices"] = (
        tampered["split"]["validation_episode_indices"],
        tampered["split"]["train_episode_indices"],
    )
    tampered["split"]["train_episode_sha256"] = TRAIN._indices_sha256(tuple(tampered["split"]["train_episode_indices"]))
    tampered["split"]["validation_episode_sha256"] = TRAIN._indices_sha256(
        tuple(tampered["split"]["validation_episode_indices"])
    )
    with pytest.raises(ValueError, match="deterministic config-declared"):
        TRAIN._validate_calvin_split_recipe(tampered, dataset, training)


def test_fixed_batch_identity_is_annotation_and_global_index() -> None:
    class _Sampler:
        population_size = 2

        def __init__(self) -> None:
            self._anchors = iter(
                (
                    CalvinAnchor(0, 9, "task"),
                    CalvinAnchor(0, 9, "task"),
                    CalvinAnchor(1, 9, "task"),
                )
            )

        def draw(self, _generator: torch.Generator) -> CalvinAnchor:
            return next(self._anchors)

    anchors = TRAIN._fixed_distinct_anchors(_Sampler(), count=2, seed=7)

    assert [(anchor.annotation_index, anchor.global_index) for anchor in anchors] == [(0, 9), (1, 9)]


def test_stop_boundary_does_not_add_an_out_of_schedule_validation() -> None:
    assert not TRAIN._validation_is_due(next_update=1, total_updates=30_000, interval=1_000)
    assert not TRAIN._validation_is_due(next_update=999, total_updates=30_000, interval=1_000)
    assert TRAIN._validation_is_due(next_update=1_000, total_updates=30_000, interval=1_000)
    assert TRAIN._validation_is_due(next_update=30_000, total_updates=30_000, interval=1_000)
    assert TRAIN._validation_is_due(next_update=500, total_updates=500, interval=1_000)
    with pytest.raises(ValueError, match="validation schedule"):
        TRAIN._validation_is_due(next_update=0, total_updates=30_000, interval=1_000)


def test_task_filter_keeps_raw_annotations_and_rejects_unknown_task() -> None:
    annotations = (
        CalvinAnnotation(0, 0, 0, 1, " raw A ", "task_a"),
        CalvinAnnotation(1, 0, 2, 3, "raw B", "task_b"),
    )

    assert TRAIN._annotations_for_task(annotations, task=None) is annotations
    assert TRAIN._annotations_for_task(annotations, task="task_a") == (annotations[0],)
    with pytest.raises(ValueError, match="unknown CALVIN task"):
        TRAIN._annotations_for_task(annotations, task="missing")


def test_processor_receives_static_then_gripper_then_raw_instruction(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Inputs(dict):
        def to(self, device: torch.device) -> _Inputs:
            self["device"] = device
            return self

    class _Processor:
        captured: object | None = None

    static = np.full((200, 200, 3), 11, dtype=np.uint8)
    gripper = np.full((84, 84, 3), 29, dtype=np.uint8)
    sample = SimpleNamespace(
        observation=CanonicalObservation(
            third_person=static,
            wrist=gripper,
            state=torch.zeros(8),
        ),
        instruction="  raw CALVIN instruction?!  ",
    )
    processor = _Processor()
    prefix_geometry = {
        "geometry": {"fixed_physical_prefix_width": 600},
        "instruction_inventory": {"records": [{"instruction": sample.instruction, "valid_prefix_length": 532}]},
        "ordered_cameras": [{}, {}],
        "tokenization": {"padding_side": "left"},
    }

    def apply_fixed(processor_arg, conversations, **kwargs):
        processor_arg.captured = conversations
        assert kwargs == {
            "expected_batch_size": 8,
            "fixed_physical_prefix_width": 600,
            "images_per_prefix": 2,
            "padding_side": "left",
        }
        return _Inputs(attention_mask=torch.ones((8, 532), dtype=torch.long))

    monkeypatch.setattr(TRAIN, "apply_fixed_prefix_chat_template", apply_fixed)

    result = TRAIN._processor_inputs(
        processor,
        (sample,) * TRAIN.PHYSICAL_BATCH_SIZE,
        torch.device("cpu"),
        prefix_geometry=prefix_geometry,
    )

    assert isinstance(processor.captured, list)
    assert len(processor.captured) == TRAIN.PHYSICAL_BATCH_SIZE
    content = processor.captured[0][0]["content"]
    assert [entry["type"] for entry in content] == ["image", "image", "text"]
    np.testing.assert_array_equal(np.asarray(content[0]["image"]), static)
    np.testing.assert_array_equal(np.asarray(content[1]["image"]), gripper)
    assert content[2]["text"] == sample.instruction
    assert result["device"] == torch.device("cpu")


def test_calvin_data_identity_pins_action_state_cameras_splits_and_sources() -> None:
    stats = _v4_stats()

    identity = TRAIN._calvin_data_identity(
        stats,
        source_revisions=TRAIN.PINNED_CALVIN_SOURCE_REVISIONS,
    )

    assert identity["protocol"] == "duovla-calvin-abc-to-d-v1"
    assert identity["archive_bytes"] == TRAIN.CALVIN_ABC_D_ARCHIVE_BYTES
    assert identity["archive_sha256"] == TRAIN.CALVIN_ABC_D_ARCHIVE_SHA256
    assert identity["central_directory_sha256"] == TRAIN.OFFICIAL_CENTRAL_DIRECTORY_SHA256
    assert identity["dataset_manifest_file_sha256"] == "5" * 64
    assert identity["dataset_manifest_schema"] == TRAIN.CALVIN_DATASET_MANIFEST_SCHEMA
    assert identity["metadata_sha256"] == "2" * 64
    assert identity["dataset_manifest_sha256"] == "3" * 64
    assert identity["member_index"] == {
        "bytes": 123_456,
        "path": TRAIN.CALVIN_INDEX_NAME,
        "schema": TRAIN.CALVIN_MEMBER_INDEX_SCHEMA,
        "sha256": "6" * 64,
    }
    assert identity["member_inventory_sha256"] == "4" * 64
    assert identity["metadata_files"] == list(TRAIN.CALVIN_CRITICAL_TRAIN_METADATA)
    assert identity["normalization_sha256"] == "1" * 64
    assert identity["reader_schema"] == TRAIN.CALVIN_ARCHIVE_READER_SCHEMA
    assert identity["storage_identity_sha256"] == "7" * 64
    assert identity["storage_mode"] == TRAIN.CALVIN_STORAGE_MODE_ARCHIVE_DIRECT
    assert identity["camera_shapes"] == {"rgb_static": [200, 200, 3], "rgb_gripper": [84, 84, 3]}
    assert identity["state_adapter"] == "robot_obs[0:7]+robot_obs[14:15]"
    assert identity["action_adapter"] == "identity_official_scaled_rel_actions"
    assert identity["calvin_source_revisions"] == TRAIN.PINNED_CALVIN_SOURCE_REVISIONS
    assert set(identity) == TRAIN._CALVIN_IDENTITY_FIELDS
    assert TRAIN._calvin_storage_run_contract(identity) == {
        "archive_bytes": str(TRAIN.CALVIN_ABC_D_ARCHIVE_BYTES),
        "archive_sha256": TRAIN.CALVIN_ABC_D_ARCHIVE_SHA256,
        "central_directory_sha256": TRAIN.OFFICIAL_CENTRAL_DIRECTORY_SHA256,
        "dataset_manifest_file_sha256": "5" * 64,
        "dataset_manifest_schema": TRAIN.CALVIN_DATASET_MANIFEST_SCHEMA,
        "dataset_manifest_sha256": "3" * 64,
        "member_index_bytes": "123456",
        "member_index_path": TRAIN.CALVIN_INDEX_NAME,
        "member_index_schema": TRAIN.CALVIN_MEMBER_INDEX_SCHEMA,
        "member_index_sha256": "6" * 64,
        "member_inventory_sha256": "4" * 64,
        "metadata_sha256": "2" * 64,
        "reader_schema": TRAIN.CALVIN_ARCHIVE_READER_SCHEMA,
        "storage_identity_sha256": "7" * 64,
        "storage_mode": TRAIN.CALVIN_STORAGE_MODE_ARCHIVE_DIRECT,
    }

    stats["action"]["transform"] = "percentile"  # type: ignore[index]
    with pytest.raises(ValueError, match="data contract"):
        TRAIN._calvin_data_identity(stats, source_revisions=TRAIN.PINNED_CALVIN_SOURCE_REVISIONS)


def test_v4_storage_identity_is_bound_to_rank_and_resume_contracts() -> None:
    identity = TRAIN._calvin_data_identity(
        _v4_stats(),
        source_revisions=TRAIN.PINNED_CALVIN_SOURCE_REVISIONS,
    )
    contract = TRAIN._calvin_storage_run_contract(identity)
    assert set(contract) == TRAIN._CALVIN_STORAGE_RUN_CONTRACT_FIELDS
    assert set(contract) <= TRAIN._CALVIN_RUN_CONTRACT_FIELDS
    manifest: dict[str, object] = {
        **contract,
        "archive_bytes": identity["archive_bytes"],
        "member_index_bytes": identity["member_index"]["bytes"],
    }

    TRAIN._validate_resume_manifest_run_contract(manifest, contract)

    for field in sorted(contract):
        changed = copy.deepcopy(manifest)
        changed[field] = changed[field] + 1 if field in TRAIN._INTEGER_MANIFEST_RUN_CONTRACT_FIELDS else "wrong"
        with pytest.raises(ValueError, match=field):
            TRAIN._validate_resume_manifest_run_contract(changed, contract)


@pytest.mark.parametrize(
    ("field", "wrong_value"),
    (
        ("schema", "duo-vla-calvin-abc-to-d-normalization-v3"),
        ("archive_bytes", TRAIN.CALVIN_ABC_D_ARCHIVE_BYTES - 1),
        ("central_directory_sha256", "a" * 64),
        ("dataset_manifest_schema", "duo-vla-calvin-dataset-manifest-v3"),
        ("metadata_files", list(reversed(TRAIN.CALVIN_CRITICAL_TRAIN_METADATA))),
        ("reader_schema", "reader-v0"),
        ("storage_mode", "verified-extraction"),
    ),
)
def test_calvin_data_identity_rejects_non_v4_storage_contract(field: str, wrong_value: object) -> None:
    stats = _v4_stats()
    if field == "schema":
        stats[field] = wrong_value
    else:
        stats["dataset"][field] = wrong_value  # type: ignore[index]

    with pytest.raises(ValueError, match=r"normalization-v4|data contract"):
        TRAIN._calvin_data_identity(stats, source_revisions=TRAIN.PINNED_CALVIN_SOURCE_REVISIONS)


@pytest.mark.parametrize(
    ("section", "field"),
    (
        ("dataset", "dataset_manifest_file_sha256"),
        ("dataset", "dataset_manifest_sha256"),
        ("dataset", "member_inventory_sha256"),
        ("dataset", "metadata_sha256"),
        ("dataset", "storage_identity_sha256"),
        ("member_index", "sha256"),
        ("stats", "content_sha256"),
    ),
)
def test_calvin_data_identity_requires_lowercase_sha256(section: str, field: str) -> None:
    stats = _v4_stats()
    if section == "dataset":
        stats["dataset"][field] = "A" * 64  # type: ignore[index]
    elif section == "member_index":
        stats["dataset"]["member_index"][field] = "A" * 64  # type: ignore[index]
    else:
        stats[field] = "A" * 64

    with pytest.raises(ValueError, match="valid lowercase"):
        TRAIN._calvin_data_identity(stats, source_revisions=TRAIN.PINNED_CALVIN_SOURCE_REVISIONS)


@pytest.mark.parametrize(("section", "field"), (("dataset", "extra"), ("member_index", "extra")))
def test_calvin_data_identity_rejects_storage_field_inventory_drift(section: str, field: str) -> None:
    stats = _v4_stats()
    if section == "dataset":
        stats["dataset"][field] = "unexpected"  # type: ignore[index]
    else:
        stats["dataset"]["member_index"][field] = "unexpected"  # type: ignore[index]

    with pytest.raises(ValueError, match="field inventory"):
        TRAIN._calvin_data_identity(stats, source_revisions=TRAIN.PINNED_CALVIN_SOURCE_REVISIONS)


def test_source_hash_tracks_calvin_trainer_modules_and_config_not_libero_script(tmp_path: Path) -> None:
    tracked = (
        "src/duo_vla/data/calvin.py",
        "src/duo_vla/data/calvin_batching.py",
        "src/duo_vla/data/calvin_stats.py",
        "configs/base.toml",
        "configs/calvin_abc_to_d.toml",
        "configs/calvin_abc_to_d_direct.toml",
        "scripts/run_calvin_train.sh",
        "scripts/train_calvin.py",
        "scripts/calvin/calvin_bridge.py",
        "scripts/calvin/prepare_archive_direct.py",
        "scripts/calvin/run_policy_server.sh",
        "scripts/calvin/revisions.env",
        "scripts/calvin/serve_policy.py",
        "pyproject.toml",
        "uv.lock",
    )
    for relative in tracked:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(relative, encoding="utf-8")
    baseline = TRAIN._source_tree_sha256(tmp_path)
    (tmp_path / "scripts/train_libero.py").write_text("unrelated", encoding="utf-8")
    assert TRAIN._source_tree_sha256(tmp_path) == baseline

    calvin_module = tmp_path / "src/duo_vla/data/calvin_batching.py"
    calvin_module.write_text("changed", encoding="utf-8")
    assert TRAIN._source_tree_sha256(tmp_path) != baseline


def test_calvin_source_inventory_v3_pins_archive_direct_preparer() -> None:
    assert TRAIN._CALVIN_SOURCE_TREE_HASH_MAGIC == b"duo-vla-calvin-training-source-tree\x00v3\x00"
    assert "scripts/calvin/prepare_archive_direct.py" in TRAIN._CALVIN_SOURCE_EXPLICIT_RELATIVE_PATHS


def _write_minimal_calvin_source_tree(root: Path, *, leaf: str, content: str) -> None:
    for relative in TRAIN._CALVIN_SOURCE_EXPLICIT_RELATIVE_PATHS:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"explicit:{relative}", encoding="utf-8")
    source = root / "src/duo_vla" / leaf
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text(content, encoding="utf-8")


def test_calvin_source_hash_framing_separates_path_content_boundaries(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    _write_minimal_calvin_source_tree(first, leaf="a", content="bc")
    _write_minimal_calvin_source_tree(second, leaf="ab", content="c")

    assert TRAIN._source_tree_sha256(first) != TRAIN._source_tree_sha256(second)


def test_calvin_source_hash_rejects_symlinks_and_missing_explicit_files(tmp_path: Path) -> None:
    _write_minimal_calvin_source_tree(tmp_path, leaf="module.py", content="source")
    target = tmp_path / "outside.py"
    target.write_text("outside", encoding="utf-8")
    symlink = tmp_path / "src/duo_vla/linked.py"
    symlink.symlink_to(target)
    with pytest.raises(RuntimeError, match="regular file"):
        TRAIN._source_tree_sha256(tmp_path)

    symlink.unlink()
    fifo = tmp_path / "src/duo_vla/nonregular"
    os.mkfifo(fifo)
    with pytest.raises(RuntimeError, match="regular file"):
        TRAIN._source_tree_sha256(tmp_path)

    fifo.unlink()
    (tmp_path / TRAIN._CALVIN_SOURCE_EXPLICIT_RELATIVE_PATHS[0]).unlink()
    with pytest.raises(RuntimeError, match="missing"):
        TRAIN._source_tree_sha256(tmp_path)


@pytest.mark.parametrize("ancestor", ("src", "configs"))
def test_calvin_source_hash_rejects_symlinked_ancestors(tmp_path: Path, ancestor: str) -> None:
    _write_minimal_calvin_source_tree(tmp_path, leaf="module.py", content="source")
    original = tmp_path / ancestor
    real = tmp_path / f"real-{ancestor}"
    original.rename(real)
    original.symlink_to(real.name, target_is_directory=True)

    with pytest.raises(RuntimeError, match="ancestor must be a real directory"):
        TRAIN._source_tree_sha256(tmp_path)


def test_calvin_source_hash_rejects_symlink_above_project_root(tmp_path: Path) -> None:
    real_parent = tmp_path / "real-parent"
    project = real_parent / "project"
    _write_minimal_calvin_source_tree(project, leaf="module.py", content="source")
    alias = tmp_path / "alias-parent"
    alias.symlink_to(real_parent.name, target_is_directory=True)

    with pytest.raises(RuntimeError, match="ancestor must be a real directory"):
        TRAIN._source_tree_sha256(alias / "project")


def test_calvin_source_hash_rejects_ancestor_replacement_during_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_minimal_calvin_source_tree(tmp_path, leaf="module.py", content="source")
    real_open = calvin_source_identity.os.open
    replaced = False

    def replace_configs_on_file_open(path, flags, *args, **kwargs):
        nonlocal replaced
        if path == "base.toml" and not flags & os.O_DIRECTORY and not replaced:
            replaced = True
            (tmp_path / "configs").rename(tmp_path / "configs-original")
            (tmp_path / "configs").mkdir()
        return real_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(calvin_source_identity.os, "open", replace_configs_on_file_open)
    with pytest.raises(RuntimeError, match="ancestor changed while reading"):
        TRAIN._source_tree_sha256(tmp_path)
    assert replaced


def test_trainer_source_uses_only_calvin_data_seam_and_metadata_checked_loader() -> None:
    source = SCRIPT.read_text(encoding="utf-8")

    assert 'parser.add_argument("training_root"' in source
    assert 'default=Path("configs/calvin_abc_to_d.toml")' in source
    assert "CalvinTaskUniformAnchorSampler" in source
    assert "expected_scenes=CALVIN_EXPECTED_SCENES" in source
    assert "authenticate_calvin_dataset_generation" in source
    assert "AuthenticatedCalvinDatasetGeneration.from_dict" in source
    assert "authenticated_generation=authenticated_generation" in source
    assert "verify_frame_files=True" not in source
    assert "verify_archive=False" not in source
    assert "training_root=args.training_root" in source
    assert "action_normalizer" not in source
    assert "load_libero" not in source
    assert '"kind": "resumable-calvin-abc-to-d-training"' in source


def test_canonical_calvin_training_launcher_scrubs_injection_and_pins_tp2() -> None:
    source = (ROOT / "scripts/run_calvin_train.sh").read_text(encoding="utf-8")

    assert "venvs/train" in source
    assert "--nproc-per-node=2" in source
    assert "exec /usr/bin/env -i" in source
    assert "PYTHONPATH" not in source
    assert '"CUBLAS_WORKSPACE_CONFIG=:4096:8"' in source
    assert '"CUDA_VISIBLE_DEVICES=0,1"' in source
    assert '"PYTHONSAFEPATH=1"' in source
    assert '"PYTHONDONTWRITEBYTECODE=1"' in source
    assert '"PYTHONPYCACHEPREFIX=/dev/null"' in source
    assert "-P -B -X pycache_prefix=/dev/null" in source
    assert '"LANG=C.UTF-8"' in source and '"LC_ALL=C.UTF-8"' in source and '"TZ=UTC"' in source


def test_single_gpu_calvin_configs_and_launcher_pin_tp1_gpu_zero() -> None:
    for base_name, single_name in (
        ("calvin_abc_to_d.toml", "calvin_abc_to_d_single_gpu.toml"),
        ("calvin_abc_to_d_direct.toml", "calvin_abc_to_d_direct_single_gpu.toml"),
    ):
        base = load_resolved_toml(ROOT / "configs" / base_name)
        single = load_resolved_toml(ROOT / "configs" / single_name)
        assert single["execution_profile"] == "duovla-single-gpu-tp1-v1"
        assert single["model"]["tensor_parallel_size"] == 1
        single.pop("execution_profile")
        single["model"]["tensor_parallel_size"] = 2
        assert single == base

    source = (ROOT / "scripts/run_calvin_train_single_gpu.sh").read_text(encoding="utf-8")
    assert "venvs/train-single-gpu" in source
    assert "--nproc-per-node=1" in source
    assert '"CUDA_VISIBLE_DEVICES=0"' in source
    assert "exec /usr/bin/env -i" in source


def test_training_runtime_rejects_inherited_pythonpath_before_import_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache_root = tmp_path / "cache"
    train_prefix = cache_root / "venvs/train"
    monkeypatch.setattr(TRAIN.platform, "python_version", lambda: TRAIN.EXPECTED_TRAIN_PYTHON)
    monkeypatch.setattr(TRAIN.sys, "prefix", str(train_prefix))
    monkeypatch.setattr(TRAIN.site, "ENABLE_USER_SITE", False)
    monkeypatch.setattr(
        TRAIN.sys,
        "flags",
        SimpleNamespace(dont_write_bytecode=1, no_user_site=1, safe_path=1),
    )
    monkeypatch.setattr(TRAIN.sys, "dont_write_bytecode", True)
    monkeypatch.setattr(TRAIN.sys, "pycache_prefix", "/dev/null")
    version = f"python{TRAIN.sys.version_info.major}.{TRAIN.sys.version_info.minor}"
    compact_version = f"python{TRAIN.sys.version_info.major}{TRAIN.sys.version_info.minor}"
    monkeypatch.setattr(
        TRAIN.sys,
        "path",
        [
            str(ROOT / "src"),
            str(Path(TRAIN.sys.base_prefix) / "lib" / f"{compact_version}.zip"),
            str(Path(TRAIN.sys.base_prefix) / "lib" / version),
            str(Path(TRAIN.sys.base_exec_prefix) / "lib" / version / "lib-dynload"),
            str(train_prefix / "lib" / version / "site-packages"),
        ],
    )
    monkeypatch.setenv("DUO_VLA_CACHE_ROOT", str(cache_root))
    for name, value in TRAIN.REQUIRED_TRAIN_ENVIRONMENT.items():
        monkeypatch.setenv(name, value)
    for name in ("LD_LIBRARY_PATH", "LD_PRELOAD", "PYTHONHOME", "PYTHONINSPECT", "PYTHONSTARTUP"):
        monkeypatch.delenv(name, raising=False)
    for name in tuple(TRAIN.os.environ):
        if name.startswith("NCCL_"):
            monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("DUO_VLA_PROJECT_ROOT", str(ROOT))
    monkeypatch.setenv("DUO_VLA_TRAIN_VENV", str(train_prefix))
    monkeypatch.setenv("HF_HOME", "/root/.cache/huggingface")
    monkeypatch.setenv("PYTHONHASHSEED", "0")
    monkeypatch.setenv("PYTHONPATH", "/tmp/untrusted")

    with pytest.raises(RuntimeError, match="PYTHONPATH"):
        TRAIN._configure_and_validate_training_runtime(ROOT)


def test_resolved_config_pins_prefix_geometry_and_rejects_unsupported_behavior(monkeypatch) -> None:
    config = TRAIN.load_resolved_toml(ROOT / "configs" / "calvin_abc_to_d.toml")
    monkeypatch.setattr(TRAIN.dist, "get_world_size", lambda: 2)

    assert config["model"]["experts_implementation"] == "grouped_mm"
    assert config["model"]["expert_batch_isolation"] == "sample_isolated_grouped_mm_v1"
    assert config["optimization"]["physical_batch_size"] == 8
    assert config["optimization"]["microbatch_size"] == 8
    assert config["optimization"]["gradient_accumulation_steps"] == 8
    assert config["optimization"]["global_batch_size"] == 64
    assert config["benchmark"]["prefix_geometry_content_sha256"] == (
        "edaef86df702e34c9be6c9103e4f3c9ccc4022df8def46df7d1f00084d0831c9"
    )
    assert config["benchmark"]["fixed_physical_prefix_width"] == 538

    interface = TRAIN._validate_and_build_interface_config(config)

    assert interface.action_horizon == 8 and interface.action_dim == 7
    direct = TRAIN.load_resolved_toml(ROOT / "configs" / "calvin_abc_to_d_direct.toml")
    TRAIN._validate_and_build_interface_config(direct)
    assert TRAIN.policy_contract_from_config(direct).objective == "direct_regression"
    for field in ("prefix_geometry_content_sha256", "fixed_physical_prefix_width"):
        changed = copy.deepcopy(config)
        del changed["benchmark"][field]
        with pytest.raises(ValueError, match=field):
            TRAIN._validate_and_build_interface_config(changed)
    for section, field, value in (
        ("optimization", "optimizer", "sgd"),
        ("optimization", "loss_accumulation_dtype", "bfloat16"),
        ("training", "data_loader_workers_per_rank", 4),
        ("training", "image_augmentation", "random_crop"),
    ):
        changed = copy.deepcopy(config)
        changed[section][field] = value
        with pytest.raises(ValueError, match=f"{section}.{field}"):
            TRAIN._validate_and_build_interface_config(changed)


def test_calvin_checkpoint_manifest_binds_parent_and_retention_uses_authenticated_config() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    parent = {
        "relative_path": "checkpoints/update-001000",
        "update": 1000,
    }

    assert '"checkpoint_retention": make_checkpoint_retention_contract(' in source
    assert "retention = apply_checkpoint_retention(output_dir)" in source
    assert "apply_checkpoint_retention(\n" not in source
    assert '"manifest_sha256": parent_manifest_sha256' not in source
    assert (
        TRAIN.make_checkpoint_retention_contract(
            permanent_checkpoint_interval=5000,
            parent_checkpoint=parent,
        )["parent_checkpoint"]
        == parent
    )
