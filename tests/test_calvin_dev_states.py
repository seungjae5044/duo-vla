from __future__ import annotations

import ast
import copy
import hashlib
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from calvin_dev_test_support import build_bundle_fixture, make_replay_objects, make_v4_fixture

import duo_vla.data.calvin_dev_states as dev


def _rehash_manifest(manifest: dict[str, Any]) -> None:
    manifest["root_sha256"] = dev._manifest_root_sha256(manifest)


def _materialized(
    replays: tuple[dev.BundledCalvinReplay, ...],
    base_seed: int,
) -> list[dev.MaterializedCalvinReset]:
    return [
        dev.MaterializedCalvinReset(
            candidate=replay.candidate,
            frame=replay.frame,
            candidate_rank_sha256=dev.candidate_rank_sha256(replay.candidate, base_seed),
            replay_actions=replay.actions.shape[0],
            replay_bundle_record_sha256=replay.record_sha256,
        )
        for replay in replays
    ]


def test_v4_inputs_authenticate_projected_bytes_without_archive_or_episode_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    training, stats, source, revisions, _members = make_v4_fixture(tmp_path, monkeypatch)
    data_root = training.parents[1]
    assert not (data_root / dev.ARCHIVE_NAME).exists()
    assert not any(training.glob("episode_*.npz"))

    inputs = dev.authenticate_dev_inputs(training, stats, source, revisions)
    candidates = dev.load_validation_candidates(inputs.metadata, inputs.split)

    assert inputs.identity["input_schema"] == dev.DEV_INPUT_SCHEMA
    assert inputs.identity["storage_mode"] == "archive-direct"
    assert inputs.identity["reader_schema"] == dev.ARCHIVE_READER_SCHEMA
    assert inputs.identity["member_index_schema"] == dev.MEMBER_INDEX_SCHEMA
    assert inputs.identity["storage_identity_sha256"] == inputs.stats["dataset"]["storage_identity_sha256"]
    assert inputs.identity["dataset_manifest_file_sha256"] == inputs.stats["dataset"]["dataset_manifest_file_sha256"]
    assert {candidate.episode_index for candidate in candidates} == set(inputs.split["validation_episode_indices"])
    assert {candidate.scene for candidate in candidates} == set(dev.ABC_SCENES)
    assert len(candidates) == 12


@pytest.mark.parametrize(
    ("mutator", "error"),
    [
        (lambda payload: payload.update(schema="duo-vla-calvin-dataset-manifest-v3"), "requires a v4"),
        (lambda payload: payload["storage"].update(mode="verified-extraction"), "content hash"),
        (
            lambda payload: payload["storage"]["member_index"].update(schema="duo-vla-calvin-member-index-v1"),
            "content hash",
        ),
    ],
)
def test_manifest_schema_storage_or_index_drift_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutator: Callable[[dict[str, Any]], None],
    error: str,
) -> None:
    training, stats, source, revisions, _members = make_v4_fixture(tmp_path, monkeypatch)
    manifest_path = training.parents[1] / dev.MANIFEST_NAME
    manifest = dev.load_strict_json(manifest_path)
    mutator(manifest)
    manifest_path.write_bytes(dev.canonical_json_bytes(manifest, pretty=True))

    with pytest.raises(dev.CalvinDevStateError, match=error):
        dev.authenticate_dev_inputs(training, stats, source, revisions)


def test_normalization_v4_exact_inventory_and_projected_metadata_are_authenticated(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    training, stats, source, revisions, _members = make_v4_fixture(tmp_path, monkeypatch)
    payload = dev.load_strict_json(stats)
    payload["schema"] = "duo-vla-calvin-abc-to-d-normalization-v3"
    payload["content_sha256"] = dev._content_sha256(payload)
    stats.write_bytes(dev.canonical_json_bytes(payload, pretty=True))
    with pytest.raises(dev.CalvinDevStateError, match="normalization schema v4"):
        dev.authenticate_dev_inputs(training, stats, source, revisions)

    training, stats, source, revisions, _members = make_v4_fixture(tmp_path / "again", monkeypatch)
    payload = dev.load_strict_json(stats)
    payload["dataset"]["unexpected"] = True
    payload["content_sha256"] = dev._content_sha256(payload)
    stats.write_bytes(dev.canonical_json_bytes(payload, pretty=True))
    with pytest.raises(dev.CalvinDevStateError, match="dataset identity fields differ"):
        dev.authenticate_dev_inputs(training, stats, source, revisions)

    training, stats, source, revisions, _members = make_v4_fixture(tmp_path / "metadata", monkeypatch)
    path = training / ".hydra/merged_config.yaml"
    path.write_bytes(path.read_bytes() + b"changed: true\n")
    with pytest.raises(dev.CalvinDevStateError, match="projected metadata byte count differs"):
        dev.authenticate_dev_inputs(training, stats, source, revisions)


def test_replay_bundle_round_trip_and_reset_bank_cross_bind_all_identities(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs, manifest, artifacts, source_replays = build_bundle_fixture(tmp_path, monkeypatch)
    output = tmp_path / "replay-bundle"
    dev.write_replay_bundle_exclusive(output, manifest, artifacts, inputs)

    loaded_manifest, replays = dev.load_replay_bundle(output, inputs)
    assert loaded_manifest == manifest
    assert len(replays) == len(source_replays) == 12
    assert np.array_equal(replays[0].actions, source_replays[0].actions)
    assert replays[0].member_identities == source_replays[0].member_identities
    assert loaded_manifest["identity"]["member_index_sha256"] == inputs.member_index["sha256"]
    assert loaded_manifest["identity"]["storage_identity_sha256"] == inputs.stats["dataset"]["storage_identity_sha256"]
    with pytest.raises(dev.CalvinDevStateError, match="overwrite"):
        dev.write_replay_bundle_exclusive(output, manifest, artifacts, inputs)

    base_seed = 7
    bank_manifest, bank_artifacts = dev.build_bank(
        _materialized(replays, base_seed),
        inputs.identity,
        inputs.split,
        manifest,
        base_seed,
    )
    assert bank_manifest["schema"] == dev.DEV_BANK_SCHEMA
    assert bank_manifest["replay_bundle"] == {
        "records_sha256": dev.canonical_sha256(manifest["records"]),
        "root_sha256": manifest["root_sha256"],
        "schema": dev.REPLAY_BUNDLE_SCHEMA,
    }
    robots, scenes = dev.validate_bank(bank_manifest, bank_artifacts)
    assert robots.shape == (12, 15)
    assert scenes.shape == (12, 24)
    bank_dir = tmp_path / "reset-bank"
    dev.write_bank_exclusive(bank_dir, bank_manifest, bank_artifacts)
    observed_manifest, observed_robots, observed_scenes = dev.load_bank(bank_dir)
    assert observed_manifest == bank_manifest
    assert np.array_equal(observed_robots, robots)
    assert np.array_equal(observed_scenes, scenes)


def test_bundle_omission_reordering_and_artifact_tamper_are_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs, manifest, artifacts, replays = build_bundle_fixture(tmp_path, monkeypatch)
    source = manifest["source"]
    with pytest.raises(dev.CalvinDevStateError, match="omitted a held-out candidate"):
        dev.build_replay_bundle(replays[:-1], inputs, source)
    with pytest.raises(dev.CalvinDevStateError, match=r"candidate.*order differs"):
        dev.build_replay_bundle(tuple(reversed(replays)), inputs, source)

    reordered = copy.deepcopy(manifest)
    reordered["records"][0], reordered["records"][1] = reordered["records"][1], reordered["records"][0]
    _rehash_manifest(reordered)
    with pytest.raises(dev.CalvinDevStateError, match=r"candidate.*order differs"):
        dev.validate_replay_bundle(reordered, artifacts, inputs, validate_member_index=False)

    omitted = copy.deepcopy(manifest)
    omitted["records"].pop()
    _rehash_manifest(omitted)
    with pytest.raises(dev.CalvinDevStateError, match="omitted or added held-out"):
        dev.validate_replay_bundle(omitted, artifacts, inputs, validate_member_index=False)

    corrupted = dict(artifacts)
    corrupted[dev.ACTION_ARTIFACT] = artifacts[dev.ACTION_ARTIFACT] + b"x"
    with pytest.raises(dev.CalvinDevStateError, match="byte count differs"):
        dev.validate_replay_bundle(manifest, corrupted, inputs, validate_member_index=False)


def test_bundle_member_identity_is_reauthenticated_against_current_v2_index(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs, manifest, artifacts, _replays = build_bundle_fixture(tmp_path, monkeypatch)
    changed = copy.deepcopy(manifest)
    record = changed["records"][0]
    wrong_sha = hashlib.sha256(b"not-the-member").hexdigest()
    record["member_identities"][0]["logical_sha256"] = wrong_sha
    record["source_frame_sha256"] = wrong_sha
    detached = dict(record)
    detached.pop("record_sha256")
    record["record_sha256"] = dev.canonical_sha256(detached)
    _rehash_manifest(changed)

    with pytest.raises(dev.CalvinDevStateError, match="current v2 index"):
        dev.validate_replay_bundle(changed, artifacts, inputs, validate_member_index=True)


def test_bundle_rejects_development_source_drift_after_input_authentication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs, manifest, artifacts, _replays = build_bundle_fixture(tmp_path, monkeypatch)
    current = dev.development_tool_source_identity()
    current["replay_generator_source_sha256"] = "0" * 64
    monkeypatch.setattr(dev, "development_tool_source_identity", lambda: current)
    with pytest.raises(dev.CalvinDevStateError, match="development source changed"):
        dev.validate_replay_bundle(manifest, artifacts, inputs, validate_member_index=False)


@pytest.mark.parametrize(
    ("section", "field", "replacement", "error"),
    [
        ("algorithm", "frame_order", "filesystem order", "algorithm contract differs"),
        ("counts", "training_frames", 11, "counts differ from authenticated metadata"),
        ("state", "q99", [1.0] * 6, "state bounds differ"),
        ("action", "continuous_max", [2.0] * 6, "action ranges differ"),
    ],
)
def test_normalization_semantic_tables_are_not_accepted_on_content_hash_alone(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    section: str,
    field: str,
    replacement: object,
    error: str,
) -> None:
    training, stats, source, revisions, _members = make_v4_fixture(tmp_path, monkeypatch)
    payload = dev.load_strict_json(stats)
    payload[section][field] = replacement
    payload["content_sha256"] = dev._content_sha256(payload)
    stats.write_bytes(dev.canonical_json_bytes(payload, pretty=True))
    with pytest.raises(dev.CalvinDevStateError, match=error):
        dev.authenticate_dev_inputs(training, stats, source, revisions)


def test_generator_side_never_opens_archive_or_episode_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    training, stats, source, revisions, members = make_v4_fixture(tmp_path, monkeypatch)
    original_open = dev.os.open
    observed: list[str] = []

    def guarded_open(path: object, flags: int, *args: object, **kwargs: object) -> int:
        rendered = os.fspath(path)
        observed.append(rendered)
        assert not rendered.endswith((".zip", ".npz"))
        assert "episode_" not in rendered
        return original_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(dev.os, "open", guarded_open)
    inputs = dev.authenticate_dev_inputs(training, stats, source, revisions)
    replays = make_replay_objects(inputs, members)
    source_identity = {
        "calvin_archive_source_sha256": inputs.identity["calvin_archive_source_sha256"],
        "dev_states_source_sha256": inputs.identity["dev_states_source_sha256"],
        "replay_exporter_source_sha256": inputs.identity["replay_exporter_source_sha256"],
        "schema": "duo-vla-calvin-dev-replay-export-source-v1",
    }
    manifest, artifacts = dev.build_replay_bundle(replays, inputs, source_identity)
    output = tmp_path / "bundle"
    dev.write_replay_bundle_exclusive(output, manifest, artifacts, inputs)
    dev.load_replay_bundle(output, inputs)
    assert observed


def test_python38_boundary_sources_parse_with_python38_grammar() -> None:
    root = Path(__file__).resolve().parents[1]
    sources = (
        root / "src/duo_vla/data/calvin_dev_states.py",
        root / "scripts/calvin/generate_calvin_dev_states.py",
        root / "scripts/calvin/evaluate_calvin_dev.py",
    )
    for path in sources:
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path), feature_version=(3, 8))


def test_bank_rejects_record_not_cross_bound_to_replay_bundle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs, manifest, _artifacts, replays = build_bundle_fixture(tmp_path, monkeypatch)
    reset = _materialized(replays, 9)[0]
    changed = dev.MaterializedCalvinReset(
        candidate=reset.candidate,
        frame=reset.frame,
        candidate_rank_sha256=reset.candidate_rank_sha256,
        replay_actions=reset.replay_actions,
        replay_bundle_record_sha256="0" * 64,
    )
    with pytest.raises(dev.CalvinDevStateError, match="not bound to a replay-bundle record"):
        dev.build_bank([changed], inputs.identity, inputs.split, manifest, 9, smoke_tasks_per_scene=1)
