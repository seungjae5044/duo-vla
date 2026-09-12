from __future__ import annotations

import copy
import json
import os
import select
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import rollout_libero_encoder_lora as ENCODER
from rollout_libero_spatial import episode_matrix, reset_physics_snapshot, summarize_reset_physics
from supervise_encoder_rollout import open_process_handle, validate_shard


def metadata():
    recipe = {
        "schema": "duo-vla-prefix-lora-continuation-v1",
        "parent_manifest_sha256": ENCODER.PARENT_HASH,
        "base_update": 15000,
        "additional_updates": 1000,
        "learning_rate": 5e-5,
        "schedule": "constant_no_warmup",
        "vision_frozen": True,
        "base_weights_frozen": True,
        "world_size": 2,
        "physical_batch_size": 64,
        "global_batch_size": 128,
        "encoder_lora_rank": 16,
        "encoder_lora_alpha": 32,
    }
    encoder = {"rank": 16, "alpha": 32, "scope": "prefix_language_attention", "vision_frozen": True}
    manifest = {
        "kind": "experimental-libero-prefix-lora-continuation",
        "encoder_adapted": True,
        "parent_manifest_sha256": ENCODER.PARENT_HASH,
        "recipe_sha256": ENCODER.canonical_hash(recipe),
        "training_suites_json": '["libero_spatial"]',
        "stage_update": 1000,
        "absolute_update": 16000,
    }
    return manifest, recipe, encoder


def test_two_shards_cover_exactly_200_unique_episodes_balanced_by_task():
    shards = [episode_matrix(20, shard_index=index, num_shards=2) for index in range(2)]
    assert all(len(shard) == 100 for shard in shards)
    assert not set(shards[0]) & set(shards[1])
    assert set(shards[0]) | set(shards[1]) == set(episode_matrix(20))
    for shard in shards:
        for task in range(10):
            assert sum(t == task for t, _ in shard) == 10


def test_pidfd_fallback_tracks_process_lifetime(monkeypatch):
    monkeypatch.delattr(os, "pidfd_open", raising=False)
    child = subprocess.Popen([sys.executable, "-c", "import sys; sys.stdin.read()"], stdin=subprocess.PIPE)
    handle = open_process_handle(child.pid)
    try:
        assert not select.select([handle], [], [], 0)[0]
        child.communicate(timeout=10)
        assert select.select([handle], [], [], 10)[0] == [handle]
    finally:
        os.close(handle)
        if child.poll() is None:
            child.kill()
            child.wait()


@pytest.mark.parametrize("args", [(0, 0, 2), (51, 0, 2), (20, 2, 2), (20, 0, 0)])
def test_invalid_shards_fail(args):
    count, index, shards = args
    with pytest.raises(ValueError):
        episode_matrix(count, shard_index=index, num_shards=shards)


def test_final_checkpoint_is_required_not_a_250_step_intermediate():
    manifest, recipe, encoder = metadata()
    ENCODER.validate_stage_metadata(manifest, recipe, encoder)
    manifest.update(stage_update=250, absolute_update=15250)
    with pytest.raises(ValueError, match="all 1000"):
        ENCODER.validate_stage_metadata(manifest, recipe, encoder)
    ENCODER.validate_stage_metadata(manifest, recipe, encoder, final=False)


@pytest.mark.parametrize("field", ["learning_rate", "vision_frozen", "additional_updates", "parent_manifest_sha256"])
def test_changed_recipe_fails_even_when_rehashed(field):
    manifest, recipe, encoder = metadata()
    recipe[field] = None
    manifest["recipe_sha256"] = ENCODER.canonical_hash(recipe)
    with pytest.raises(ValueError):
        ENCODER.validate_stage_metadata(manifest, recipe, encoder)


def test_reset_audit_is_read_only_and_reports_flying_object():
    model = SimpleNamespace(
        njnt=2, jnt_type=[0, 3], jnt_qposadr=[0, 7], jnt_dofadr=[0, 6], joint_id2name=lambda index: f"joint-{index}"
    )
    data = SimpleNamespace(qpos=np.array([0.0, 0.0, 0.8, 1.0, 0.0, 0.0, 0.0, 0.0]), qvel=np.zeros(7))
    env = SimpleNamespace(
        sim=SimpleNamespace(model=model, data=data), get_sim_state=lambda: np.concatenate((data.qpos, data.qvel))
    )
    before = data.qpos.copy()
    first = reset_physics_snapshot(env)
    np.testing.assert_array_equal(before, data.qpos)
    settled = copy.deepcopy(first)
    settled["objects"]["joint-0"]["position"][2] += 0.2
    settled["objects"]["joint-0"]["linear_speed"] = 4.0
    result = summarize_reset_physics([first, settled])
    assert result["warning"]
    assert result["max_free_object_displacement_m"] == pytest.approx(0.2)
    assert result["max_free_object_linear_speed_m_s"] == 4.0
    assert not summarize_reset_physics([first, first])["warning"]
    data.qvel[0] = np.nan
    with pytest.raises(FloatingPointError):
        reset_physics_snapshot(env)


def shard_fixture(tmp_path):
    evaluation, policy = tmp_path / "evaluation", tmp_path / "policy"
    evaluation.mkdir()
    policy.mkdir()
    rows = [
        {
            "task_id": t,
            "reset_id": r,
            "nfe": 4,
            "execution_horizon": 8,
            "shard_index": 0,
            "success": r == 0,
            "steps": 220,
            "calls": 28,
        }
        for t, r in episode_matrix(20, shard_index=0, num_shards=2)
    ]
    (evaluation / "episodes.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    (evaluation / "reset_audit.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    summary = {
        "status": "complete",
        "target": 100,
        "completed": 100,
        "successes": 10,
        "success_rate": 0.1,
        "nfe": 4,
        "execution_horizon": 8,
    }
    serving = {
        "checkpoint_manifest_sha256": "checkpoint",
        "encoder_adapter_sha256": "encoder",
        "encoder_adapter_loaded": True,
        "nfe": 4,
        "absolute_update": 16000,
        "stage_update": 1000,
        "cuda_visible_devices": "0",
        "prefix_autocast": False,
    }
    parity = {"repeat_max_abs_error": 0.0, "permutation_max_abs_error": 0.0, "singleton_max_abs_error": 0.0}
    for path, value in (
        (evaluation / "summary.json", summary),
        (policy / "serving.json", serving),
        (evaluation / "inference_validation.json", parity),
    ):
        path.write_text(json.dumps(value))
    return rows


def test_rollout_completion_requires_exact_matrix_and_encoder_weights(tmp_path):
    rows = shard_fixture(tmp_path)
    actual, summary, audit = validate_shard(tmp_path, 0, "checkpoint", "encoder")
    assert actual == rows and summary["completed"] == len(audit) == 100
    with pytest.raises(RuntimeError, match="exact final"):
        validate_shard(tmp_path, 0, "checkpoint", "wrong-encoder")
    with pytest.raises(RuntimeError, match="off-shard"):
        validate_shard(tmp_path, 1, "checkpoint", "encoder")
    rows[-1] = rows[0]
    (tmp_path / "evaluation/episodes.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    with pytest.raises(RuntimeError, match="duplicated"):
        validate_shard(tmp_path, 0, "checkpoint", "encoder")
