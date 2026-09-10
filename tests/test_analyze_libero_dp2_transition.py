from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any

import pytest
import torch
from safetensors.torch import save_file

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/analyze_libero_dp2_transition.py"
SPEC = importlib.util.spec_from_file_location("analyze_libero_dp2_transition", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
ANALYZE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = ANALYZE
SPEC.loader.exec_module(ANALYZE)
PROJECT_ROOT = SCRIPT.parents[1]

from duo_vla.dp2_fork import dp2_semantic_recipe_sha256  # noqa: E402
from duo_vla.run_config import load_resolved_toml, save_resolved_config  # noqa: E402


def test_tensor_delta_statistics_reports_identical_and_rotated_updates(tmp_path: Path) -> None:
    parent = tmp_path / "parent.safetensors"
    reference = tmp_path / "reference.safetensors"
    candidate = tmp_path / "candidate.safetensors"
    save_file({"weight": torch.tensor([0.0, 0.0], dtype=torch.float32)}, parent)
    save_file({"weight": torch.tensor([1.0, 2.0], dtype=torch.float32)}, reference)
    save_file({"weight": torch.tensor([1.0, 2.0], dtype=torch.float32)}, candidate)

    exact = ANALYZE._tensor_delta_statistics(parent, reference, candidate)

    assert exact["delta_cosine_similarity"] == pytest.approx(1.0)
    assert exact["delta_relative_l2_difference"] == pytest.approx(0.0)
    assert exact["max_absolute_final_difference"] == pytest.approx(0.0)
    assert exact["elements"] == 2
    assert exact["tensors"] == 1

    save_file({"weight": torch.tensor([2.0, -1.0], dtype=torch.float32)}, candidate)
    rotated = ANALYZE._tensor_delta_statistics(parent, reference, candidate)
    assert rotated["delta_cosine_similarity"] == pytest.approx(0.0)
    assert rotated["delta_relative_l2_difference"] == pytest.approx(2.0**0.5)


def test_exact_nested_accepts_tensor_state_and_rejects_replica_drift() -> None:
    left = {"groups": [{"step": torch.tensor(1001), "value": torch.tensor([1.0, 2.0])}]}
    right = {"groups": [{"step": torch.tensor(1001), "value": torch.tensor([1.0, 2.0])}]}

    ANALYZE._exact_nested(left, right, path="state")
    right["groups"][0]["value"][1] = 3.0
    with pytest.raises(ANALYZE.AnalysisError, match="tensor values differ"):
        ANALYZE._exact_nested(left, right, path="state")


def test_transition_thresholds_are_predeclared_and_nontrivial() -> None:
    assert ANALYZE.LOSS_RELATIVE_DIFFERENCE_MAX == 0.01
    assert ANALYZE.GRADIENT_NORM_RELATIVE_DIFFERENCE_MAX == 0.05
    assert ANALYZE.PARAMETER_DELTA_COSINE_MIN == 0.99
    assert ANALYZE.PARAMETER_DELTA_RELATIVE_L2_MAX == 0.15
    assert ANALYZE.MEDIAN_SPEEDUP_MIN == 0.02
    assert ANALYZE.PEAK_MEMORY_GIB_MAX == 75.0


def _metric(update: int, *, seconds: float) -> dict[str, Any]:
    return {
        "examples_seen": update * 64,
        "gradient_norm": 2.0,
        "interface_learning_rate": 0.001,
        "lora_learning_rate": 0.0001,
        "objective": "rectified_flow",
        "train_loss": 1.0,
        "update": update,
        "update_seconds": seconds,
    }


def _write_metrics(path: Path, records: list[dict[str, Any]]) -> bytes:
    raw = "".join(json.dumps(record, allow_nan=False, sort_keys=True) + "\n" for record in records).encode()
    path.write_bytes(raw)
    return raw


def test_metrics_are_stably_bound_and_require_the_exact_ordered_range(tmp_path: Path) -> None:
    path = tmp_path / "metrics.jsonl"
    raw = _write_metrics(path, [_metric(update, seconds=1.0) for update in range(1, 4)])

    records, identity = ANALYZE.load_metrics(
        path,
        name="test",
        expected_start=1,
        expected_end=3,
    )

    assert list(records) == [1, 2, 3]
    assert identity == {
        "bytes": len(raw),
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }
    _write_metrics(path, [_metric(1, seconds=1.0), _metric(3, seconds=1.0)])
    with pytest.raises(ANALYZE.AnalysisError, match="range or ordering differs"):
        ANALYZE.load_metrics(path, name="test", expected_start=1, expected_end=3)


def test_checkpoint_semantic_recipe_authenticates_resolved_config_artifact(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint"
    artifact = checkpoint / "artifacts/resolved_config.json"
    config = load_resolved_toml(PROJECT_ROOT / "configs/libero_single_gpu_fused_v2_b64.toml")
    config["run"] = {"max_cached_files": 377, "seed": 0, "task": None}
    config_sha256 = save_resolved_config(artifact, config)
    raw_sha256 = hashlib.sha256(artifact.read_bytes()).hexdigest()
    manifest = {
        "artifacts": {
            "resolved_config": {
                "bytes": artifact.stat().st_size,
                "path": "artifacts/resolved_config.json",
                "sha256": raw_sha256,
            }
        },
        "config_sha256": config_sha256,
    }

    assert ANALYZE._checkpoint_semantic_recipe_sha256(checkpoint, manifest, name="parent") == (
        raw_sha256,
        dp2_semantic_recipe_sha256(config),
    )


@pytest.mark.parametrize(
    "manifest",
    (
        {
            "cuda_allocator_peak_memory_bytes_by_rank": [0, 1],
            "cuda_allocator_peak_memory_bytes_max": 1,
        },
        {
            "cuda_allocator_peak_memory_bytes_by_rank": [1, True],
            "cuda_allocator_peak_memory_bytes_max": 1,
        },
        {
            "cuda_allocator_peak_memory_bytes_by_rank": [1, 2],
            "cuda_allocator_peak_memory_bytes_max": 1,
        },
    ),
)
def test_peak_memory_comes_only_from_two_positive_cross_linked_manifest_counters(
    manifest: dict[str, Any],
) -> None:
    with pytest.raises(ANALYZE.AnalysisError, match="allocator peak"):
        ANALYZE._candidate_peak_memory(manifest)


def test_run_seals_semantic_and_performance_checkpoints_metrics_and_memory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent_path = (tmp_path / "parent/checkpoints/update-001000").resolve()
    reference_path = (tmp_path / "parent/checkpoints/update-001001").resolve()
    candidate_path = (tmp_path / "candidate/checkpoints/update-001001").resolve()
    performance_path = (tmp_path / "candidate/checkpoints/update-001100").resolve()
    for path in (parent_path, reference_path, candidate_path, performance_path):
        path.mkdir(parents=True)

    reference_records = [_metric(update, seconds=10.0 if update <= 1000 else 9.0) for update in range(1, 1002)]
    candidate_records = [_metric(update, seconds=9.0 if update == 1001 else 8.0) for update in range(1001, 1101)]
    reference_metrics = tmp_path / "reference.jsonl"
    candidate_metrics = tmp_path / "candidate.jsonl"
    reference_raw = _write_metrics(reference_metrics, reference_records)
    candidate_raw = _write_metrics(candidate_metrics, candidate_records)

    parent_sha = "1" * 64
    reference_sha = "2" * 64
    candidate_sha = "3" * 64
    performance_sha = "4" * 64
    parent_run_uuid = "00000000-0000-0000-0000-000000000001"
    candidate_run_uuid = "00000000-0000-0000-0000-000000000002"
    parent_config = "5" * 64
    parent_source = "6" * 64
    optimizer_schema = "7" * 64
    candidate_config = "8" * 64
    candidate_source = "9" * 64
    train_venv = {"root_sha256": "a" * 64}
    parent_stat = os.stat(parent_path, follow_symlinks=False)
    lineage = {
        "fork_manifest_sha256": "b" * 64,
        "parent_checkpoint": str(parent_path),
        "parent_checkpoint_device": parent_stat.st_dev,
        "parent_checkpoint_inode": parent_stat.st_ino,
        "parent_config_sha256": parent_config,
        "parent_manifest_sha256": parent_sha,
        "parent_optimizer_parameter_schema_sha256": optimizer_schema,
        "parent_resolved_config_sha256": "c" * 64,
        "parent_run_uuid": parent_run_uuid,
        "parent_source_tree_sha256": parent_source,
        "parent_update": 1000,
        "semantic_recipe_sha256": "d" * 64,
    }
    parent = {
        "config_sha256": parent_config,
        "execution_environment": {"authenticated_runtime": {"train_venv": train_venv}},
        "execution_profile": ANALYZE.TP1_PROFILE,
        "last_metrics": reference_records[999],
        "optimizer_parameter_schema_sha256": optimizer_schema,
        "run_uuid": parent_run_uuid,
        "source_tree_sha256": parent_source,
    }
    reference = {
        **parent,
        "last_metrics": reference_records[1000],
        "parent_manifest_sha256": parent_sha,
    }
    dp_environment = {
        "authenticated_runtime": {"train_venv": train_venv},
        "cuda_runtime": "12.9",
        "gpu_uuids": [value.removeprefix("GPU-") for value in ANALYZE.DP2_GPU_UUIDS],
    }
    candidate = {
        "config_sha256": candidate_config,
        "cuda_allocator_peak_memory_bytes_by_rank": [60 * 2**30, 61 * 2**30],
        "cuda_allocator_peak_memory_bytes_max": 61 * 2**30,
        "execution_environment": dp_environment,
        "execution_profile": ANALYZE.DP2_PROFILE,
        "fork_lineage": lineage,
        "last_metrics": candidate_records[0],
        "parent_manifest_sha256": None,
        "run_uuid": candidate_run_uuid,
        "source_tree_sha256": candidate_source,
    }
    performance = {
        **candidate,
        "cuda_allocator_peak_memory_bytes_by_rank": [70 * 2**30, 72 * 2**30],
        "cuda_allocator_peak_memory_bytes_max": 72 * 2**30,
        "last_metrics": candidate_records[-1],
        "parent_manifest_sha256": candidate_sha,
    }
    manifests = {
        parent_path: (parent, parent_sha, 1000),
        reference_path: (reference, reference_sha, 1001),
        candidate_path: (candidate, candidate_sha, 1001),
        performance_path: (performance, performance_sha, 1100),
    }

    def authenticate(path: Path, *, expected_update: int, name: str):
        canonical = path.resolve(strict=True)
        manifest, digest, update = manifests[canonical]
        assert expected_update == update
        assert name in {"parent", "reference", "candidate", "performance"}
        return canonical, manifest, digest

    lineage_calls: list[dict[str, Any]] = []

    def validate_lineage(manifest: dict[str, Any]):
        lineage_calls.append(manifest)
        return {"fork": "same"}, dict(manifest["fork_lineage"])

    monkeypatch.setattr(ANALYZE, "authenticate_checkpoint", authenticate)
    monkeypatch.setattr(ANALYZE, "validate_dp2_checkpoint_lineage", validate_lineage)
    monkeypatch.setattr(
        ANALYZE,
        "_checkpoint_semantic_recipe_sha256",
        lambda *_args, **_kwargs: ("c" * 64, "d" * 64),
    )
    monkeypatch.setattr(
        ANALYZE,
        "trainable_delta_statistics",
        lambda *_args: {
            "combined": {
                "delta_cosine_similarity": 1.0,
                "delta_relative_l2_difference": 0.0,
            }
        },
    )
    monkeypatch.setattr(
        ANALYZE,
        "validate_candidate_rank_replicas",
        lambda *_args: {
            "optimizer_scheduler_trainer_bitwise_replicated": True,
            "rng_states_distinct": True,
        },
    )
    output = tmp_path / "report/analysis.json"
    args = argparse.Namespace(
        candidate_checkpoint=candidate_path,
        candidate_metrics=candidate_metrics,
        candidate_performance_checkpoint=None,
        output=output,
        parent_checkpoint=parent_path,
        reference_checkpoint=reference_path,
        reference_metrics=reference_metrics,
    )

    report, digest = ANALYZE.run(args)

    assert report["schema"] == "duovla-libero-dp2-transition-analysis-v2"
    assert report["status"] == "passed"
    assert lineage_calls == [candidate, performance]
    assert report["inputs"]["candidate_performance_checkpoint"] == str(performance_path)
    assert report["inputs"]["candidate_performance_manifest_sha256"] == performance_sha
    assert report["inputs"]["reference_metrics"] == {
        "bytes": len(reference_raw),
        "path": str(reference_metrics.resolve()),
        "sha256": hashlib.sha256(reference_raw).hexdigest(),
    }
    assert report["inputs"]["candidate_metrics"]["bytes"] == len(candidate_raw)
    assert report["inputs"]["candidate_metrics"]["sha256"] == hashlib.sha256(candidate_raw).hexdigest()
    assert report["measurements"]["candidate_peak_memory_bytes_by_rank"] == [70 * 2**30, 72 * 2**30]
    assert report["measurements"]["candidate_peak_memory_gib_by_rank"] == [70.0, 72.0]
    assert report["measurements"]["candidate_peak_memory_gib"] == 72.0
    assert report["measurements"]["timing"]["baseline_window"] == [901, 1000]
    assert report["measurements"]["timing"]["candidate_window"] == [1002, 1100]
    assert report["measurements"]["timing"]["median_speedup"] == pytest.approx(0.2)
    assert hashlib.sha256(output.read_bytes()).hexdigest() == digest


def test_manifest_metric_and_run_crosslinks_reject_drift() -> None:
    metric = _metric(1001, seconds=1.0)
    ANALYZE._crosscheck_manifest_metric({"last_metrics": metric}, metric, name="candidate")
    with pytest.raises(ANALYZE.AnalysisError, match=r"last_metrics.*value differs"):
        ANALYZE._crosscheck_manifest_metric(
            {"last_metrics": {**metric, "train_loss": 2.0}},
            metric,
            name="candidate",
        )
    identity = {"run_uuid": "run", "config_sha256": "a" * 64, "source_tree_sha256": "b" * 64}
    ANALYZE._crosscheck_run_identity(identity, dict(identity), name="candidate/performance")
    with pytest.raises(ANALYZE.AnalysisError, match="config_sha256 differs"):
        ANALYZE._crosscheck_run_identity(
            identity,
            {**identity, "config_sha256": "c" * 64},
            name="candidate/performance",
        )


def test_cli_removes_manual_memory_and_timing_inputs_and_keeps_performance_checkpoint_optional(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(SCRIPT),
            "--parent-checkpoint",
            "/parent",
            "--reference-checkpoint",
            "/reference",
            "--candidate-checkpoint",
            "/candidate",
            "--reference-metrics",
            "/reference-metrics",
            "--candidate-metrics",
            "/candidate-metrics",
            "--output",
            "/output",
        ],
    )

    args = ANALYZE.parse_args()

    assert args.candidate_performance_checkpoint is None
    assert not hasattr(args, "candidate_peak_memory_gib")
    assert not hasattr(args, "baseline_window_start")
    assert not hasattr(args, "candidate_window_start")
