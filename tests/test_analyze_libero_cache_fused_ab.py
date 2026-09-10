from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "analyze_libero_cache_fused_ab.py"
SPEC = importlib.util.spec_from_file_location("analyze_libero_cache_fused_ab", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
ANALYZER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = ANALYZER
SPEC.loader.exec_module(ANALYZER)

SOURCE_SHA256 = "a" * 64
KERNEL_SHA256 = "b" * 64
BASELINE_UUID = "GPU-baseline"
CANDIDATE_UUID = "GPU-candidate"


def _raw_json(value: Any, *, newline: bool = True) -> bytes:
    raw = json.dumps(value, allow_nan=False, separators=(",", ":"), sort_keys=True).encode("ascii")
    return raw + (b"\n" if newline else b"")


def _write_json(path: Path, value: Any) -> bytes:
    raw = _raw_json(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return raw


def _config(arm: str, *, candidate_cache: int = 377) -> dict[str, Any]:
    baseline = arm == "baseline"
    backend = "sample_isolated_grouped_mm_v1" if baseline else "sample_isolated_grouped_mm_v2"
    profile = (
        "duovla-single-gpu-tp1-sequential-v1-train-b64-serve-b8-v1"
        if baseline
        else "duovla-single-gpu-tp1-fused-v2-train-b64-serve-b8-v1"
    )
    gpu_index = 0 if baseline else 1
    gpu_uuid = BASELINE_UUID if baseline else CANDIDATE_UUID
    geometry: dict[str, Any] = {
        "execution_profile": profile,
        "expert_batch_isolation": backend,
        "experts_implementation": "grouped_mm",
        "fixed_physical_prefix_width": 545,
        "physical_batch_size": 64,
        "prefix_geometry_content_sha256": "c" * 64,
        "serving_batch_size": 8,
        "tensor_parallel_size": 1,
    }
    if not baseline:
        geometry["shared_weight_kernel_sha256"] = KERNEL_SHA256
    return {
        "benchmark": {"dataset_revision": "revision"},
        "execution_environment": {
            "authenticated_runtime": {
                "environment": {"CUDA_VISIBLE_DEVICES": str(gpu_index), "UNCHANGED": "yes"},
                "static_environment_sha256": f"static-{arm}",
            },
            "cuda_runtime": "12.9",
            "gpu_uuids": [gpu_uuid],
        },
        "execution_geometry": geometry,
        "execution_profile": profile,
        "model": {
            "expert_batch_isolation": backend,
            "experts_implementation": "grouped_mm",
            "id": "model",
        },
        "optimization": {
            "global_batch_size": 64,
            "gradient_accumulation_steps": 1,
            "microbatch_size": 64,
            "physical_batch_size": 64,
            "serving_batch_size": 8,
            "total_updates": 30_000,
        },
        "policy": {"objective": "rectified_flow", "nfe": 10},
        "reproducibility": {"seed": 0},
        "run": {
            "max_cached_files": 128 if baseline else candidate_cache,
            "seed": 0,
            "task": None,
        },
        "source_tree_sha256": SOURCE_SHA256,
    }


def _metrics(arm: str) -> list[dict[str, Any]]:
    candidate = arm == "candidate"
    records = []
    for update in range(1, 101):
        baseline_seconds = 10.0 + update / 100.0
        baseline_loss = 1.0 + update / 1000.0
        baseline_gradient = 10.0 + update / 100.0
        records.append(
            {
                "examples_seen": update * 64,
                "gradient_norm": baseline_gradient * (1.02 if candidate else 1.0),
                "interface_learning_rate": update / 1_001_000.0,
                "lora_learning_rate": update / 10_010_000.0,
                "objective": "rectified_flow",
                "train_loss": baseline_loss * (1.01 if candidate else 1.0),
                "update": update,
                "update_seconds": baseline_seconds * (0.7 if candidate else 1.0),
            }
        )
    return records


def _common_manifest_fields() -> dict[str, Any]:
    return {
        "dataset_content_inventory_sha256": "d" * 64,
        "dataset_files_verified": 382,
        "dataset_id": "HuggingFaceVLA/libero",
        "dataset_revision": "revision",
        "dataset_total_bytes": 123,
        "dataset_tree_sha256": "e" * 64,
        "experts_implementation": "grouped_mm",
        "fixed_physical_prefix_width": 545,
        "model_content_inventory_sha256": "f" * 64,
        "model_files_verified": 21,
        "model_id": "model",
        "model_revision": "model-revision",
        "model_total_bytes": 456,
        "model_tree_sha256": "1" * 64,
        "normalization_sha256": "2" * 64,
        "optimizer_parameter_schema_sha256": "3" * 64,
        "policy_contract": {"objective": "rectified_flow"},
        "policy_contract_sha256": "4" * 64,
        "prefix_geometry_content_sha256": "c" * 64,
        "task": None,
        "tensor_parallel_size": 1,
        "train_episode_count": 90,
        "validation_episode_count": 10,
    }


def _criteria(baseline: Path, candidate: Path) -> dict[str, Any]:
    return {
        "acceptance": {
            "candidate_peak_memory_gib_max": 90.0,
            "finite_loss_and_gradient_every_update": True,
            "gradient_norm_ratio_interval": [0.75, 4.0 / 3.0],
            "gradient_norm_ratio_min_fraction": 0.95,
            "mean_speedup_updates_51_100_min": 0.2,
            "median_relative_loss_difference_max": 0.05,
            "p95_candidate_update_seconds_below_baseline": True,
            "tail_mean_relative_loss_difference_max": 0.05,
            "tail_updates": [91, 100],
        },
        "arms": {
            "baseline": {
                "backend": "sample_isolated_grouped_mm_v1",
                "config": "configs/libero_single_gpu_ab_v1_b64.toml",
                "gpu_index": 0,
                "gpu_uuid": BASELINE_UUID,
                "max_cached_files": 128,
                "output_dir": str(baseline),
            },
            "candidate": {
                "backend": "sample_isolated_grouped_mm_v2",
                "config": "configs/libero_single_gpu_fused_v2_b64.toml",
                "gpu_index": 1,
                "gpu_uuid": CANDIDATE_UUID,
                "max_cached_files": 377,
                "output_dir": str(candidate),
            },
        },
        "execution": {
            "canonical_stream_batch_size": 8,
            "canonical_stream_plans_per_update": 8,
            "global_batch_size": 64,
            "physical_batch_size": 64,
            "seed": 0,
            "serving_batch_size": 8,
            "stop_after_updates": 100,
            "total_updates": 30_000,
        },
        "qualification": {},
        "schema": "duovla-libero-cache-fused-ab-source-freeze-v1",
        "shared_weight_kernel_sha256": KERNEL_SHA256,
        "source_tree_sha256": SOURCE_SHA256,
        "status": "frozen",
    }


def _make_arm(root: Path, arm: str, *, candidate_cache: int = 377) -> tuple[Path, list[dict[str, Any]]]:
    run = root / arm
    run.mkdir()
    config = _config(arm, candidate_cache=candidate_cache)
    config_sha256 = hashlib.sha256(_raw_json(config, newline=False)).hexdigest()
    resolved = {"config": config, "config_sha256": config_sha256}
    resolved_raw = _write_json(run / "resolved_config.json", resolved)
    metrics = _metrics(arm)
    (run / "metrics.jsonl").write_bytes(b"".join(_raw_json(record) for record in metrics))

    checkpoint = run / "checkpoints" / "update-000100"
    artifact_dir = checkpoint / "artifacts"
    artifact_dir.mkdir(parents=True)
    (artifact_dir / "resolved_config.json").write_bytes(resolved_raw)
    (artifact_dir / "payload.bin").write_bytes(f"payload-{arm}".encode())
    artifacts = {}
    for name, relative in (
        ("resolved_config", "artifacts/resolved_config.json"),
        ("payload", "artifacts/payload.bin"),
    ):
        raw = (checkpoint / relative).read_bytes()
        artifacts[name] = {"bytes": len(raw), "path": relative, "sha256": hashlib.sha256(raw).hexdigest()}

    backend = "sample_isolated_grouped_mm_v1" if arm == "baseline" else "sample_isolated_grouped_mm_v2"
    manifest = {
        **_common_manifest_fields(),
        "artifacts": artifacts,
        "config_sha256": config_sha256,
        "execution_environment_sha256": f"environment-{arm}",
        "execution_profile": config["execution_profile"],
        "expert_batch_isolation": backend,
        "last_metrics": metrics[-1],
        "physical_batch_size": 64,
        "run_seed": 0,
        "run_uuid": f"run-{arm}",
        "schema": "duo-vla-checkpoint-v1",
        "serving_batch_size": 8,
        "source_tree_sha256": SOURCE_SHA256,
        "trainer_state": {
            "examples_seen": 6400,
            "next_update": 100,
            "schema": "duo-vla-trainer-state-v1",
        },
    }
    if arm == "candidate":
        manifest["shared_weight_kernel_sha256"] = KERNEL_SHA256
    manifest_raw = _write_json(checkpoint / "manifest.json", manifest)
    _write_json(
        run / "run_journal.json",
        {
            "config_sha256": config_sha256,
            "latest_checkpoint": {
                "last_metrics": metrics[-1],
                "manifest_sha256": hashlib.sha256(manifest_raw).hexdigest(),
                "parent_manifest_sha256": None,
                "relative_path": "checkpoints/update-000100",
                "update": 100,
            },
            "run_uuid": f"run-{arm}",
            "schema": "duo-vla-run-journal-v1",
        },
    )
    return run, metrics


def _make_fixture(tmp_path: Path, *, candidate_cache: int = 377) -> tuple[Path, Path, Path]:
    baseline, _ = _make_arm(tmp_path, "baseline", candidate_cache=candidate_cache)
    candidate, _ = _make_arm(tmp_path, "candidate", candidate_cache=candidate_cache)
    criteria = tmp_path / "criteria.json"
    _write_json(criteria, _criteria(baseline, candidate))
    return baseline, candidate, criteria


def _analyze(baseline: Path, candidate: Path, criteria: Path, **kwargs: Any) -> dict[str, Any]:
    return ANALYZER.analyze(
        baseline,
        candidate,
        criteria,
        baseline_peak_memory_gib=70.0,
        candidate_peak_memory_gib=80.0,
        bootstrap_replicates=200,
        **kwargs,
    )


def test_analyze_authenticates_aligned_runs_and_passes_frozen_criteria(tmp_path: Path) -> None:
    baseline, candidate, criteria = _make_fixture(tmp_path)

    report = _analyze(baseline, candidate, criteria)
    repeated = _analyze(baseline, candidate, criteria)

    assert report == repeated
    assert report["status"] == "pass"
    assert report["acceptance"]["all_passed"]
    assert all(report["acceptance"]["checks"].values())
    assert report["validation"]["aligned_updates"] == 100
    assert report["timing"]["updates_1_100"]["speedup_fraction"]["mean"] == pytest.approx(0.3)
    assert report["timing"]["updates_51_100"]["speedup_fraction"]["median"] == pytest.approx(0.3)
    assert report["loss_relative_difference"]["all_updates"]["median"] == pytest.approx(0.01)
    assert report["gradient_norm_ratio"]["pass_fraction"] == 1.0
    assert report["manifests"]["baseline"]["artifact_count"] == 2
    unsigned = dict(report)
    digest = unsigned.pop("report_sha256")
    assert digest == hashlib.sha256(ANALYZER.canonical_json_bytes(unsigned)).hexdigest()


def test_missing_peak_memory_rejects_instead_of_silently_passing(tmp_path: Path) -> None:
    baseline, candidate, criteria = _make_fixture(tmp_path)

    report = ANALYZER.analyze(
        baseline,
        candidate,
        criteria,
        bootstrap_replicates=10,
    )

    assert report["status"] == "rejected"
    assert not report["acceptance"]["checks"]["candidate_peak_memory_gib_max"]
    assert report["inputs"]["candidate"]["peak_memory"]["available"] is False


def test_discovers_authenticated_completion_peak_files(tmp_path: Path) -> None:
    baseline, candidate, criteria = _make_fixture(tmp_path)
    for run, peak in ((baseline, 71.0), (candidate, 81.0)):
        _write_json(
            run / "completion.json",
            {
                "final_update": 100,
                "output_dir": str(run),
                "peak_memory_gib": peak,
            },
        )

    report = ANALYZER.analyze(
        baseline,
        candidate,
        criteria,
        bootstrap_replicates=10,
    )

    assert report["status"] == "pass"
    peak = report["inputs"]["candidate"]["peak_memory"]
    assert peak["peak_memory_gib"] == 81.0
    assert peak["source"] == "run_completion_file"


def test_rejects_noncontiguous_or_unaligned_metrics(tmp_path: Path) -> None:
    baseline, candidate, criteria = _make_fixture(tmp_path)
    path = candidate / "metrics.jsonl"
    records = [_decode for _decode in map(json.loads, path.read_text().splitlines())]
    records[49]["update"] = 49
    path.write_bytes(b"".join(_raw_json(record) for record in records))

    with pytest.raises(ANALYZER.AnalysisError, match="not contiguous"):
        _analyze(baseline, candidate, criteria)


def test_rejects_nonfinite_metrics_before_statistics(tmp_path: Path) -> None:
    baseline, candidate, criteria = _make_fixture(tmp_path)
    path = candidate / "metrics.jsonl"
    lines = path.read_text().splitlines()
    record = json.loads(lines[10])
    record["train_loss"] = math.nan
    lines[10] = json.dumps(record, allow_nan=True, sort_keys=True)
    path.write_text("\n".join(lines) + "\n")

    with pytest.raises(ANALYZER.AnalysisError, match="non-finite"):
        _analyze(baseline, candidate, criteria)


def test_rejects_candidate_cache_other_than_frozen_377(tmp_path: Path) -> None:
    baseline, candidate, criteria = _make_fixture(tmp_path, candidate_cache=376)

    with pytest.raises(ANALYZER.AnalysisError, match="candidate cache mismatch"):
        _analyze(baseline, candidate, criteria)


def test_rejects_tampered_checkpoint_artifact(tmp_path: Path) -> None:
    baseline, candidate, criteria = _make_fixture(tmp_path)
    (candidate / "checkpoints/update-000100/artifacts/payload.bin").write_bytes(b"tampered")

    with pytest.raises(ANALYZER.AnalysisError, match=r"payload.*(byte count|SHA-256) mismatch"):
        _analyze(baseline, candidate, criteria)


def test_atomic_writer_is_canonical_and_never_replaces(tmp_path: Path) -> None:
    output = tmp_path / "report.json"
    value = {"status": "pass", "value": 1}

    assert ANALYZER.write_canonical_json_atomic_exclusive(output, value) == output
    assert output.read_bytes() == ANALYZER.canonical_json_bytes(value)
    with pytest.raises(ANALYZER.AnalysisError, match="refusing to replace"):
        ANALYZER.write_canonical_json_atomic_exclusive(output, {"status": "different"})
    assert output.read_bytes() == ANALYZER.canonical_json_bytes(value)


def test_block_bootstrap_is_deterministic_and_paired() -> None:
    baseline = [float(value) for value in range(1, 11)]
    candidate = [0.8 * value for value in baseline]

    first = ANALYZER.block_bootstrap_speedup_ci(
        baseline,
        candidate,
        block_size=5,
        replicates=100,
        seed=7,
    )
    second = ANALYZER.block_bootstrap_speedup_ci(
        baseline,
        candidate,
        block_size=5,
        replicates=100,
        seed=7,
    )

    assert first == second
    assert first["lower"] == pytest.approx(0.2)
    assert first["upper"] == pytest.approx(0.2)
