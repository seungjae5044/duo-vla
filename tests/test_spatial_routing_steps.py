from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

from duo_vla.expert_routing import DecoderRoutingTrace, RoutingAccumulator

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from libero_bridge import libero_replan_seed
from plot_spatial_routing_steps import load_traces


@pytest.fixture
def traced_run(tmp_path: Path) -> Path:
    """Produce known routes with variable episode lengths and partial batches."""
    policy, evaluation = tmp_path / "k8-policy", tmp_path / "k8-evaluation"
    policy.mkdir()
    evaluation.mkdir()
    recorder = DecoderRoutingTrace(policy / "routing_trace", 2)
    accumulator = RoutingAccumulator()
    episodes = [
        {
            "task_id": task,
            "reset_id": reset,
            "calls": 1 + reset % 3,
            "steps": (1 + reset % 3) * 8,
            "success": True,
            "execution_horizon": 8,
            "nfe": 2,
        }
        for task in range(10)
        for reset in range(10)
    ]
    for step in range(3):
        active = [row for row in episodes if row["calls"] > step]
        for start in range(0, len(active), 8):
            batch = active[start : start + 8]
            identities = [
                {
                    "task_id": row["task_id"],
                    "reset_id": row["reset_id"],
                    "policy_step": step,
                    "environment_step": step * 8,
                    "inference_seed": libero_replan_seed(
                        0, "libero_spatial", row["task_id"], "official", row["reset_id"], None, step
                    ),
                }
                for row in batch
            ]
            recorder.begin(identities)
            mask = np.zeros((8, 8), dtype=bool)
            mask[: len(batch)] = True
            task_ids = [row["task_id"] for row in batch] + [0] * (8 - len(batch))
            weights = np.broadcast_to(np.arange(1, 9) / 36.0, (8, 8, 8)).copy()
            for phase in range(2):
                for layer in range(30):
                    routes = np.broadcast_to(np.arange(8), (8, 8, 8)).copy()
                    for index, row in enumerate(batch):
                        routes[index] = (routes[index] + row["task_id"] + row["reset_id"] + step + layer + phase) % 128
                    recorder.observe(layer, phase, routes, weights, mask)
                    accumulator.observe(
                        f"model.decoder.layers.{layer}", f"denoise_{phase}", routes, weights, mask, task_ids
                    )
            recorder.finish()
    (evaluation / "episodes.jsonl").write_text("".join(json.dumps(row) + "\n" for row in episodes))
    records = {
        evaluation / "summary.json": {
            "status": "complete",
            "completed": 100,
            "target": 100,
            "successes": 100,
            "nfe": 2,
            "execution_horizon": 8,
            "batch_calls": recorder.batch_index,
        },
        evaluation / "experiment.json": {
            "evaluation_seed": 0,
            "environment_seed": 7,
            "max_policy_steps": 220,
            "settle_steps": 10,
        },
        evaluation / "inference_validation.json": {
            "repeat_max_abs_error": 0,
            "permutation_max_abs_error": 0,
            "warmup_included_in_routing": False,
        },
        policy / "serving.json": {"decoder_trace": {"enabled": True}, "cuda_visible_devices": "6,7"},
        policy / "expert_routing.json": accumulator.report(),
    }
    for path, data in records.items():
        path.write_text(json.dumps(data))
    return tmp_path


def test_complete_trace_recovers_varying_policy_step_support(traced_run: Path) -> None:
    result = load_traces(traced_run, 8)
    np.testing.assert_array_equal(result["support"], [100, 60, 30])
    assert result["by_step"]["counts"].shape == (2, 30, 3, 128)
    np.testing.assert_array_equal(
        result["by_step"]["counts"].sum(axis=(0, 1, 3)), np.array([100, 60, 30]) * 2 * 30 * 64
    )


@pytest.mark.parametrize("corruption", ["seed", "duplicate", "aggregate", "missing"])
def test_invalid_trace_cannot_produce_final_figures(traced_run: Path, corruption: str) -> None:
    policy = traced_run / "k8-policy"
    first = policy / "routing_trace/batch_000000.npz"
    if corruption in ("seed", "duplicate"):
        with np.load(first, allow_pickle=False) as source:
            data = dict(source)
        if corruption == "seed":
            data["inference_seed"][0] += 1
        else:
            for key in ("task_id", "reset_id", "policy_step", "environment_step", "inference_seed"):
                data[key][1] = data[key][0]
        np.savez_compressed(first, **data)
    elif corruption == "missing":
        first.unlink()
    else:
        path = policy / "expert_routing.json"
        report = json.loads(path.read_text())
        report["per_task"][0]["counts"][0] += 1
        path.write_text(json.dumps(report))
    with pytest.raises((ValueError, AssertionError)):
        load_traces(traced_run, 8)
