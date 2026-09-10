from __future__ import annotations

import numpy as np
import pytest

from duo_vla.expert_routing import DecoderRoutingTrace, RoutingAccumulator, routing_metrics


def test_balanced_and_collapsed_routing_have_distinct_metrics() -> None:
    balanced = routing_metrics(np.ones(8), np.ones(8), np.ones(8), 8)
    collapsed = routing_metrics(np.array([8, 0, 0, 0]), np.array([8, 0, 0, 0]), np.array([8, 0, 0, 0]), 8)
    assert balanced["normalized_entropy"] == pytest.approx(1)
    assert balanced["effective_experts"] == pytest.approx(8)
    assert collapsed["normalized_entropy"] == 0
    assert collapsed["max_top1_fraction"] == 1


def test_padding_and_inactive_rows_do_not_affect_routing_counts() -> None:
    accumulator = RoutingAccumulator(4)
    routes = np.array([[[0, 1], [3, 3]], [[2, 3], [3, 3]], [[3, 3], [3, 3]]])
    weights = np.array([[[0.2, 0.8], [1, 1]], [[0.6, 0.4], [1, 1]], [[1, 1], [1, 1]]])
    accumulator.observe("encoder.0", "prefix", routes, weights, [[1, 0], [1, 0], [0, 0]], [0, 1, 0])
    report = accumulator.report()
    layer = report["layers"][0]
    assert layer["counts"] == [1, 1, 1, 1]
    assert layer["top1_counts"] == [0, 1, 1, 0]
    assert layer["tokens"] == 2
    assert layer["max_assignment_share"] == 0.25
    assert layer["max_token_selection_fraction"] == 0.5
    assert len(report["per_task"]) == 2


def test_out_of_range_routes_fail_without_changing_statistics() -> None:
    accumulator = RoutingAccumulator(4)
    with pytest.raises(ValueError, match="outside"):
        accumulator.observe("decoder.0", "denoise_0", np.array([[[4]]]), np.ones((1, 1, 1)), [[1]], [0])
    assert accumulator.report()["layers"] == []


def test_trace_preserves_episode_time_and_matches_aggregate(tmp_path) -> None:
    trace = DecoderRoutingTrace(tmp_path / "trace", nfe=2, layers=1, experts=4)
    rows = [
        {"task_id": 0, "reset_id": 3, "policy_step": 2, "environment_step": 8, "inference_seed": 1},
        {"task_id": 0, "reset_id": 4, "policy_step": 1, "environment_step": 4, "inference_seed": 2},
    ]
    routes = np.array([[[0, 1], [2, 3]], [[2, 3], [1, 3]], [[3, 3], [3, 3]]])
    weights = np.array([[[0.2, 0.8], [0.6, 0.4]], [[0.6, 0.4], [1, 1]], [[1, 1], [1, 1]]])
    mask = np.array([[1, 1], [1, 0], [0, 0]], dtype=bool)
    trace.begin(rows)
    accumulator = RoutingAccumulator(4)
    for phase in range(2):
        trace.observe(0, phase, routes, weights, mask)
        accumulator.observe("model.decoder.layers.0", f"denoise_{phase}", routes, weights, mask, [0, 0, 0])
    with np.load(trace.finish(), allow_pickle=False) as data:
        assert data["counts"].shape == (2, 2, 1, 4)
        np.testing.assert_array_equal(data["reset_id"], [3, 4])
        np.testing.assert_array_equal(data["policy_step"], [2, 1])
        np.testing.assert_array_equal(data["environment_step"], [8, 4])
        for phase, aggregate in enumerate(accumulator.report()["layers"]):
            for key in ("counts", "top1_counts", "gate_weight_sums"):
                np.testing.assert_allclose(data[key][:, phase, 0].sum(axis=0), aggregate[key])
            assert data["tokens"][:, phase, 0].sum() == aggregate["tokens"]


def test_trace_rejects_incomplete_duplicate_and_inactive_rows(tmp_path) -> None:
    trace = DecoderRoutingTrace(tmp_path / "trace", nfe=2, layers=1, experts=4)
    trace.begin([{"task_id": 0, "reset_id": 0, "policy_step": 0, "environment_step": 0, "inference_seed": 0}])
    indices, weights = np.zeros((2, 1, 1), dtype=int), np.ones((2, 1, 1))
    with pytest.raises(ValueError, match="inactive"):
        trace.observe(0, 0, indices, weights, np.ones((2, 1), dtype=bool))
    mask = np.array([[1], [0]], dtype=bool)
    trace.observe(0, 0, indices, weights, mask)
    with pytest.raises(ValueError, match="duplicate"):
        trace.observe(0, 0, indices, weights, mask)
    with pytest.raises(ValueError, match="incomplete"):
        trace.finish()
    assert list((tmp_path / "trace").iterdir()) == []
    trace.observe(0, 1, indices, weights, mask)
    assert trace.finish().is_file()
