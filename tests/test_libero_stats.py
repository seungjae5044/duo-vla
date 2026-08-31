from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest

from duo_vla.data.libero_stats import (
    LIBERO_DATASET_REVISION,
    LIBERO_STATS_SCHEMA,
    load_libero_normalizers,
    percentile_bounds,
    save_libero_normalization_artifact,
)


def test_percentile_bounds_are_exact_and_exclude_gripper() -> None:
    states = np.arange(800, dtype=np.float32).reshape(100, 8)
    continuous = np.linspace(-1.0, 1.0, 600, dtype=np.float32).reshape(100, 6)
    gripper = np.where(np.arange(100) % 2, 1.0, -1.0).astype(np.float32)[:, None]
    actions = np.concatenate((continuous, gripper), axis=1)

    state_q01, state_q99, action_q01, action_q99 = percentile_bounds(states, actions)

    expected_state_q01, expected_state_q99 = np.quantile(states, (0.01, 0.99), axis=0, method="linear")
    expected_action_q01, expected_action_q99 = np.quantile(continuous, (0.01, 0.99), axis=0, method="linear")
    np.testing.assert_array_equal(state_q01, expected_state_q01)
    np.testing.assert_array_equal(state_q99, expected_state_q99)
    np.testing.assert_array_equal(action_q01, expected_action_q01)
    np.testing.assert_array_equal(action_q99, expected_action_q99)


@pytest.mark.parametrize(
    ("states", "actions", "message"),
    [
        (np.zeros((2, 7), np.float32), np.zeros((2, 7), np.float32), "states"),
        (np.zeros((2, 8), np.float32), np.zeros((2, 6), np.float32), "actions"),
        (np.full((2, 8), np.nan, np.float32), np.zeros((2, 7), np.float32), "non-finite"),
    ],
)
def test_percentile_bounds_reject_invalid_inputs(states: np.ndarray, actions: np.ndarray, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        percentile_bounds(states, actions)


def test_artifact_save_load_and_hash_verification(tmp_path: Path) -> None:
    payload = {
        "schema": LIBERO_STATS_SCHEMA,
        "dataset": {"id": "HuggingFaceVLA/libero", "revision": LIBERO_DATASET_REVISION},
        "state": {"q01": [0.0] * 8, "q99": [1.0] * 8},
        "action": {"q01": [0.0] * 6, "q99": [1.0] * 6},
    }
    from duo_vla.data.libero_stats import _content_hash

    payload["content_sha256"] = _content_hash(payload)
    path = tmp_path / "stats.json"
    save_libero_normalization_artifact(path, payload)
    state, action, loaded = load_libero_normalizers(path)
    assert state.dim == 8 and action.continuous.dim == 6
    assert loaded == payload

    corrupted = copy.deepcopy(payload)
    corrupted["state"]["q01"][0] = 0.5
    path.write_text(__import__("json").dumps(corrupted), encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_libero_normalizers(path)
