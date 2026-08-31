from __future__ import annotations

import numpy as np
import pytest
import torch

from duo_vla.benchmarks.calvin import (
    calvin_env_action,
    calvin_long_horizon_metrics,
    calvin_replan_seed,
    calvin_state,
    make_calvin_action_chunk,
)
from duo_vla.benchmarks.libero import (
    libero_env_action,
    make_libero_action_chunk,
    rotate_eval_rgb_180,
    validate_libero_dataset_actions,
)


def _actions(length: int, *, gripper: float = 1.0) -> torch.Tensor:
    values = torch.zeros(length, 7)
    values[:, :6] = torch.linspace(-0.5, 0.5, length)[:, None]
    values[:, 6] = gripper
    return values


def test_libero_camera_rotation_is_exactly_180_degrees_and_owned() -> None:
    image = np.arange(2 * 3 * 3, dtype=np.uint8).reshape(2, 3, 3)
    rotated = rotate_eval_rgb_180(image)
    np.testing.assert_array_equal(rotated, image[::-1, ::-1])
    assert rotated.flags.c_contiguous and not np.shares_memory(rotated, image)


def test_libero_chunk_stops_at_episode_and_zero_pads() -> None:
    chunk = make_libero_action_chunk(_actions(5), 3, horizon=4)
    assert chunk.valid_mask.tolist() == [True, True, False, False]
    torch.testing.assert_close(chunk.actions[2:], torch.zeros(2, 7))


def test_libero_rejects_zero_one_gripper_conversion() -> None:
    actions = _actions(2)
    actions[0, 6] = 0
    with pytest.raises(ValueError, match="exactly"):
        validate_libero_dataset_actions(actions)


def test_benchmarks_have_opposite_gripper_meanings_at_adapter_boundary() -> None:
    generated = torch.tensor([2.0, -2.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    libero = libero_env_action(generated)
    calvin = calvin_env_action(generated)
    assert libero.dtype == np.float32 and libero[0] == 1 and libero[1] == -1 and libero[6] == 1
    assert calvin.dtype == np.float32 and calvin[0] == 1 and calvin[1] == -1 and calvin[6] == -1
    # CALVIN's official strict-positive threshold maps zero to close; LIBERO maps it to its +1 side.


def test_calvin_state_selects_robot_no_joints_dimensions() -> None:
    robot_obs = torch.arange(15, dtype=torch.float32)
    robot_obs[14] = -1
    state = calvin_state(robot_obs)
    torch.testing.assert_close(state, torch.tensor([0, 1, 2, 3, 4, 5, 6, -1], dtype=torch.float32))


def test_calvin_chunk_respects_both_boundaries_and_repeats_padded_gripper() -> None:
    actions = _actions(10, gripper=-1)
    chunk = make_calvin_action_chunk(
        actions,
        4,
        annotation_end_exclusive=7,
        episode_end_inclusive=8,
        horizon=5,
    )
    assert chunk.valid_mask.tolist() == [True, True, True, False, False]
    torch.testing.assert_close(chunk.actions[3:, :6], torch.zeros(2, 6))
    torch.testing.assert_close(chunk.actions[3:, 6], -torch.ones(2))


def test_calvin_metrics_match_official_identity() -> None:
    metrics = calvin_long_horizon_metrics([0, 1, 2, 3, 4, 5])
    assert metrics.success_rates == pytest.approx((5 / 6, 4 / 6, 3 / 6, 2 / 6, 1 / 6))
    assert metrics.average_length == pytest.approx(2.5)


def test_calvin_replan_seed_is_index_sensitive() -> None:
    assert calvin_replan_seed(0, 1, 2, 3) != calvin_replan_seed(0, 1, 2, 4)


def test_calvin_env_action_returns_copy_safe_from_in_place_simulator_scaling() -> None:
    source = np.zeros(7, dtype=np.float32)
    action = calvin_env_action(source)
    action[:6] *= 0.02
    np.testing.assert_array_equal(source, np.zeros(7, dtype=np.float32))
