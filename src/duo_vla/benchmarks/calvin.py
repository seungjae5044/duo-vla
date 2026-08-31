"""Canonical CALVIN ABC-to-D training and long-horizon evaluation semantics."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor

from duo_vla.benchmarks.common import CanonicalObservation, PaddedActionChunk


@dataclass(frozen=True, slots=True)
class CalvinProtocol:
    name: str = "duovla-calvin-abc-to-d-v1"
    dataset: str = "task_ABC_D"
    training_environments: tuple[str, ...] = ("A", "B", "C")
    evaluation_environment: str = "D"
    action_horizon: int = 8
    action_dim: int = 7
    state_dim: int = 8
    control_frequency_hz: int = 30
    num_sequences: int = 1000
    subtasks_per_sequence: int = 5
    max_steps_per_subtask: int = 360
    evaluation_seed: int = 0


def calvin_state(robot_obs: Tensor) -> Tensor:
    """Select the official ``robot_no_joints`` 8D proprioceptive state."""

    if robot_obs.shape[-1:] != (15,) or not robot_obs.is_floating_point():
        raise ValueError("CALVIN robot_obs must be floating with final dimension 15")
    if not bool(torch.isfinite(robot_obs).all()):
        raise ValueError("CALVIN robot_obs contains non-finite values")
    state = torch.cat((robot_obs[..., :7], robot_obs[..., 14:15]), dim=-1)
    gripper_state = state[..., -1]
    if not bool(((gripper_state == -1) | (gripper_state == 1)).all()):
        raise ValueError("CALVIN robot_obs[14] gripper state must be exactly {-1, +1}")
    return state


def calvin_training_observation(sample: Mapping[str, object]) -> CanonicalObservation:
    return CanonicalObservation(
        third_person=_required_rgb(sample, "rgb_static"),
        wrist=_required_rgb(sample, "rgb_gripper"),
        state=calvin_state(torch.as_tensor(sample["robot_obs"])),
    )


def calvin_rollout_observation(observation: Mapping[str, object]) -> CanonicalObservation:
    try:
        rgb = observation["rgb_obs"]
        if not isinstance(rgb, Mapping):
            raise TypeError
        static = rgb["rgb_static"]
        gripper = rgb["rgb_gripper"]
        robot_obs = observation["robot_obs"]
    except (KeyError, TypeError) as exc:
        raise ValueError("rollout observation does not match the CALVIN nested schema") from exc
    return CanonicalObservation(
        third_person=_as_rgb(static, "rgb_static"),
        wrist=_as_rgb(gripper, "rgb_gripper"),
        state=calvin_state(torch.as_tensor(robot_obs)),
    )


def validate_calvin_rel_actions(actions: Tensor) -> Tensor:
    if actions.ndim < 2 or actions.shape[-1] != 7 or not actions.is_floating_point():
        raise ValueError("CALVIN rel_actions must be floating with final dimension 7")
    if not bool(torch.isfinite(actions).all()):
        raise ValueError("CALVIN rel_actions contain non-finite values")
    if bool((actions[..., :6].abs() > 1.0 + 1e-6).any()):
        raise ValueError("CALVIN scaled relative pose inputs must be in [-1, 1]")
    gripper = actions[..., 6]
    if not bool(((gripper == -1) | (gripper == 1)).all()):
        raise ValueError("CALVIN dataset gripper must be exactly {-1, +1}")
    return actions


def make_calvin_action_chunk(
    actions: Tensor,
    anchor: int,
    *,
    annotation_end_exclusive: int,
    episode_end_inclusive: int,
    horizon: int = 8,
) -> PaddedActionChunk:
    """Build a chunk within a half-open annotation and an inclusive episode."""

    validate_calvin_rel_actions(actions)
    if actions.ndim != 2:
        raise ValueError("trajectory actions must have shape [time, 7]")
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    if annotation_end_exclusive <= 0:
        raise ValueError("annotation_end_exclusive must be positive")
    last_valid = min(annotation_end_exclusive - 1, episode_end_inclusive, actions.shape[0] - 1)
    if not 0 <= anchor <= last_valid:
        raise IndexError("anchor lies outside the annotation/episode intersection")
    valid_length = min(horizon, last_valid - anchor + 1)
    chunk = actions.new_zeros(horizon, 7)
    chunk[:valid_length] = actions[anchor : anchor + valid_length]
    # The padded gripper value is irrelevant to attention/loss but preserving the last command avoids an artificial 0.
    chunk[valid_length:, 6] = chunk[valid_length - 1, 6]
    mask = torch.arange(horizon, device=actions.device) < valid_length
    return PaddedActionChunk(actions=chunk, valid_mask=mask)


def calvin_env_action(action: Tensor | np.ndarray) -> np.ndarray:
    """Return a fresh CALVIN rel_action copy using -close/+open polarity."""

    values = torch.as_tensor(action).detach().float().reshape(-1)
    if values.shape != (7,) or not bool(torch.isfinite(values).all()):
        raise ValueError("action must contain seven finite values")
    output = values.clone()
    output[:6].clamp_(-1.0, 1.0)
    # Match the pinned official CALVIN wrappers exactly: zero is close.
    output[6] = 1.0 if output[6] > 0 else -1.0
    return output.cpu().numpy().astype(np.float32, copy=True)


@dataclass(frozen=True, slots=True)
class CalvinLongHorizonMetrics:
    success_rates: tuple[float, float, float, float, float]
    average_length: float


def calvin_long_horizon_metrics(successful_subtasks: Sequence[int]) -> CalvinLongHorizonMetrics:
    counts = np.asarray(successful_subtasks, dtype=np.int64)
    if counts.ndim != 1 or counts.size == 0 or bool(((counts < 0) | (counts > 5)).any()):
        raise ValueError("successful_subtasks must be a non-empty 1D sequence in [0, 5]")
    rates = tuple(float(np.mean(counts >= k)) for k in range(1, 6))
    average_length = float(np.mean(counts))
    if not np.isclose(sum(rates), average_length):
        raise AssertionError("CALVIN metric identity AvgLen = sum(SR1..SR5) was violated")
    return CalvinLongHorizonMetrics(success_rates=rates, average_length=average_length)  # type: ignore[arg-type]


def calvin_replan_seed(global_seed: int, sequence_id: int, subtask_id: int, replan_id: int) -> int:
    digest = hashlib.blake2b(
        f"calvin\x1f{global_seed}\x1f{sequence_id}\x1f{subtask_id}\x1f{replan_id}".encode(),
        digest_size=8,
    ).digest()
    return int.from_bytes(digest, "little") & ((1 << 63) - 1)


def _required_rgb(sample: Mapping[str, object], key: str) -> np.ndarray:
    try:
        value = sample[key]
    except KeyError as exc:
        raise ValueError(f"training sample is missing {key}") from exc
    return _as_rgb(value, key)


def _as_rgb(value: object, name: str) -> np.ndarray:
    if not isinstance(value, np.ndarray) or value.ndim != 3 or value.shape[-1] != 3 or value.dtype != np.uint8:
        raise ValueError(f"{name} must be an HWC uint8 numpy image")
    return value
