"""Canonical LIBERO training and rollout boundary semantics."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
from torch import Tensor

from duo_vla.benchmarks.common import PaddedActionChunk


@dataclass(frozen=True, slots=True)
class LiberoProtocol:
    name: str = "duovla-libero-v1"
    dataset_id: str = "HuggingFaceVLA/libero"
    dataset_revision: str = "86958911c0f959db2bbbdb107eb3e17c5f9c798e"
    suites: tuple[str, ...] = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
    tasks_per_suite: int = 10
    action_horizon: int = 8
    action_dim: int = 7
    state_dim: int = 8
    control_frequency_hz: int = 20
    settle_steps: int = 10
    environment_seed: int = 7
    max_policy_steps: dict[str, int] = field(
        default_factory=lambda: {
            "libero_spatial": 220,
            "libero_object": 280,
            "libero_goal": 300,
            "libero_10": 520,
        }
    )


def rotate_eval_rgb_180(image: np.ndarray) -> np.ndarray:
    """Apply the one required simulator-to-regenerated-dataset camera transform."""

    if not isinstance(image, np.ndarray) or image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError("image must be an HWC numpy RGB array")
    if image.dtype != np.uint8:
        raise TypeError("image must use uint8")
    return np.flip(image, axis=(0, 1)).copy()


def validate_libero_state(state: Tensor) -> Tensor:
    if state.shape[-1:] != (8,) or not state.is_floating_point():
        raise ValueError("LIBERO state must be floating with final dimension 8")
    if not bool(torch.isfinite(state).all()):
        raise ValueError("LIBERO state contains non-finite values")
    return state


def validate_libero_dataset_actions(actions: Tensor) -> Tensor:
    """Reject silently converted gripper labels and out-of-range controller inputs."""

    if actions.ndim < 2 or actions.shape[-1] != 7 or not actions.is_floating_point():
        raise ValueError("LIBERO actions must be floating with final dimension 7")
    if not bool(torch.isfinite(actions).all()):
        raise ValueError("LIBERO actions contain non-finite values")
    if bool((actions[..., :6].abs() > 1.0 + 1e-6).any()):
        raise ValueError("LIBERO OSC_POSE controller inputs must be in [-1, 1]")
    gripper = actions[..., 6]
    if not bool(((gripper == -1) | (gripper == 1)).all()):
        raise ValueError("LIBERO dataset gripper must be exactly {-1, +1}")
    return actions


def make_libero_action_chunk(actions: Tensor, anchor: int, *, horizon: int = 8) -> PaddedActionChunk:
    """Construct ``[a_i, ..., a_(i+H-1)]`` without crossing the supplied episode tensor."""

    validate_libero_dataset_actions(actions)
    if actions.ndim != 2:
        raise ValueError("one episode's actions must have shape [time, 7]")
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    if not 0 <= anchor < actions.shape[0]:
        raise IndexError("anchor is outside the episode")
    valid_length = min(horizon, actions.shape[0] - anchor)
    chunk = actions.new_zeros(horizon, 7)
    chunk[:valid_length] = actions[anchor : anchor + valid_length]
    mask = torch.arange(horizon, device=actions.device) < valid_length
    return PaddedActionChunk(actions=chunk, valid_mask=mask)


def libero_env_action(action: Tensor | np.ndarray) -> np.ndarray:
    """Return an owned float32 OSC_POSE action using LIBERO's -open/+close polarity."""

    values = torch.as_tensor(action).detach().float().reshape(-1)
    if values.shape != (7,) or not bool(torch.isfinite(values).all()):
        raise ValueError("action must contain seven finite values")
    output = values.clone()
    output[:6].clamp_(-1.0, 1.0)
    output[6] = 1.0 if output[6] >= 0 else -1.0
    return output.cpu().numpy().astype(np.float32, copy=True)
