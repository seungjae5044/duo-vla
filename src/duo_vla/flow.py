"""Straight-line conditional flow matching for continuous action chunks."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True, slots=True)
class FlowTrainingPair:
    """A sampled point and target on the path from Gaussian noise to clean action."""

    noisy_actions: Tensor
    target_velocity: Tensor
    timesteps: Tensor
    noise: Tensor


def _validate_action_chunk(actions: Tensor) -> None:
    if actions.ndim != 3:
        raise ValueError(f"actions must have shape [batch, horizon, action_dim], got {tuple(actions.shape)}")
    if not actions.is_floating_point():
        raise TypeError("actions must use a floating-point dtype")


def make_flow_training_pair(
    clean_actions: Tensor,
    *,
    timesteps: Tensor | None = None,
    noise: Tensor | None = None,
    generator: torch.Generator | None = None,
) -> FlowTrainingPair:
    """Sample one scalar timestep per action chunk and construct the rectified-flow target.

    The path is ``A_t = (1 - t) A_0 + t A_1`` with ``A_0 = noise`` and ``A_1 = clean_actions``.
    """

    _validate_action_chunk(clean_actions)
    batch_size = clean_actions.shape[0]

    if timesteps is None:
        timesteps = torch.rand(
            batch_size,
            device=clean_actions.device,
            dtype=clean_actions.dtype,
            generator=generator,
        )
    else:
        timesteps = timesteps.to(device=clean_actions.device, dtype=clean_actions.dtype)
        if timesteps.shape not in {(batch_size,), (batch_size, 1), (batch_size, 1, 1)}:
            raise ValueError(f"timesteps must contain one scalar per batch item, got {tuple(timesteps.shape)}")
        timesteps = timesteps.reshape(batch_size)
        if bool(((timesteps < 0) | (timesteps > 1)).any()):
            raise ValueError("timesteps must be in [0, 1]")

    if noise is None:
        noise = torch.randn(
            clean_actions.shape,
            device=clean_actions.device,
            dtype=clean_actions.dtype,
            generator=generator,
        )
    else:
        noise = noise.to(device=clean_actions.device, dtype=clean_actions.dtype)
        if noise.shape != clean_actions.shape:
            raise ValueError(f"noise shape {tuple(noise.shape)} does not match actions {tuple(clean_actions.shape)}")

    broadcast_t = timesteps[:, None, None]
    noisy_actions = (1.0 - broadcast_t) * noise + broadcast_t * clean_actions
    target_velocity = clean_actions - noise
    return FlowTrainingPair(
        noisy_actions=noisy_actions,
        target_velocity=target_velocity,
        timesteps=timesteps,
        noise=noise,
    )


def masked_velocity_mse(prediction: Tensor, target: Tensor, valid_mask: Tensor | None = None) -> Tensor:
    """Mean squared velocity error over valid action positions, accumulated in FP32."""

    if prediction.shape != target.shape:
        raise ValueError(f"prediction {tuple(prediction.shape)} and target {tuple(target.shape)} must match")
    _validate_action_chunk(prediction)

    squared_error = (prediction.float() - target.float()).square()
    if valid_mask is None:
        return squared_error.mean()
    if valid_mask.shape != prediction.shape[:2]:
        raise ValueError(
            f"valid_mask must have shape [batch, horizon]={tuple(prediction.shape[:2])}, got {tuple(valid_mask.shape)}"
        )
    valid = valid_mask.to(device=prediction.device, dtype=torch.bool)
    valid_count = valid.sum()
    if int(valid_count.item()) == 0:
        raise ValueError("valid_mask contains no valid action positions")
    return (squared_error * valid[..., None]).sum() / (valid_count * prediction.shape[-1])


@torch.no_grad()
def euler_sample(
    velocity_fn: Callable[[Tensor, Tensor], Tensor],
    *,
    initial_noise: Tensor,
    num_steps: int,
    schedule: Sequence[float] | Tensor | None = None,
    return_trajectory: bool = False,
) -> Tensor | tuple[Tensor, tuple[Tensor, ...]]:
    """Integrate the learned velocity field from ``t=0`` to ``t=1`` without intermediate clipping."""

    _validate_action_chunk(initial_noise)
    if num_steps <= 0:
        raise ValueError("num_steps must be positive")

    if schedule is None:
        times = torch.linspace(0.0, 1.0, num_steps + 1, device=initial_noise.device, dtype=torch.float32)
    else:
        times = torch.as_tensor(schedule, device=initial_noise.device, dtype=torch.float32)
        if times.ndim != 1 or times.numel() != num_steps + 1:
            raise ValueError("schedule must be one-dimensional with num_steps + 1 entries")
        if not torch.isclose(times[0], torch.tensor(0.0, device=times.device)):
            raise ValueError("schedule must start at zero")
        if not torch.isclose(times[-1], torch.tensor(1.0, device=times.device)):
            raise ValueError("schedule must end at one")
        if bool((times[1:] <= times[:-1]).any()):
            raise ValueError("schedule must be strictly increasing")

    actions = initial_noise.clone()
    trajectory: list[Tensor] = [actions.clone()] if return_trajectory else []
    batch_size = actions.shape[0]
    for index in range(num_steps):
        t = times[index].expand(batch_size).to(dtype=actions.dtype)
        velocity = velocity_fn(actions, t)
        if velocity.shape != actions.shape:
            raise ValueError(f"velocity_fn returned {tuple(velocity.shape)}, expected {tuple(actions.shape)}")
        delta_t = (times[index + 1] - times[index]).to(dtype=actions.dtype)
        actions = actions + delta_t * velocity
        if return_trajectory:
            trajectory.append(actions.clone())

    if return_trajectory:
        return actions, tuple(trajectory)
    return actions
