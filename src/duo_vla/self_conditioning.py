"""Versioned continuous endpoint self-conditioning; no categorical/native-SC equivalence implied."""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from typing import Any

import torch
from torch import Tensor

SC_VERSION = "action_endpoint_v1"


def enabled(denoiser: Any) -> bool:
    projector = getattr(denoiser, "action_projector", None)
    return getattr(getattr(projector, "config", None), "self_conditioning", "none") == SC_VERSION


def bootstrap_for_update(seed: int, update: int) -> bool:
    """Stateless independent Bernoulli(1/2), shared by all ranks/microbatches in an update."""
    if type(seed) is not int or type(update) is not int or update < 0:
        raise ValueError("SC selection needs an integer seed and nonnegative update")
    digest = hashlib.sha256(f"duo-vla-sc-v1:{seed}:{update}".encode()).digest()
    return bool(digest[0] & 1)


def endpoint_estimate(actions: Tensor, timesteps: Tensor, velocity: Tensor) -> Tensor:
    if actions.ndim != 3 or velocity.shape != actions.shape or timesteps.shape != (actions.shape[0],):
        raise ValueError("endpoint estimation requires [B,H,D] actions/velocity and [B] times")
    return actions.float() + (1.0 - timesteps.float()[:, None, None]) * velocity.float()


def training_velocity(
    denoiser: Any,
    actions: Tensor,
    timesteps: Tensor,
    state: Tensor,
    *,
    bootstrap: bool,
    **conditioning: Any,
) -> Tensor:
    """Reuse the exact pair and prefix; stop only the candidate pass, not the main prefix graph."""
    if type(bootstrap) is not bool:
        raise ValueError("bootstrap must be one update-level boolean")
    if not enabled(denoiser):
        if bootstrap:
            raise ValueError("bootstrap requires action_endpoint_v1")
        return denoiser(actions, timesteps, state, **conditioning)
    candidate = None
    if bootstrap:
        with torch.no_grad():
            preliminary = denoiser(actions, timesteps, state, sc_actions=None, sc_present=False, **conditioning)
            candidate = endpoint_estimate(actions, timesteps, preliminary).detach()
    return denoiser(actions, timesteps, state, sc_actions=candidate, sc_present=bootstrap, **conditioning)


@torch.no_grad()
def sample_action_flow(
    denoiser: Any,
    initial_noise: Tensor,
    state: Tensor,
    *,
    num_steps: int,
    use_self_conditioning: bool = True,
    before_step: Callable[[int], None] | None = None,
    after_velocity: Callable[[Tensor], None] | None = None,
    **conditioning: Any,
) -> Tensor:
    """FP32 uniform Euler, fresh query-local history; exactly NFE decoder calls, no clipping."""
    if type(num_steps) is not int or num_steps <= 0 or initial_noise.ndim != 3:
        raise ValueError("sampling requires positive integer NFE and [B,H,D] noise")
    if not enabled(denoiser):
        raise ValueError("versioned SC sampling requires action_endpoint_v1")
    if type(use_self_conditioning) is not bool:
        raise ValueError("use_self_conditioning must be boolean")
    actions = initial_noise.float().clone()
    candidate = None
    for index in range(num_steps):
        if before_step is not None:
            before_step(index)
        time = torch.full((actions.shape[0],), index / num_steps, device=actions.device, dtype=torch.float32)
        velocity = denoiser(
            actions,
            time,
            state,
            sc_actions=candidate,
            sc_present=use_self_conditioning and index > 0,
            **conditioning,
        ).float()
        if velocity.shape != actions.shape:
            raise ValueError("velocity shape does not match the action chunk")
        if after_velocity is not None:
            after_velocity(velocity)
        # Both estimates use PRE-update actions/time. No in-place cache/history mutation.
        next_candidate = endpoint_estimate(actions, time, velocity) if use_self_conditioning else None
        actions = actions + velocity / num_steps
        candidate = next_candidate
    if not bool(actions.isfinite().all()):
        raise FloatingPointError("nonfinite sampled actions")
    return actions
