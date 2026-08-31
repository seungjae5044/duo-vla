"""Shared train/validation input-target pairs for supported policy objectives."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from duo_vla.flow import make_flow_training_pair
from duo_vla.policy_contract import DIRECT_REGRESSION, RECTIFIED_FLOW, PolicyContract


@dataclass(frozen=True, slots=True)
class PolicyTrainingPair:
    """Decoder inputs and regression target for one complete action chunk."""

    input_actions: Tensor
    timesteps: Tensor
    target: Tensor


def make_policy_training_pair(
    clean_actions: Tensor,
    contract: PolicyContract,
    *,
    generator: torch.Generator | None = None,
) -> PolicyTrainingPair:
    """Construct the canonical pair for flow or deterministic direct regression."""

    if clean_actions.ndim != 3:
        raise ValueError("clean_actions must have shape [batch, horizon, action_dim]")
    if not clean_actions.is_floating_point():
        raise TypeError("clean_actions must use a floating-point dtype")
    if clean_actions.shape[1:] != (contract.action_horizon, contract.action_dim):
        raise ValueError("clean_actions shape differs from the policy contract")

    if contract.objective == RECTIFIED_FLOW:
        pair = make_flow_training_pair(clean_actions, generator=generator)
        return PolicyTrainingPair(
            input_actions=pair.noisy_actions,
            timesteps=pair.timesteps,
            target=pair.target_velocity,
        )
    if contract.objective == DIRECT_REGRESSION:
        return PolicyTrainingPair(
            input_actions=torch.zeros_like(clean_actions),
            timesteps=torch.ones(
                clean_actions.shape[0],
                device=clean_actions.device,
                dtype=clean_actions.dtype,
            ),
            target=clean_actions,
        )
    raise ValueError(f"unsupported policy objective: {contract.objective!r}")


def make_seeded_policy_training_pair(
    clean_actions: Tensor,
    contract: PolicyContract,
    *,
    seed: int,
) -> PolicyTrainingPair:
    """Construct a deterministic pair while avoiding any RNG object for direct regression."""

    if type(seed) is not int or not 0 <= seed < 2**63:
        raise ValueError("objective seed must be an integer in [0, 2^63)")
    if contract.objective == DIRECT_REGRESSION:
        return make_policy_training_pair(clean_actions, contract)
    if contract.objective == RECTIFIED_FLOW:
        generator = torch.Generator(device=clean_actions.device).manual_seed(seed)
        return make_policy_training_pair(clean_actions, contract, generator=generator)
    raise ValueError(f"unsupported policy objective: {contract.objective!r}")
