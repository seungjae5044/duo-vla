from __future__ import annotations

import torch
from torch import nn

from duo_vla.config import FlowConfig
from duo_vla.policy import DirectRegressionPolicy, RectifiedFlowPolicy


class _OracleDenoiser(nn.Module):
    def __init__(self, clean: torch.Tensor, initial_noise: torch.Tensor) -> None:
        super().__init__()
        self.clean = clean
        self.initial_noise = initial_noise
        self.calls = 0

    def forward(self, noisy_actions: torch.Tensor, timesteps: torch.Tensor, state: torch.Tensor, **kwargs):
        del noisy_actions, timesteps, state, kwargs
        self.calls += 1
        return self.clean - self.initial_noise


class _ConstantRegressor(nn.Module):
    def __init__(self, actions: torch.Tensor) -> None:
        super().__init__()
        self.actions = actions
        self.calls = 0
        self.last_inputs: tuple[torch.Tensor, torch.Tensor] | None = None

    def forward(self, noisy_actions: torch.Tensor, timesteps: torch.Tensor, state: torch.Tensor, **kwargs):
        del state, kwargs
        self.calls += 1
        self.last_inputs = (noisy_actions.clone(), timesteps.clone())
        return self.actions


def _context(batch_size: int = 1) -> dict[str, object]:
    return {
        "state": torch.zeros(batch_size, 8),
        "prefix_cache": object(),
        "prefix_attention_mask": torch.ones(batch_size, 3, dtype=torch.bool),
        "action_valid_mask": torch.ones(batch_size, 8, dtype=torch.bool),
    }


def test_flow_policy_integrates_then_clips_only_final_chunk() -> None:
    initial = torch.full((1, 8, 7), 2.0)
    clean = torch.full_like(initial, 0.5)
    denoiser = _OracleDenoiser(clean, initial)
    policy = RectifiedFlowPolicy(denoiser, FlowConfig())
    output = policy.sample_normalized(initial_noise=initial, **_context())
    torch.testing.assert_close(output, clean)
    assert denoiser.calls == 10


def test_direct_regression_policy_is_deterministic_and_clips() -> None:
    predicted = torch.full((1, 8, 7), 2.0)
    denoiser = _ConstantRegressor(predicted)
    policy = DirectRegressionPolicy(denoiser, FlowConfig())
    first = policy.sample_normalized(**_context())
    assert denoiser.calls == 1
    assert denoiser.last_inputs is not None
    torch.testing.assert_close(denoiser.last_inputs[0], torch.zeros_like(predicted))
    torch.testing.assert_close(denoiser.last_inputs[1], torch.ones(1))
    second = policy.sample_normalized(**_context())
    assert denoiser.calls == 2
    torch.testing.assert_close(first, second)
    torch.testing.assert_close(first, torch.ones_like(first))
