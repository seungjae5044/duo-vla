from __future__ import annotations

import pytest
import torch

from duo_vla.flow import euler_sample, make_flow_training_pair, masked_velocity_mse


def test_flow_path_endpoints_and_target() -> None:
    clean = torch.tensor([[[0.5, -0.5], [1.0, -1.0]]])
    noise = torch.tensor([[[2.0, 3.0], [-2.0, -3.0]]])

    at_zero = make_flow_training_pair(clean, timesteps=torch.tensor([0.0]), noise=noise)
    at_one = make_flow_training_pair(clean, timesteps=torch.tensor([1.0]), noise=noise)

    torch.testing.assert_close(at_zero.noisy_actions, noise)
    torch.testing.assert_close(at_one.noisy_actions, clean)
    torch.testing.assert_close(at_zero.target_velocity, clean - noise)
    torch.testing.assert_close(at_one.target_velocity, clean - noise)


def test_oracle_velocity_recovers_clean_actions() -> None:
    clean = torch.randn(3, 8, 7)
    noise = torch.randn_like(clean)

    def oracle(_actions: torch.Tensor, _t: torch.Tensor) -> torch.Tensor:
        return clean - noise

    generated = euler_sample(oracle, initial_noise=noise, num_steps=10)
    torch.testing.assert_close(generated, clean, atol=2e-6, rtol=2e-6)


def test_masked_velocity_loss_ignores_invalid_positions() -> None:
    target = torch.zeros(1, 3, 2)
    prediction = torch.tensor([[[1.0, 1.0], [100.0, 100.0], [3.0, 3.0]]])
    valid = torch.tensor([[True, False, True]])
    loss = masked_velocity_mse(prediction, target, valid)
    assert loss.item() == pytest.approx(5.0)


def test_masked_velocity_loss_rejects_all_padding() -> None:
    values = torch.zeros(1, 2, 3)
    with pytest.raises(ValueError, match="no valid"):
        masked_velocity_mse(values, values, torch.zeros(1, 2, dtype=torch.bool))
