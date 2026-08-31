from __future__ import annotations

import torch

from duo_vla.action_interface import ActionInputProjector, VelocityHead
from duo_vla.backbones import TinyActionDecoder
from duo_vla.config import ActionInterfaceConfig
from duo_vla.modeling import DuoVLADenoiser


def _make_model() -> DuoVLADenoiser:
    config = ActionInterfaceConfig(hidden_size=32, state_dim=8)
    return DuoVLADenoiser(
        ActionInputProjector(config),
        TinyActionDecoder(config.hidden_size),
        VelocityHead(config.hidden_size, config.action_dim),
    )


def test_invalid_action_values_cannot_affect_valid_predictions() -> None:
    torch.manual_seed(0)
    model = _make_model().eval()
    actions = torch.randn(1, 8, 7)
    perturbed = actions.clone()
    perturbed[:, 5:] = 1e6
    valid = torch.tensor([[True] * 5 + [False] * 3])
    common = {
        "timesteps": torch.tensor([0.4]),
        "state": torch.randn(1, 8),
        "prefix_cache": torch.randn(1, 4, 32),
        "prefix_attention_mask": torch.tensor([[True, True, True, False]]),
        "action_valid_mask": valid,
    }

    with torch.no_grad():
        prediction = model(actions, **common)
        changed = model(perturbed, **common)
    torch.testing.assert_close(prediction[:, :5], changed[:, :5])


def test_valid_action_values_can_affect_other_valid_predictions() -> None:
    torch.manual_seed(1)
    model = _make_model().eval()
    actions = torch.randn(1, 8, 7)
    perturbed = actions.clone()
    perturbed[:, 2] += 10.0
    valid = torch.ones(1, 8, dtype=torch.bool)
    common = {
        "timesteps": torch.tensor([0.4]),
        "state": torch.randn(1, 8),
        "prefix_cache": torch.randn(1, 4, 32),
        "prefix_attention_mask": torch.ones(1, 4, dtype=torch.bool),
        "action_valid_mask": valid,
    }

    with torch.no_grad():
        prediction = model(actions, **common)
        changed = model(perturbed, **common)
    assert not torch.allclose(prediction[:, 0], changed[:, 0])
