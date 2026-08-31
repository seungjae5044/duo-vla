from __future__ import annotations

import torch

from duo_vla.action_interface import ActionInputProjector, VelocityHead, sinusoidal_timestep_embedding
from duo_vla.config import ActionInterfaceConfig


def test_timestep_embedding_shape_and_endpoint_difference() -> None:
    embedding = sinusoidal_timestep_embedding(torch.tensor([0.0, 1.0]), dim=256, scale=1000.0)
    assert embedding.shape == (2, 256)
    assert not torch.allclose(embedding[0], embedding[1])


def test_action_interface_shapes_and_invalid_slot_zeroing() -> None:
    config = ActionInterfaceConfig(hidden_size=32, state_dim=8)
    projector = ActionInputProjector(config)
    actions = torch.randn(2, 8, 7)
    timesteps = torch.tensor([0.25, 0.75])
    state = torch.randn(2, 8)
    valid = torch.tensor([[True] * 8, [True] * 5 + [False] * 3])

    embeddings = projector(actions, timesteps, state, valid_mask=valid)
    assert embeddings.shape == (2, 8, 32)
    torch.testing.assert_close(embeddings[1, 5:], torch.zeros_like(embeddings[1, 5:]))

    head = VelocityHead(32, 7)
    assert head(embeddings).shape == (2, 8, 7)


def test_action_projector_has_no_extra_trainable_normalization() -> None:
    projector = ActionInputProjector(ActionInterfaceConfig(hidden_size=32, state_dim=8))
    assert not any(isinstance(module, torch.nn.RMSNorm) for module in projector.modules())
