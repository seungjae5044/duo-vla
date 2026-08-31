"""Benchmark-independent Duo-VLA denoiser composition."""

from __future__ import annotations

from typing import Any, Protocol

from torch import Tensor, nn

from duo_vla.action_interface import ActionInputProjector, VelocityHead


class ActionDecoderBackend(Protocol):
    """Minimal contract implemented by DiffusionGemma and test backends."""

    def decode_actions(
        self,
        action_embeddings: Tensor,
        *,
        prefix_cache: Any,
        prefix_attention_mask: Tensor,
        action_valid_mask: Tensor,
    ) -> Tensor: ...


class DuoVLADenoiser(nn.Module):
    def __init__(
        self,
        action_projector: ActionInputProjector,
        decoder_backend: nn.Module,
        velocity_head: VelocityHead,
    ) -> None:
        super().__init__()
        self.action_projector = action_projector
        self.decoder_backend = decoder_backend
        self.velocity_head = velocity_head

    def forward(
        self,
        noisy_actions: Tensor,
        timesteps: Tensor,
        state: Tensor,
        *,
        prefix_cache: Any,
        prefix_attention_mask: Tensor,
        action_valid_mask: Tensor,
    ) -> Tensor:
        action_embeddings = self.action_projector(
            noisy_actions,
            timesteps,
            state,
            valid_mask=action_valid_mask,
        )
        hidden_states = self.decoder_backend.decode_actions(
            action_embeddings,
            prefix_cache=prefix_cache,
            prefix_attention_mask=prefix_attention_mask,
            action_valid_mask=action_valid_mask,
        )
        if hidden_states.shape[:-1] != noisy_actions.shape[:-1]:
            raise ValueError("decoder backend returned incompatible hidden-state shape")
        return self.velocity_head(hidden_states)
