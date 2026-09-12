"""Training objectives and normalized-space inference policies."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn

from duo_vla.config import FlowConfig
from duo_vla.flow import euler_sample, make_flow_training_pair, masked_velocity_mse
from duo_vla.self_conditioning import bootstrap_for_update, enabled, sample_action_flow, training_velocity


class RectifiedFlowPolicy(nn.Module):
    def __init__(self, denoiser: nn.Module, config: FlowConfig) -> None:
        super().__init__()
        self.denoiser = denoiser
        self.config = config

    def training_loss(
        self,
        clean_actions: Tensor,
        state: Tensor,
        *,
        prefix_cache: Any,
        prefix_attention_mask: Tensor,
        action_valid_mask: Tensor,
        generator: torch.Generator | None = None,
        optimizer_update: int | None = None,
        sc_seed: int = 0,
        sc_bootstrap: bool | None = None,
    ) -> Tensor:
        if enabled(self.denoiser) and sc_bootstrap is None:
            if optimizer_update is None:
                raise ValueError("SC training requires optimizer_update or an explicit sc_bootstrap decision")
            sc_bootstrap = bootstrap_for_update(sc_seed, optimizer_update)
        pair = make_flow_training_pair(clean_actions, generator=generator)
        prediction = training_velocity(
            self.denoiser,
            pair.noisy_actions,
            pair.timesteps,
            state,
            prefix_cache=prefix_cache,
            prefix_attention_mask=prefix_attention_mask,
            action_valid_mask=action_valid_mask,
            bootstrap=False if sc_bootstrap is None else sc_bootstrap,
        )
        return masked_velocity_mse(prediction, pair.target_velocity, action_valid_mask)

    @torch.no_grad()
    def sample_normalized(
        self,
        state: Tensor,
        *,
        prefix_cache: Any,
        prefix_attention_mask: Tensor,
        action_valid_mask: Tensor | None = None,
        initial_noise: Tensor | None = None,
        generator: torch.Generator | None = None,
        num_steps: int | None = None,
    ) -> Tensor:
        batch_size = state.shape[0]
        if action_valid_mask is None:
            action_valid_mask = torch.ones(
                batch_size,
                self.config.action_horizon,
                dtype=torch.bool,
                device=state.device,
            )
        if initial_noise is None:
            initial_noise = torch.randn(
                batch_size,
                self.config.action_horizon,
                self.config.action_dim,
                dtype=state.dtype,
                device=state.device,
                generator=generator,
            )
        steps = self.config.inference_steps if num_steps is None else num_steps
        if enabled(self.denoiser):
            return sample_action_flow(
                self.denoiser,
                initial_noise,
                state,
                num_steps=steps,
                prefix_cache=prefix_cache,
                prefix_attention_mask=prefix_attention_mask,
                action_valid_mask=action_valid_mask,
            ).clamp(-1.0, 1.0)

        def velocity(actions: Tensor, timesteps: Tensor) -> Tensor:
            return self.denoiser(
                actions,
                timesteps,
                state,
                prefix_cache=prefix_cache,
                prefix_attention_mask=prefix_attention_mask,
                action_valid_mask=action_valid_mask,
            )

        result = euler_sample(velocity, initial_noise=initial_noise, num_steps=steps)
        return result.clamp(-1.0, 1.0)


class DirectRegressionPolicy(nn.Module):
    """One-pass control with the same decoder/interface, used as the primary non-flow baseline."""

    def __init__(self, denoiser: nn.Module, config: FlowConfig) -> None:
        super().__init__()
        self.denoiser = denoiser
        self.config = config

    def predict_normalized(
        self,
        state: Tensor,
        *,
        prefix_cache: Any,
        prefix_attention_mask: Tensor,
        action_valid_mask: Tensor,
    ) -> Tensor:
        batch_size = state.shape[0]
        # Deterministic empty action canvas; horizon/type embeddings provide distinct learned queries.
        empty_canvas = state.new_zeros(batch_size, self.config.action_horizon, self.config.action_dim)
        fixed_time = state.new_ones(batch_size)
        return self.denoiser(
            empty_canvas,
            fixed_time,
            state,
            prefix_cache=prefix_cache,
            prefix_attention_mask=prefix_attention_mask,
            action_valid_mask=action_valid_mask,
        )

    def training_loss(
        self,
        clean_actions: Tensor,
        state: Tensor,
        *,
        prefix_cache: Any,
        prefix_attention_mask: Tensor,
        action_valid_mask: Tensor,
    ) -> Tensor:
        prediction = self.predict_normalized(
            state,
            prefix_cache=prefix_cache,
            prefix_attention_mask=prefix_attention_mask,
            action_valid_mask=action_valid_mask,
        )
        return masked_velocity_mse(prediction, clean_actions, action_valid_mask)

    @torch.no_grad()
    def sample_normalized(
        self,
        state: Tensor,
        *,
        prefix_cache: Any,
        prefix_attention_mask: Tensor,
        action_valid_mask: Tensor,
    ) -> Tensor:
        return self.predict_normalized(
            state,
            prefix_cache=prefix_cache,
            prefix_attention_mask=prefix_attention_mask,
            action_valid_mask=action_valid_mask,
        ).clamp(-1.0, 1.0)
