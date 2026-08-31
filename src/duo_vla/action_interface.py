"""Trainable continuous action, state, and flow-time interfaces."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from duo_vla.config import ActionInterfaceConfig


def sinusoidal_timestep_embedding(
    timesteps: Tensor,
    *,
    dim: int,
    scale: float = 1000.0,
    max_period: float = 10_000.0,
) -> Tensor:
    """DiT-style Fourier features with an explicit normalized-time scale."""

    if timesteps.ndim != 1:
        raise ValueError(f"timesteps must have shape [batch], got {tuple(timesteps.shape)}")
    if dim <= 0 or dim % 2:
        raise ValueError("dim must be a positive even integer")
    half = dim // 2
    frequencies = torch.exp(
        -math.log(max_period) * torch.arange(half, device=timesteps.device, dtype=torch.float32) / half
    )
    arguments = timesteps.float()[:, None] * scale * frequencies[None]
    return torch.cat((torch.cos(arguments), torch.sin(arguments)), dim=-1)


class TwoLayerMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.net(inputs)


class ActionInputProjector(nn.Module):
    """Construct ``H`` continuous decoder embeddings from action, time, and state."""

    def __init__(self, config: ActionInterfaceConfig) -> None:
        super().__init__()
        self.config = config
        self.action_projection = nn.Linear(config.action_dim, config.hidden_size)
        self.timestep_mlp = TwoLayerMLP(
            config.timestep_embedding_dim,
            config.hidden_size,
            config.hidden_size,
        )
        self.state_mlp = TwoLayerMLP(config.state_dim, config.hidden_size, config.hidden_size)
        self.horizon_embedding = nn.Parameter(torch.empty(config.action_horizon, config.hidden_size))
        self.action_type_embedding = nn.Parameter(torch.empty(config.hidden_size))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.action_projection.weight)
        nn.init.zeros_(self.action_projection.bias)
        nn.init.normal_(self.horizon_embedding, std=0.02)
        nn.init.normal_(self.action_type_embedding, std=0.02)

    def forward(
        self,
        noisy_actions: Tensor,
        timesteps: Tensor,
        state: Tensor,
        *,
        valid_mask: Tensor | None = None,
    ) -> Tensor:
        expected = (noisy_actions.shape[0], self.config.action_horizon, self.config.action_dim)
        if noisy_actions.shape != expected:
            raise ValueError(f"noisy_actions must have shape {expected}, got {tuple(noisy_actions.shape)}")
        if timesteps.shape != (noisy_actions.shape[0],):
            raise ValueError("timesteps must contain one scalar per batch item")
        if state.shape != (noisy_actions.shape[0], self.config.state_dim):
            raise ValueError(
                f"state must have shape {(noisy_actions.shape[0], self.config.state_dim)}, got {tuple(state.shape)}"
            )

        time_features = sinusoidal_timestep_embedding(
            timesteps,
            dim=self.config.timestep_embedding_dim,
            scale=self.config.timestep_scale,
            max_period=self.config.timestep_max_period,
        ).to(dtype=self.timestep_mlp.net[0].weight.dtype)
        action_embedding = self.action_projection(noisy_actions)
        time_embedding = self.timestep_mlp(time_features).to(dtype=action_embedding.dtype)[:, None]
        state_embedding = self.state_mlp(state).to(dtype=action_embedding.dtype)[:, None]
        horizon_embedding = self.horizon_embedding.to(dtype=action_embedding.dtype)[None]
        type_embedding = self.action_type_embedding.to(dtype=action_embedding.dtype)[None, None]
        output = action_embedding + time_embedding + state_embedding + horizon_embedding + type_embedding

        if valid_mask is not None:
            if valid_mask.shape != noisy_actions.shape[:2]:
                raise ValueError("valid_mask must have shape [batch, horizon]")
            output = output * valid_mask.to(device=output.device, dtype=output.dtype)[..., None]
        return output


class VelocityHead(nn.Module):
    """Linear decoder hidden-state projection with no bounded activation."""

    def __init__(self, hidden_size: int, action_dim: int, *, init_std: float = 1e-3) -> None:
        super().__init__()
        if hidden_size <= 0 or action_dim <= 0 or init_std <= 0:
            raise ValueError("hidden_size, action_dim, and init_std must be positive")
        self.projection = nn.Linear(hidden_size, action_dim)
        nn.init.normal_(self.projection.weight, std=init_std)
        nn.init.zeros_(self.projection.bias)

    def forward(self, hidden_states: Tensor) -> Tensor:
        return self.projection(hidden_states)
