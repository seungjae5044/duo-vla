"""Validated configuration objects shared by training and inference."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class FlowConfig:
    """Rectified-flow and action chunk configuration."""

    action_horizon: int = 8
    action_dim: int = 7
    inference_steps: int = 10

    def __post_init__(self) -> None:
        if self.action_horizon <= 0:
            raise ValueError("action_horizon must be positive")
        if self.action_dim <= 0:
            raise ValueError("action_dim must be positive")
        if self.inference_steps <= 0:
            raise ValueError("inference_steps must be positive")


@dataclass(frozen=True, slots=True)
class ActionInterfaceConfig:
    """Continuous action/state interface around the frozen decoder."""

    hidden_size: int
    state_dim: int
    action_horizon: int = 8
    action_dim: int = 7
    timestep_embedding_dim: int = 256
    timestep_scale: float = 1000.0
    timestep_max_period: float = 10_000.0
    output_init_std: float = 1e-3
    self_conditioning: str = "none"

    def __post_init__(self) -> None:
        for name in ("hidden_size", "state_dim", "action_horizon", "action_dim", "timestep_embedding_dim"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.timestep_embedding_dim % 2:
            raise ValueError("timestep_embedding_dim must be even")
        if self.timestep_scale <= 0:
            raise ValueError("timestep_scale must be positive")
        if self.timestep_max_period <= 1:
            raise ValueError("timestep_max_period must be greater than one")
        if self.output_init_std <= 0:
            raise ValueError("output_init_std must be positive")
        if self.self_conditioning not in {"none", "action_endpoint_v1"}:
            raise ValueError("unsupported action self-conditioning architecture")
