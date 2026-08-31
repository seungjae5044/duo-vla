"""Percentile normalization with an explicit binary gripper contract."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True, slots=True)
class PercentileNormalizer:
    """Per-dimension affine map between percentile bounds and ``[-1, 1]``."""

    lower: Tensor
    upper: Tensor
    eps: float = 1e-6

    def __post_init__(self) -> None:
        if self.lower.ndim != 1 or self.upper.ndim != 1:
            raise ValueError("lower and upper must be one-dimensional")
        if self.lower.shape != self.upper.shape:
            raise ValueError("lower and upper must have the same shape")
        if not self.lower.is_floating_point() or not self.upper.is_floating_point():
            raise TypeError("percentile bounds must be floating point")
        if bool((self.upper < self.lower).any()):
            raise ValueError("upper percentile must not be smaller than lower percentile")
        if self.eps <= 0:
            raise ValueError("eps must be positive")

    @property
    def dim(self) -> int:
        return self.lower.numel()

    @property
    def constant_mask(self) -> Tensor:
        return (self.upper - self.lower).abs() < self.eps

    @classmethod
    def fit(
        cls,
        values: Tensor,
        *,
        valid_mask: Tensor | None = None,
        lower_quantile: float = 0.01,
        upper_quantile: float = 0.99,
        eps: float = 1e-6,
    ) -> PercentileNormalizer:
        if values.ndim < 2 or not values.is_floating_point():
            raise ValueError("values must be a floating tensor with a final feature dimension")
        if not 0 <= lower_quantile < upper_quantile <= 1:
            raise ValueError("quantiles must satisfy 0 <= lower < upper <= 1")
        flattened = values.reshape(-1, values.shape[-1]).float()
        if valid_mask is not None:
            if valid_mask.shape != values.shape[:-1]:
                raise ValueError("valid_mask must match all non-feature dimensions")
            flattened = flattened[valid_mask.reshape(-1).bool()]
        if flattened.shape[0] == 0:
            raise ValueError("cannot fit percentile statistics without valid values")
        lower = torch.quantile(flattened, lower_quantile, dim=0)
        upper = torch.quantile(flattened, upper_quantile, dim=0)
        return cls(lower=lower, upper=upper, eps=eps)

    def to(
        self,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> PercentileNormalizer:
        return PercentileNormalizer(
            lower=self.lower.to(device=device, dtype=dtype),
            upper=self.upper.to(device=device, dtype=dtype),
            eps=self.eps,
        )

    def normalize(self, values: Tensor, *, clip: bool = True) -> Tensor:
        self._validate_values(values)
        lower = self.lower.to(device=values.device, dtype=values.dtype)
        upper = self.upper.to(device=values.device, dtype=values.dtype)
        constant = self.constant_mask.to(device=values.device)
        source = values.clamp(lower, upper) if clip else values
        normalized = 2.0 * (source - lower) / (upper - lower).clamp_min(self.eps) - 1.0
        return torch.where(constant, torch.zeros_like(normalized), normalized)

    def unnormalize(self, values: Tensor, *, clip: bool = True) -> Tensor:
        self._validate_values(values)
        lower = self.lower.to(device=values.device, dtype=values.dtype)
        upper = self.upper.to(device=values.device, dtype=values.dtype)
        normalized = values.clamp(-1.0, 1.0) if clip else values
        restored = lower + 0.5 * (normalized + 1.0) * (upper - lower)
        constant = self.constant_mask.to(device=values.device)
        return torch.where(constant, lower.expand_as(restored), restored)

    def _validate_values(self, values: Tensor) -> None:
        if not values.is_floating_point():
            raise TypeError("values must be floating point")
        if values.shape[-1] != self.dim:
            raise ValueError(f"expected final dimension {self.dim}, got {values.shape[-1]}")


@dataclass(frozen=True, slots=True)
class ActionNormalizer:
    """Normalize continuous action channels while preserving a binary gripper channel."""

    continuous: PercentileNormalizer
    action_dim: int = 7
    gripper_index: int = -1

    def __post_init__(self) -> None:
        index = self.gripper_index % self.action_dim
        if self.action_dim <= 1:
            raise ValueError("action_dim must include continuous channels and a gripper channel")
        if not 0 <= index < self.action_dim:
            raise ValueError("gripper_index is out of range")
        if self.continuous.dim != self.action_dim - 1:
            raise ValueError("continuous normalizer must cover every non-gripper action dimension")

    @property
    def resolved_gripper_index(self) -> int:
        return self.gripper_index % self.action_dim

    def normalize(self, actions: Tensor) -> Tensor:
        self._validate(actions)
        continuous, gripper = self._split(actions)
        normalized_continuous = self.continuous.normalize(continuous, clip=True)
        binary_gripper = torch.where(gripper >= 0, torch.ones_like(gripper), -torch.ones_like(gripper))
        return self._merge(normalized_continuous, binary_gripper)

    def unnormalize(self, actions: Tensor) -> Tensor:
        """Clip the final generated chunk, invert continuous stats, then threshold gripper at zero."""

        self._validate(actions)
        continuous, gripper = self._split(actions.clamp(-1.0, 1.0))
        restored_continuous = self.continuous.unnormalize(continuous, clip=False)
        binary_gripper = torch.where(gripper >= 0, torch.ones_like(gripper), -torch.ones_like(gripper))
        return self._merge(restored_continuous, binary_gripper)

    def _split(self, actions: Tensor) -> tuple[Tensor, Tensor]:
        index = self.resolved_gripper_index
        continuous = torch.cat((actions[..., :index], actions[..., index + 1 :]), dim=-1)
        return continuous, actions[..., index : index + 1]

    def _merge(self, continuous: Tensor, gripper: Tensor) -> Tensor:
        index = self.resolved_gripper_index
        return torch.cat((continuous[..., :index], gripper, continuous[..., index:]), dim=-1)

    def _validate(self, actions: Tensor) -> None:
        if not actions.is_floating_point():
            raise TypeError("actions must be floating point")
        if actions.shape[-1] != self.action_dim:
            raise ValueError(f"expected action dimension {self.action_dim}, got {actions.shape[-1]}")
