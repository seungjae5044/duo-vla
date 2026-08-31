"""Dependency-light canonical records shared by simulator adapters."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor


@dataclass(frozen=True, slots=True)
class CanonicalObservation:
    """Two raw RGB views and benchmark-canonical proprioception at one control time."""

    third_person: np.ndarray
    wrist: np.ndarray
    state: Tensor

    def __post_init__(self) -> None:
        _validate_rgb(self.third_person, "third_person")
        _validate_rgb(self.wrist, "wrist")
        if self.state.ndim != 1 or not self.state.is_floating_point():
            raise ValueError("state must be a one-dimensional floating tensor")


@dataclass(frozen=True, slots=True)
class PaddedActionChunk:
    actions: Tensor
    valid_mask: Tensor

    def __post_init__(self) -> None:
        if self.actions.ndim != 2 or not self.actions.is_floating_point():
            raise ValueError("actions must have floating shape [horizon, action_dim]")
        if self.valid_mask.shape != self.actions.shape[:1] or self.valid_mask.dtype != torch.bool:
            raise ValueError("valid_mask must have bool shape [horizon]")
        if not bool(self.valid_mask.any()):
            raise ValueError("an action chunk must contain at least one valid action")


def _validate_rgb(image: np.ndarray, name: str) -> None:
    if not isinstance(image, np.ndarray) or image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"{name} must be an HWC numpy RGB image")
    if image.dtype != np.uint8:
        raise TypeError(f"{name} must use uint8")
