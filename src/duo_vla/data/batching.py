"""Validated CPU collation for real LIBERO training samples."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from duo_vla.data.libero import LiberoTrainingSample
from duo_vla.normalization import ActionNormalizer, PercentileNormalizer


@dataclass(frozen=True, slots=True)
class LiberoBatch:
    samples: tuple[LiberoTrainingSample, ...]
    states: Tensor
    clean_actions: Tensor
    action_valid_mask: Tensor

    @property
    def batch_size(self) -> int:
        return len(self.samples)


def collate_libero_samples(
    samples: tuple[LiberoTrainingSample, ...],
    *,
    state_normalizer: PercentileNormalizer,
    action_normalizer: ActionNormalizer,
) -> LiberoBatch:
    if not samples:
        raise ValueError("samples must not be empty")
    identities = [(sample.episode_index, sample.frame_index) for sample in samples]
    if len(set(identities)) != len(identities):
        raise ValueError("a fixed LIBERO batch must contain distinct anchors")
    states = torch.stack([sample.observation.state for sample in samples]).float()
    actions = torch.stack([sample.action_chunk.actions for sample in samples]).float()
    valid = torch.stack([sample.action_chunk.valid_mask for sample in samples]).bool()
    if states.shape != (len(samples), 8) or actions.shape != (len(samples), 8, 7):
        raise ValueError("LIBERO batch has an invalid state or action shape")
    if valid.shape != actions.shape[:2] or bool((valid.sum(dim=1) == 0).any()):
        raise ValueError("every LIBERO sample must have at least one valid action")
    normalized_states = state_normalizer.normalize(states)
    normalized_actions = action_normalizer.normalize(actions)
    normalized_actions = normalized_actions * valid[..., None].to(dtype=normalized_actions.dtype)
    return LiberoBatch(
        samples=samples,
        states=normalized_states,
        clean_actions=normalized_actions,
        action_valid_mask=valid,
    )
