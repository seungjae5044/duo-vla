"""Validated collation for raw CALVIN ABC training samples."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from duo_vla.data.calvin import CalvinTrainingSample
from duo_vla.normalization import ActionNormalizer


@dataclass(frozen=True, slots=True)
class CalvinBatch:
    samples: tuple[CalvinTrainingSample, ...]
    states: Tensor
    clean_actions: Tensor
    action_valid_mask: Tensor

    @property
    def batch_size(self) -> int:
        return len(self.samples)


def coalesce_calvin_batches(batches: tuple[CalvinBatch, ...], *, physical_batch_size: int) -> CalvinBatch:
    """Join already-validated B8 chunks without rejecting cross-chunk repeats.

    Distinctness applies within each canonical RNG plan, not across plans. A
    repeated anchor in another plan has its own noise and remains a real sample.
    """
    if physical_batch_size not in (8, 16, 32, 64) or len(batches) * 8 != physical_batch_size:
        raise ValueError("physical CALVIN batches must contain complete canonical B8 chunks")
    if any(batch.batch_size != 8 for batch in batches):
        raise ValueError("only canonical B8 batches may be coalesced")
    return CalvinBatch(
        samples=tuple(sample for batch in batches for sample in batch.samples),
        states=torch.cat([batch.states for batch in batches]),
        clean_actions=torch.cat([batch.clean_actions for batch in batches]),
        action_valid_mask=torch.cat([batch.action_valid_mask for batch in batches]),
    )


def collate_calvin_samples(
    samples: tuple[CalvinTrainingSample, ...],
    *,
    state_normalizer: ActionNormalizer,
) -> CalvinBatch:
    """Normalize only state channels; CALVIN ``rel_actions`` are already simulator-scaled."""

    if not samples:
        raise ValueError("samples must not be empty")
    identities = [(sample.annotation_index, sample.global_index) for sample in samples]
    if len(set(identities)) != len(identities):
        raise ValueError("a fixed CALVIN batch must contain distinct annotation/frame anchors")
    if state_normalizer.action_dim != 8 or state_normalizer.resolved_gripper_index != 7:
        raise ValueError("CALVIN state normalizer must cover seven continuous channels and gripper channel 7")
    states = torch.stack([sample.observation.state for sample in samples]).float()
    actions = torch.stack([sample.action_chunk.actions for sample in samples]).float()
    valid = torch.stack([sample.action_chunk.valid_mask for sample in samples]).bool()
    if states.shape != (len(samples), 8) or actions.shape != (len(samples), 8, 7):
        raise ValueError("CALVIN batch has an invalid state or action shape")
    if valid.shape != actions.shape[:2] or bool((valid.sum(dim=1) == 0).any()):
        raise ValueError("every CALVIN sample must have at least one valid action")
    if not bool(torch.isfinite(states).all()) or not bool(torch.isfinite(actions).all()):
        raise ValueError("CALVIN batch contains non-finite state or action values")
    if bool((actions[..., :6].abs() > 1.0 + 1e-6).any()):
        raise ValueError("CALVIN rel_actions must already be scaled to [-1, 1]")
    valid_gripper = actions[..., 6][valid]
    if not bool(((valid_gripper == -1) | (valid_gripper == 1)).all()):
        raise ValueError("valid CALVIN gripper actions must be exactly {-1, +1}")
    normalized_states = state_normalizer.normalize(states)
    if not bool(torch.isfinite(normalized_states).all()):
        raise ValueError("CALVIN state normalization produced non-finite values")
    # Keep the official action scaling exactly; only padded slots are zeroed for loss/attention safety.
    clean_actions = actions * valid[..., None].to(dtype=actions.dtype)
    return CalvinBatch(
        samples=samples,
        states=normalized_states,
        clean_actions=clean_actions,
        action_valid_mask=valid,
    )
