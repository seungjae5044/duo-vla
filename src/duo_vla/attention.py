"""Logical attention masks for the prefix-plus-action block structure."""

from __future__ import annotations

import torch
from torch import Tensor


def build_action_suffix_mask(prefix_valid: Tensor, action_valid: Tensor) -> Tensor:
    """Return allowed attention edges for action queries.

    The result has shape ``[batch, horizon, prefix + horizon]`` and uses ``True`` for an allowed edge. Invalid action
    queries have no outgoing edges; invalid prefix/action keys are unavailable to every action query.
    """

    _validate_valid_masks(prefix_valid, action_valid)
    keys = torch.cat((prefix_valid.bool(), action_valid.bool()), dim=1)
    return action_valid.bool()[..., None] & keys[:, None, :]


def build_prefix_action_block_mask(
    prefix_valid: Tensor,
    action_valid: Tensor,
    *,
    prefix_pairwise_mask: Tensor | None = None,
) -> Tensor:
    """Build a reference full block mask without allowing prefix-to-action attention.

    ``prefix_pairwise_mask`` can preserve a backbone-specific prefix relation. If omitted, valid prefix tokens attend
    bidirectionally to other valid prefix tokens. Production DiffusionGemma code should pass its native prefix mask.
    """

    _validate_valid_masks(prefix_valid, action_valid)
    batch_size, prefix_length = prefix_valid.shape
    horizon = action_valid.shape[1]
    if prefix_pairwise_mask is None:
        prefix_pairwise_mask = prefix_valid.bool()[:, :, None] & prefix_valid.bool()[:, None, :]
    elif prefix_pairwise_mask.shape != (batch_size, prefix_length, prefix_length):
        raise ValueError(
            "prefix_pairwise_mask must have shape "
            f"{(batch_size, prefix_length, prefix_length)}, got {tuple(prefix_pairwise_mask.shape)}"
        )
    else:
        prefix_pairwise_mask = (
            prefix_pairwise_mask.bool() & prefix_valid.bool()[:, :, None] & prefix_valid.bool()[:, None, :]
        )

    output = torch.zeros(
        batch_size,
        prefix_length + horizon,
        prefix_length + horizon,
        dtype=torch.bool,
        device=prefix_valid.device,
    )
    output[:, :prefix_length, :prefix_length] = prefix_pairwise_mask
    output[:, prefix_length:, :] = build_action_suffix_mask(prefix_valid, action_valid)
    return output


def bool_mask_to_additive(mask: Tensor, *, dtype: torch.dtype) -> Tensor:
    """Convert a true-is-allowed logical mask to an additive attention bias."""

    if mask.dtype != torch.bool:
        raise TypeError("mask must use bool dtype")
    if not torch.empty((), dtype=dtype).is_floating_point():
        raise TypeError("additive attention mask dtype must be floating point")
    zero = torch.zeros((), device=mask.device, dtype=dtype)
    blocked = torch.full((), torch.finfo(dtype).min, device=mask.device, dtype=dtype)
    return torch.where(mask, zero, blocked)


def _validate_valid_masks(prefix_valid: Tensor, action_valid: Tensor) -> None:
    if prefix_valid.ndim != 2 or action_valid.ndim != 2:
        raise ValueError("prefix_valid and action_valid must have shape [batch, positions]")
    if prefix_valid.shape[0] != action_valid.shape[0]:
        raise ValueError("prefix_valid and action_valid batch sizes must match")
    if prefix_valid.device != action_valid.device:
        raise ValueError("prefix_valid and action_valid must be on the same device")
    if bool((action_valid.bool().sum(dim=1) == 0).any()):
        raise ValueError("every batch item must contain at least one valid action")
