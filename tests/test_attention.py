from __future__ import annotations

import pytest
import torch

from duo_vla.attention import build_action_suffix_mask, build_prefix_action_block_mask


def test_block_mask_has_no_prefix_to_action_edges_and_masks_padding() -> None:
    prefix = torch.tensor([[True, True, False]])
    action = torch.tensor([[True, True, False]])
    mask = build_prefix_action_block_mask(prefix, action)

    assert mask.shape == (1, 6, 6)
    assert not bool(mask[0, :3, 3:].any())
    assert bool(mask[0, 3, :2].all())
    assert bool(mask[0, 3, 3:5].all())
    assert not bool(mask[0, :, 2].any())
    assert not bool(mask[0, :, 5].any())
    assert not bool(mask[0, 5].any())


def test_action_suffix_mask_rejects_all_padded_sample() -> None:
    with pytest.raises(ValueError, match="at least one valid action"):
        build_action_suffix_mask(torch.ones(1, 2, dtype=torch.bool), torch.zeros(1, 3, dtype=torch.bool))
