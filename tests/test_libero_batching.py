from __future__ import annotations

import numpy as np
import pytest
import torch

from duo_vla.benchmarks.common import CanonicalObservation, PaddedActionChunk
from duo_vla.data.batching import collate_libero_samples
from duo_vla.data.libero import LiberoTrainingSample
from duo_vla.normalization import ActionNormalizer, PercentileNormalizer


def _sample(episode: int, frame: int, valid_length: int) -> LiberoTrainingSample:
    actions = torch.zeros(8, 7)
    actions[:valid_length, :6] = 0.25
    actions[:valid_length, 6] = -1.0
    return LiberoTrainingSample(
        observation=CanonicalObservation(
            third_person=np.zeros((256, 256, 3), dtype=np.uint8),
            wrist=np.zeros((256, 256, 3), dtype=np.uint8),
            state=torch.full((8,), 0.25),
        ),
        instruction="task",
        action_chunk=PaddedActionChunk(
            actions=actions,
            valid_mask=torch.arange(8) < valid_length,
        ),
        episode_index=episode,
        frame_index=frame,
        task_index=0,
    )


def _normalizers() -> tuple[PercentileNormalizer, ActionNormalizer]:
    state = PercentileNormalizer(torch.zeros(8), torch.ones(8))
    action = ActionNormalizer(PercentileNormalizer(torch.zeros(6), torch.ones(6)))
    return state, action


def test_collation_zeros_padding_after_normalization() -> None:
    state, action = _normalizers()
    batch = collate_libero_samples(
        (_sample(0, 0, 8), _sample(1, 3, 3)),
        state_normalizer=state,
        action_normalizer=action,
    )
    assert batch.states.dtype == torch.float32
    assert batch.clean_actions.dtype == torch.float32
    torch.testing.assert_close(batch.clean_actions[1, 3:], torch.zeros(5, 7))
    torch.testing.assert_close(batch.clean_actions[1, :3, 6], -torch.ones(3))


def test_collation_rejects_duplicate_anchors() -> None:
    state, action = _normalizers()
    sample = _sample(0, 0, 8)
    with pytest.raises(ValueError, match="distinct"):
        collate_libero_samples((sample, sample), state_normalizer=state, action_normalizer=action)
