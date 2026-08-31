from __future__ import annotations

import numpy as np
import pytest
import torch

from duo_vla.benchmarks.common import CanonicalObservation, PaddedActionChunk
from duo_vla.data.calvin import CalvinTrainingSample
from duo_vla.data.calvin_batching import collate_calvin_samples
from duo_vla.normalization import ActionNormalizer, PercentileNormalizer


def _sample(annotation: int, frame: int, valid_length: int) -> CalvinTrainingSample:
    state = torch.tensor([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, -1.0])
    actions = torch.zeros(8, 7)
    actions[:valid_length, :6] = 0.25
    actions[:valid_length, 6] = 1.0
    actions[valid_length:, 6] = 1.0
    return CalvinTrainingSample(
        observation=CanonicalObservation(
            third_person=np.zeros((200, 200, 3), dtype=np.uint8),
            wrist=np.zeros((84, 84, 3), dtype=np.uint8),
            state=state,
        ),
        instruction="instruction",
        action_chunk=PaddedActionChunk(actions, torch.arange(8) < valid_length),
        annotation_index=annotation,
        episode_index=annotation,
        global_index=frame,
        task="task",
    )


def _state_normalizer() -> ActionNormalizer:
    return ActionNormalizer(
        PercentileNormalizer(torch.zeros(7), torch.arange(1, 8, dtype=torch.float32)),
        action_dim=8,
        gripper_index=7,
    )


def test_calvin_collation_normalizes_state_but_preserves_official_action_scaling() -> None:
    batch = collate_calvin_samples(
        (_sample(0, 0, 8), _sample(1, 9, 3)),
        state_normalizer=_state_normalizer(),
    )

    torch.testing.assert_close(batch.states[:, :7], torch.zeros(2, 7))
    torch.testing.assert_close(batch.states[:, 7], -torch.ones(2))
    torch.testing.assert_close(batch.clean_actions[0, :, :6], torch.full((8, 6), 0.25))
    torch.testing.assert_close(batch.clean_actions[1, :3, 6], torch.ones(3))
    torch.testing.assert_close(batch.clean_actions[1, 3:], torch.zeros(5, 7))


def test_calvin_collation_rejects_duplicate_annotation_frame_identity() -> None:
    sample = _sample(0, 0, 8)
    with pytest.raises(ValueError, match="distinct"):
        collate_calvin_samples((sample, sample), state_normalizer=_state_normalizer())
