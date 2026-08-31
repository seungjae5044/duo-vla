from __future__ import annotations

import torch

from duo_vla.normalization import ActionNormalizer, PercentileNormalizer


def test_percentile_normalization_round_trip_and_clipping() -> None:
    normalizer = PercentileNormalizer(lower=torch.tensor([-2.0, 10.0]), upper=torch.tensor([2.0, 20.0]))
    values = torch.tensor([[-3.0, 12.5], [1.0, 30.0]])
    normalized = normalizer.normalize(values)
    torch.testing.assert_close(normalized, torch.tensor([[-1.0, -0.5], [0.5, 1.0]]))
    restored = normalizer.unnormalize(normalized)
    torch.testing.assert_close(restored, torch.tensor([[-2.0, 12.5], [1.0, 20.0]]))


def test_constant_dimension_maps_to_zero_and_back() -> None:
    normalizer = PercentileNormalizer(lower=torch.tensor([4.0]), upper=torch.tensor([4.0]))
    normalized = normalizer.normalize(torch.tensor([[4.0], [100.0]]))
    torch.testing.assert_close(normalized, torch.zeros_like(normalized))
    torch.testing.assert_close(normalizer.unnormalize(normalized), torch.full_like(normalized, 4.0))


def test_fit_ignores_padded_values() -> None:
    values = torch.tensor([[[0.0], [1.0], [1000.0]]])
    valid = torch.tensor([[True, True, False]])
    normalizer = PercentileNormalizer.fit(
        values,
        valid_mask=valid,
        lower_quantile=0.0,
        upper_quantile=1.0,
    )
    torch.testing.assert_close(normalizer.lower, torch.tensor([0.0]))
    torch.testing.assert_close(normalizer.upper, torch.tensor([1.0]))


def test_action_normalizer_thresholds_gripper_with_zero_tie_break() -> None:
    continuous = PercentileNormalizer(lower=torch.full((6,), -2.0), upper=torch.full((6,), 2.0))
    normalizer = ActionNormalizer(continuous=continuous)
    raw = torch.tensor([[[0.0, 1.0, -1.0, 2.0, -2.0, 0.5, 0.0]]])
    normalized = normalizer.normalize(raw)
    assert normalized[..., -1].item() == 1.0
    restored = normalizer.unnormalize(normalized)
    torch.testing.assert_close(restored[..., :6], raw[..., :6])
    assert restored[..., -1].item() == 1.0
