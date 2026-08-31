from __future__ import annotations

from duo_vla.experiments.synthetic_overfit import run_fixed_batch_overfit


def test_tiny_fixed_batch_loss_decreases() -> None:
    result = run_fixed_batch_overfit(steps=20, seed=7, device="cpu", batch_size=4, hidden_size=16)
    assert result.final_loss < result.initial_loss
