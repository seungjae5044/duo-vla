from __future__ import annotations

import pytest
import torch

from duo_vla.training import (
    TRAINER_STATE_SCHEMA,
    TrainerState,
    combine_masked_sse,
    make_microbatch_plan,
    make_update_plan,
    masked_element_count,
    masked_sse,
)


def test_microbatch_plans_are_stateless_and_domain_separated() -> None:
    expected = make_microbatch_plan(seed=1729, update=4, microstep=2)
    assert make_microbatch_plan(seed=1729, update=4, microstep=2) == expected
    assert (expected.data_seed, expected.flow_seed) == (3068469086136291493, 4588896706338224119)
    assert expected.data_seed != expected.flow_seed

    variants = {
        make_microbatch_plan(seed=1730, update=4, microstep=2),
        make_microbatch_plan(seed=1729, update=5, microstep=2),
        make_microbatch_plan(seed=1729, update=4, microstep=3),
    }
    assert expected not in variants
    assert make_update_plan(1729, 4, gradient_accumulation_steps=3) == tuple(
        make_microbatch_plan(1729, 4, microstep) for microstep in range(3)
    )


def test_resumed_state_produces_the_same_next_update_plans() -> None:
    seed = 90210
    uninterrupted = tuple(make_update_plan(seed, update, gradient_accumulation_steps=3) for update in range(5))

    state = TrainerState()
    state = state.advance(examples=12)
    state = state.advance(examples=9)
    resumed = TrainerState.from_json(state.to_json())

    assert resumed == TrainerState(next_update=2, examples_seen=21)
    assert (
        tuple(make_update_plan(seed, update, gradient_accumulation_steps=3) for update in range(resumed.next_update, 5))
        == uninterrupted[2:]
    )


def test_masked_sse_uses_the_exact_cross_microbatch_denominator() -> None:
    one_valid = masked_sse(
        torch.tensor([[[10.0]]]),
        torch.zeros(1, 1, 1),
        torch.tensor([[True]]),
    )
    three_valid = masked_sse(
        torch.zeros(1, 3, 1),
        torch.zeros(1, 3, 1),
        torch.tensor([[True, True, True]]),
    )

    naive_mean_of_means = (one_valid.mean + three_valid.mean) / 2
    exact = combine_masked_sse([one_valid, three_valid])

    assert naive_mean_of_means.item() == pytest.approx(50.0)
    assert exact.element_count == 4
    assert exact.mean.item() == pytest.approx(25.0)
    assert exact.mean.item() != pytest.approx(naive_mean_of_means.item())


def test_per_microbatch_backward_matches_one_full_masked_loss() -> None:
    first_prediction = torch.tensor([[[2.0]]], requires_grad=True)
    second_prediction = torch.tensor([[[3.0], [4.0], [5.0]]], requires_grad=True)
    first_mask = torch.tensor([[True]])
    second_mask = torch.tensor([[True, False, True]])
    first = masked_sse(first_prediction, torch.zeros_like(first_prediction), first_mask)
    second = masked_sse(second_prediction, torch.zeros_like(second_prediction), second_mask)
    total_elements = first.element_count + second.element_count

    first.loss_for_total(total_elements).backward()
    second.loss_for_total(total_elements).backward()

    full_prediction = torch.tensor([[[2.0], [3.0], [4.0], [5.0]]], requires_grad=True)
    full = masked_sse(
        full_prediction,
        torch.zeros_like(full_prediction),
        torch.tensor([[True, True, False, True]]),
    )
    full.mean.backward()

    torch.testing.assert_close(first_prediction.grad, full_prediction.grad[:, :1])
    torch.testing.assert_close(second_prediction.grad, full_prediction.grad[:, 1:])


def test_trainer_state_serialization_is_canonical_and_strict() -> None:
    state = TrainerState(next_update=7, examples_seen=205)
    encoded = state.to_json()
    assert encoded == ('{"examples_seen":205,"next_update":7,"schema":"duo-vla-trainer-state-v1"}')
    assert TrainerState.from_json(encoded) == state
    assert state.to_dict() == {
        "schema": TRAINER_STATE_SCHEMA,
        "next_update": 7,
        "examples_seen": 205,
    }


@pytest.mark.parametrize(
    "value",
    [
        {"schema": TRAINER_STATE_SCHEMA, "next_update": 1},
        {
            "schema": TRAINER_STATE_SCHEMA,
            "next_update": 1,
            "examples_seen": 2,
            "extra": 3,
        },
        {"schema": "future-state", "next_update": 1, "examples_seen": 2},
        {"schema": TRAINER_STATE_SCHEMA, "next_update": True, "examples_seen": 2},
        {"schema": TRAINER_STATE_SCHEMA, "next_update": 1.0, "examples_seen": 2},
        {"schema": TRAINER_STATE_SCHEMA, "next_update": -1, "examples_seen": 2},
        {"schema": TRAINER_STATE_SCHEMA, "next_update": 1, "examples_seen": "2"},
    ],
)
def test_trainer_state_rejects_noncanonical_mappings(value: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        TrainerState.from_dict(value)


def test_trainer_state_rejects_malformed_or_duplicate_json() -> None:
    with pytest.raises(ValueError, match="valid JSON"):
        TrainerState.from_json("{")
    with pytest.raises(ValueError, match="duplicate JSON key"):
        TrainerState.from_json(
            '{"schema":"duo-vla-trainer-state-v1","next_update":1,"next_update":2,"examples_seen":3}'
        )


def test_training_primitives_reject_invalid_counts_and_masks() -> None:
    with pytest.raises(ValueError, match="positive integer"):
        make_update_plan(1, 2, gradient_accumulation_steps=0)
    with pytest.raises(ValueError, match="seed"):
        make_microbatch_plan(True, 1, 0)
    with pytest.raises(ValueError, match="no valid"):
        masked_sse(
            torch.zeros(1, 2, 3),
            torch.zeros(1, 2, 3),
            torch.zeros(1, 2, dtype=torch.bool),
        )
    with pytest.raises(ValueError, match="bool shape"):
        masked_element_count(torch.ones(1, 2), action_dim=3)
    with pytest.raises(ValueError, match="must not be empty"):
        combine_masked_sse([])
    with pytest.raises(ValueError, match="covering"):
        masked_sse(
            torch.ones(1, 1, 1),
            torch.zeros(1, 1, 1),
            torch.ones(1, 1, dtype=torch.bool),
        ).loss_for_total(0)
    with pytest.raises(ValueError, match="positive integer"):
        TrainerState().advance(examples=0)
