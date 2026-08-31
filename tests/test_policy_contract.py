from __future__ import annotations

import copy
from pathlib import Path

import pytest
import torch

from duo_vla.flow import make_flow_training_pair
from duo_vla.objectives import make_policy_training_pair, make_seeded_policy_training_pair
from duo_vla.policy_contract import (
    DIRECT_REGRESSION,
    RECTIFIED_FLOW,
    PolicyContract,
    policy_contract_from_config,
    validate_manifest_policy_contract,
)
from duo_vla.run_config import load_resolved_toml
from duo_vla.training import combine_masked_sse, masked_sse

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _contract(name: str) -> PolicyContract:
    config_name = "libero.toml" if name == RECTIFIED_FLOW else "libero_direct_regression.toml"
    return policy_contract_from_config(load_resolved_toml(PROJECT_ROOT / "configs" / config_name))


def test_committed_flow_and_direct_configs_resolve_to_canonical_contracts() -> None:
    flow = _contract(RECTIFIED_FLOW)
    direct = _contract(DIRECT_REGRESSION)

    assert flow.objective == RECTIFIED_FLOW
    assert flow.sampler == "euler_uniform"
    assert flow.nfe == 10
    assert flow.inference_seed_behavior == "episode_identity_gaussian_noise"
    assert direct.objective == DIRECT_REGRESSION
    assert direct.training_input == "zero_action_canvas"
    assert direct.training_target == "clean_action"
    assert direct.training_timestep == "fixed_one"
    assert direct.sampler == "single_forward"
    assert direct.nfe == 1
    assert direct.inference_seed_behavior == "episode_identity_echo_only"
    assert PolicyContract.from_dict(flow.to_dict()) == flow
    assert PolicyContract.from_dict(direct.to_dict()) == direct


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value["policy"].update(extra="ambiguous"),
        lambda value: value["policy"].update(nfe=True),
        lambda value: value["action"].update(flow_steps=5),
        lambda value: value["sampling"].update(integrator="single_forward"),
        lambda value: value["sampling"].update(clip_intermediate_actions=True),
    ],
)
def test_policy_config_rejects_partial_or_mixed_semantics(mutation) -> None:
    config = copy.deepcopy(load_resolved_toml(PROJECT_ROOT / "configs" / "libero.toml"))
    mutation(config)
    with pytest.raises(ValueError):
        policy_contract_from_config(config)


def test_manifest_policy_contract_must_exactly_match_resolved_config() -> None:
    config = load_resolved_toml(PROJECT_ROOT / "configs" / "libero.toml")
    contract = policy_contract_from_config(config)
    assert validate_manifest_policy_contract({"policy_contract": contract.to_dict()}, config) == contract

    mismatched = _contract(DIRECT_REGRESSION).to_dict()
    with pytest.raises(ValueError, match="differs"):
        validate_manifest_policy_contract({"policy_contract": mismatched}, config)
    with pytest.raises((TypeError, ValueError)):
        validate_manifest_policy_contract({}, config)


def test_common_objective_pair_preserves_flow_math_and_direct_query_contract() -> None:
    clean = torch.linspace(-1.0, 1.0, 2 * 8 * 7).reshape(2, 8, 7)
    golden = make_flow_training_pair(clean, generator=torch.Generator().manual_seed(19))
    flow = make_policy_training_pair(
        clean,
        _contract(RECTIFIED_FLOW),
        generator=torch.Generator().manual_seed(19),
    )
    torch.testing.assert_close(flow.input_actions, golden.noisy_actions)
    torch.testing.assert_close(flow.timesteps, golden.timesteps)
    torch.testing.assert_close(flow.target, golden.target_velocity)
    recovered_noise = flow.input_actions - flow.timesteps[:, None, None] * clean
    recovered_noise = recovered_noise / (1.0 - flow.timesteps[:, None, None])
    torch.testing.assert_close(flow.target, clean - recovered_noise)

    generator = torch.Generator().manual_seed(31)
    before = generator.get_state().clone()
    direct = make_policy_training_pair(clean, _contract(DIRECT_REGRESSION), generator=generator)
    torch.testing.assert_close(direct.input_actions, torch.zeros_like(clean))
    torch.testing.assert_close(direct.timesteps, torch.ones(2))
    torch.testing.assert_close(direct.target, clean)
    torch.testing.assert_close(generator.get_state(), before)


def test_seeded_direct_pair_does_not_construct_a_torch_generator(monkeypatch: pytest.MonkeyPatch) -> None:
    import duo_vla.objectives

    def forbidden_generator(*args, **kwargs):
        del args, kwargs
        raise AssertionError("direct regression must not construct a random generator")

    monkeypatch.setattr(duo_vla.objectives.torch, "Generator", forbidden_generator)
    clean = torch.zeros(2, 8, 7)
    pair = make_seeded_policy_training_pair(clean, _contract(DIRECT_REGRESSION), seed=91)
    torch.testing.assert_close(pair.input_actions, torch.zeros_like(clean))


def test_direct_objective_uses_the_same_global_masked_sse_denominator() -> None:
    contract = _contract(DIRECT_REGRESSION)
    first_clean = torch.full((1, 8, 7), 99.0)
    first_clean[:, 0] = 10.0
    second_clean = torch.zeros(1, 8, 7)
    first = make_policy_training_pair(first_clean, contract)
    second = make_policy_training_pair(second_clean, contract)
    components = [
        masked_sse(
            torch.zeros_like(first.target),
            first.target,
            torch.tensor([[True, False, False, False, False, False, False, False]]),
        ),
        masked_sse(
            torch.zeros_like(second.target),
            second.target,
            torch.tensor([[True, True, True, False, False, False, False, False]]),
        ),
    ]
    combined = combine_masked_sse(components)
    assert combined.element_count == 28
    assert combined.mean.item() == pytest.approx(25.0)
