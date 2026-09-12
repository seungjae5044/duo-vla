from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from train_libero_sc_encoder import LR, STATE_SCHEMA, validate_state, validation_plan_groups

from duo_vla.training import make_microbatch_plan
from duo_vla.training_checkpoint import optimizer_parameter_inventory


@pytest.mark.parametrize("batch", [8, 16, 32, 64, 72, 80])
def test_validation_padding_never_changes_sample_identity_or_count(batch):
    scored = []
    for rank in range(2):
        for plans, active in validation_plan_groups(0x5A17, 2048, batch, rank, 2):
            assert len(plans) * 8 == batch
            assert 0 < active <= batch
            scored.extend(plans[: active // 8])
            assert all(p == plans[active // 8 - 1] for p in plans[active // 8 :])
    assert scored == [make_microbatch_plan(0x5A17, 0, i) for i in range(256)]


def state_fixture():
    p = torch.nn.Parameter(torch.ones(2, 3))
    optimizer = torch.optim.AdamW(
        [{"name": "interface", "params": [p]}], lr=LR, betas=(0.9, 0.95), eps=1e-8, weight_decay=1e-10
    )
    (p * 0).sum().backward()
    optimizer.step()
    inventory = optimizer_parameter_inventory(optimizer, [("action_projector.sc_projection.weight", p)])
    payload = {
        "schema": STATE_SCHEMA,
        "update": 1,
        "recipe_sha256": "recipe",
        "rank": 0,
        "world_size": 2,
        "optimizer_parameter_inventory": inventory,
        "optimizer": optimizer.state_dict(),
    }
    manifest = {"update": 1, "recipe_sha256": "recipe"}
    return payload, inventory, manifest


def test_fresh_zero_gradient_optimizer_counter_is_still_updated_and_resumable():
    state, inventory, manifest = state_fixture()
    assert validate_state(state, inventory, recipe_hash="recipe", rank=0, world=2, manifest=manifest) == 1


@pytest.mark.parametrize("change", ["counter", "lr", "rank", "recipe", "inventory", "moment"])
def test_resume_rejects_changed_identity_hyperparameters_and_state(change):
    state, inventory, manifest = state_fixture()
    state = copy.deepcopy(state)
    if change == "counter":
        state["optimizer"]["state"][0]["step"].fill_(10001)
    elif change == "lr":
        state["optimizer"]["param_groups"][0]["lr"] = 1e-3
    elif change == "rank":
        state["rank"] = 1
    elif change == "recipe":
        state["recipe_sha256"] = "different"
    elif change == "inventory":
        state["optimizer_parameter_inventory"]["groups"][0]["parameters"][0]["name"] = "different"
    else:
        state["optimizer"]["state"][0]["exp_avg"].fill_(float("nan"))
    with pytest.raises(ValueError):
        validate_state(state, inventory, recipe_hash="recipe", rank=0, world=2, manifest=manifest)
