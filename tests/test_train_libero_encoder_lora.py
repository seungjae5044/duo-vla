from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import train_libero_encoder_lora as TRAIN

from duo_vla.training_checkpoint import optimizer_parameter_inventory


def continuation_fixture():
    params = [(name, torch.nn.Parameter(torch.tensor([1.0]))) for name in ("lora", "interface", "encoder")]
    optimizer = torch.optim.AdamW(
        [{"name": name, "params": [p], "initial_lr": TRAIN.LR} for name, p in params],
        lr=TRAIN.LR,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=1e-10,
    )
    for _, p in params:
        p.grad = torch.ones_like(p)
    optimizer.step()
    for name, p in params:
        optimizer.state[p]["step"].fill_(3 if name == "encoder" else 15003)
    inventory = optimizer_parameter_inventory(optimizer, params)
    payload = {
        "schema": "duo-vla-prefix-lora-state-v1",
        "stage_update": 3,
        "absolute_update": 15003,
        "recipe_sha256": "recipe",
        "rank": 0,
        "world_size": 2,
        "optimizer_parameter_inventory": inventory,
        "optimizer": optimizer.state_dict(),
    }
    kwargs = {
        "recipe_hash": "recipe",
        "rank": 0,
        "world": 2,
        "manifest": {"stage_update": 3, "absolute_update": 15003},
        "optimization": {"adam_beta1": 0.9, "adam_beta2": 0.95, "adam_epsilon": 1e-8, "weight_decay": 1e-10},
    }
    return payload, inventory, kwargs


def test_resume_validates_independent_encoder_and_inherited_adam_counters():
    payload, inventory, kwargs = continuation_fixture()
    assert TRAIN.validate_continuation_state(payload, inventory, **kwargs) == 3


@pytest.mark.parametrize("corruption", ["rank", "stage", "lr", "encoder_step", "decoder_step", "moment", "order"])
def test_resume_rejects_changed_recipe_or_optimizer(corruption):
    payload, inventory, kwargs = continuation_fixture()
    if corruption == "rank":
        payload["rank"] = 1
    elif corruption == "stage":
        payload["stage_update"] = 4
    elif corruption == "lr":
        payload["optimizer"]["param_groups"][2]["lr"] = 1e-4
    elif corruption == "encoder_step":
        payload["optimizer"]["state"][2]["step"].fill_(15003)
    elif corruption == "decoder_step":
        payload["optimizer"]["state"][0]["step"].fill_(3)
    elif corruption == "moment":
        payload["optimizer"]["state"][1]["exp_avg"].fill_(torch.nan)
    else:
        payload["optimizer"]["param_groups"][0]["params"] = [1]
    with pytest.raises(ValueError):
        TRAIN.validate_continuation_state(payload, inventory, **kwargs)


def test_parent_optimizer_preserves_moments_but_uses_constant_lr():
    params = [(name, torch.nn.Parameter(torch.tensor([1.0]))) for name in ("lora", "interface")]
    optimizer = torch.optim.AdamW(
        [{"name": name, "params": [p]} for name, p in params],
        lr=1e-4,
    )
    for _, p in params:
        p.grad = torch.ones_like(p)
    optimizer.step()
    for _, p in params:
        optimizer.state[p]["step"].fill_(15000)
    payload = {
        "optimizer": copy.deepcopy(optimizer.state_dict()),
        "optimizer_parameter_inventory": optimizer_parameter_inventory(optimizer, params),
    }
    TRAIN.restore_parent_optimizer(optimizer, params, payload)
    assert all(group["lr"] == group["initial_lr"] == TRAIN.LR for group in optimizer.param_groups)
    for index, (_, p) in enumerate(params):
        for key, value in optimizer.state[p].items():
            torch.testing.assert_close(value, payload["optimizer"]["state"][index][key], rtol=0, atol=0)
