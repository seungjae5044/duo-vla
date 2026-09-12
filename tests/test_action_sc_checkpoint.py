from __future__ import annotations

import json
from dataclasses import asdict

import pytest
from test_checkpointing import FakeAdapter
from torch import nn

from duo_vla.action_sc_checkpoint import RUN_SCHEMA, SC_CONTRACT, canonical_sha256, read_sc_checkpoint
from duo_vla.checkpointing import load_checkpoint_manifest, save_trainable_checkpoint
from duo_vla.config import ActionInterfaceConfig
from duo_vla.self_conditioning import SC_VERSION


def fixture(tmp_path, *, change=None):
    config = asdict(ActionInterfaceConfig(hidden_size=2816, state_dim=8, self_conditioning=SC_VERSION))
    encoder = {"rank": 16, "alpha": 32, "targets": [f"target_{i}" for i in range(113)], "vision_frozen": True}
    recipe = {
        "schema": RUN_SCHEMA,
        "self_conditioning": SC_CONTRACT,
        "initialization": "pretrained_base_fresh_decoder_encoder_interface_optimizer",
        "total_updates": 7500,
        "learning_rate": 5e-5,
        "vision_frozen": True,
        "base_weights_frozen": True,
        "world_size": 2,
        "physical_batch_size": 64,
        "global_batch_size": 128,
        "action_interface_config": config,
        "model": {"id": "model", "revision": "revision"},
        "artifact_sha256": {},
        "source_files_sha256": {},
    }
    if change == "lr":
        recipe["learning_rate"] = 1e-3
    elif change == "batch":
        recipe["global_batch_size"] = 64
    elif change == "scratch":
        recipe["initialization"] = "inherited"
    elif change == "signal":
        recipe["self_conditioning"] = {**SC_CONTRACT, "signal": "ground_truth_action"}
    elif change == "vision":
        recipe["vision_frozen"] = False
    elif change == "interface":
        recipe["action_interface_config"] = {**config, "self_conditioning": "none"}
    elif change == "encoder":
        encoder["targets"] = []
    extras = {}
    for name, value in (
        ("recipe", recipe),
        ("action_interface_config", config),
        ("encoder_config", encoder),
        ("normalization", {}),
        ("prefix_geometry", {}),
        ("encoder_adapter", {}),
    ):
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(value))
        extras[name] = path
    checkpoint = tmp_path / "checkpoint"
    save_trainable_checkpoint(
        checkpoint,
        adapted_model=FakeAdapter(),
        interface_modules={"head": nn.Linear(2, 2)},
        manifest={
            "kind": RUN_SCHEMA,
            "action_self_conditioning": SC_VERSION,
            "recipe_sha256": canonical_sha256(recipe),
            "encoder_adapted": True,
            "training_suites_json": '["libero_spatial"]',
            "update": 2,
            "model_id": "model",
            "model_revision": "revision",
        },
        additional_artifacts=extras,
    )
    return checkpoint


def test_recipe_interface_encoder_contract_roundtrip(tmp_path):
    checkpoint = fixture(tmp_path)
    manifest, recipe, config, encoder = read_sc_checkpoint(checkpoint)
    assert manifest["update"] == 2 and recipe["total_updates"] == 7500
    assert config.self_conditioning == SC_VERSION and len(encoder["targets"]) == 113


@pytest.mark.parametrize("change", ["lr", "batch", "scratch", "signal", "vision", "interface", "encoder"])
def test_authenticated_but_incompatible_metadata_is_rejected(tmp_path, change):
    checkpoint = fixture(tmp_path, change=change)
    with pytest.raises(ValueError):
        read_sc_checkpoint(checkpoint)


def test_artifact_corruption_is_rejected_before_use(tmp_path):
    checkpoint = fixture(tmp_path)
    manifest = load_checkpoint_manifest(checkpoint)
    (checkpoint / manifest["artifacts"]["recipe"]["path"]).write_text("{}")
    with pytest.raises(ValueError):
        read_sc_checkpoint(checkpoint)
