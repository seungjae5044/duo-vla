"""Fail-closed identity and architecture contracts for fresh Spatial action-SC runs."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

from duo_vla.checkpointing import load_checkpoint_manifest
from duo_vla.config import ActionInterfaceConfig
from duo_vla.self_conditioning import SC_VERSION

RUN_SCHEMA = "duo-vla-spatial-sc-encoder-scratch-v1"
STATE_SCHEMA = "duo-vla-spatial-sc-encoder-training-state-v1"
SC_CONTRACT = {
    "version": SC_VERSION,
    "signal": "detached_normalized_clean_endpoint",
    "features": "per_slot_action_plus_per_sample_presence_bit",
    "injection": "zero_initialized_biasless_linear_added_before_native_post_norm",
    "bootstrap_probability": 0.5,
    "bootstrap_scope": "optimizer_update_shared_all_ranks_and_microbatches",
    "bootstrap_pair": "same_action_time_observation_prefix",
    "rng": "sha256_duo-vla-sc-v1_seed_update_low_bit",
    "inference": "previous_step_preupdate_endpoint_reset_each_query",
    "integration": "fp32_uniform_index_over_nfe_velocity_div_nfe",
    "intermediate_clipping": False,
}
CRITICAL_SOURCES = (
    "src/duo_vla/action_interface.py",
    "src/duo_vla/modeling.py",
    "src/duo_vla/self_conditioning.py",
    "src/duo_vla/backbones/diffusion_gemma.py",
    "src/duo_vla/backbones/encoder_lora.py",
    "src/duo_vla/backbones/sample_isolated_experts.py",
    "src/duo_vla/backbones/sample_isolated_experts_v2.py",
    "src/duo_vla/backbones/shared_weight_grouped_mm_triton.py",
    "src/duo_vla/backbones/loading.py",
    "src/duo_vla/normalization.py",
    "src/duo_vla/prefix_geometry.py",
    "scripts/rollout_libero_encoder_lora.py",
)


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def read_sc_checkpoint(checkpoint: Path, *, source_root: Path | None = None):
    manifest = load_checkpoint_manifest(checkpoint)
    if manifest.get("kind") != RUN_SCHEMA or manifest.get("action_self_conditioning") != SC_VERSION:
        raise ValueError("not a versioned action-SC scratch checkpoint")
    artifacts = manifest["artifacts"]
    required = {
        "recipe",
        "action_interface_config",
        "encoder_config",
        "encoder_adapter",
        "normalization",
        "prefix_geometry",
    }
    if not required.issubset(artifacts):
        raise ValueError("SC checkpoint is missing required architecture/data artifacts")

    def read(name):
        return json.loads((checkpoint / artifacts[name]["path"]).read_text())

    recipe = read("recipe")
    interface = read("action_interface_config")
    encoder = read("encoder_config")
    if (
        recipe.get("schema") != RUN_SCHEMA
        or canonical_sha256(recipe) != manifest.get("recipe_sha256")
        or recipe.get("self_conditioning") != SC_CONTRACT
        or recipe.get("initialization") != "pretrained_base_fresh_decoder_encoder_interface_optimizer"
        or recipe.get("total_updates") != 7500
        or recipe.get("learning_rate") != 5e-5
        or recipe.get("vision_frozen") is not True
        or recipe.get("base_weights_frozen") is not True
        or recipe.get("world_size") != 2
        or recipe.get("physical_batch_size") not in (8, 16, 32, 64, 72, 80)
        or recipe.get("global_batch_size") != 2 * recipe["physical_batch_size"]
        or manifest.get("encoder_adapted") is not True
        or json.loads(manifest.get("training_suites_json", "[]")) != ["libero_spatial"]
    ):
        raise ValueError("SC checkpoint recipe identity/semantics mismatch")
    config = ActionInterfaceConfig(**interface)
    if (
        asdict(config) != interface
        or interface != recipe.get("action_interface_config")
        or config.self_conditioning != SC_VERSION
        or (config.hidden_size, config.state_dim, config.action_horizon, config.action_dim) != (2816, 8, 8, 7)
        or encoder.get("rank") != 16
        or encoder.get("alpha") != 32
        or len(encoder.get("targets", [])) != 113
        or encoder.get("vision_frozen") is not True
    ):
        raise ValueError("SC checkpoint architecture mismatch")
    if (
        manifest.get("model_id") != recipe["model"]["id"]
        or manifest.get("model_revision") != recipe["model"]["revision"]
        or any(artifacts[name]["sha256"] != digest for name, digest in recipe["artifact_sha256"].items())
    ):
        raise ValueError("SC checkpoint model/data artifacts differ from the recipe")
    update = manifest.get("update")
    if type(update) is not int or not 0 < update <= 7500:
        raise ValueError("SC checkpoint update is invalid")
    if source_root is not None:
        for name in CRITICAL_SOURCES:
            if file_sha256(source_root / name) != recipe["source_files_sha256"].get(name):
                raise ValueError(f"SC policy source changed: {name}")
    return manifest, recipe, config, encoder
