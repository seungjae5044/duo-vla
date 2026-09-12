#!/usr/bin/env python3
"""Explicit encoder-aware, one-replica-per-GPU Spatial subset evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from multiprocessing.connection import Listener
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rollout_libero_spatial import (  # noqa: E402
    BATCH_SIZE,
    decode_image,
    receive,
    run_evaluation,
    send,
    sha256,
    write_json,
)

PARENT_HASH = "f03384d957778a0eac49c98bf5d37c7c167e7a318b1a29958a0832ee799cfb23"
POLICY_SOURCES = (
    "src/duo_vla/backbones/encoder_lora.py",
    "src/duo_vla/backbones/diffusion_gemma.py",
    "src/duo_vla/backbones/loading.py",
    "src/duo_vla/backbones/sample_isolated_experts.py",
    "src/duo_vla/backbones/sample_isolated_experts_v2.py",
    "src/duo_vla/backbones/shared_weight_grouped_mm_triton.py",
    "src/duo_vla/action_interface.py",
    "src/duo_vla/modeling.py",
    "src/duo_vla/normalization.py",
    "src/duo_vla/prefix_geometry.py",
)


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def validate_stage_metadata(manifest, recipe, encoder_config, *, final=True):
    if (
        manifest.get("kind") != "experimental-libero-prefix-lora-continuation"
        or manifest.get("encoder_adapted") is not True
        or manifest.get("parent_manifest_sha256") != PARENT_HASH
        or manifest.get("recipe_sha256") != canonical_hash(recipe)
        or json.loads(manifest.get("training_suites_json", "null")) != ["libero_spatial"]
    ):
        raise ValueError("checkpoint is not the authorized encoder-adapted Spatial continuation")
    required = {
        "schema": "duo-vla-prefix-lora-continuation-v1",
        "parent_manifest_sha256": PARENT_HASH,
        "base_update": 15000,
        "additional_updates": 1000,
        "learning_rate": 5e-5,
        "schedule": "constant_no_warmup",
        "vision_frozen": True,
        "base_weights_frozen": True,
        "world_size": 2,
        "physical_batch_size": 64,
        "global_batch_size": 128,
        "encoder_lora_rank": 16,
        "encoder_lora_alpha": 32,
    }
    if any(recipe.get(key) != value for key, value in required.items()):
        raise ValueError("continuation recipe changed")
    stage = manifest.get("stage_update")
    if type(stage) is not int or not 0 < stage <= 1000 or manifest.get("absolute_update") != 15000 + stage:
        raise ValueError("continuation update identity is invalid")
    if final and stage != 1000:
        raise ValueError("rollout must wait for all 1000 additional updates")
    if (
        encoder_config.get("rank") != 16
        or encoder_config.get("alpha") != 32
        or encoder_config.get("scope") != "prefix_language_attention"
        or encoder_config.get("vision_frozen") is not True
    ):
        raise ValueError("encoder adapter configuration changed")


def read_stage_checkpoint(checkpoint, expected_hash, *, final=True):
    from duo_vla.checkpointing import load_checkpoint_manifest

    if sha256(checkpoint / "manifest.json") != expected_hash:
        raise ValueError("checkpoint manifest does not match the expected hash")
    manifest = load_checkpoint_manifest(checkpoint)
    for name, path in {
        "encoder_adapter": "artifacts/encoder_adapter.safetensors",
        "encoder_config": "artifacts/encoder_config.json",
        "recipe": "artifacts/recipe.json",
        "resolved_config": "artifacts/resolved_config.json",
        "normalization": "artifacts/normalization.json",
        "prefix_geometry": "artifacts/prefix_geometry.json",
        "interface": "interface.safetensors",
    }.items():
        if manifest["artifacts"].get(name, {}).get("path") != path:
            raise ValueError(f"noncanonical or missing checkpoint artifact: {name}")
    recipe = json.loads((checkpoint / "artifacts/recipe.json").read_text())
    encoder_config = json.loads((checkpoint / "artifacts/encoder_config.json").read_text())
    validate_stage_metadata(manifest, recipe, encoder_config, final=final)
    for path in POLICY_SOURCES:
        if sha256(ROOT / path) != recipe["source_files_sha256"].get(path):
            raise ValueError(f"policy implementation differs from training: {path}")
    return manifest, recipe, encoder_config


class EncoderSpatialPolicy:
    def __init__(self, checkpoint, expected_hash, *, action_sc=False, nfe=4):
        import torch
        from safetensors.torch import load_file
        from transformers import AutoProcessor

        from duo_vla.action_interface import ActionInputProjector, VelocityHead
        from duo_vla.backbones.diffusion_gemma import DiffusionGemmaActionDecoder
        from duo_vla.backbones.encoder_lora import install_encoder_lora, load_encoder_adapter_state
        from duo_vla.backbones.loading import load_diffusion_gemma_bf16_tp
        from duo_vla.backbones.sample_isolated_experts_v2 import (
            install_sample_isolated_grouped_mm_experts_v2,
            verify_sample_isolated_grouped_mm_experts_v2,
        )
        from duo_vla.checkpointing import load_interface_state_dict, load_lora_checkpoint
        from duo_vla.config import ActionInterfaceConfig
        from duo_vla.data.libero_stats import load_libero_normalizers
        from duo_vla.modeling import DuoVLADenoiser
        from duo_vla.policy_contract import validate_manifest_policy_contract

        if os.environ.get("CUDA_VISIBLE_DEVICES") not in {"0", "1"} or torch.cuda.device_count() != 1:
            raise RuntimeError("each server must use exactly physical GPU 0 or 1")
        torch.cuda.set_device(0)
        self.torch, self.device = torch, torch.device("cuda:0")
        if "A6000" in torch.cuda.get_device_name(0):
            raise RuntimeError("A6000 use is prohibited")
        self.nfe = nfe
        if type(nfe) is not int or nfe <= 0:
            raise ValueError("NFE must be a positive integer")
        sc_interface = None
        if action_sc:
            from duo_vla.action_sc_checkpoint import read_sc_checkpoint

            if sha256(checkpoint / "manifest.json") != expected_hash:
                raise ValueError("SC checkpoint manifest hash differs")
            self.manifest, self.recipe, sc_interface, encoder_config = read_sc_checkpoint(checkpoint, source_root=ROOT)
        else:
            self.manifest, self.recipe, encoder_config = read_stage_checkpoint(checkpoint, expected_hash)
        # Low-precision encoder projections can differ across physical GEMM widths.
        # The SC recipe therefore keeps training geometry even for a singleton query.
        self.physical_batch_size = self.recipe["physical_batch_size"] if action_sc else BATCH_SIZE
        resolved = json.loads((checkpoint / "artifacts/resolved_config.json").read_text())
        self.config = config = resolved["config"]
        if canonical_hash(config) != self.manifest["config_sha256"]:
            raise ValueError("inherited base configuration changed")
        contract = validate_manifest_policy_contract(self.manifest, config)
        if contract.objective != "rectified_flow" or contract.action_horizon != 8 or contract.action_dim != 7:
            raise ValueError("expected H8 continuous rectified-flow policy")
        self.prefix_geometry = json.loads((checkpoint / "artifacts/prefix_geometry.json").read_text())
        self.valid_lengths = {
            record["instruction"]: record["valid_prefix_length"]
            for record in self.prefix_geometry["instruction_inventory"]["records"]
        }
        self.processor = AutoProcessor.from_pretrained(
            config["model"]["id"],
            revision=config["model"]["revision"],
            local_files_only=True,
        )
        self.model = load_diffusion_gemma_bf16_tp(tp_size=1, local_files_only=True)
        install_sample_isolated_grouped_mm_experts_v2(self.model, physical_batch_size=self.physical_batch_size)
        self.adapted, _ = load_lora_checkpoint(
            checkpoint,
            self.model,
            validate_decoder_contract=True,
            allow_encoder_adapter=True,
            allow_action_self_conditioning=action_sc,
        )
        targets = install_encoder_lora(self.model)
        if list(targets) != encoder_config["targets"] or len(targets) != 113:
            raise ValueError("encoder projection inventory changed")
        load_encoder_adapter_state(self.model, load_file(str(checkpoint / "artifacts/encoder_adapter.safetensors")))
        self.adapted.requires_grad_(False).eval()
        if any(parameter.requires_grad for parameter in self.model.parameters()):
            raise RuntimeError("inference model must be completely frozen")
        verify_sample_isolated_grouped_mm_experts_v2(self.model, physical_batch_size=self.physical_batch_size)
        action = config["action"]
        interface = sc_interface or ActionInterfaceConfig(
            hidden_size=2816,
            state_dim=8,
            action_horizon=8,
            action_dim=7,
            timestep_embedding_dim=action["timestep_embedding_dimension"],
            timestep_scale=action["timestep_scale"],
            timestep_max_period=action["timestep_max_period"],
            output_init_std=action["output_head_initialization_std"],
        )
        projector = ActionInputProjector(interface).to(self.device)
        head = VelocityHead(2816, 7).to(self.device)
        load_interface_state_dict(
            checkpoint / "interface.safetensors",
            {
                "action_projector": projector,
                "velocity_head": head,
            },
        )
        self.denoiser = (
            DuoVLADenoiser(
                projector,
                DiffusionGemmaActionDecoder.from_block_diffusion_model(self.model),
                head,
            )
            .requires_grad_(False)
            .eval()
        )
        state_norm, self.action_norm, _ = load_libero_normalizers(checkpoint / "artifacts/normalization.json")
        self.state_norm = state_norm.to(device=self.device, dtype=torch.float32)

    def predict(self, rows, *, record=True, return_normalized=False):
        from PIL import Image

        from duo_vla.backbones.diffusion_gemma import encode_diffusion_gemma_prefix
        from duo_vla.prefix_geometry import apply_fixed_prefix_chat_template
        from duo_vla.self_conditioning import enabled, sample_action_flow

        torch = self.torch
        physical_batch_size = self.physical_batch_size
        if not 1 <= len(rows) <= physical_batch_size:
            raise ValueError(f"expected one to {physical_batch_size} observations")
        active = len(rows)
        padded = rows + [rows[-1]] * (physical_batch_size - active)
        conversations = [
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": Image.fromarray(decode_image(row["agentview"]))},
                        {"type": "image", "image": Image.fromarray(decode_image(row["wrist"]))},
                        {"type": "text", "text": row["instruction"]},
                    ],
                }
            ]
            for row in padded
        ]
        if any(row["instruction"] not in self.valid_lengths for row in rows):
            raise ValueError("instruction absent from the pinned prefix geometry")
        torch.cuda.synchronize(self.device)
        started = time.perf_counter()
        with torch.inference_mode():
            inputs = apply_fixed_prefix_chat_template(
                self.processor,
                conversations,
                fixed_physical_prefix_width=self.config["benchmark"]["fixed_physical_prefix_width"],
                padding_side=self.prefix_geometry["tokenization"]["padding_side"],
                expected_batch_size=physical_batch_size,
                images_per_prefix=2,
            ).to(self.device)
            if inputs["attention_mask"].sum(dim=1).tolist() != [
                self.valid_lengths[row["instruction"]] for row in padded
            ]:
                raise ValueError("prefix token geometry changed")
            # Match training: BF16 frozen encoder with FP32 router/LoRA math,
            # outside autocast. Only the action decoder uses BF16 autocast.
            prefix = encode_diffusion_gemma_prefix(self.model, inputs)
            states = self.state_norm.normalize(torch.tensor([row["state"] for row in padded], device=self.device))
            actions = torch.cat(
                [
                    torch.randn(
                        (1, 8, 7),
                        device=self.device,
                        dtype=torch.float32,
                        generator=torch.Generator(device=self.device).manual_seed(row["inference_seed"]),
                    )
                    for row in padded
                ]
            )
            mask = torch.ones((physical_batch_size, 8), dtype=torch.bool, device=self.device)
            if enabled(self.denoiser):
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    actions = sample_action_flow(
                        self.denoiser,
                        actions,
                        states,
                        num_steps=self.nfe,
                        prefix_cache=prefix.past_key_values,
                        prefix_attention_mask=prefix.attention_mask,
                        action_valid_mask=mask,
                    )
            else:
                for step in range(self.nfe):
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        velocity = self.denoiser(
                            actions,
                            torch.full((physical_batch_size,), step / self.nfe, device=self.device),
                            states,
                            prefix_cache=prefix.past_key_values,
                            prefix_attention_mask=prefix.attention_mask,
                            action_valid_mask=mask,
                        )
                    actions = actions + velocity.float() / self.nfe
            if not actions.isfinite().all():
                raise FloatingPointError("nonfinite policy actions")
            clip = (actions.abs() > 1).float().mean(dim=(1, 2))[:active].cpu().tolist()
            output = self.action_norm.unnormalize(actions.clamp(-1, 1))[:active].float().cpu().tolist()
            raw = actions[:active].float().cpu().tolist() if return_normalized else None
        torch.cuda.synchronize(self.device)
        result = {"actions": output, "clip_fraction": clip, "seconds": time.perf_counter() - started}
        if return_normalized:
            result["raw_normalized_actions"] = raw
        return result


def run_server(args):
    import torch

    from duo_vla.runtime_determinism import configure_strict_cuda_determinism

    configure_strict_cuda_determinism(torch)
    args.output.mkdir(parents=True, exist_ok=False)
    action_sc = getattr(args, "action_self_conditioning", False)
    policy = EncoderSpatialPolicy(args.checkpoint, args.expected_manifest_sha256, action_sc=action_sc, nfe=args.nfe)
    write_json(
        args.output / "serving.json",
        {
            "schema": "duo-vla-sc-encoder-spatial-serving-v1"
            if action_sc
            else "duo-vla-prefix-lora-spatial-serving-v1",
            "action_self_conditioning": "action_endpoint_v1" if action_sc else "none",
            "nfe": args.nfe,
            "checkpoint": str(args.checkpoint),
            "checkpoint_manifest_sha256": args.expected_manifest_sha256,
            "encoder_adapter_sha256": policy.manifest["artifacts"]["encoder_adapter"]["sha256"],
            "encoder_targets": 113,
            "encoder_adapter_loaded": True,
            "vision_frozen": True,
            "absolute_update": policy.manifest["update"] if action_sc else policy.manifest["absolute_update"],
            "stage_update": policy.manifest["update"] if action_sc else policy.manifest["stage_update"],
            "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
            "serving_topology": "one full TP1 replica per GPU; independent disjoint episode shards",
            "physical_batch_size": policy.physical_batch_size,
            "prefix_autocast": False,
            "decoder_autocast": "bfloat16",
            "routing_instrumented": False,
            "torch": torch.__version__,
            "source_files_sha256": {
                str(p.relative_to(ROOT)): sha256(p)
                for folder in (ROOT / "src", ROOT / "scripts")
                for p in sorted(folder.rglob("*.py"))
            },
        },
    )
    with Listener(str(args.socket), family="AF_UNIX", authkey=args.authkey.read_bytes()) as listener:
        os.chmod(args.socket, 0o600)
        print(json.dumps({"status": "ready", "checkpoint": str(args.checkpoint)}), flush=True)
        with listener.accept() as connection:
            while True:
                message = receive(connection)
                if message["operation"] == "shutdown":
                    send(connection, {"status": "stopped"})
                    break
                if message["operation"] == "routing":
                    send(connection, {"status": "disabled"})
                    continue
                if message["operation"] != "predict":
                    raise ValueError("unsupported policy operation")
                send(connection, policy.predict(message["rows"], record=message.get("record", True)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("server", "evaluate"))
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--expected-manifest-sha256")
    parser.add_argument("--socket", type=Path, required=True)
    parser.add_argument("--authkey", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shard-index", type=int, choices=(0, 1), required=True)
    parser.add_argument(
        "--action-self-conditioning",
        action="store_true",
        help="Explicitly load a versioned fresh SC+encoder checkpoint",
    )
    parser.add_argument("--nfe", type=int, default=4)
    args = parser.parse_args()
    if args.nfe <= 0 or (not args.action_self_conditioning and args.nfe != 4):
        parser.error("legacy continuation evaluation is fixed NFE4; SC requires positive NFE")
    args.execution_horizon, args.evaluation_seed, args.resets_per_task = 8, 0, 20
    args.num_shards, args.reset_audit, args.verify_singleton, args.routing_instrumented = 2, True, True, False
    if args.mode == "server":
        if args.checkpoint is None or args.expected_manifest_sha256 is None:
            parser.error("server requires the final checkpoint and its manifest hash")
        run_server(args)
    else:
        if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
            raise RuntimeError("simulator CUDA compute must be disabled; use the verified EGL device for rendering")
        run_evaluation(args)


if __name__ == "__main__":
    main()
