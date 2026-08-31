#!/usr/bin/env python3
"""Two-rank BF16 DiffusionGemma continuous-action forward/backward smoke gate."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from PIL import Image
from transformers import AutoProcessor

from duo_vla.action_interface import ActionInputProjector, VelocityHead
from duo_vla.backbones.diffusion_gemma import (
    DiffusionGemmaActionDecoder,
    apply_decoder_attention_lora,
    decoder_lora_parameter_partition,
    encode_diffusion_gemma_prefix,
)
from duo_vla.backbones.loading import DEFAULT_DIFFUSION_GEMMA_SPEC, load_diffusion_gemma_bf16_tp
from duo_vla.checkpointing import load_interface_state_dict, load_lora_checkpoint, save_trainable_checkpoint
from duo_vla.config import ActionInterfaceConfig
from duo_vla.flow import make_flow_training_pair, masked_velocity_mse
from duo_vla.modeling import DuoVLADenoiser
from duo_vla.optimization import assert_replicated_tensor, clip_tensor_parallel_grad_norm_


def _processor_inputs(processor, *, device: torch.device, text_only: bool, batch_size: int):
    conversations = []
    for index in range(batch_size):
        instruction = "move the red object to the left" + (" carefully" * index)
        if text_only:
            message = [{"role": "user", "content": instruction}]
        else:
            first = Image.fromarray(np.full((64, 64, 3), (180, 30 + index, 30), dtype=np.uint8))
            wrist = Image.fromarray(np.full((64, 64, 3), (30, 180, 30 + index), dtype=np.uint8))
            message = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": first},
                        {"type": "image", "image": wrist},
                        {"type": "text", "text": instruction},
                    ],
                }
            ]
        conversations.append(message)
    inputs = processor.apply_chat_template(
        conversations[0] if batch_size == 1 else conversations,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
        processor_kwargs={"padding": batch_size > 1},
    )
    return inputs.to(device)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--text-only", action="store_true")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument("--load-checkpoint", type=Path)
    args = parser.parse_args()
    if args.batch_size <= 0:
        raise ValueError("batch size must be positive")
    if args.checkpoint_dir is not None and args.load_checkpoint is not None:
        raise ValueError("checkpoint-dir and load-checkpoint are mutually exclusive")

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    if not dist.is_initialized():
        dist.init_process_group("nccl", device_id=device)
    torch.manual_seed(1234)
    torch.cuda.reset_peak_memory_stats(device)

    try:
        processor = AutoProcessor.from_pretrained(
            DEFAULT_DIFFUSION_GEMMA_SPEC.model_id,
            revision=DEFAULT_DIFFUSION_GEMMA_SPEC.revision,
            local_files_only=args.local_files_only,
        )
        base_model = load_diffusion_gemma_bf16_tp(
            local_files_only=args.local_files_only,
            tp_size=dist.get_world_size(),
        )
        processor_inputs = _processor_inputs(
            processor,
            device=device,
            text_only=args.text_only,
            batch_size=args.batch_size,
        )
        prefix = encode_diffusion_gemma_prefix(base_model, dict(processor_inputs))

        decoder_backend = DiffusionGemmaActionDecoder.from_block_diffusion_model(base_model)
        torch.manual_seed(1235)
        if args.load_checkpoint is None:
            adapted_model = apply_decoder_attention_lora(base_model, rank=16, alpha=32, dropout=0.0)
            loaded_manifest = None
        else:
            adapted_model, loaded_manifest = load_lora_checkpoint(args.load_checkpoint, base_model)
        interface_config = ActionInterfaceConfig(hidden_size=2816, state_dim=8)
        projector = ActionInputProjector(interface_config).to(device=device)
        head = VelocityHead(2816, 7).to(device=device)
        if args.load_checkpoint is not None:
            load_interface_state_dict(
                args.load_checkpoint / "interface.safetensors",
                {"action_projector": projector, "velocity_head": head},
            )
        denoiser = DuoVLADenoiser(projector, decoder_backend, head)

        clean = torch.empty(args.batch_size, 8, 7, device=device).uniform_(-0.8, 0.8)
        pair = make_flow_training_pair(clean)
        state = torch.empty(args.batch_size, 8, device=device).uniform_(-1, 1)
        valid = torch.ones(args.batch_size, 8, dtype=torch.bool, device=device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            prediction = denoiser(
                pair.noisy_actions,
                pair.timesteps,
                state,
                prefix_cache=prefix.past_key_values,
                prefix_attention_mask=prefix.attention_mask,
                action_valid_mask=valid,
            )
        loss = masked_velocity_mse(prediction, pair.target_velocity, valid)
        if not torch.isfinite(loss):
            raise FloatingPointError("real-model smoke loss is not finite")
        loss.backward()

        lora_parameters = [parameter for parameter in adapted_model.parameters() if parameter.requires_grad]
        if loaded_manifest is None:
            lora_partition = decoder_lora_parameter_partition(adapted_model)
            replicated_lora = [parameter for _, parameter in lora_partition.replicated]
            sharded_lora = [parameter for _, parameter in lora_partition.sharded]
        else:
            replicated_lora = []
            sharded_lora = []
        interface_parameters = [*projector.parameters(), *head.parameters()]
        missing_grad = sum(parameter.grad is None for parameter in [*lora_parameters, *interface_parameters])
        nonfinite_grad = sum(
            parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all())
            for parameter in [*lora_parameters, *interface_parameters]
        )
        if (missing_grad or nonfinite_grad) and args.load_checkpoint is None:
            raise RuntimeError(f"gradient audit failed: missing={missing_grad}, nonfinite={nonfinite_grad}")

        gradient_norm = clip_tensor_parallel_grad_norm_(
            [*replicated_lora, *interface_parameters],
            sharded_lora,
            max_norm=1.0,
            error_if_nonfinite=True,
        )

        prediction_sha256 = assert_replicated_tensor("smoke_prediction", prediction)
        if loaded_manifest is not None:
            expected_hash = loaded_manifest.get("reference_prediction_sha256")
            if expected_hash is not None and prediction_sha256 != expected_hash:
                raise RuntimeError("checkpoint round-trip changed the reference prediction")

        checkpoint_manifest = None
        if args.checkpoint_dir is not None:
            checkpoint_manifest = save_trainable_checkpoint(
                args.checkpoint_dir,
                adapted_model=adapted_model,
                interface_modules={"action_projector": projector, "velocity_head": head},
                manifest={
                    "kind": "unoptimized-real-model-smoke",
                    "model_id": DEFAULT_DIFFUSION_GEMMA_SPEC.model_id,
                    "model_revision": DEFAULT_DIFFUSION_GEMMA_SPEC.revision,
                    "reference_batch_size": args.batch_size,
                    "reference_prediction_sha256": prediction_sha256,
                    "reference_seed": 1234,
                },
            )

        dist.barrier()
        if dist.get_rank() == 0:
            print(
                json.dumps(
                    {
                        "batch_size": args.batch_size,
                        "checkpoint_saved": checkpoint_manifest is not None,
                        "checkpoint_loaded": loaded_manifest is not None,
                        "gradient_norm_before_clip": float(gradient_norm),
                        "loss": float(loss.detach()),
                        "prediction_shape": list(prediction.shape),
                        "prefix_length": int(prefix.attention_mask.shape[1]),
                        "prediction_sha256": prediction_sha256,
                        "lora_parameter_tensors": len(lora_parameters),
                        "replicated_lora_parameter_tensors": len(replicated_lora),
                        "sharded_lora_parameter_tensors": len(sharded_lora),
                        "interface_parameter_tensors": len(interface_parameters),
                        "peak_memory_gib": torch.cuda.max_memory_allocated(device) / 2**30,
                        "world_size": dist.get_world_size(),
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
