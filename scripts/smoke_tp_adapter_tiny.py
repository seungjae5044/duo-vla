#!/usr/bin/env python3
"""Exercise DiffusionGemma continuous embeddings and PEFT LoRA under native TP=2 using tiny random weights."""

from __future__ import annotations

import json
import os
from pathlib import Path

import torch
import torch.distributed as dist
from transformers import DiffusionGemmaConfig, DiffusionGemmaForBlockDiffusion

try:
    from transformers import DistributedConfig
except ImportError:  # Transformers 5.15
    from transformers.distributed import DistributedConfig

from duo_vla.action_interface import ActionInputProjector, VelocityHead
from duo_vla.backbones.diffusion_gemma import (
    DiffusionGemmaActionDecoder,
    apply_decoder_attention_lora,
    encode_diffusion_gemma_prefix,
)
from duo_vla.backbones.loading import symmetric_diffusion_gemma_tp_plan
from duo_vla.config import ActionInterfaceConfig
from duo_vla.flow import make_flow_training_pair, masked_velocity_mse
from duo_vla.modeling import DuoVLADenoiser


def _config() -> DiffusionGemmaConfig:
    config = DiffusionGemmaConfig(
        text_config={
            "vocab_size": 128,
            "hidden_size": 32,
            "intermediate_size": 48,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 8,
            "global_head_dim": 8,
            "num_global_key_value_heads": 2,
            "max_position_embeddings": 128,
            "sliding_window": 16,
            "layer_types": ["sliding_attention", "full_attention"],
            "num_experts": 4,
            "top_k_experts": 2,
            "moe_intermediate_size": 16,
            "use_bidirectional_attention": "vision",
        },
        vision_config={
            "hidden_size": 32,
            "intermediate_size": 48,
            "num_hidden_layers": 1,
            "num_attention_heads": 4,
            "num_key_value_heads": 4,
            "head_dim": 8,
            "max_position_embeddings": 128,
            "position_embedding_size": 64,
            "patch_size": 4,
            "pooling_kernel_size": 1,
        },
        canvas_length=8,
    )
    config._attn_implementation = "eager"
    config.text_config._attn_implementation = "eager"
    return config


def main() -> None:
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    device = torch.device("cuda", local_rank)
    checkpoint = Path("/root/.cache/duo-vla/models/tiny-diffusion-gemma-tp")
    try:
        if dist.get_rank() == 0:
            torch.manual_seed(0)
            DiffusionGemmaForBlockDiffusion(_config()).save_pretrained(checkpoint)
        dist.barrier()
        config = DiffusionGemmaConfig.from_pretrained(checkpoint)
        model = DiffusionGemmaForBlockDiffusion.from_pretrained(
            checkpoint,
            config=config,
            dtype=torch.bfloat16,
            distributed_config=DistributedConfig(
                tp_size=dist.get_world_size(),
                tp_plan=symmetric_diffusion_gemma_tp_plan(config),
            ),
            attn_implementation="eager",
        )
        input_ids = torch.randint(3, 128, (1, 5), device=device)
        prefix_mask = torch.ones_like(input_ids, dtype=torch.bool)
        prefix = encode_diffusion_gemma_prefix(
            model,
            {"input_ids": input_ids, "attention_mask": prefix_mask},
        )
        backend = DiffusionGemmaActionDecoder.from_block_diffusion_model(model)
        adapted = apply_decoder_attention_lora(model, rank=2, alpha=4)
        interface_config = ActionInterfaceConfig(hidden_size=32, state_dim=8)
        projector = ActionInputProjector(interface_config).to(device=device, dtype=torch.bfloat16)
        head = VelocityHead(32, 7).to(device=device, dtype=torch.bfloat16)
        denoiser = DuoVLADenoiser(projector, backend, head)
        clean = torch.empty(1, 8, 7, device=device, dtype=torch.bfloat16).uniform_(-0.8, 0.8)
        pair = make_flow_training_pair(clean)
        state = torch.empty(1, 8, device=device, dtype=torch.bfloat16).uniform_(-1, 1)
        valid = torch.ones(1, 8, dtype=torch.bool, device=device)
        prediction = denoiser(
            pair.noisy_actions,
            pair.timesteps,
            state,
            prefix_cache=prefix.past_key_values,
            prefix_attention_mask=prefix.attention_mask,
            action_valid_mask=valid,
        )
        loss = masked_velocity_mse(prediction, pair.target_velocity, valid)
        loss.backward()
        lora = [parameter for parameter in adapted.parameters() if parameter.requires_grad]
        missing = sum(parameter.grad is None for parameter in [*lora, *projector.parameters(), *head.parameters()])
        if missing:
            raise RuntimeError(f"{missing} trainable tensors did not receive gradients")
        dist.barrier()
        if dist.get_rank() == 0:
            print(
                json.dumps(
                    {
                        "loss": float(loss.detach()),
                        "lora_parameter_tensors": len(lora),
                        "prediction_shape": list(prediction.shape),
                        "world_size": dist.get_world_size(),
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
