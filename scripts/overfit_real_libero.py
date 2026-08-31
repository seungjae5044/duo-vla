#!/usr/bin/env python3
"""Overfit one real pinned LIBERO action chunk through the full TP=2 DiffusionGemma model."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

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
from duo_vla.data.libero import LiberoParquetDataset
from duo_vla.flow import make_flow_training_pair, masked_velocity_mse
from duo_vla.modeling import DuoVLADenoiser
from duo_vla.normalization import ActionNormalizer, PercentileNormalizer
from duo_vla.optimization import (
    assert_replicated_parameter_values,
    assert_replicated_tensor,
    clip_tensor_parallel_grad_norm_,
)


def _prefix_inputs(processor, sample, device: torch.device):
    message = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": Image.fromarray(sample.observation.third_person)},
                {"type": "image", "image": Image.fromarray(sample.observation.wrist)},
                {"type": "text", "text": sample.instruction},
            ],
        }
    ]
    inputs = processor.apply_chat_template(
        message,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    return inputs.to(device)


def _smoke_normalizers(dataset: LiberoParquetDataset, episode_index: int):
    """Fit only the referenced local shard; these statistics are never a benchmark artifact."""

    episode = dataset.episodes[episode_index]
    table = dataset._read_data_file(episode.chunk_index, episode.file_index)
    states = torch.tensor(table.column("observation.state").to_pylist(), dtype=torch.float32)
    actions = torch.tensor(table.column("action").to_pylist(), dtype=torch.float32)
    return (
        PercentileNormalizer.fit(states),
        ActionNormalizer(PercentileNormalizer.fit(actions[:, :6])),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot_root", type=Path)
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--lora-learning-rate", type=float, default=1e-4)
    parser.add_argument("--interface-learning-rate", type=float, default=1e-3)
    parser.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument("--load-checkpoint", type=Path)
    args = parser.parse_args()
    if args.steps <= 0:
        raise ValueError("steps must be positive")
    if args.lora_learning_rate <= 0 or args.interface_learning_rate <= 0:
        raise ValueError("learning rates must be positive")
    if args.checkpoint_dir is not None and args.load_checkpoint is not None:
        raise ValueError("checkpoint-dir and load-checkpoint are mutually exclusive")

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", device_id=device)
    torch.manual_seed(2026)
    torch.cuda.reset_peak_memory_stats(device)
    try:
        dataset = LiberoParquetDataset(args.snapshot_root)
        sample = dataset.sample(args.episode, args.frame)
        state_normalizer, action_normalizer = _smoke_normalizers(dataset, args.episode)
        state = state_normalizer.normalize(sample.observation.state[None]).to(device=device)
        clean = action_normalizer.normalize(sample.action_chunk.actions[None]).to(device=device)
        valid = sample.action_chunk.valid_mask[None].to(device)
        clean = clean * valid[..., None].to(dtype=clean.dtype)

        processor = AutoProcessor.from_pretrained(
            DEFAULT_DIFFUSION_GEMMA_SPEC.model_id,
            revision=DEFAULT_DIFFUSION_GEMMA_SPEC.revision,
            local_files_only=True,
        )
        model = load_diffusion_gemma_bf16_tp(local_files_only=True, tp_size=dist.get_world_size())
        prefix_inputs = _prefix_inputs(processor, sample, device)
        prefix = encode_diffusion_gemma_prefix(model, dict(prefix_inputs))
        backend = DiffusionGemmaActionDecoder.from_block_diffusion_model(model)
        torch.manual_seed(2027)
        if args.load_checkpoint is None:
            adapted = apply_decoder_attention_lora(model, rank=16, alpha=32)
            loaded_manifest = None
        else:
            adapted, loaded_manifest = load_lora_checkpoint(args.load_checkpoint, model)

        interface_config = ActionInterfaceConfig(hidden_size=2816, state_dim=8)
        projector = ActionInputProjector(interface_config).to(device=device)
        head = VelocityHead(2816, 7).to(device=device)
        if args.load_checkpoint is not None:
            load_interface_state_dict(
                args.load_checkpoint / "interface.safetensors",
                {"action_projector": projector, "velocity_head": head},
            )
        denoiser = DuoVLADenoiser(projector, backend, head).train()
        adapted.train()

        flow_generator = torch.Generator(device=device).manual_seed(99)
        pair = make_flow_training_pair(clean, generator=flow_generator)
        if loaded_manifest is not None:
            denoiser.eval()
            adapted.eval()
            with torch.no_grad():
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    prediction = denoiser(
                        pair.noisy_actions,
                        pair.timesteps,
                        state,
                        prefix_cache=prefix.past_key_values,
                        prefix_attention_mask=prefix.attention_mask,
                        action_valid_mask=valid,
                    )
                loss = float(masked_velocity_mse(prediction, pair.target_velocity, valid))
            prediction_hash = assert_replicated_tensor("final_prediction", prediction)
            if prediction_hash != loaded_manifest.get("reference_prediction_sha256"):
                raise RuntimeError("trained checkpoint round-trip changed the reference prediction")
            dist.barrier()
            if dist.get_rank() == 0:
                print(
                    json.dumps(
                        {
                            "checkpoint_loaded": str(args.load_checkpoint),
                            "loss": loss,
                            "prediction_sha256": prediction_hash,
                            "roundtrip_match": True,
                        },
                        indent=2,
                        sort_keys=True,
                    )
                )
            return

        lora_parameters = [parameter for parameter in adapted.parameters() if parameter.requires_grad]
        lora_partition = decoder_lora_parameter_partition(adapted)
        interface_parameters = [*projector.parameters(), *head.parameters()]
        named_interface = [(f"projector.{name}", parameter) for name, parameter in projector.named_parameters()]
        named_interface.extend((f"head.{name}", parameter) for name, parameter in head.named_parameters())
        optimizer = torch.optim.AdamW(
            [
                {"params": lora_parameters, "lr": args.lora_learning_rate},
                {"params": interface_parameters, "lr": args.interface_learning_rate},
            ],
            betas=(0.9, 0.95),
            eps=1e-8,
            weight_decay=1e-10,
        )

        initial_loss: float | None = None
        best_loss = float("inf")
        best_step = -1
        maximum_gradient_norm = 0.0
        started = time.perf_counter()
        for step in range(args.steps):
            optimizer.zero_grad(set_to_none=True)
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
                raise FloatingPointError("real fixed-batch overfit produced a non-finite loss")
            if initial_loss is None:
                initial_loss = float(loss.detach())
            loss.backward()
            if step == 0:
                for name, parameter in [*lora_partition.replicated, *named_interface]:
                    if parameter.grad is None:
                        raise RuntimeError(f"replicated gradient is missing: {name}")
                    assert_replicated_tensor(f"gradient.{name}", parameter.grad)
            gradient_norm = clip_tensor_parallel_grad_norm_(
                [*(parameter for _, parameter in lora_partition.replicated), *interface_parameters],
                [parameter for _, parameter in lora_partition.sharded],
                max_norm=1.0,
                error_if_nonfinite=True,
            )
            maximum_gradient_norm = max(maximum_gradient_norm, float(gradient_norm))
            optimizer.step()
            current_loss = float(loss.detach())
            if current_loss < best_loss:
                best_loss = current_loss
                best_step = step
        elapsed = time.perf_counter() - started
        assert initial_loss is not None

        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                final_prediction = denoiser(
                    pair.noisy_actions,
                    pair.timesteps,
                    state,
                    prefix_cache=prefix.past_key_values,
                    prefix_attention_mask=prefix.attention_mask,
                    action_valid_mask=valid,
                )
            final_loss = float(masked_velocity_mse(final_prediction, pair.target_velocity, valid))
        interface_hash = assert_replicated_parameter_values(named_interface)
        prediction_hash = assert_replicated_tensor("final_prediction", final_prediction)
        checkpoint_manifest = None
        if args.checkpoint_dir is not None:
            checkpoint_manifest = save_trainable_checkpoint(
                args.checkpoint_dir,
                adapted_model=adapted,
                interface_modules={"action_projector": projector, "velocity_head": head},
                manifest={
                    "dataset_id": "HuggingFaceVLA/libero",
                    "dataset_revision": "86958911c0f959db2bbbdb107eb3e17c5f9c798e",
                    "episode": args.episode,
                    "final_loss": final_loss,
                    "frame": args.frame,
                    "initial_loss": initial_loss,
                    "interface_learning_rate": args.interface_learning_rate,
                    "kind": "real-libero-fixed-batch-overfit-smoke",
                    "lora_learning_rate": args.lora_learning_rate,
                    "model_id": DEFAULT_DIFFUSION_GEMMA_SPEC.model_id,
                    "model_revision": DEFAULT_DIFFUSION_GEMMA_SPEC.revision,
                    "reference_prediction_sha256": prediction_hash,
                    "stats_scope": "smoke-only local parquet shard",
                    "steps": args.steps,
                },
            )
        dist.barrier()
        if dist.get_rank() == 0:
            print(
                json.dumps(
                    {
                        "best_loss": best_loss,
                        "best_step": best_step,
                        "checkpoint_saved": checkpoint_manifest is not None,
                        "elapsed_seconds": elapsed,
                        "episode": args.episode,
                        "final_loss": final_loss,
                        "frame": args.frame,
                        "initial_loss": initial_loss,
                        "instruction": sample.instruction,
                        "maximum_gradient_norm_before_clip": maximum_gradient_norm,
                        "replicated_lora_parameter_tensors": len(lora_partition.replicated),
                        "sharded_lora_parameter_tensors": len(lora_partition.sharded),
                        "peak_memory_gib": torch.cuda.max_memory_allocated(device) / 2**30,
                        "reduction": initial_loss / max(final_loss, torch.finfo(torch.float32).tiny),
                        "stats_scope": "smoke-only local parquet shard",
                        "steps": args.steps,
                        "interface_parameter_sha256": interface_hash,
                        "interface_learning_rate": args.interface_learning_rate,
                        "lora_learning_rate": args.lora_learning_rate,
                        "prediction_sha256": prediction_hash,
                        "valid_actions": int(valid.sum()),
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
