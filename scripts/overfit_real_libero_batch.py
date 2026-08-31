#!/usr/bin/env python3
"""G3: overfit a deterministic batch of distinct real LIBERO chunks with the full TP=2 model."""

from __future__ import annotations

import argparse
import hashlib
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
from duo_vla.data.batching import collate_libero_samples
from duo_vla.data.libero import LiberoParquetDataset
from duo_vla.data.libero_stats import LIBERO_DATASET_REVISION, load_libero_normalizers
from duo_vla.data.sampling import LiberoAnchor, TaskUniformAnchorSampler
from duo_vla.flow import make_flow_training_pair, masked_velocity_mse
from duo_vla.modeling import DuoVLADenoiser
from duo_vla.optimization import (
    assert_replicated_parameter_values,
    assert_replicated_tensor,
    clip_tensor_parallel_grad_norm_,
)


def _stable_json_hash(value: object) -> str:
    serialized = json.dumps(value, allow_nan=False, separators=(",", ":"), sort_keys=True).encode()
    return hashlib.sha256(serialized).hexdigest()


def _assert_replicated_object(name: str, value: object) -> str:
    digest = _stable_json_hash(value)
    observed: list[str | None] = [None] * dist.get_world_size()
    dist.all_gather_object(observed, digest)
    if any(item != digest for item in observed):
        raise RuntimeError(f"replicated object {name!r} differs across TP ranks: {observed}")
    return digest


def _fixed_distinct_anchors(
    dataset: LiberoParquetDataset,
    train_episode_indices: tuple[int, ...],
    *,
    batch_size: int,
    seed: int,
) -> tuple[LiberoAnchor, ...]:
    sampler = TaskUniformAnchorSampler(dataset.episodes, train_episode_indices)
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if batch_size > sampler.population_size:
        raise ValueError(f"cannot draw {batch_size} distinct anchors from a population of {sampler.population_size}")
    generator = torch.Generator().manual_seed(seed)
    anchors: list[LiberoAnchor] = []
    seen: set[tuple[int, int]] = set()
    while len(anchors) < batch_size:
        anchor = sampler.draw(generator)
        identity = (anchor.episode_index, anchor.frame_index)
        if identity not in seen:
            seen.add(identity)
            anchors.append(anchor)
    return tuple(anchors)


def _anchor_records(anchors: tuple[LiberoAnchor, ...]) -> list[dict[str, object]]:
    return [
        {"episode_index": anchor.episode_index, "frame_index": anchor.frame_index, "task": anchor.task}
        for anchor in anchors
    ]


def _processor_inputs(processor, samples, device: torch.device):
    conversations = []
    for sample in samples:
        conversations.append(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": Image.fromarray(sample.observation.third_person)},
                        {"type": "image", "image": Image.fromarray(sample.observation.wrist)},
                        {"type": "text", "text": sample.instruction},
                    ],
                }
            ]
        )
    inputs = processor.apply_chat_template(
        conversations,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
        processor_kwargs={"padding": True},
    )
    return inputs.to(device)


def _trainable_snapshot(named_parameters) -> dict[str, torch.Tensor]:
    return {name: parameter.detach().cpu().clone() for name, parameter in named_parameters}


def _changed_count(named_parameters, before: dict[str, torch.Tensor]) -> int:
    return sum(not torch.equal(parameter.detach().cpu(), before[name]) for name, parameter in named_parameters)


def _assert_fp32_optimizer_state(optimizer: torch.optim.Optimizer) -> tuple[str, ...]:
    dtypes: set[str] = set()
    for state in optimizer.state.values():
        for value in state.values():
            if isinstance(value, torch.Tensor) and value.is_floating_point():
                dtypes.add(str(value.dtype))
                if value.dtype != torch.float32:
                    raise RuntimeError(f"optimizer state must be FP32, found {value.dtype}")
    if not dtypes:
        raise RuntimeError("optimizer has no materialized floating-point state")
    return tuple(sorted(dtypes))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot_root", type=Path)
    parser.add_argument("normalization_artifact", type=Path)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--anchor-seed", type=int, default=314159)
    parser.add_argument("--flow-seed", type=int, default=271828)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--lora-learning-rate", type=float, default=1e-4)
    parser.add_argument("--interface-learning-rate", type=float, default=1e-3)
    parser.add_argument("--minimum-reduction", type=float, default=20.0)
    parser.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument("--load-checkpoint", type=Path)
    args = parser.parse_args()
    if args.batch_size < 32:
        raise ValueError("G3 requires at least 32 distinct real chunks")
    if args.steps <= 0 or args.minimum_reduction <= 1:
        raise ValueError("steps must be positive and minimum reduction must exceed one")
    if not 0 <= args.warmup_steps < args.steps:
        raise ValueError("warmup steps must be nonnegative and shorter than total steps")
    if args.lora_learning_rate <= 0 or args.interface_learning_rate <= 0:
        raise ValueError("learning rates must be positive")
    if args.checkpoint_dir is not None and args.load_checkpoint is not None:
        raise ValueError("checkpoint-dir and load-checkpoint are mutually exclusive")

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", device_id=device)
    if dist.get_world_size() != 2:
        raise RuntimeError("the real LIBERO G3 gate requires TP world size 2")
    torch.manual_seed(2026)
    torch.cuda.reset_peak_memory_stats(device)
    try:
        dataset = LiberoParquetDataset(args.snapshot_root)
        state_normalizer, action_normalizer, stats_manifest = load_libero_normalizers(
            args.normalization_artifact,
            expected_revision=LIBERO_DATASET_REVISION,
        )
        train_episode_indices = tuple(int(index) for index in stats_manifest["split"]["train_episode_indices"])
        anchors = _fixed_distinct_anchors(
            dataset,
            train_episode_indices,
            batch_size=args.batch_size,
            seed=args.anchor_seed,
        )
        anchor_records = _anchor_records(anchors)
        anchor_hash = _assert_replicated_object("anchors", anchor_records)
        samples = dataset.sample_many(anchors)
        batch = collate_libero_samples(
            samples,
            state_normalizer=state_normalizer,
            action_normalizer=action_normalizer,
        )
        state = batch.states.to(device=device)
        clean = batch.clean_actions.to(device=device)
        valid = batch.action_valid_mask.to(device=device)

        processor = AutoProcessor.from_pretrained(
            DEFAULT_DIFFUSION_GEMMA_SPEC.model_id,
            revision=DEFAULT_DIFFUSION_GEMMA_SPEC.revision,
            local_files_only=True,
        )
        model = load_diffusion_gemma_bf16_tp(local_files_only=True, tp_size=dist.get_world_size())
        processor_inputs = _processor_inputs(processor, samples, device)
        prefix = encode_diffusion_gemma_prefix(model, dict(processor_inputs))
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

        flow_generator = torch.Generator(device=device).manual_seed(args.flow_seed)
        pair = make_flow_training_pair(clean, generator=flow_generator)
        flow_hashes = {
            "noisy_actions": assert_replicated_tensor("g3.noisy_actions", pair.noisy_actions),
            "target_velocity": assert_replicated_tensor("g3.target_velocity", pair.target_velocity),
            "timesteps": assert_replicated_tensor("g3.timesteps", pair.timesteps),
        }
        if loaded_manifest is not None:
            expected = {
                "batch_anchor_sha256": anchor_hash,
                "normalization_content_sha256": stats_manifest["content_sha256"],
                "flow_seed": args.flow_seed,
                "batch_size": args.batch_size,
            }
            mismatches = {
                key: (value, loaded_manifest.get(key))
                for key, value in expected.items()
                if loaded_manifest.get(key) != value
            }
            if mismatches:
                raise ValueError(f"checkpoint reference batch contract mismatch: {mismatches}")
            denoiser.eval()
            adapted.eval()
            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                prediction = denoiser(
                    pair.noisy_actions,
                    pair.timesteps,
                    state,
                    prefix_cache=prefix.past_key_values,
                    prefix_attention_mask=prefix.attention_mask,
                    action_valid_mask=valid,
                )
            loss = float(masked_velocity_mse(prediction, pair.target_velocity, valid))
            prediction_hash = assert_replicated_tensor("g3.final_prediction", prediction)
            if prediction_hash != loaded_manifest.get("reference_prediction_sha256"):
                raise RuntimeError("G3 checkpoint round-trip changed the reference prediction")
            if loss != loaded_manifest.get("final_loss"):
                raise RuntimeError("G3 checkpoint round-trip changed the reference loss")
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

        lora_partition = decoder_lora_parameter_partition(adapted)
        named_lora = [(name, parameter) for name, parameter in adapted.named_parameters() if parameter.requires_grad]
        named_interface = [(f"projector.{name}", parameter) for name, parameter in projector.named_parameters()]
        named_interface.extend((f"head.{name}", parameter) for name, parameter in head.named_parameters())
        lora_parameters = [parameter for _, parameter in named_lora]
        interface_parameters = [parameter for _, parameter in named_interface]
        declared_trainable_ids = {id(parameter) for parameter in [*lora_parameters, *interface_parameters]}
        if len(declared_trainable_ids) != len(lora_parameters) + len(interface_parameters):
            raise RuntimeError("declared trainable parameter groups overlap")
        frozen_versions = [
            (name, parameter, parameter._version)
            for name, parameter in adapted.named_parameters()
            if not parameter.requires_grad
        ]
        lora_before = _trainable_snapshot(named_lora)
        interface_before = _trainable_snapshot(named_interface)
        optimizer = torch.optim.AdamW(
            [
                {"name": "lora", "params": lora_parameters, "lr": args.lora_learning_rate},
                {"name": "interface", "params": interface_parameters, "lr": args.interface_learning_rate},
            ],
            betas=(0.9, 0.95),
            eps=1e-8,
            weight_decay=1e-10,
        )
        optimizer_parameter_ids = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}
        if optimizer_parameter_ids != declared_trainable_ids:
            raise RuntimeError("optimizer parameter inventory differs from the declared trainable inventory")

        initial_loss: float | None = None
        best_loss = float("inf")
        best_step = -1
        maximum_gradient_norm = 0.0
        optimizer_state_dtypes: tuple[str, ...] = ()
        started = time.perf_counter()
        for step in range(args.steps):
            learning_rate_scale = min(1.0, (step + 1) / max(1, args.warmup_steps))
            optimizer.param_groups[0]["lr"] = args.lora_learning_rate * learning_rate_scale
            optimizer.param_groups[1]["lr"] = args.interface_learning_rate * learning_rate_scale
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
                raise FloatingPointError("G3 fixed-batch overfit produced a non-finite loss")
            if initial_loss is None:
                initial_loss = float(loss.detach())
            loss.backward()
            if step in {0, 1}:
                for name, parameter in [*lora_partition.replicated, *named_interface]:
                    if parameter.grad is None:
                        raise RuntimeError(f"replicated gradient is missing: {name}")
                    assert_replicated_tensor(f"gradient.{step}.{name}", parameter.grad)
            gradient_norm = clip_tensor_parallel_grad_norm_(
                [*(parameter for _, parameter in lora_partition.replicated), *interface_parameters],
                [parameter for _, parameter in lora_partition.sharded],
                max_norm=1.0,
                error_if_nonfinite=True,
            )
            maximum_gradient_norm = max(maximum_gradient_norm, float(gradient_norm))
            optimizer.step()
            if step == 0:
                optimizer_state_dtypes = _assert_fp32_optimizer_state(optimizer)
            current_loss = float(loss.detach())
            if current_loss < best_loss:
                best_loss = current_loss
                best_step = step
            if dist.get_rank() == 0 and (step == 0 or (step + 1) % 25 == 0):
                print(json.dumps({"step": step + 1, "loss": current_loss}, sort_keys=True), flush=True)
        elapsed = time.perf_counter() - started
        assert initial_loss is not None

        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            final_prediction = denoiser(
                pair.noisy_actions,
                pair.timesteps,
                state,
                prefix_cache=prefix.past_key_values,
                prefix_attention_mask=prefix.attention_mask,
                action_valid_mask=valid,
            )
        final_loss = float(masked_velocity_mse(final_prediction, pair.target_velocity, valid))
        reduction = initial_loss / max(final_loss, torch.finfo(torch.float32).tiny)
        if reduction < args.minimum_reduction:
            raise RuntimeError(
                f"G3 failed: fixed-batch loss reduction {reduction:.3f}x is below {args.minimum_reduction:.3f}x"
            )
        changed_lora = _changed_count(named_lora, lora_before)
        changed_interface = _changed_count(named_interface, interface_before)
        if changed_lora == 0 or changed_interface == 0:
            raise RuntimeError("G3 optimizer did not change both declared trainable parameter groups")
        modified_frozen = [name for name, parameter, version in frozen_versions if parameter._version != version]
        if modified_frozen:
            raise RuntimeError(f"frozen base parameters were modified: {modified_frozen[:10]}")
        replicated_parameter_hash = assert_replicated_parameter_values([*lora_partition.replicated, *named_interface])
        prediction_hash = assert_replicated_tensor("g3.final_prediction", final_prediction)
        checkpoint_manifest = None
        if args.checkpoint_dir is not None:
            checkpoint_manifest = save_trainable_checkpoint(
                args.checkpoint_dir,
                adapted_model=adapted,
                interface_modules={"action_projector": projector, "velocity_head": head},
                additional_artifacts={"normalization": args.normalization_artifact},
                manifest={
                    "batch_anchor_sha256": anchor_hash,
                    "batch_anchors": anchor_records,
                    "batch_size": args.batch_size,
                    "dataset_id": "HuggingFaceVLA/libero",
                    "dataset_revision": LIBERO_DATASET_REVISION,
                    "final_loss": final_loss,
                    "flow_seed": args.flow_seed,
                    "flow_tensor_sha256": flow_hashes,
                    "initial_loss": initial_loss,
                    "interface_learning_rate": args.interface_learning_rate,
                    "kind": "real-libero-distinct-fixed-batch-overfit-g3",
                    "lora_learning_rate": args.lora_learning_rate,
                    "model_id": DEFAULT_DIFFUSION_GEMMA_SPEC.model_id,
                    "model_revision": DEFAULT_DIFFUSION_GEMMA_SPEC.revision,
                    "normalization_content_sha256": stats_manifest["content_sha256"],
                    "reference_prediction_sha256": prediction_hash,
                    "steps": args.steps,
                    "warmup_steps": args.warmup_steps,
                },
            )
        dist.barrier()
        if dist.get_rank() == 0:
            print(
                json.dumps(
                    {
                        "batch_anchor_sha256": anchor_hash,
                        "batch_size": args.batch_size,
                        "best_loss": best_loss,
                        "best_step": best_step,
                        "changed_interface_parameter_tensors": changed_interface,
                        "changed_lora_parameter_tensors": changed_lora,
                        "checkpoint_saved": checkpoint_manifest is not None,
                        "distinct_tasks": len({anchor.task for anchor in anchors}),
                        "elapsed_seconds": elapsed,
                        "final_loss": final_loss,
                        "initial_loss": initial_loss,
                        "interface_parameter_tensors": len(named_interface),
                        "maximum_gradient_norm_before_clip": maximum_gradient_norm,
                        "normalization_content_sha256": stats_manifest["content_sha256"],
                        "optimizer_state_dtypes": optimizer_state_dtypes,
                        "peak_memory_gib": torch.cuda.max_memory_allocated(device) / 2**30,
                        "prediction_sha256": prediction_hash,
                        "reduction": reduction,
                        "replicated_parameter_sha256": replicated_parameter_hash,
                        "replicated_lora_parameter_tensors": len(lora_partition.replicated),
                        "sharded_lora_parameter_tensors": len(lora_partition.sharded),
                        "steps": args.steps,
                        "valid_actions": int(valid.sum()),
                        "warmup_steps": args.warmup_steps,
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
