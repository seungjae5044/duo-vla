#!/usr/bin/env python3
"""G3: overfit immutable real LIBERO chunks in fixed physical-B=8 microbatches."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

import torch
import torch.distributed as dist
from transformers import AutoProcessor

from duo_vla.action_interface import ActionInputProjector, VelocityHead
from duo_vla.backbones.diffusion_gemma import (
    DiffusionGemmaActionDecoder,
    apply_decoder_attention_lora,
    decoder_lora_parameter_partition,
    encode_diffusion_gemma_prefix,
)
from duo_vla.backbones.loading import DEFAULT_DIFFUSION_GEMMA_SPEC, load_diffusion_gemma_bf16_tp
from duo_vla.backbones.sample_isolated_experts import (
    GROUPED_MM_EXPERTS_IMPLEMENTATION,
    install_sample_isolated_grouped_mm_experts,
    verify_sample_isolated_grouped_mm_experts,
)
from duo_vla.checkpointing import load_interface_state_dict, load_lora_checkpoint, save_trainable_checkpoint
from duo_vla.data.batching import collate_libero_samples
from duo_vla.data.libero import LiberoParquetDataset
from duo_vla.data.libero_stats import LIBERO_DATASET_REVISION, load_libero_normalizers
from duo_vla.data.sampling import LiberoAnchor, TaskUniformAnchorSampler
from duo_vla.flow import make_flow_training_pair
from duo_vla.modeling import DuoVLADenoiser
from duo_vla.optimization import (
    assert_replicated_parameter_values,
    assert_replicated_tensor,
    clip_tensor_parallel_grad_norm_,
)
from duo_vla.training import masked_element_count, masked_sse


def _load_canonical_trainer() -> ModuleType:
    path = Path(__file__).resolve().with_name("train_libero.py")
    spec = importlib.util.spec_from_file_location("_duo_vla_canonical_libero_trainer", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load canonical LIBERO trainer contracts: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(spec.name, module)
    spec.loader.exec_module(module)
    return module


TRAIN = _load_canonical_trainer()
PHYSICAL_BATCH_SIZE = TRAIN.PHYSICAL_BATCH_SIZE


@dataclass(frozen=True, slots=True)
class PreparedFixedMicrobatch:
    state: torch.Tensor
    clean: torch.Tensor
    valid: torch.Tensor
    pair: Any
    prefix: Any


def _stable_json_hash(value: object) -> str:
    serialized = json.dumps(value, allow_nan=False, separators=(",", ":"), sort_keys=True).encode()
    return hashlib.sha256(serialized).hexdigest()


def _qualification_source_sha256(project_root: Path) -> str:
    digest = hashlib.sha256()
    digest.update(TRAIN._source_tree_sha256(project_root).encode())
    for relative in (
        "scripts/overfit_real_libero_batch.py",
        "scripts/run_overfit_real_libero_batch_single_gpu.sh",
    ):
        path = project_root / relative
        digest.update(relative.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _report_target(path: Path | None) -> Path | None:
    if path is None:
        return None
    absolute = path.absolute()
    if absolute.exists() or absolute.is_symlink():
        raise FileExistsError(f"G3 report already exists: {absolute}")
    parent = absolute.parent.resolve(strict=True)
    if not parent.is_dir():
        raise NotADirectoryError(f"G3 report parent is not a directory: {parent}")
    return parent / absolute.name


def _write_report(path: Path | None, value: dict[str, Any]) -> None:
    if path is None:
        return
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, allow_nan=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


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
    parser.add_argument("prefix_geometry_artifact", type=Path)
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
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if not 32 <= args.batch_size <= 128 or args.batch_size % PHYSICAL_BATCH_SIZE:
        raise ValueError("G3 requires 32--128 distinct chunks divisible into physical-B=8 microbatches")
    if args.steps <= 0 or args.minimum_reduction <= 1:
        raise ValueError("steps must be positive and minimum reduction must exceed one")
    if not 0 <= args.warmup_steps < args.steps:
        raise ValueError("warmup steps must be nonnegative and shorter than total steps")
    if args.lora_learning_rate <= 0 or args.interface_learning_rate <= 0:
        raise ValueError("learning rates must be positive")
    if args.checkpoint_dir is not None and args.load_checkpoint is not None:
        raise ValueError("checkpoint-dir and load-checkpoint are mutually exclusive")

    project_root = Path(__file__).resolve().parents[1]
    source_identity_start = _qualification_source_sha256(project_root)
    report_path = _report_target(args.report)
    runtime_preflight = TRAIN._configure_and_validate_training_runtime(project_root)
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", device_id=device)
    if dist.get_world_size() not in {1, 2}:
        raise RuntimeError("the real LIBERO G3 gate requires TP world size one or two")
    torch.manual_seed(2026)
    torch.cuda.reset_peak_memory_stats(device)
    try:
        config_name = "libero_single_gpu.toml" if dist.get_world_size() == 1 else "libero.toml"
        config = TRAIN.load_resolved_toml(project_root / "configs" / config_name)
        interface_config = TRAIN._validate_and_build_interface_config(config)
        dataset_snapshot_report = TRAIN._authenticate_dataset_snapshot_distributed(args.snapshot_root)
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

        model_snapshot = (
            Path(os.environ["HF_HOME"])
            / "hub/models--google--diffusiongemma-26B-A4B-it/snapshots"
            / DEFAULT_DIFFUSION_GEMMA_SPEC.revision
        )
        prefix_sha256, fixed_prefix_width = TRAIN._prefix_geometry_pins(config)
        prefix_geometry, model_snapshot_report = TRAIN._authenticate_prefix_geometry_distributed(
            args.prefix_geometry_artifact,
            expected_content_sha256=prefix_sha256,
            expected_fixed_physical_prefix_width=fixed_prefix_width,
            model_snapshot=model_snapshot,
            instructions=tuple(dataset.task_by_index.values()),
        )

        processor = AutoProcessor.from_pretrained(
            DEFAULT_DIFFUSION_GEMMA_SPEC.model_id,
            revision=DEFAULT_DIFFUSION_GEMMA_SPEC.revision,
            local_files_only=True,
        )
        model = load_diffusion_gemma_bf16_tp(local_files_only=True, tp_size=dist.get_world_size())
        installed_experts = install_sample_isolated_grouped_mm_experts(
            model,
            physical_batch_size=PHYSICAL_BATCH_SIZE,
        )
        if installed_experts.experts_implementation != GROUPED_MM_EXPERTS_IMPLEMENTATION:
            raise RuntimeError("installed LIBERO G3 expert implementation differs from the production contract")
        verify_sample_isolated_grouped_mm_experts(model, physical_batch_size=PHYSICAL_BATCH_SIZE)
        backend = DiffusionGemmaActionDecoder.from_block_diffusion_model(model)
        torch.manual_seed(2027)
        if args.load_checkpoint is None:
            adapted = apply_decoder_attention_lora(
                model,
                rank=int(config["lora"]["rank"]),
                alpha=int(config["lora"]["alpha"]),
                dropout=float(config["lora"]["dropout"]),
            )
            loaded_manifest = None
        else:
            adapted, loaded_manifest = load_lora_checkpoint(args.load_checkpoint, model)
        projector = ActionInputProjector(interface_config).to(device=device)
        head = VelocityHead(
            interface_config.hidden_size,
            interface_config.action_dim,
            init_std=interface_config.output_init_std,
        ).to(device=device)
        if args.load_checkpoint is not None:
            load_interface_state_dict(
                args.load_checkpoint / "interface.safetensors",
                {"action_projector": projector, "velocity_head": head},
            )
        denoiser = DuoVLADenoiser(projector, backend, head).train()
        adapted.train()
        model.model.encoder.eval()

        flow_generator = torch.Generator(device=device).manual_seed(args.flow_seed)
        fixed_batches_list: list[PreparedFixedMicrobatch] = []
        flow_hashes: dict[str, dict[str, str]] = {}
        for microstep, start in enumerate(range(0, args.batch_size, PHYSICAL_BATCH_SIZE)):
            batch = collate_libero_samples(
                samples[start : start + PHYSICAL_BATCH_SIZE],
                state_normalizer=state_normalizer,
                action_normalizer=action_normalizer,
            )
            state = batch.states.to(device=device)
            clean = batch.clean_actions.to(device=device)
            valid = batch.action_valid_mask.to(device=device)
            processor_inputs = TRAIN._processor_inputs(
                processor,
                batch.samples,
                device,
                prefix_geometry=prefix_geometry,
            )
            prefix = encode_diffusion_gemma_prefix(model, dict(processor_inputs))
            pair = make_flow_training_pair(clean, generator=flow_generator)
            flow_hashes[str(microstep)] = {
                "noisy_actions": assert_replicated_tensor(f"g3.{microstep}.noisy_actions", pair.noisy_actions),
                "target_velocity": assert_replicated_tensor(f"g3.{microstep}.target_velocity", pair.target_velocity),
                "timesteps": assert_replicated_tensor(f"g3.{microstep}.timesteps", pair.timesteps),
            }
            fixed_batches_list.append(
                PreparedFixedMicrobatch(state=state, clean=clean, valid=valid, pair=pair, prefix=prefix)
            )
        fixed_batches = tuple(fixed_batches_list)
        total_elements = sum(
            masked_element_count(fixed.valid, action_dim=interface_config.action_dim) for fixed in fixed_batches
        )
        if loaded_manifest is not None:
            expected = {
                "batch_anchor_sha256": anchor_hash,
                "normalization_content_sha256": stats_manifest["content_sha256"],
                "flow_seed": args.flow_seed,
                "batch_size": args.batch_size,
                "qualification_source_sha256": source_identity_start,
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
            loss_numerator = 0.0
            prediction_hashes: list[str] = []
            with torch.no_grad():
                for microstep, fixed in enumerate(fixed_batches):
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        prediction = denoiser(
                            fixed.pair.noisy_actions,
                            fixed.pair.timesteps,
                            fixed.state,
                            prefix_cache=fixed.prefix.past_key_values,
                            prefix_attention_mask=fixed.prefix.attention_mask,
                            action_valid_mask=fixed.valid,
                        )
                    component = masked_sse(prediction, fixed.pair.target_velocity, fixed.valid)
                    loss_numerator += float(component.squared_error_sum)
                    prediction_hashes.append(assert_replicated_tensor(f"g3.{microstep}.final_prediction", prediction))
            loss = loss_numerator / total_elements
            prediction_hash = _stable_json_hash(prediction_hashes)
            if prediction_hash != loaded_manifest.get("reference_prediction_sha256"):
                raise RuntimeError("G3 checkpoint round-trip changed the reference prediction")
            if loss != loaded_manifest.get("final_loss"):
                raise RuntimeError("G3 checkpoint round-trip changed the reference loss")
            source_identity_end = _qualification_source_sha256(project_root)
            if source_identity_end != source_identity_start:
                raise RuntimeError("G3 qualification sources changed during checkpoint round-trip")
            if dist.get_rank() == 0:
                result = {
                    "checkpoint_loaded": str(args.load_checkpoint),
                    "gate": "G3",
                    "loss": loss,
                    "prediction_sha256": prediction_hash,
                    "qualification_source_sha256": source_identity_end,
                    "roundtrip_match": True,
                    "schema": "duo-vla-libero-fixed-anchor-overfit-g3-v1",
                    "status": "pass",
                }
                _write_report(report_path, result)
                print(
                    json.dumps(
                        result,
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
            loss_numerator = 0.0
            for fixed in fixed_batches:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    prediction = denoiser(
                        fixed.pair.noisy_actions,
                        fixed.pair.timesteps,
                        fixed.state,
                        prefix_cache=fixed.prefix.past_key_values,
                        prefix_attention_mask=fixed.prefix.attention_mask,
                        action_valid_mask=fixed.valid,
                    )
                component = masked_sse(prediction, fixed.pair.target_velocity, fixed.valid)
                if not torch.isfinite(component.squared_error_sum):
                    raise FloatingPointError("G3 fixed-batch overfit produced a non-finite loss")
                component.loss_for_total(total_elements).backward()
                loss_numerator += float(component.squared_error_sum.detach())
            loss = loss_numerator / total_elements
            if initial_loss is None:
                initial_loss = loss
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
            current_loss = loss
            if current_loss < best_loss:
                best_loss = current_loss
                best_step = step
            if dist.get_rank() == 0 and (step == 0 or (step + 1) % 25 == 0):
                print(json.dumps({"step": step + 1, "loss": current_loss}, sort_keys=True), flush=True)
        elapsed = time.perf_counter() - started
        assert initial_loss is not None

        final_numerator = 0.0
        final_prediction_hashes: list[str] = []
        with torch.no_grad():
            for microstep, fixed in enumerate(fixed_batches):
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    final_prediction = denoiser(
                        fixed.pair.noisy_actions,
                        fixed.pair.timesteps,
                        fixed.state,
                        prefix_cache=fixed.prefix.past_key_values,
                        prefix_attention_mask=fixed.prefix.attention_mask,
                        action_valid_mask=fixed.valid,
                    )
                component = masked_sse(final_prediction, fixed.pair.target_velocity, fixed.valid)
                final_numerator += float(component.squared_error_sum)
                final_prediction_hashes.append(
                    assert_replicated_tensor(f"g3.{microstep}.final_prediction", final_prediction)
                )
        final_loss = final_numerator / total_elements
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
        prediction_hash = _stable_json_hash(final_prediction_hashes)
        checkpoint_manifest = None
        if args.checkpoint_dir is not None:
            checkpoint_manifest = save_trainable_checkpoint(
                args.checkpoint_dir,
                adapted_model=adapted,
                interface_modules={"action_projector": projector, "velocity_head": head},
                additional_artifacts={
                    "normalization": args.normalization_artifact,
                    "prefix_geometry": args.prefix_geometry_artifact,
                },
                manifest={
                    "batch_anchor_sha256": anchor_hash,
                    "batch_anchors": anchor_records,
                    "batch_size": args.batch_size,
                    "dataset_id": "HuggingFaceVLA/libero",
                    "dataset_revision": LIBERO_DATASET_REVISION,
                    "execution_profile": config.get("execution_profile"),
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
                    "prefix_geometry_content_sha256": prefix_sha256,
                    "qualification_source_sha256": source_identity_start,
                    "reference_prediction_sha256": prediction_hash,
                    "steps": args.steps,
                    "tensor_parallel_size": dist.get_world_size(),
                    "warmup_steps": args.warmup_steps,
                },
            )
        dist.barrier()
        source_identity_end = _qualification_source_sha256(project_root)
        if source_identity_end != source_identity_start:
            raise RuntimeError("G3 qualification sources changed during the optimizer run")
        if dist.get_rank() == 0:
            result = {
                "batch_anchor_sha256": anchor_hash,
                "batch_size": args.batch_size,
                "best_loss": best_loss,
                "best_step": best_step,
                "changed_interface_parameter_tensors": changed_interface,
                "changed_lora_parameter_tensors": changed_lora,
                "checkpoint_saved": checkpoint_manifest is not None,
                "dataset_content_inventory_sha256": dataset_snapshot_report["content_inventory_sha256"],
                "distinct_tasks": len({anchor.task for anchor in anchors}),
                "elapsed_seconds": elapsed,
                "execution_profile": config.get("execution_profile"),
                "final_loss": final_loss,
                "gate": "G3",
                "initial_loss": initial_loss,
                "interface_parameter_tensors": len(named_interface),
                "maximum_gradient_norm_before_clip": maximum_gradient_norm,
                "model_content_inventory_sha256": model_snapshot_report["content_inventory_sha256"],
                "normalization_content_sha256": stats_manifest["content_sha256"],
                "optimizer_state_dtypes": optimizer_state_dtypes,
                "pass_criteria": {
                    "declared_interface_tensors_changed": changed_interface > 0,
                    "declared_lora_tensors_changed": changed_lora > 0,
                    "loss_reduction_at_least_minimum": reduction >= args.minimum_reduction,
                    "only_declared_parameter_tensors_changed": not modified_frozen,
                },
                "peak_memory_gib": torch.cuda.max_memory_allocated(device) / 2**30,
                "prediction_sha256": prediction_hash,
                "qualification_source_sha256": source_identity_end,
                "reduction": reduction,
                "replicated_parameter_sha256": replicated_parameter_hash,
                "replicated_lora_parameter_tensors": len(lora_partition.replicated),
                "schema": "duo-vla-libero-fixed-anchor-overfit-g3-v1",
                "sharded_lora_parameter_tensors": len(lora_partition.sharded),
                "status": "pass",
                "steps": args.steps,
                "tensor_parallel_size": dist.get_world_size(),
                "torchrun": runtime_preflight["torchrun"],
                "valid_actions": sum(int(fixed.valid.sum()) for fixed in fixed_batches),
                "warmup_steps": args.warmup_steps,
            }
            _write_report(report_path, result)
            print(
                json.dumps(
                    result,
                    indent=2,
                    sort_keys=True,
                )
            )
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
