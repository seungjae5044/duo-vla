#!/usr/bin/env python3
"""Explicit Spatial prefix-LoRA continuation; not the qualified frozen-prefix recipe.

Frozen vision/base weights, independent encoder adapters, inherited decoder and
interface AdamW moments, fixed LR, manual globally normalized DP2 gradients.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import signal
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402
from safetensors.torch import load_file, save_file  # noqa: E402

# Reuse the exact data/flow stream, input geometry, validation, and reduction helpers.
from train_libero import (  # noqa: E402
    _assert_distributed_gradient_health,
    _assert_replicated_gradient_values,
    _assert_replicated_optimizer_state,
    _canonical_training_pair,
    _coalesce_canonical_batches,
    _materialize_canonical_batches,
    _processor_inputs,
    _run_validation,
    _sum_data_parallel_gradients_,
)
from transformers import AutoProcessor  # noqa: E402

from duo_vla.action_interface import ActionInputProjector, VelocityHead  # noqa: E402
from duo_vla.backbones.diffusion_gemma import (  # noqa: E402
    DiffusionGemmaActionDecoder,
    encode_diffusion_gemma_prefix,
    encode_diffusion_gemma_prefix_trainable,
)
from duo_vla.backbones.encoder_lora import (  # noqa: E402
    encoder_adapter_parameters,
    encoder_adapter_state,
    install_checkpointed_prefix,
    install_encoder_lora,
    load_encoder_adapter_state,
)
from duo_vla.backbones.loading import load_diffusion_gemma_bf16_tp  # noqa: E402
from duo_vla.backbones.sample_isolated_experts_v2 import (  # noqa: E402
    install_sample_isolated_grouped_mm_experts_v2,
)
from duo_vla.checkpointing import (  # noqa: E402
    load_checkpoint_manifest,
    load_interface_state_dict,
    load_lora_checkpoint,
    save_trainable_checkpoint,
)
from duo_vla.config import ActionInterfaceConfig  # noqa: E402
from duo_vla.data.libero import LiberoParquetDataset  # noqa: E402
from duo_vla.data.libero_stats import load_libero_normalizers  # noqa: E402
from duo_vla.data.sampling import TaskUniformAnchorSampler  # noqa: E402
from duo_vla.modeling import DuoVLADenoiser  # noqa: E402
from duo_vla.optimization import assert_replicated_parameter_values  # noqa: E402
from duo_vla.policy_contract import validate_manifest_policy_contract  # noqa: E402
from duo_vla.training import make_update_plan, masked_element_count, masked_sse  # noqa: E402
from duo_vla.training_checkpoint import (  # noqa: E402
    capture_rng_state,
    optimizer_parameter_inventory,
    restore_rng_state,
    validate_optimizer_state_dict,
)

PARENT_HASH = "f03384d957778a0eac49c98bf5d37c7c167e7a318b1a29958a0832ee799cfb23"
BASE_UPDATE = 15000
LR = 5e-5
STOP_REQUESTED = False


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w") as handle:
        handle.write(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


def emit(**value):
    if int(os.environ.get("RANK", "0")) == 0:
        print(json.dumps({"utc": datetime.now(UTC).isoformat(), **value}, allow_nan=False), flush=True)


def restore_parent_optimizer(optimizer, named_parameters, payload):
    inventory = optimizer_parameter_inventory(optimizer, named_parameters)
    if inventory != payload["optimizer_parameter_inventory"]:
        raise ValueError("parent optimizer parameter names/order/shapes changed")
    validate_optimizer_state_dict(payload["optimizer"], inventory, expected_update=BASE_UPDATE)
    optimizer.load_state_dict(payload["optimizer"])
    for group in optimizer.param_groups:
        group["lr"] = LR
        group["initial_lr"] = LR


def validate_continuation_state(payload, inventory, *, recipe_hash, rank, world, manifest, optimization):
    """Resume old Adam counters at 15000+k, new encoder counters at k."""
    stage = payload.get("stage_update")
    if (
        payload.get("schema") != "duo-vla-prefix-lora-state-v1"
        or type(stage) is not int
        or not 0 < stage <= 1000
        or payload.get("absolute_update") != BASE_UPDATE + stage
        or payload.get("rank") != rank
        or payload.get("world_size") != world
        or payload.get("recipe_sha256") != recipe_hash
        or payload.get("optimizer_parameter_inventory") != inventory
        or manifest.get("stage_update") != stage
        or manifest.get("absolute_update") != BASE_UPDATE + stage
    ):
        raise ValueError("continuation state identity/progress mismatch")
    state = payload["optimizer"]
    if set(state) != {"param_groups", "state"} or len(state["param_groups"]) != 3:
        raise ValueError("continuation optimizer schema mismatch")
    if [group["name"] for group in inventory["groups"]] != ["lora", "interface", "encoder"]:
        raise ValueError("continuation optimizer groups mismatch")
    offset = 0
    for group, metadata in zip(state["param_groups"], inventory["groups"], strict=True):
        ids = list(range(offset, offset + len(metadata["parameters"])))
        if group["params"] != ids:
            raise ValueError("continuation optimizer parameter order mismatch")
        if (
            group["lr"] != LR
            or group["initial_lr"] != LR
            or tuple(group["betas"]) != (optimization["adam_beta1"], optimization["adam_beta2"])
            or group["eps"] != optimization["adam_epsilon"]
            or group["weight_decay"] != optimization["weight_decay"]
        ):
            raise ValueError("continuation optimizer hyperparameters changed")
        subgroup = {
            "param_groups": [{**group, "params": list(range(len(ids)))}],
            "state": {index: state["state"][identifier] for index, identifier in enumerate(ids)},
        }
        validate_optimizer_state_dict(
            subgroup,
            {"schema": inventory["schema"], "groups": [metadata]},
            expected_update=stage if group["name"] == "encoder" else BASE_UPDATE + stage,
        )
        for parameter_state in subgroup["state"].values():
            if any(not value.isfinite().all() for value in parameter_state.values()):
                raise ValueError("nonfinite continuation optimizer state")
        offset += len(ids)
    if set(state["state"]) != set(range(offset)):
        raise ValueError("unexpected continuation optimizer state")
    return stage


def synchronize_scalar(value, device, *, op=dist.ReduceOp.SUM):
    tensor = torch.tensor(value, device=device, dtype=torch.float64)
    if dist.is_initialized():
        dist.all_reduce(tensor, op=op)
    return tensor.item()


def request_stop(signum, frame):
    global STOP_REQUESTED
    STOP_REQUESTED = True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, choices=(8, 16, 32, 64), required=True)
    parser.add_argument("--checkpoint-prefix", action="store_true")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--probe-steps", type=int, default=0)
    parser.add_argument("--stop-after", type=int)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--checkpoint-interval", type=int, default=250)
    args = parser.parse_args()
    if args.steps != 1000 or args.probe_steps < 0 or args.checkpoint_interval <= 0:
        parser.error("this authorized experiment is exactly 1000 additional updates")
    if args.stop_after is not None and not 0 < args.stop_after <= args.steps:
        parser.error("stop-after must be within the 1000-update stage")
    if args.probe_steps and args.resume:
        parser.error("capacity probes must start from the original parent")
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if (world, visible) not in {(1, "0"), (1, "1"), (2, "0,1")}:
        raise RuntimeError("only explicitly selected physical GPU 0/1, or DP2 on 0,1, are authorized")
    if not args.probe_steps and world != 2:
        raise RuntimeError("production continuation requires 2-GPU data parallel")
    torch.set_num_threads(1)
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    if "A6000" in torch.cuda.get_device_name(device):
        raise RuntimeError("A6000 use is prohibited")
    if world == 2:
        dist.init_process_group("nccl", timeout=timedelta(minutes=10))
    torch.manual_seed(20260910)
    if file_hash(args.parent / "manifest.json") != PARENT_HASH:
        raise ValueError("parent is not the authorized Spatial 15000 checkpoint")
    parent = load_checkpoint_manifest(args.parent)
    if json.loads(parent["training_suites_json"]) != ["libero_spatial"]:
        raise ValueError("parent suite mismatch")
    resolved = json.loads((args.parent / "artifacts/resolved_config.json").read_text())
    config = resolved["config"]
    if (
        hashlib.sha256(json.dumps(config, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        != parent["config_sha256"]
    ):
        raise ValueError("parent configuration identity mismatch")
    contract = validate_manifest_policy_contract(parent, config)
    prefix_geometry = json.loads((args.parent / "artifacts/prefix_geometry.json").read_text())
    sources = {
        str(path.relative_to(ROOT)): file_hash(path)
        for folder in (ROOT / "src", ROOT / "scripts")
        for path in sorted(folder.rglob("*.py"))
    }
    recipe = {
        "schema": "duo-vla-prefix-lora-continuation-v1",
        "parent_manifest_sha256": PARENT_HASH,
        "parent_checkpoint": str(args.parent.resolve()),
        "base_update": BASE_UPDATE,
        "additional_updates": args.steps,
        "learning_rate": LR,
        "schedule": "constant_no_warmup",
        "physical_batch_size": args.batch_size,
        "world_size": world,
        "global_batch_size": args.batch_size * world,
        "gradient_accumulation_steps": 1,
        "encoder_lora_rank": 16,
        "encoder_lora_alpha": 32,
        "vision_frozen": True,
        "base_weights_frozen": True,
        "encoder_checkpointing": args.checkpoint_prefix,
        "cuda_allocator_config": os.environ.get("PYTORCH_ALLOC_CONF", ""),
        "decoder_interface_optimizer_moments": "inherited",
        "encoder_optimizer_moments": "new",
        "data_seed": 0,
        "encoder_init_seed": 20260910,
        "validation_samples": 2048,
        "checkpoint_interval": args.checkpoint_interval,
        "dataset": str(args.dataset.resolve()),
        "source_files_sha256": sources,
        "versions": {name: importlib.metadata.version(name) for name in ("torch", "transformers", "peft", "triton")},
        "result_scope": "experimental_prefix_adaptation_not_official_benchmark",
    }
    recipe_hash = hashlib.sha256(json.dumps(recipe, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if rank == 0:
        if args.resume:
            if json.loads((args.output / "recipe.json").read_text()) != recipe:
                raise ValueError("resume recipe changed")
        else:
            args.output.mkdir(parents=True, exist_ok=False)
            write_json(args.output / "recipe.json", recipe)
    if world == 2:
        dist.barrier()
    emit(event="loading_model", physical_batch=args.batch_size, checkpoint_prefix=args.checkpoint_prefix)
    model = load_diffusion_gemma_bf16_tp(
        tp_size=1, replica_mode="data_parallel" if world == 2 else None, local_files_only=True
    )
    model.requires_grad_(False)
    install_sample_isolated_grouped_mm_experts_v2(model, physical_batch_size=args.batch_size)
    checkpoint_path = args.resume or args.parent
    adapted, loaded_manifest = load_lora_checkpoint(
        checkpoint_path,
        model,
        is_trainable=True,
        validate_decoder_contract=True,
        allow_encoder_adapter=bool(args.resume),
    )
    decoder_parameters = [
        (name, parameter) for name, parameter in adapted.named_parameters() if parameter.requires_grad
    ]
    action = config["action"]
    interface_config = ActionInterfaceConfig(
        hidden_size=2816,
        state_dim=8,
        action_horizon=8,
        action_dim=7,
        timestep_embedding_dim=action["timestep_embedding_dimension"],
        timestep_scale=action["timestep_scale"],
        timestep_max_period=action["timestep_max_period"],
        output_init_std=action["output_head_initialization_std"],
    )
    projector = ActionInputProjector(interface_config).to(device)
    head = VelocityHead(2816, 7, init_std=interface_config.output_init_std).to(device)
    modules = {"action_projector": projector, "velocity_head": head}
    load_interface_state_dict(checkpoint_path / "interface.safetensors", modules)
    interface_parameters = [
        (f"{name}.{key}", parameter) for name, module in modules.items() for key, parameter in module.named_parameters()
    ]
    old_parameters = decoder_parameters + interface_parameters
    optimization = config["optimization"]
    optimizer = torch.optim.AdamW(
        [
            {"name": "lora", "params": [p for _, p in decoder_parameters]},
            {"name": "interface", "params": [p for _, p in interface_parameters]},
        ],
        lr=LR,
        betas=(optimization["adam_beta1"], optimization["adam_beta2"]),
        eps=optimization["adam_epsilon"],
        weight_decay=optimization["weight_decay"],
    )
    if not args.resume:
        parent_state = torch.load(
            args.parent / f"artifacts/training_rank_{rank:03d}.pt", map_location="cpu", weights_only=True
        )
        restore_parent_optimizer(optimizer, old_parameters, parent_state)
        del parent_state
    targets = install_encoder_lora(model)
    if len(targets) != 113:
        raise RuntimeError("pinned encoder adapter topology changed")
    encoder_parameters = encoder_adapter_parameters(model)
    optimizer.add_param_group(
        {"name": "encoder", "params": [p for _, p in encoder_parameters], "lr": LR, "initial_lr": LR}
    )
    named_parameters = old_parameters + encoder_parameters
    inventory = optimizer_parameter_inventory(optimizer, named_parameters)
    if any(p.dtype != torch.float32 for _, p in named_parameters):
        raise RuntimeError("all trainables must remain FP32")
    allowed_ids = {id(p) for _, p in named_parameters}
    if any(p.requires_grad and id(p) not in allowed_ids for p in model.parameters()):
        raise RuntimeError("unexpected trainable base/vision parameter")
    if args.checkpoint_prefix:
        install_checkpointed_prefix(model)
    start = 0
    if args.resume:
        if loaded_manifest["recipe_sha256"] != recipe_hash:
            raise ValueError("checkpoint belongs to a different recipe")
        load_encoder_adapter_state(model, load_file(str(args.resume / "artifacts/encoder_adapter.safetensors")))
        state = torch.load(
            args.resume / f"artifacts/training_rank_{rank:03d}.pt", map_location="cpu", weights_only=True
        )
        start = validate_continuation_state(
            state,
            inventory,
            recipe_hash=recipe_hash,
            rank=rank,
            world=world,
            manifest=loaded_manifest,
            optimization=optimization,
        )
        optimizer.load_state_dict(state["optimizer"])
        restore_rng_state(state["rng"], device)
        del state
    if start >= min(args.steps, args.stop_after or args.steps):
        raise ValueError("resume has already reached the requested stop boundary")
    if world == 2:
        parameter_hash = assert_replicated_parameter_values(named_parameters)
        optimizer_hash = _assert_replicated_optimizer_state(
            optimizer, named_parameters if args.resume else old_parameters
        )
        emit(event="initial_replicas_verified", parameter_sha256=parameter_hash, optimizer_sha256=optimizer_hash)
    denoiser = DuoVLADenoiser(projector, DiffusionGemmaActionDecoder.from_block_diffusion_model(model), head).train()
    adapted.train()
    model.model.encoder.eval()  # deterministic dropout=0; eval does not disable autograd
    processor = AutoProcessor.from_pretrained(
        config["model"]["id"], revision=config["model"]["revision"], local_files_only=True
    )
    dataset = LiberoParquetDataset(args.dataset, max_cached_files=377)
    state_norm, action_norm, stats = load_libero_normalizers(args.parent / "artifacts/normalization.json")
    train_sampler = TaskUniformAnchorSampler(dataset.episodes, stats["split"]["train_episode_indices"])
    validation_sampler = TaskUniformAnchorSampler(dataset.episodes, stats["split"]["validation_episode_indices"])
    if len(train_sampler.tasks) != 10 or set(train_sampler.tasks) != set(validation_sampler.tasks):
        raise ValueError("Spatial-only ten-task split required")
    if len(stats["split"]["train_episode_indices"]) != 389:
        raise ValueError("Spatial training split changed")
    encoder_count = sum(p.numel() for _, p in encoder_parameters)
    emit(
        event="model_ready",
        encoder_targets=len(targets),
        encoder_parameters=encoder_count,
        inherited_optimizer_parameters=len(old_parameters),
        total_trainable_parameters=sum(p.numel() for _, p in named_parameters),
    )
    if rank == 0:
        write_json(
            args.output / "encoder_config.json",
            {
                "rank": 16,
                "alpha": 32,
                "targets": targets,
                "scope": "prefix_language_attention",
                "last_layer_q_o_excluded": "not reachable from action loss",
                "vision_frozen": True,
            },
        )

    def prepare(stage_index):
        per_rank = args.batch_size // 8
        plans = make_update_plan(0, BASE_UPDATE + stage_index, gradient_accumulation_steps=per_rank * world)
        plans = plans[rank * per_rank : (rank + 1) * per_rank]
        chunks = _materialize_canonical_batches(
            dataset, train_sampler, plans, state_normalizer=state_norm, action_normalizer=action_norm
        )
        batch = _coalesce_canonical_batches(chunks, physical_batch_size=args.batch_size)
        if any(sample.instruction not in train_sampler.tasks for sample in batch.samples):
            raise ValueError("off-suite sample")
        prefix_inputs = _processor_inputs(
            processor, batch.samples, torch.device("cpu"), prefix_geometry, expected_batch_size=args.batch_size
        )
        return batch, plans, prefix_inputs

    def validate():
        return _run_validation(
            denoiser=denoiser,
            adapted=adapted,
            model=model,
            processor=processor,
            dataset=dataset,
            sampler=validation_sampler,
            state_normalizer=state_norm,
            action_normalizer=action_norm,
            device=device,
            samples=2048,
            physical_batch_size=args.batch_size,
            validation_seed=0x5A17,
            policy_contract=contract,
            prefix_geometry=prefix_geometry,
            data_parallel_size=world,
        )

    def save(stage_update, metric):
        parameter_hash = assert_replicated_parameter_values(named_parameters)
        optimizer_hash = _assert_replicated_optimizer_state(optimizer, named_parameters)
        scratch = args.output / "checkpoint_staging"
        scratch.mkdir(exist_ok=True)
        rank_path = scratch / f"training_rank_{rank:03d}.pt"
        payload = {
            "schema": "duo-vla-prefix-lora-state-v1",
            "stage_update": stage_update,
            "absolute_update": BASE_UPDATE + stage_update,
            "recipe_sha256": recipe_hash,
            "optimizer": optimizer.state_dict(),
            "optimizer_parameter_inventory": inventory,
            "rng": capture_rng_state(device),
            "rank": rank,
            "world_size": world,
        }
        torch.save(payload, rank_path)
        if rank == 0:
            save_file(encoder_adapter_state(model), str(scratch / "encoder_adapter.safetensors"))
        if world == 2:
            dist.barrier()
        extras = {f"training_rank_{index:03d}": scratch / f"training_rank_{index:03d}.pt" for index in range(world)}
        extras.update(
            encoder_adapter=scratch / "encoder_adapter.safetensors",
            encoder_config=args.output / "encoder_config.json",
            recipe=args.output / "recipe.json",
            resolved_config=args.parent / "artifacts/resolved_config.json",
            normalization=args.parent / "artifacts/normalization.json",
            prefix_geometry=args.parent / "artifacts/prefix_geometry.json",
        )
        path = args.output / "checkpoints" / f"update-{BASE_UPDATE + stage_update:06d}"
        manifest = {
            "kind": "experimental-libero-prefix-lora-continuation",
            "recipe_sha256": recipe_hash,
            "parent_checkpoint": str(args.parent),
            "parent_manifest_sha256": PARENT_HASH,
            "model_id": parent["model_id"],
            "model_revision": parent["model_revision"],
            "config_sha256": parent["config_sha256"],
            "base_config_role": "inherited_frozen_prefix_recipe_only",
            "effective_recipe_artifact": "artifacts/recipe.json",
            "encoder_adapted": True,
            "policy_contract": parent["policy_contract"],
            "policy_contract_sha256": parent["policy_contract_sha256"],
            "training_suites_json": parent["training_suites_json"],
            "last_metrics": metric,
            "stage_update": stage_update,
            "absolute_update": BASE_UPDATE + stage_update,
            "world_size": world,
            "physical_batch_size": args.batch_size,
            "global_batch_size": args.batch_size * world,
            "replicated_parameter_sha256": parameter_hash,
            "replicated_optimizer_sha256": optimizer_hash,
        }
        save_trainable_checkpoint(
            path, adapted_model=adapted, interface_modules=modules, manifest=manifest, additional_artifacts=extras
        )
        if rank == 0:
            load_checkpoint_manifest(path)
            write_json(
                args.output / "latest_checkpoint.json",
                {
                    "path": str(path),
                    "manifest_sha256": file_hash(path / "manifest.json"),
                    "stage_update": stage_update,
                    "absolute_update": BASE_UPDATE + stage_update,
                },
            )
        emit(event="checkpoint_saved", path=str(path), stage_update=stage_update)

    for signum in (signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, request_stop)
    end = args.probe_steps or min(args.steps, args.stop_after or args.steps)
    torch.cuda.reset_peak_memory_stats(device)
    latest = None
    with ThreadPoolExecutor(max_workers=1) as executor:
        pending = executor.submit(prepare, start)
        for stage_index in range(start, end):
            beginning = time.perf_counter()
            if pending is None:
                pending = executor.submit(prepare, stage_index)
            batch, plans, prefix_inputs = pending.result()
            stage_update = stage_index + 1
            boundary = stage_update % args.checkpoint_interval == 0 or stage_update == end
            pending = executor.submit(prepare, stage_index + 1) if not boundary else None
            clean, states, valid = (
                batch.clean_actions.to(device),
                batch.states.to(device),
                batch.action_valid_mask.to(device),
            )
            inputs = prefix_inputs.to(device)
            pair = _canonical_training_pair(clean, contract, plans)
            count = int(synchronize_scalar(masked_element_count(valid, action_dim=7), device))
            optimizer.zero_grad(set_to_none=True)
            if any(group["lr"] != LR for group in optimizer.param_groups):
                raise RuntimeError("constant learning rate changed")
            # Compare the parent-compatible path with the zero-init encoder on the first probe update.
            reference = None
            if args.probe_steps and stage_index == 0:
                with torch.no_grad():
                    frozen = encode_diffusion_gemma_prefix(model, dict(inputs))
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        reference = denoiser(
                            pair.input_actions,
                            pair.timesteps,
                            states,
                            prefix_cache=frozen.past_key_values,
                            prefix_attention_mask=frozen.attention_mask,
                            action_valid_mask=valid,
                        ).detach()
                    del frozen
            native_gradients = None
            if args.probe_steps and args.checkpoint_prefix and args.batch_size == 8:
                # Real-model parity at the measured native-fit batch, including the
                # second update where both low-rank factors receive gradients.
                language = model.model.encoder.language_model
                recompute_forward = language.forward
                language.forward = language._duovla_original_forward
                try:
                    native_prefix = encode_diffusion_gemma_prefix_trainable(model, dict(inputs))
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        native_prediction = denoiser(
                            pair.input_actions,
                            pair.timesteps,
                            states,
                            prefix_cache=native_prefix.past_key_values,
                            prefix_attention_mask=native_prefix.attention_mask,
                            action_valid_mask=valid,
                        )
                    native_component = masked_sse(native_prediction, pair.target, valid)
                    native_component.loss_for_total(count).backward()
                    native_gradients = {name: p.grad.detach().cpu().clone() for name, p in named_parameters}
                    reference = native_prediction.detach()
                    del native_prefix, native_prediction, native_component
                    optimizer.zero_grad(set_to_none=True)
                finally:
                    language.forward = recompute_forward
            prefix = encode_diffusion_gemma_prefix_trainable(model, dict(inputs))
            if not prefix.past_key_values.layers[0].keys.requires_grad:
                raise RuntimeError("encoder KV gradients are disconnected")
            with torch.autocast("cuda", dtype=torch.bfloat16):
                prediction = denoiser(
                    pair.input_actions,
                    pair.timesteps,
                    states,
                    prefix_cache=prefix.past_key_values,
                    prefix_attention_mask=prefix.attention_mask,
                    action_valid_mask=valid,
                )
            if reference is not None:
                error = float((reference.float() - prediction.detach().float()).abs().max())
                if error != 0:
                    raise RuntimeError(f"zero-init encoder changes policy output: {error}")
                emit(
                    event="native_recompute_policy_parity"
                    if native_gradients is not None
                    else "zero_init_policy_parity",
                    stage_update=stage_update,
                    max_abs_error=error,
                )
            component = masked_sse(prediction, pair.target, valid)
            component.loss_for_total(count).backward()
            if native_gradients is not None:
                largest_error = 0.0
                for name, parameter in named_parameters:
                    actual = parameter.grad.detach().cpu()
                    expected = native_gradients[name]
                    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6, msg=name)
                    largest_error = max(largest_error, float((actual - expected).abs().max()))
                emit(
                    event="native_recompute_gradients_verified",
                    stage_update=stage_update,
                    tensors=len(native_gradients),
                    max_abs_error=largest_error,
                )
                del native_gradients
            if world == 2:
                _assert_distributed_gradient_health([p for _, p in named_parameters])
            else:
                invalid = [name for name, p in named_parameters if p.grad is None or not p.grad.isfinite().all()]
                if invalid:
                    raise RuntimeError(f"missing/nonfinite gradients: {invalid}")
            if world == 2:
                _sum_data_parallel_gradients_(named_parameters)
                if stage_index == start:
                    _assert_replicated_gradient_values(named_parameters)
            encoder_norm = float(
                torch.linalg.vector_norm(torch.stack([p.grad.float().norm() for _, p in encoder_parameters]))
            )
            if not encoder_norm > 0:
                raise RuntimeError("encoder LoRA receives no gradient")
            grad_norm = float(
                torch.nn.utils.clip_grad_norm_([p for _, p in named_parameters], 1.0, error_if_nonfinite=True)
            )
            optimizer.step()
            if world == 2 and stage_index == start:
                assert_replicated_parameter_values(named_parameters)
                _assert_replicated_optimizer_state(optimizer, named_parameters)
                emit(event="first_update_replicas_verified", stage_update=stage_update)
            loss = synchronize_scalar(float(component.squared_error_sum.detach()), device) / count
            torch.cuda.synchronize(device)
            seconds = synchronize_scalar(time.perf_counter() - beginning, device, op=dist.ReduceOp.MAX)
            peak = synchronize_scalar(torch.cuda.max_memory_allocated(device), device, op=dist.ReduceOp.MAX)
            latest = {
                "update": BASE_UPDATE + stage_update,
                "stage_update": stage_update,
                "train_loss": loss,
                "encoder_learning_rate": LR,
                "lora_learning_rate": LR,
                "interface_learning_rate": LR,
                "encoder_gradient_norm": encoder_norm,
                "gradient_norm": grad_norm,
                "update_seconds": seconds,
                "global_batch_size": args.batch_size * world,
                "peak_memory_gib": peak / 2**30,
                "utc": datetime.now(UTC).isoformat(),
            }
            del prefix, prediction, component, reference
            if boundary and not args.probe_steps:
                emit(event="validation_started", stage_update=stage_update, samples=2048)
                validation_start = time.perf_counter()
                latest["validation_loss"] = validate()
                latest["validation_seconds"] = time.perf_counter() - validation_start
                emit(
                    event="validation_finished",
                    stage_update=stage_update,
                    validation_loss=latest["validation_loss"],
                    seconds=latest["validation_seconds"],
                )
            if rank == 0:
                with (args.output / "metrics.jsonl").open("a") as handle:
                    handle.write(json.dumps(latest, allow_nan=False) + "\n")
                write_json(
                    args.output / "progress.json", {"status": "running", **latest, "target_stage_updates": args.steps}
                )
            emit(**latest)
            stopped = bool(synchronize_scalar(int(STOP_REQUESTED), device, op=dist.ReduceOp.MAX))
            if (boundary or stopped) and not args.probe_steps:
                if pending is not None:
                    pending.result()
                save(stage_update, latest)
            if stopped:
                break
    if rank == 0:
        write_json(
            args.output / "progress.json",
            {
                "status": "probe_complete"
                if args.probe_steps
                else "complete"
                if latest["stage_update"] == args.steps
                else "stopped",
                **latest,
                "target_stage_updates": args.steps,
            },
        )
    emit(event="finished", stage_update=latest["stage_update"], probe=bool(args.probe_steps))
    if world == 2:
        dist.destroy_process_group()


if __name__ == "__main__":
    try:
        main()
    except torch.OutOfMemoryError:
        emit(event="out_of_memory", rank=int(os.environ.get("RANK", "0")))
        raise
