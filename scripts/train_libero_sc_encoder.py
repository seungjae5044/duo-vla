#!/usr/bin/env python3
"""Fresh 7,500-update Spatial flow training with endpoint SC and frozen-base encoder/decoder LoRA."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import signal
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from datetime import UTC, datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402
from safetensors.torch import load_file, save_file  # noqa: E402
from train_libero import (  # noqa: E402
    _assert_distributed_gradient_health,
    _assert_replicated_gradient_values,
    _assert_replicated_optimizer_state,
    _canonical_training_pair,
    _coalesce_canonical_batches,
    _materialize_canonical_batches,
    _processor_inputs,
    _sum_data_parallel_gradients_,
)
from train_libero_encoder_lora import emit, synchronize_scalar, write_json  # noqa: E402
from transformers import AutoProcessor  # noqa: E402

from duo_vla.action_interface import ActionInputProjector, VelocityHead  # noqa: E402
from duo_vla.action_sc_checkpoint import (  # noqa: E402
    RUN_SCHEMA,
    SC_CONTRACT,
    STATE_SCHEMA,
    canonical_sha256,
    file_sha256,
    read_sc_checkpoint,
)
from duo_vla.backbones.diffusion_gemma import (  # noqa: E402
    DiffusionGemmaActionDecoder,
    apply_decoder_attention_lora,
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
from duo_vla.backbones.sample_isolated_experts_v2 import install_sample_isolated_grouped_mm_experts_v2  # noqa: E402
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
from duo_vla.runtime_determinism import configure_strict_cuda_determinism  # noqa: E402
from duo_vla.self_conditioning import (  # noqa: E402
    SC_VERSION,
    bootstrap_for_update,
    sample_action_flow,
    training_velocity,
)
from duo_vla.training import make_microbatch_plan, make_update_plan, masked_element_count, masked_sse  # noqa: E402
from duo_vla.training_checkpoint import (  # noqa: E402
    capture_rng_state,
    optimizer_parameter_inventory,
    restore_rng_state,
    validate_optimizer_state_dict,
)

ARTIFACT_REFERENCE_HASH = "f03384d957778a0eac49c98bf5d37c7c167e7a318b1a29958a0832ee799cfb23"
TOTAL_UPDATES = 7500
LR = 5e-5
STOP = False


def stop_request(signum, frame):
    global STOP
    STOP = True


def validation_plan_groups(seed, samples, batch_size, rank, world):
    """Fixed held-out identities across physical widths; only forward padding is duplicated."""
    if samples % (8 * world) or batch_size % 8 or not 0 <= rank < world:
        raise ValueError("invalid canonical validation partition")
    plans = [make_microbatch_plan(seed, 0, i) for i in range(samples // 8)]
    local = plans[rank * (len(plans) // world) : (rank + 1) * (len(plans) // world)]
    chunks = batch_size // 8
    for index in range(0, len(local), chunks):
        selected = local[index : index + chunks]
        active = len(selected) * 8
        yield selected + [selected[-1]] * (chunks - len(selected)), active


def validate_state(payload, inventory, *, recipe_hash, rank, world, manifest):
    update = payload.get("update")
    if (
        payload.get("schema") != STATE_SCHEMA
        or type(update) is not int
        or not 0 < update <= TOTAL_UPDATES
        or manifest.get("update") != update
        or payload.get("recipe_sha256") != recipe_hash
        or manifest.get("recipe_sha256") != recipe_hash
        or payload.get("rank") != rank
        or payload.get("world_size") != world
        or payload.get("optimizer_parameter_inventory") != inventory
    ):
        raise ValueError("SC optimizer resume identity differs")
    validate_optimizer_state_dict(payload["optimizer"], inventory, expected_update=update)
    for group in payload["optimizer"]["param_groups"]:
        if (
            group["lr"] != LR
            or tuple(group["betas"]) != (0.9, 0.95)
            or group["eps"] != 1e-8
            or group["weight_decay"] != 1e-10
        ):
            raise ValueError("SC optimizer hyperparameters changed")
    for state in payload["optimizer"]["state"].values():
        if any(not value.isfinite().all() for value in state.values()):
            raise ValueError("nonfinite optimizer state")
    return update


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact-reference",
        type=Path,
        required=True,
        help="Authenticated data/config artifacts ONLY; never loads its trainable weights or optimizer",
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, choices=(8, 16, 32, 64, 72, 80), required=True)
    parser.add_argument("--checkpoint-prefix", action="store_true")
    parser.add_argument("--probe-steps", type=int, default=0)
    parser.add_argument("--verify-native", action="store_true")
    parser.add_argument("--stop-after", type=int)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    if args.probe_steps < 0 or (args.probe_steps and args.resume):
        parser.error("probes require fresh state and nonnegative probe steps")
    if args.verify_native and not (args.probe_steps >= 2 and args.batch_size == 8 and args.checkpoint_prefix):
        parser.error("native qualification requires B8, prefix recomputation and at least two probe updates")
    if args.stop_after is not None and not 0 < args.stop_after <= TOTAL_UPDATES:
        parser.error("stop-after must lie within 1..7500")
    rank, local, world = (
        int(os.environ.get(key, default)) for key, default in (("RANK", "0"), ("LOCAL_RANK", "0"), ("WORLD_SIZE", "1"))
    )
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if (world, visible) not in {(1, "0"), (1, "1"), (2, "0,1")}:
        raise RuntimeError("only physical GPUs 0/1 are authorized")
    if not args.probe_steps and world != 2:
        raise RuntimeError("production requires two data-parallel replicas")
    torch.set_num_threads(1)
    configure_strict_cuda_determinism(torch)
    torch.cuda.set_device(local)
    device = torch.device("cuda", local)
    if "A6000" in torch.cuda.get_device_name(device):
        raise RuntimeError("A6000 use is prohibited")
    if world == 2:
        dist.init_process_group("nccl", timeout=timedelta(minutes=10), device_id=device)
    if file_sha256(args.artifact_reference / "manifest.json") != ARTIFACT_REFERENCE_HASH:
        raise ValueError("unexpected data-artifact reference")
    reference = load_checkpoint_manifest(args.artifact_reference)
    resolved = json.loads((args.artifact_reference / "artifacts/resolved_config.json").read_text())
    base_config = resolved["config"]
    if canonical_sha256(base_config) != reference["config_sha256"]:
        raise ValueError("reference configuration hash differs")
    if json.loads(reference["training_suites_json"]) != ["libero_spatial"]:
        raise ValueError("Spatial-only data required")
    contract = validate_manifest_policy_contract(reference, base_config)
    if args.dataset.name != base_config["benchmark"]["dataset_revision"]:
        raise ValueError("dataset snapshot revision differs")
    artifact_paths = {
        name: args.artifact_reference / reference["artifacts"][name]["path"]
        for name in ("normalization", "prefix_geometry", "resolved_config")
    }
    prefix_geometry = json.loads(artifact_paths["prefix_geometry"].read_text())
    interface_config = ActionInterfaceConfig(hidden_size=2816, state_dim=8, self_conditioning=SC_VERSION)
    run_id = json.loads((args.output / "recipe.json").read_text())["run_uuid"] if args.resume else None
    if not args.resume:
        value = [str(uuid.uuid4()) if rank == 0 else None]
        if world == 2:
            dist.broadcast_object_list(value, src=0, device=device)
        run_id = value[0]
    if str(uuid.UUID(run_id)) != run_id:
        raise ValueError("invalid run UUID")
    recipe = {
        "schema": RUN_SCHEMA,
        "run_uuid": run_id,
        "total_updates": TOTAL_UPDATES,
        "initialization": "pretrained_base_fresh_decoder_encoder_interface_optimizer",
        "model": base_config["model"],
        "artifact_reference_manifest_sha256": ARTIFACT_REFERENCE_HASH,
        "artifact_reference_role": "data_geometry_base_configuration_only_no_weight_or_optimizer_initialization",
        "artifact_sha256": {name: file_sha256(path) for name, path in artifact_paths.items()},
        "dataset": str(args.dataset.resolve()),
        "dataset_revision": base_config["benchmark"]["dataset_revision"],
        "learning_rate": LR,
        "schedule": "constant_no_warmup",
        "optimizer": "adamw",
        "betas": [0.9, 0.95],
        "epsilon": 1e-8,
        "weight_decay": 1e-10,
        "gradient_clip_norm": 1.0,
        "physical_batch_size": args.batch_size,
        "world_size": world,
        "global_batch_size": args.batch_size * world,
        "gradient_accumulation_steps": 1,
        "gradient_reduction": "sum_globally_normalized_masked_sse",
        "self_conditioning": SC_CONTRACT,
        "encoder_rank": 16,
        "encoder_alpha": 32,
        "decoder_rank": 16,
        "decoder_alpha": 32,
        "lora_dropout": 0.0,
        "vision_frozen": True,
        "base_weights_frozen": True,
        "multimodal_projector_frozen": True,
        "checkpoint_prefix": args.checkpoint_prefix,
        "prefix_autocast": False,
        "data_seed": 0,
        "initialization_seed": 0,
        "sc_seed": 0,
        "validation_seed": 0x5A17,
        "validation_samples": 2048,
        "validation_interval": 500,
        "validation_sampled_action_nfe": 4,
        "validation_padding": "repeat_last_canonical_chunk_forward_only_excluded_from_metrics",
        "checkpoint_interval": 2500,
        "action_interface_config": asdict(interface_config),
        "cuda_allocator_config": os.environ.get("PYTORCH_ALLOC_CONF", ""),
        "source_files_sha256": {
            str(p.relative_to(ROOT)): file_sha256(p)
            for folder in (ROOT / "src", ROOT / "scripts")
            for p in sorted(folder.rglob("*.py"))
        },
        "versions": {name: importlib.metadata.version(name) for name in ("torch", "transformers", "peft", "triton")},
        "scope": "experimental_spatial_training_not_an_official_benchmark",
    }
    recipe_hash = canonical_sha256(recipe)
    if rank == 0:
        if args.resume:
            if json.loads((args.output / "recipe.json").read_text()) != recipe:
                raise ValueError("resume recipe/source differs")
        else:
            args.output.mkdir(parents=True, exist_ok=False)
            write_json(args.output / "recipe.json", recipe)
            write_json(args.output / "action_interface_config.json", asdict(interface_config))
    if world == 2:
        dist.barrier()
    torch.manual_seed(0)
    emit(event="loading_fresh_base", batch_size=args.batch_size, probe=bool(args.probe_steps))
    model = load_diffusion_gemma_bf16_tp(
        tp_size=1, replica_mode="data_parallel" if world == 2 else None, local_files_only=True
    )
    model.requires_grad_(False)
    install_sample_isolated_grouped_mm_experts_v2(model, physical_batch_size=args.batch_size)
    # Loading base weights must not change the explicitly seeded adapter/interface initialization stream.
    torch.manual_seed(0)
    resumed = None
    if args.resume:
        resumed, loaded_recipe, loaded_interface, encoder_config = read_sc_checkpoint(args.resume, source_root=ROOT)
        if loaded_recipe != recipe or loaded_interface != interface_config:
            raise ValueError("resume architecture/recipe differs")
        adapted, _ = load_lora_checkpoint(
            args.resume,
            model,
            is_trainable=True,
            validate_decoder_contract=True,
            allow_encoder_adapter=True,
            allow_action_self_conditioning=True,
        )
    else:
        adapted = apply_decoder_attention_lora(model, rank=16, alpha=32, dropout=0.0)
    decoder_parameters = [(name, p) for name, p in adapted.named_parameters() if p.requires_grad]
    if len(decoder_parameters) != 230:
        raise RuntimeError("pinned decoder LoRA topology changed")
    projector = ActionInputProjector(interface_config).to(device)
    head = VelocityHead(2816, 7, init_std=interface_config.output_init_std).to(device)
    modules = {"action_projector": projector, "velocity_head": head}
    interface_parameters = [
        (f"{name}.{key}", p) for name, module in modules.items() for key, p in module.named_parameters()
    ]
    targets = install_encoder_lora(model)
    if len(targets) != 113:
        raise RuntimeError("pinned encoder topology changed")
    encoder_parameters = encoder_adapter_parameters(model)
    encoder_config = {
        "rank": 16,
        "alpha": 32,
        "targets": list(targets),
        "vision_frozen": True,
        "scope": "prefix_language_attention_action_reachable_projections",
    }
    if rank == 0 and not args.resume:
        write_json(args.output / "encoder_config.json", encoder_config)
    named = decoder_parameters + interface_parameters + encoder_parameters
    allowed = {id(p) for _, p in named}
    if any(p.requires_grad and id(p) not in allowed for p in model.parameters()):
        raise RuntimeError("unexpected trainable base/vision weight")
    if any(p.dtype != torch.float32 for _, p in named):
        raise RuntimeError("all trainables must be FP32")
    if args.checkpoint_prefix:
        install_checkpointed_prefix(model)
    optimizer = torch.optim.AdamW(
        [
            {"name": name, "params": [p for _, p in parameters]}
            for name, parameters in (
                ("decoder", decoder_parameters),
                ("interface", interface_parameters),
                ("encoder", encoder_parameters),
            )
        ],
        lr=LR,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=1e-10,
    )
    inventory = optimizer_parameter_inventory(optimizer, named)
    start = 0
    if args.resume:
        artifacts = resumed["artifacts"]
        load_interface_state_dict(args.resume / artifacts["interface"]["path"], modules)
        load_encoder_adapter_state(model, load_file(str(args.resume / artifacts["encoder_adapter"]["path"])))
        # Authenticated local RNG state includes numpy objects; never load unverified pickle input.
        state = torch.load(
            args.resume / artifacts[f"training_rank_{rank:03d}"]["path"], map_location="cpu", weights_only=False
        )
        start = validate_state(state, inventory, recipe_hash=recipe_hash, rank=rank, world=world, manifest=resumed)
        history = [json.loads(line) for line in (args.output / "metrics.jsonl").read_text().splitlines()]
        if [row["update"] for row in history] != list(range(1, start + 1)):
            raise ValueError("resume requires exactly the committed metric prefix, without gaps or duplicate updates")
        if history[-1] != resumed["last_metrics"]:
            raise ValueError("last metric differs from the checkpoint")
        optimizer.load_state_dict(state["optimizer"])
        restore_rng_state(state["rng"], device)
        del state
    elif optimizer.state or any(
        torch.count_nonzero(p)
        for name, p in named
        if name.endswith(("lora_B.default.weight", "adapter_b", "sc_projection.weight"))
    ):
        raise RuntimeError("scratch initialization is not fresh/zero-B")
    if world == 2:
        initial_hash = assert_replicated_parameter_values(named)
        if args.resume:
            optimizer_hash = _assert_replicated_optimizer_state(optimizer, named)
            if (
                initial_hash != resumed["replicated_parameter_sha256"]
                or optimizer_hash != resumed["replicated_optimizer_sha256"]
            ):
                raise ValueError("restored trainables/optimizer do not match the committed fingerprints")
    else:
        initial_hash = None
    denoiser = DuoVLADenoiser(projector, DiffusionGemmaActionDecoder.from_block_diffusion_model(model), head).train()
    adapted.train()
    model.model.encoder.eval()
    processor = AutoProcessor.from_pretrained(
        base_config["model"]["id"], revision=base_config["model"]["revision"], local_files_only=True
    )
    dataset = LiberoParquetDataset(args.dataset, max_cached_files=377)
    state_norm, action_norm, stats = load_libero_normalizers(artifact_paths["normalization"])
    train_sampler = TaskUniformAnchorSampler(dataset.episodes, stats["split"]["train_episode_indices"])
    val_sampler = TaskUniformAnchorSampler(dataset.episodes, stats["split"]["validation_episode_indices"])
    if len(train_sampler.tasks) != 10 or set(train_sampler.tasks) != set(val_sampler.tasks):
        raise ValueError("Spatial-only ten-task split required")
    if len(stats["split"]["train_episode_indices"]) != 389 or len(stats["split"]["validation_episode_indices"]) != 43:
        raise ValueError("Spatial split changed")
    emit(
        event="model_ready",
        trainable_parameters=sum(p.numel() for _, p in named),
        trainable_tensors=len(named),
        sc_parameters=projector.sc_projection.weight.numel(),
        encoder_targets=len(targets),
        initial_parameter_sha256=initial_hash,
        resumed_update=start,
        fresh_optimizer=not bool(optimizer.state),
        pretrained_trainable_weights_loaded=bool(args.resume),
    )

    def materialize(plans, sampler):
        chunks = _materialize_canonical_batches(
            dataset, sampler, plans, state_normalizer=state_norm, action_normalizer=action_norm
        )
        batch = _coalesce_canonical_batches(chunks, physical_batch_size=args.batch_size, experimental_sc_batch=True)
        prefix_inputs = _processor_inputs(
            processor,
            batch.samples,
            torch.device("cpu"),
            prefix_geometry,
            expected_batch_size=args.batch_size,
            experimental_sc_batch=True,
        )
        return batch, plans, prefix_inputs

    def prepare(index):
        per_rank = args.batch_size // 8
        plans = make_update_plan(0, index, gradient_accumulation_steps=per_rank * world)
        return materialize(plans[rank * per_rank : (rank + 1) * per_rank], train_sampler)

    def predict(pair, states, valid, prefix, bootstrap):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return training_velocity(
                denoiser,
                pair.input_actions,
                pair.timesteps,
                states,
                bootstrap=bootstrap,
                prefix_cache=prefix.past_key_values,
                prefix_attention_mask=prefix.attention_mask,
                action_valid_mask=valid,
            )

    @torch.no_grad()
    def validate():
        denoiser.eval()
        totals = {"no_candidate": 0.0, "bootstrap": 0.0, "sampled_action_nfe4": 0.0}
        count = 0
        for selected, active_rows in validation_plan_groups(0x5A17, 2048, args.batch_size, rank, world):
            batch, selected, inputs = materialize(selected, val_sampler)
            clean, states, valid = (
                batch.clean_actions.to(device),
                batch.states.to(device),
                batch.action_valid_mask.to(device),
            )
            pair = _canonical_training_pair(clean, contract, selected)
            score_mask = valid.clone()
            score_mask[active_rows:] = False
            prefix = encode_diffusion_gemma_prefix(model, inputs.to(device))
            for name, bootstrap in (("no_candidate", False), ("bootstrap", True)):
                prediction = predict(pair, states, valid, prefix, bootstrap)
                component = masked_sse(prediction, pair.target, score_mask)
                totals[name] += float(component.squared_error_sum)
            count += component.element_count
            noise = torch.cat(
                [
                    torch.randn(
                        (8, 8, 7),
                        device=device,
                        dtype=torch.float32,
                        generator=torch.Generator(device=device).manual_seed(plan.flow_seed ^ 0x5343),
                    )
                    for plan in selected
                ]
            )
            with torch.autocast("cuda", dtype=torch.bfloat16):
                sampled = sample_action_flow(
                    denoiser,
                    noise,
                    states,
                    num_steps=4,
                    prefix_cache=prefix.past_key_values,
                    prefix_attention_mask=prefix.attention_mask,
                    action_valid_mask=valid,
                ).clamp(-1, 1)
            totals["sampled_action_nfe4"] += float(masked_sse(sampled, clean, score_mask).squared_error_sum)
        count = synchronize_scalar(count, device)
        result = {
            f"validation_{name}_loss": synchronize_scalar(value, device) / count for name, value in totals.items()
        }
        denoiser.train()
        model.model.encoder.eval()
        return result

    @torch.no_grad()
    def record_serving_reference():
        """Known raw policy output for independent save/load checks at the same physical width."""
        from rollout_libero_spatial import encode_image

        samples = list(batch.samples[:8])
        padded = samples + [samples[-1]] * (args.batch_size - len(samples))
        inputs = _processor_inputs(
            processor, padded, device, prefix_geometry, expected_batch_size=args.batch_size, experimental_sc_batch=True
        )
        reference_prefix = encode_diffusion_gemma_prefix(model, inputs)
        reference_states = state_norm.normalize(torch.stack([sample.observation.state for sample in padded])).to(device)
        noise_rows = [
            torch.randn(
                (1, 8, 7),
                device=device,
                dtype=torch.float32,
                generator=torch.Generator(device=device).manual_seed(7 + index),
            )
            for index in range(8)
        ]
        noise = torch.cat(noise_rows + [noise_rows[-1]] * (args.batch_size - 8))
        with torch.autocast("cuda", dtype=torch.bfloat16):
            raw = sample_action_flow(
                denoiser,
                noise,
                reference_states,
                num_steps=4,
                prefix_cache=reference_prefix.past_key_values,
                prefix_attention_mask=reference_prefix.attention_mask,
                action_valid_mask=torch.ones((args.batch_size, 8), device=device, dtype=torch.bool),
            )[:8]
        rows = [
            {
                "instruction": sample.instruction,
                "task_id": sample.task_index,
                "state": sample.observation.state.tolist(),
                "inference_seed": 7 + index,
                "agentview": encode_image(sample.observation.third_person),
                "wrist": encode_image(sample.observation.wrist),
            }
            for index, sample in enumerate(samples)
        ]
        write_json(
            args.output / "serving_reference.json",
            {
                "rows": rows,
                "raw_normalized_actions": raw.cpu().tolist(),
                "nfe": 4,
                "training_physical_batch_size": args.batch_size,
            },
        )

    def save(update, metric):
        if world == 2:
            parameter_hash = assert_replicated_parameter_values(named)
            optimizer_hash = _assert_replicated_optimizer_state(optimizer, named)
        else:
            parameter_hash = optimizer_hash = None
        validate_optimizer_state_dict(optimizer.state_dict(), inventory, expected_update=update)
        if rank == 0:
            record_serving_reference()
        scratch = args.output / "checkpoint_staging"
        scratch.mkdir(exist_ok=True)
        rank_path = scratch / f"training_rank_{rank:03d}.pt"
        torch.save(
            {
                "schema": STATE_SCHEMA,
                "update": update,
                "recipe_sha256": recipe_hash,
                "optimizer": optimizer.state_dict(),
                "optimizer_parameter_inventory": inventory,
                "rng": capture_rng_state(device),
                "rank": rank,
                "world_size": world,
            },
            rank_path,
        )
        if rank == 0:
            save_file(encoder_adapter_state(model), str(scratch / "encoder_adapter.safetensors"))
        if world == 2:
            dist.barrier()
        extras = {f"training_rank_{i:03d}": scratch / f"training_rank_{i:03d}.pt" for i in range(world)}
        extras.update(artifact_paths)
        extras.update(
            encoder_adapter=scratch / "encoder_adapter.safetensors",
            encoder_config=args.output / "encoder_config.json",
            recipe=args.output / "recipe.json",
            action_interface_config=args.output / "action_interface_config.json",
            serving_reference=args.output / "serving_reference.json",
        )
        checkpoint = args.output / "checkpoints" / f"update-{update:06d}"
        manifest = {
            "kind": RUN_SCHEMA,
            "recipe_sha256": recipe_hash,
            "update": update,
            "model_id": reference["model_id"],
            "model_revision": reference["model_revision"],
            "config_sha256": reference["config_sha256"],
            "base_config_role": "data_geometry_reference_only",
            "effective_recipe_artifact": "artifacts/recipe.json",
            "action_self_conditioning": SC_VERSION,
            "encoder_adapted": True,
            "policy_contract": reference["policy_contract"],
            "policy_contract_sha256": reference["policy_contract_sha256"],
            "training_suites_json": reference["training_suites_json"],
            "last_metrics": metric,
            "replicated_parameter_sha256": parameter_hash,
            "replicated_optimizer_sha256": optimizer_hash,
            "complete": update == TOTAL_UPDATES,
        }
        save_trainable_checkpoint(
            checkpoint, adapted_model=adapted, interface_modules=modules, manifest=manifest, additional_artifacts=extras
        )
        if rank == 0:
            read_sc_checkpoint(checkpoint, source_root=ROOT)
            write_json(
                args.output / "latest_checkpoint.json",
                {
                    "path": str(checkpoint),
                    "update": update,
                    "manifest_sha256": file_sha256(checkpoint / "manifest.json"),
                },
            )
        emit(event="checkpoint_saved", update=update, path=str(checkpoint))

    def qualification(pair, states, valid, prefix, prediction, bootstrap, count, index):
        if not args.verify_native:
            return
        native_gradients = {name: p.grad.detach().cpu().clone() for name, p in named}
        optimizer.zero_grad(set_to_none=True)
        language = model.model.encoder.language_model
        recompute = language.forward
        language.forward = language._duovla_original_forward
        try:
            native = encode_diffusion_gemma_prefix_trainable(model, current_inputs)
            output = predict(pair, states, valid, native, bootstrap)
            masked_sse(output, pair.target, valid).loss_for_total(count).backward()
            torch.testing.assert_close(prediction.detach(), output.detach(), rtol=1e-5, atol=1e-6)
            maximum = 0.0
            for name, parameter in named:
                expected = native_gradients[name]
                torch.testing.assert_close(parameter.grad.detach().cpu(), expected, rtol=1e-5, atol=1e-6, msg=name)
                maximum = max(maximum, float((parameter.grad.detach().cpu() - expected).abs().max()))
            emit(
                event="native_recompute_sc_parity",
                update=index + 1,
                bootstrap=bootstrap,
                tensors=len(named),
                gradient_max_abs_error=maximum,
                output_max_abs_error=float((prediction.detach().float() - output.detach().float()).abs().max()),
            )
        finally:
            language.forward = recompute

    for signum in (signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, stop_request)
    end = args.probe_steps or min(TOTAL_UPDATES, args.stop_after or TOTAL_UPDATES)
    if start >= end:
        raise ValueError("checkpoint already reached requested boundary")
    torch.cuda.reset_peak_memory_stats(device)
    latest = None
    with ThreadPoolExecutor(max_workers=1) as executor:
        pending = executor.submit(prepare, start)
        for index in range(start, end):
            began = time.perf_counter()
            if pending is None:
                pending = executor.submit(prepare, index)
            batch, plans, cpu_inputs = pending.result()
            update = index + 1
            boundary = update % 500 == 0 or update == end
            pending = executor.submit(prepare, index + 1) if not boundary else None
            clean, states, valid = (
                batch.clean_actions.to(device),
                batch.states.to(device),
                batch.action_valid_mask.to(device),
            )
            current_inputs = cpu_inputs.to(device)
            pair = _canonical_training_pair(clean, contract, plans)
            count = int(synchronize_scalar(masked_element_count(valid, action_dim=7), device))
            bootstrap = (index % 2 == 0) if args.probe_steps else bootstrap_for_update(0, index)
            optimizer.zero_grad(set_to_none=True)
            if any(group["lr"] != LR for group in optimizer.param_groups):
                raise RuntimeError("fixed learning rate changed")
            prefix = encode_diffusion_gemma_prefix_trainable(model, current_inputs)
            cache_versions = [
                (layer.keys._version, layer.values._version, layer.keys.shape)
                for layer in prefix.past_key_values.layers
            ]
            if not prefix.past_key_values.layers[0].keys.requires_grad:
                raise RuntimeError("encoder gradients are disconnected")
            prediction = predict(pair, states, valid, prefix, bootstrap)
            if args.probe_steps and index == 0:
                with torch.no_grad():
                    absent = predict(pair, states, valid, prefix, False)
                if not torch.equal(prediction.detach(), absent):
                    raise RuntimeError("zero-init SC output does not match absent candidate")
                emit(event="zero_init_sc_parity", max_abs_error=0.0)
                del absent
            component = masked_sse(prediction, pair.target, valid)
            component.loss_for_total(count).backward()
            qualification(pair, states, valid, prefix, prediction, bootstrap, count, index)
            if cache_versions != [
                (layer.keys._version, layer.values._version, layer.keys.shape)
                for layer in prefix.past_key_values.layers
            ]:
                raise RuntimeError("SC passes mutated the prefix cache")
            if any(p.grad is not None for p in model.parameters() if not p.requires_grad):
                raise RuntimeError("frozen base received gradients")
            if world == 2:
                _assert_distributed_gradient_health([p for _, p in named])
                _sum_data_parallel_gradients_(named)
                if index == start:
                    _assert_replicated_gradient_values(named)
            elif any(p.grad is None or not p.grad.isfinite().all() for _, p in named):
                raise RuntimeError("missing/nonfinite training gradients")
            sc_gradient = float(projector.sc_projection.weight.grad.norm())
            encoder_gradient = float(
                torch.linalg.vector_norm(torch.stack([p.grad.norm() for _, p in encoder_parameters]))
            )
            if (bootstrap and sc_gradient == 0) or (not bootstrap and sc_gradient != 0) or encoder_gradient == 0:
                raise RuntimeError("SC/encoder gradient contract failed")
            grad_norm = float(torch.nn.utils.clip_grad_norm_([p for _, p in named], 1.0, error_if_nonfinite=True))
            optimizer.step()
            if world == 2 and index == start:
                assert_replicated_parameter_values(named)
                _assert_replicated_optimizer_state(optimizer, named)
                emit(event="first_update_replicas_verified", update=update)
            loss = synchronize_scalar(float(component.squared_error_sum.detach()), device) / count
            torch.cuda.synchronize(device)
            seconds = synchronize_scalar(time.perf_counter() - began, device, op=dist.ReduceOp.MAX)
            peak = synchronize_scalar(torch.cuda.max_memory_allocated(device), device, op=dist.ReduceOp.MAX)
            memory_stats = torch.cuda.memory_stats(device)
            retries = synchronize_scalar(memory_stats.get("num_alloc_retries", 0), device, op=dist.ReduceOp.MAX)
            reserved = synchronize_scalar(torch.cuda.max_memory_reserved(device), device, op=dist.ReduceOp.MAX)
            latest = {
                "update": update,
                "train_loss": loss,
                "sc_bootstrap": bootstrap,
                "learning_rate": LR,
                "global_batch_size": args.batch_size * world,
                "encoder_gradient_norm": encoder_gradient,
                "sc_gradient_norm": sc_gradient,
                "gradient_norm": grad_norm,
                "update_seconds": seconds,
                "peak_memory_gib": peak / 2**30,
                "peak_reserved_memory_gib": reserved / 2**30,
                "allocator_retries": int(retries),
                "utc": datetime.now(UTC).isoformat(),
            }
            del prefix, prediction, component
            if boundary and not args.probe_steps:
                emit(event="validation_started", update=update)
                before = time.perf_counter()
                latest.update(validate())
                latest["validation_seconds"] = time.perf_counter() - before
            if rank == 0:
                with (args.output / "metrics.jsonl").open("a") as handle:
                    handle.write(json.dumps(latest, allow_nan=False) + "\n")
                write_json(
                    args.output / "progress.json", {"status": "running", "target_updates": TOTAL_UPDATES, **latest}
                )
            emit(**latest)
            stopped = bool(synchronize_scalar(int(STOP), device, op=dist.ReduceOp.MAX))
            if not args.probe_steps and (update % 2500 == 0 or update == end or stopped):
                if pending is not None:
                    pending.result()
                save(update, latest)
            if stopped:
                break
    if rank == 0:
        status = (
            "probe_complete" if args.probe_steps else "complete" if latest["update"] == TOTAL_UPDATES else "stopped"
        )
        write_json(args.output / "progress.json", {"status": status, "target_updates": TOTAL_UPDATES, **latest})
    emit(event="finished", update=latest["update"], probe=bool(args.probe_steps))
    if world == 2:
        dist.destroy_process_group()


if __name__ == "__main__":
    try:
        main()
    except torch.OutOfMemoryError:
        emit(event="out_of_memory", rank=int(os.environ.get("RANK", "0")))
        raise
