"""Pinned DiffusionGemma loading for the qualified one- and two-GPU BF16 setups."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from torch import nn

from duo_vla.backbones.diffusion_gemma import decoder_attention_lora_targets
from duo_vla.backbones.sample_isolated_experts import (
    GROUPED_MM_EXPERTS_IMPLEMENTATION,
    validate_diffusion_gemma_grouped_mm_expert_topology,
)


@dataclass(frozen=True, slots=True)
class DiffusionGemmaModelSpec:
    model_id: str = "google/diffusiongemma-26B-A4B-it"
    revision: str = "f7f5b7f5fa82ffc52addd066915886d497f5517b"
    expected_hidden_size: int = 2816
    expected_num_layers: int = 30
    expected_canvas_length: int = 256
    expected_experts_implementation: str = GROUPED_MM_EXPERTS_IMPLEMENTATION
    expected_decoder_attention_lora_target_count: int = 115
    expected_decoder_attention_lora_projection_histogram: tuple[tuple[str, int], ...] = (
        ("q_proj", 30),
        ("k_proj", 30),
        ("v_proj", 25),
        ("o_proj", 30),
    )
    expected_decoder_attention_lora_v_projection_layers: tuple[int, ...] = (
        0,
        1,
        2,
        3,
        4,
        6,
        7,
        8,
        9,
        10,
        12,
        13,
        14,
        15,
        16,
        18,
        19,
        20,
        21,
        22,
        24,
        25,
        26,
        27,
        28,
    )


DEFAULT_DIFFUSION_GEMMA_SPEC = DiffusionGemmaModelSpec()
DATA_PARALLEL_REPLICA_MODE = "data_parallel"


def expected_decoder_attention_lora_targets(
    spec: DiffusionGemmaModelSpec = DEFAULT_DIFFUSION_GEMMA_SPEC,
) -> tuple[str, ...]:
    """Construct the exact PEFT target names pinned by a model topology contract."""

    v_layers = set(spec.expected_decoder_attention_lora_v_projection_layers)
    targets = tuple(
        f"model.decoder.layers.{layer}.self_attn.{projection}"
        for layer in range(spec.expected_num_layers)
        for projection in ("q_proj", "k_proj", "v_proj", "o_proj")
        if projection != "v_proj" or layer in v_layers
    )
    if len(targets) != spec.expected_decoder_attention_lora_target_count:
        raise ValueError("pinned LoRA target names disagree with the expected target count")
    histogram = tuple(
        (projection, sum(target.endswith(f".{projection}") for target in targets))
        for projection in ("q_proj", "k_proj", "v_proj", "o_proj")
    )
    if histogram != spec.expected_decoder_attention_lora_projection_histogram:
        raise ValueError("pinned LoRA target names disagree with the expected projection histogram")
    return targets


def expected_decoder_attention_lora_adapter_config(
    *,
    spec: DiffusionGemmaModelSpec = DEFAULT_DIFFUSION_GEMMA_SPEC,
    rank: int = 16,
    alpha: int = 32,
    dropout: float = 0.0,
    peft_version: str = "0.20.0",
) -> dict[str, Any]:
    """Return the complete pinned PEFT serialization contract, excluding target order."""

    return {
        "alora_invocation_tokens": None,
        "alpha_pattern": {},
        "arrow_config": None,
        "auto_mapping": {
            "base_model_class": "DiffusionGemmaForBlockDiffusion",
            "parent_library": "transformers.models.diffusion_gemma.modeling_diffusion_gemma",
        },
        "base_model_name_or_path": spec.model_id,
        "bias": "none",
        "corda_config": None,
        "ensure_weight_tying": False,
        "eva_config": None,
        "exclude_modules": None,
        "fan_in_fan_out": False,
        "inference_mode": True,
        "init_lora_weights": True,
        "layer_replication": None,
        "layers_pattern": None,
        "layers_to_transform": None,
        "loftq_config": {},
        "lora_alpha": alpha,
        "lora_bias": False,
        "lora_dropout": dropout,
        "lora_ga_config": None,
        "megatron_config": None,
        "megatron_core": "megatron.core",
        "modules_to_save": None,
        "monteclora_config": None,
        "peft_type": "LORA",
        "peft_version": peft_version,
        "qalora_group_size": 16,
        "r": rank,
        "rank_pattern": {},
        "revision": None,
        "target_modules": list(expected_decoder_attention_lora_targets(spec)),
        "target_parameters": None,
        "task_type": None,
        "trainable_token_indices": None,
        "use_bdlora": None,
        "use_dora": False,
        "use_qalora": False,
        "use_rslora": False,
        "velora_config": None,
    }


def validate_decoder_attention_lora_adapter_config(
    path: str | Path,
    *,
    spec: DiffusionGemmaModelSpec = DEFAULT_DIFFUSION_GEMMA_SPEC,
    rank: int = 16,
    alpha: int = 32,
    dropout: float = 0.0,
    peft_version: str = "0.20.0",
) -> dict[str, Any]:
    """Authenticate the semantics of a serialized PEFT adapter before it is loaded."""

    config_path = Path(path)
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot parse LoRA adapter config: {config_path}") from exc
    if not isinstance(payload, dict):
        raise ValueError("LoRA adapter config must be a JSON object")
    required = expected_decoder_attention_lora_adapter_config(
        spec=spec,
        rank=rank,
        alpha=alpha,
        dropout=dropout,
        peft_version=peft_version,
    )
    expected_targets = set(required.pop("target_modules"))
    mismatches = {
        name: {"expected": expected, "observed": payload.get(name)}
        for name, expected in required.items()
        if payload.get(name) != expected
    }
    unexpected_fields = sorted(set(payload) - set(required) - {"target_modules"})
    missing_fields = sorted(set(required) - set(payload))
    observed_targets = payload.get("target_modules")
    target_valid = (
        isinstance(observed_targets, list)
        and all(isinstance(value, str) for value in observed_targets)
        and len(observed_targets) == len(set(observed_targets))
        and set(observed_targets) == expected_targets
    )
    if mismatches or unexpected_fields or missing_fields or not target_valid:
        missing = sorted(expected_targets - set(observed_targets or ())) if isinstance(observed_targets, list) else []
        extra = sorted(set(observed_targets or ()) - expected_targets) if isinstance(observed_targets, list) else []
        raise ValueError(
            "LoRA adapter config violates the decoder-attention contract: "
            f"fields={mismatches}, "
            f"missing_fields={missing_fields}, unexpected_fields={unexpected_fields}, "
            f"target_count={len(observed_targets) if isinstance(observed_targets, list) else None}, "
            f"missing_targets={missing}, extra_targets={extra}"
        )
    return payload


def expected_decoder_attention_lora_weight_schema(
    *,
    spec: DiffusionGemmaModelSpec = DEFAULT_DIFFUSION_GEMMA_SPEC,
    rank: int = 16,
) -> dict[str, tuple[int, int]]:
    """Return exact consolidated PEFT tensor names and global shapes for the pinned topology."""

    if rank <= 0:
        raise ValueError("LoRA rank must be positive")
    v_layers = set(spec.expected_decoder_attention_lora_v_projection_layers)
    schema: dict[str, tuple[int, int]] = {}
    for target in expected_decoder_attention_lora_targets(spec):
        layer = int(target.split(".layers.", maxsplit=1)[1].split(".", maxsplit=1)[0])
        projection = target.rsplit(".", maxsplit=1)[1]
        sliding = layer in v_layers
        if projection == "q_proj":
            input_features, output_features = spec.expected_hidden_size, 4096 if sliding else 8192
        elif projection in {"k_proj", "v_proj"}:
            input_features, output_features = spec.expected_hidden_size, 2048 if sliding else 1024
        elif projection == "o_proj":
            input_features, output_features = 4096 if sliding else 8192, spec.expected_hidden_size
        else:  # pragma: no cover - target construction makes this unreachable
            raise AssertionError(f"unsupported LoRA projection: {projection}")
        prefix = f"base_model.model.{target}"
        schema[f"{prefix}.lora_A.weight"] = (rank, input_features)
        schema[f"{prefix}.lora_B.weight"] = (output_features, rank)
    if len(schema) != 2 * spec.expected_decoder_attention_lora_target_count:
        raise AssertionError("LoRA tensor schema count differs from the pinned topology")
    return schema


def validate_decoder_attention_lora_weights(
    path: str | Path,
    *,
    spec: DiffusionGemmaModelSpec = DEFAULT_DIFFUSION_GEMMA_SPEC,
    rank: int = 16,
) -> dict[str, tuple[int, int]]:
    """Fail closed on missing, extra, malformed, non-FP32, or non-finite saved LoRA tensors."""

    try:
        import torch
        from safetensors import safe_open
    except ImportError as exc:  # pragma: no cover - train environment always pins both
        raise ImportError("install torch and safetensors before validating LoRA weights") from exc
    expected = expected_decoder_attention_lora_weight_schema(spec=spec, rank=rank)
    weights_path = Path(path)
    try:
        with safe_open(weights_path, framework="pt", device="cpu") as handle:
            observed_names = set(handle.keys())
            if observed_names != set(expected):
                raise ValueError(
                    "LoRA weight tensor names violate the decoder-attention contract: "
                    f"missing={sorted(set(expected) - observed_names)}, "
                    f"extra={sorted(observed_names - set(expected))}"
                )
            for name, expected_shape in expected.items():
                tensor = handle.get_tensor(name)
                if tuple(tensor.shape) != expected_shape or tensor.dtype != torch.float32:
                    raise ValueError(
                        f"LoRA tensor schema mismatch for {name}: shape={tuple(tensor.shape)}, dtype={tensor.dtype}"
                    )
                if not bool(torch.isfinite(tensor).all()):
                    raise ValueError(f"LoRA tensor contains non-finite values: {name}")
    except (OSError, RuntimeError) as exc:
        raise ValueError(f"cannot parse LoRA safetensors: {weights_path}") from exc
    return expected


def symmetric_diffusion_gemma_tp_plan(config: Any) -> dict[str, str]:
    """Build an explicit plan for both sides of DiffusionGemma's tied text stack.

    Transformers 5.16.1's recursively collected automatic plan can contain only the encoder aliases. Checkpoint
    finalization then ties those weights to unsharded decoder parameters, leaving TP forward hooks around ordinary
    tensors. Explicit symmetric rules keep the source (decoder) and alias (encoder) modules compatible after tying.
    """

    text_plan = dict(config.text_config.base_model_tp_plan or {})
    if not text_plan:
        raise RuntimeError("DiffusionGemma text config does not define a tensor-parallel plan")
    plan: dict[str, str] = {}
    for prefix in ("model.encoder.language_model", "model.decoder"):
        plan.update({f"{prefix}.{name}": style for name, style in text_plan.items()})
    # Gemma4 vision projections are clippable wrapper modules around an inner Linear. In Transformers 5.16.1 the
    # published vision plan installs a DTensor input hook on the wrapper while checkpoint loading can leave its inner
    # Linear local, producing a mixed Tensor/DTensor matmul. The frozen tower is small enough to replicate safely.
    return plan


def load_diffusion_gemma_bf16_tp(
    spec: DiffusionGemmaModelSpec = DEFAULT_DIFFUSION_GEMMA_SPEC,
    *,
    tp_size: int = 2,
    replica_mode: str | None = None,
    local_files_only: bool = False,
    **from_pretrained_kwargs: Any,
) -> nn.Module:
    """Load the pinned checkpoint with either native TP or explicit full-model DP replication."""

    if tp_size <= 0:
        raise ValueError("tp_size must be positive")
    if replica_mode not in {None, DATA_PARALLEL_REPLICA_MODE}:
        raise ValueError(f"unsupported DiffusionGemma replica mode: {replica_mode!r}")
    if replica_mode is not None and tp_size != 1:
        raise ValueError("data-parallel replica loading requires tp_size=1")
    if "device_map" in from_pretrained_kwargs:
        raise ValueError("device_map must not be combined with native tensor parallel loading")
    if "experts_implementation" in from_pretrained_kwargs:
        raise ValueError("experts_implementation is pinned by DiffusionGemmaModelSpec")
    if spec.expected_experts_implementation != GROUPED_MM_EXPERTS_IMPLEMENTATION:
        raise ValueError(
            "the pinned DiffusionGemma loader supports only "
            f"experts_implementation={GROUPED_MM_EXPERTS_IMPLEMENTATION!r}"
        )
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if tp_size > 1 and world_size != tp_size:
        raise RuntimeError(
            f"tp_size={tp_size} requires torchrun with WORLD_SIZE={tp_size}; observed WORLD_SIZE={world_size}"
        )
    if replica_mode == DATA_PARALLEL_REPLICA_MODE and world_size != 2:
        raise RuntimeError(f"data-parallel replica loading requires WORLD_SIZE=2; observed WORLD_SIZE={world_size}")
    if tp_size == 1 and replica_mode is None and world_size != 1:
        raise RuntimeError("TP=1 loading inside a multi-process job requires explicit replica_mode='data_parallel'")

    try:
        import torch
        from transformers import DiffusionGemmaConfig, DiffusionGemmaForBlockDiffusion

        try:
            from transformers import DistributedConfig
        except ImportError:  # Transformers 5.15 exposes this from the distributed package only.
            from transformers.distributed import DistributedConfig
    except ImportError as exc:  # pragma: no cover - only minimal installs omit these dependencies
        raise ImportError("install duo-vla's train dependencies before loading DiffusionGemma") from exc

    if not torch.cuda.is_available():
        raise RuntimeError("DiffusionGemma BF16 loading requires CUDA")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("the selected CUDA device does not support BF16")

    config = DiffusionGemmaConfig.from_pretrained(
        spec.model_id,
        revision=spec.revision,
        local_files_only=local_files_only,
    )
    topology_kwargs: dict[str, Any]
    if tp_size == 1:
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        expected_local_ranks = {0} if replica_mode is None else set(range(world_size))
        if local_rank not in expected_local_ranks:
            raise RuntimeError(
                f"TP=1 {replica_mode or 'standalone'} loading received invalid LOCAL_RANK={local_rank}; "
                f"expected one of {sorted(expected_local_ranks)}"
            )
        topology_kwargs = {"device_map": {"": local_rank}}
    else:
        topology_kwargs = {
            "distributed_config": DistributedConfig(
                tp_size=tp_size,
                tp_plan=symmetric_diffusion_gemma_tp_plan(config),
            )
        }
    model = DiffusionGemmaForBlockDiffusion.from_pretrained(
        spec.model_id,
        revision=spec.revision,
        config=config,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        local_files_only=local_files_only,
        attn_implementation="sdpa",
        experts_implementation=spec.expected_experts_implementation,
        **topology_kwargs,
        **from_pretrained_kwargs,
    )
    model._tp_size = tp_size
    model._replica_mode = replica_mode
    model._data_parallel_world_size = world_size if replica_mode == DATA_PARALLEL_REPLICA_MODE else 1
    config = model.config
    observed = (
        config.text_config.hidden_size,
        config.text_config.num_hidden_layers,
        config.canvas_length,
    )
    expected = (spec.expected_hidden_size, spec.expected_num_layers, spec.expected_canvas_length)
    if observed != expected:
        raise RuntimeError(f"pinned model architecture changed: expected {expected}, observed {observed}")
    try:
        validate_diffusion_gemma_grouped_mm_expert_topology(
            model,
            expected_num_layers=spec.expected_num_layers,
            expected_experts_implementation=spec.expected_experts_implementation,
        )
    except (TypeError, ValueError, RuntimeError) as exc:
        raise RuntimeError(
            f"pinned model revision {spec.revision} has an incompatible grouped-MM expert topology: {exc}"
        ) from exc
    try:
        decoder_attention_lora_targets(
            model,
            expected_target_count=spec.expected_decoder_attention_lora_target_count,
            expected_projection_histogram=dict(spec.expected_decoder_attention_lora_projection_histogram),
            expected_v_projection_layers=spec.expected_decoder_attention_lora_v_projection_layers,
        )
    except (TypeError, ValueError, RuntimeError) as exc:
        raise RuntimeError(
            f"pinned model revision {spec.revision} has an incompatible decoder attention LoRA topology: {exc}"
        ) from exc
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model
