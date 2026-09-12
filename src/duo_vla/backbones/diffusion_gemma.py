"""Continuous-input adapter and decoder-only LoRA utilities for Hugging Face DiffusionGemma."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn

_ATTENTION_PROJECTION_NAMES = ("q_proj", "k_proj", "v_proj", "o_proj")


@dataclass(frozen=True, slots=True)
class DiffusionGemmaPrefix:
    """Read-only layer-wise prompt KV and the corresponding valid-position mask."""

    past_key_values: Any
    attention_mask: Tensor


@dataclass(frozen=True, slots=True)
class LoRAParameterPartition:
    """Named LoRA parameters split by their TP storage semantics."""

    replicated: tuple[tuple[str, nn.Parameter], ...]
    sharded: tuple[tuple[str, nn.Parameter], ...]


class DiffusionGemmaActionDecoder(nn.Module):
    """Run the native DiffusionGemma decoder stack from continuous embeddings.

    Transformers 5.15 exposes ``inputs_embeds`` for the encoder but not the diffusion decoder. This adapter mirrors the
    small portion of the upstream decoder forward between token embedding and ``BaseModelOutput``. Keeping it in a
    separate module makes the compatibility surface explicit and testable against the native token-ID path.
    """

    def __init__(self, decoder: nn.Module) -> None:
        super().__init__()
        required = (
            "text_config",
            "layers",
            "norm",
            "rotary_emb",
            "self_conditioning",
            "unique_layer_types",
            "create_diffusion_decoder_attention_mask",
        )
        missing = [name for name in required if not hasattr(decoder, name)]
        if missing:
            raise TypeError(f"unsupported DiffusionGemma decoder; missing {missing}")
        self.decoder = decoder

    @classmethod
    def from_block_diffusion_model(cls, model: nn.Module) -> DiffusionGemmaActionDecoder:
        try:
            decoder = model.model.decoder
        except AttributeError as exc:
            raise TypeError("expected DiffusionGemmaForBlockDiffusion-compatible model.model.decoder") from exc
        return cls(decoder)

    def decode_actions(
        self,
        action_embeddings: Tensor,
        *,
        prefix_cache: Any,
        prefix_attention_mask: Tensor,
        action_valid_mask: Tensor,
    ) -> Tensor:
        if action_embeddings.ndim != 3:
            raise ValueError("action_embeddings must have shape [batch, horizon, hidden]")
        batch_size, horizon, hidden_size = action_embeddings.shape
        if hidden_size != self.decoder.text_config.hidden_size:
            raise ValueError(
                f"action embedding width {hidden_size} does not match decoder width "
                f"{self.decoder.text_config.hidden_size}"
            )
        if prefix_cache is None:
            raise ValueError("prefix_cache must not be None")
        if prefix_attention_mask.ndim != 2 or prefix_attention_mask.shape[0] != batch_size:
            raise ValueError("prefix_attention_mask must have shape [batch, prefix]")
        if action_valid_mask.shape != (batch_size, horizon):
            raise ValueError("action_valid_mask must have shape [batch, horizon]")
        valid_actions = action_valid_mask.bool()
        if bool((valid_actions.sum(dim=1) == 0).any()):
            raise ValueError("every batch item must contain at least one valid action")

        # Match the native first-step path: zero self-conditioning still applies the decoder's post RMS normalization.
        hidden_states = self.decoder.self_conditioning(action_embeddings, torch.zeros_like(action_embeddings))

        cache_length = prefix_cache.get_seq_length(layer_idx=0)
        if isinstance(cache_length, Tensor):
            cache_length = int(cache_length.item())
        prefix_length = prefix_attention_mask.shape[1]
        if cache_length != prefix_length:
            raise ValueError(
                f"prefix attention-mask width {prefix_length} does not match cache sequence length {cache_length}"
            )
        layer_types = tuple(self.decoder.text_config.layer_types)
        sliding_window = getattr(self.decoder.text_config, "sliding_window", None)
        if "sliding_attention" in layer_types and (
            type(sliding_window) is not int or sliding_window <= 0 or prefix_length > sliding_window
        ):
            raise ValueError(
                f"prefix length {prefix_length} exceeds or cannot validate the native sliding window {sliding_window!r}"
            )
        # The cache width is the padded batch width, not the semantic prefix length.  Positioning actions after the
        # physical cache width makes the same example depend on the longest instruction in its training microbatch.
        # Continue each sample from its number of valid prefix tokens so left padding is positionally invisible and
        # batched training matches the unpadded batch-size-one serving path.
        valid_prefix_lengths = prefix_attention_mask.to(device=hidden_states.device, dtype=torch.long).sum(dim=1)
        decoder_position_ids = (
            valid_prefix_lengths[:, None]
            + torch.arange(
                horizon,
                device=hidden_states.device,
                dtype=torch.long,
            )[None, :]
        )

        decoder_attention_mask = torch.cat(
            (
                prefix_attention_mask.to(device=hidden_states.device, dtype=torch.bool),
                valid_actions.to(device=hidden_states.device),
            ),
            dim=1,
        )
        mask_mapping = self.decoder.create_diffusion_decoder_attention_mask(
            config=self.decoder.text_config,
            inputs_embeds=hidden_states,
            past_key_values=prefix_cache,
            decoder_attention_mask=decoder_attention_mask,
        )

        position_embeddings = {
            layer_type: self.decoder.rotary_emb(hidden_states, decoder_position_ids, layer_type)
            for layer_type in self.decoder.unique_layer_types
        }
        for index, decoder_layer in enumerate(self.decoder.layers[: self.decoder.text_config.num_hidden_layers]):
            layer_type = self.decoder.text_config.layer_types[index]
            hidden_states = decoder_layer(
                hidden_states,
                position_embeddings=position_embeddings[layer_type],
                attention_mask=mask_mapping[layer_type],
                position_ids=decoder_position_ids,
                past_key_values=prefix_cache,
            )
        hidden_states = self.decoder.norm(hidden_states)
        return hidden_states * valid_actions.to(device=hidden_states.device, dtype=hidden_states.dtype)[..., None]


@torch.no_grad()
def encode_diffusion_gemma_prefix(
    model: nn.Module,
    processor_inputs: Mapping[str, Any],
) -> DiffusionGemmaPrefix:
    """Encode processor output once and return a detached, read-only conditioning cache."""

    return encode_diffusion_gemma_prefix_trainable(model, processor_inputs)


def encode_diffusion_gemma_prefix_trainable(
    model: nn.Module,
    processor_inputs: Mapping[str, Any],
) -> DiffusionGemmaPrefix:
    """Explicit opt-in differentiable prefill; preserve gradients through the prefix KV cache.

    The existing frozen entry point above remains the default for all qualified recipes.
    """

    if "input_ids" not in processor_inputs or "attention_mask" not in processor_inputs:
        raise ValueError("processor_inputs must include input_ids and attention_mask")
    try:
        encoder = model.model.encoder
    except AttributeError as exc:
        raise TypeError("expected DiffusionGemmaForBlockDiffusion-compatible model.model.encoder") from exc
    encoder_kwargs = dict(processor_inputs)
    attention_mask = encoder_kwargs.get("attention_mask")
    if not isinstance(attention_mask, Tensor) or attention_mask.ndim != 2:
        raise ValueError("processor attention_mask must be a rank-2 tensor")
    valid_prefix = attention_mask.bool()
    if bool((valid_prefix.sum(dim=1) == 0).any()):
        raise ValueError("every prefix must contain at least one valid token")
    # Transformers otherwise uses arange(batch_width), so left padding shifts RoPE positions.  Canonical per-sample
    # positions make a sample invariant to its batch companions.  Keep padding at position zero; it remains masked.
    canonical_position_ids = valid_prefix.long().cumsum(dim=1) - 1
    canonical_position_ids.masked_fill_(~valid_prefix, 0)
    supplied_position_ids = encoder_kwargs.get("position_ids")
    if supplied_position_ids is not None and (
        not isinstance(supplied_position_ids, Tensor)
        or supplied_position_ids.shape != canonical_position_ids.shape
        or not torch.equal(supplied_position_ids.to(canonical_position_ids.device), canonical_position_ids)
    ):
        raise ValueError("processor position_ids do not match canonical padding-invariant prefix positions")
    encoder_kwargs["position_ids"] = canonical_position_ids
    outputs = encoder(**encoder_kwargs)
    if outputs.past_key_values is None:
        raise RuntimeError("DiffusionGemma encoder did not return a KV cache")
    return DiffusionGemmaPrefix(
        past_key_values=outputs.past_key_values,
        attention_mask=valid_prefix,
    )


def _validate_expected_decoder_lora_topology(
    *,
    expected_target_count: int | None,
    expected_projection_histogram: Mapping[str, int] | None,
    expected_v_projection_layers: Sequence[int] | None,
) -> tuple[int, tuple[tuple[str, int], ...], tuple[int, ...]] | None:
    expectations = (expected_target_count, expected_projection_histogram, expected_v_projection_layers)
    if all(value is None for value in expectations):
        return None
    if any(value is None for value in expectations):
        raise ValueError(
            "expected decoder LoRA count, projection histogram, and v-projection layers must be provided together"
        )
    if type(expected_target_count) is not int or expected_target_count <= 0:
        raise ValueError("expected decoder LoRA target count must be a positive integer")
    assert expected_projection_histogram is not None
    expected_names = set(_ATTENTION_PROJECTION_NAMES)
    observed_names = set(expected_projection_histogram)
    if observed_names != expected_names:
        raise ValueError(
            "expected decoder LoRA projection histogram fields differ: "
            f"missing={sorted(expected_names - observed_names)}, extra={sorted(observed_names - expected_names)}"
        )
    normalized_histogram: list[tuple[str, int]] = []
    for name in _ATTENTION_PROJECTION_NAMES:
        count = expected_projection_histogram[name]
        if type(count) is not int or count < 0:
            raise ValueError(f"expected decoder LoRA {name} count must be a nonnegative integer")
        normalized_histogram.append((name, count))
    if sum(count for _, count in normalized_histogram) != expected_target_count:
        raise ValueError("expected decoder LoRA target count does not equal the projection histogram total")

    assert expected_v_projection_layers is not None
    normalized_v_layers = tuple(expected_v_projection_layers)
    if any(type(index) is not int or index < 0 for index in normalized_v_layers):
        raise ValueError("expected decoder LoRA v-projection layers must be nonnegative integers")
    if normalized_v_layers != tuple(sorted(set(normalized_v_layers))):
        raise ValueError("expected decoder LoRA v-projection layers must be sorted and unique")
    if dict(normalized_histogram)["v_proj"] != len(normalized_v_layers):
        raise ValueError("expected decoder LoRA v_proj count does not equal the v-projection layer count")
    return expected_target_count, tuple(normalized_histogram), normalized_v_layers


def decoder_attention_lora_targets(
    model: nn.Module,
    *,
    expected_target_count: int | None = None,
    expected_projection_histogram: Mapping[str, int] | None = None,
    expected_v_projection_layers: Sequence[int] | None = None,
) -> tuple[str, ...]:
    """Return exact decoder attention linear names and optionally enforce a pinned topology."""

    expected_topology = _validate_expected_decoder_lora_topology(
        expected_target_count=expected_target_count,
        expected_projection_histogram=expected_projection_histogram,
        expected_v_projection_layers=expected_v_projection_layers,
    )

    try:
        decoder = model.model.decoder
    except AttributeError as exc:
        raise TypeError("expected DiffusionGemmaForBlockDiffusion-compatible model.model.decoder") from exc

    decoder_prefix = next((name for name, module in model.named_modules() if module is decoder), None)
    if decoder_prefix is None:
        raise RuntimeError("could not resolve the decoder's fully-qualified module name")
    targets: list[str] = []
    projection_counts = dict.fromkeys(_ATTENTION_PROJECTION_NAMES, 0)
    v_projection_layers: list[int] = []
    for index, layer in enumerate(decoder.layers):
        attention = layer.self_attn
        for projection_name in _ATTENTION_PROJECTION_NAMES:
            projection = getattr(attention, projection_name, None)
            if projection is not None:
                if not isinstance(projection, nn.Linear):
                    raise TypeError(f"{decoder_prefix}.layers.{index}.self_attn.{projection_name} is not nn.Linear")
                targets.append(f"{decoder_prefix}.layers.{index}.self_attn.{projection_name}")
                projection_counts[projection_name] += 1
                if projection_name == "v_proj":
                    v_projection_layers.append(index)
    if not targets:
        raise RuntimeError("no decoder attention projections were found")
    if any("encoder" in target.split(".") for target in targets):
        raise RuntimeError("decoder LoRA target resolution unexpectedly reached the encoder")
    if expected_topology is not None:
        expected_count, expected_histogram, expected_v_layers = expected_topology
        observed_histogram = tuple((name, projection_counts[name]) for name in _ATTENTION_PROJECTION_NAMES)
        observed_v_layers = tuple(v_projection_layers)
        mismatches: list[str] = []
        if len(targets) != expected_count:
            mismatches.append(f"target count expected {expected_count}, observed {len(targets)}")
        if observed_histogram != expected_histogram:
            mismatches.append(f"projection histogram expected {expected_histogram}, observed {observed_histogram}")
        if observed_v_layers != expected_v_layers:
            mismatches.append(f"v-projection layers expected {expected_v_layers}, observed {observed_v_layers}")
        if mismatches:
            raise RuntimeError(f"decoder attention LoRA topology changed: {'; '.join(mismatches)}")
    return tuple(targets)


def apply_decoder_attention_lora(
    model: nn.Module,
    *,
    rank: int = 16,
    alpha: int = 32,
    dropout: float = 0.0,
) -> nn.Module:
    """Freeze a DiffusionGemma model and attach LoRA only to exact decoder attention projections."""

    if rank <= 0 or alpha <= 0 or not 0 <= dropout < 1:
        raise ValueError("rank/alpha must be positive and dropout must be in [0, 1)")
    try:
        from peft import LoraConfig, get_peft_model
    except ImportError as exc:  # pragma: no cover - exercised only in minimal installs
        raise ImportError("install duo-vla's train dependencies to enable LoRA") from exc

    targets = decoder_attention_lora_targets(model)
    tp_size = int(getattr(model, "_tp_size", None) or 1)
    if tp_size > 1:
        missing_tp_metadata = [
            target
            for target in targets
            if getattr(model.get_submodule(target), "_hf_tp_plan", None) not in {"colwise", "rowwise"}
            or getattr(model.get_submodule(target), "_hf_device_mesh", None) is None
        ]
        if missing_tp_metadata:
            raise RuntimeError(
                "tensor-parallel LoRA targets have no PEFT-compatible TP metadata; "
                "the installed Transformers/PEFT pair cannot train adapters safely"
            )
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        target_modules=list(targets),
    )
    adapted = get_peft_model(model, config)
    if tp_size > 1:
        tp_lora_layers = [module for module in adapted.modules() if hasattr(module, "lora_A")]
        missing_tp_hooks = [module for module in tp_lora_layers if not hasattr(module, "_tp_info")]
        if len(tp_lora_layers) != len(targets) or missing_tp_hooks:
            raise RuntimeError(
                "PEFT did not install TP metadata/hooks on every decoder LoRA layer; refusing unsafe training"
            )
    trainable = tuple(name for name, parameter in adapted.named_parameters() if parameter.requires_grad)
    if not trainable:
        raise RuntimeError("LoRA injection produced no trainable parameters")
    invalid = [
        name
        for name in trainable
        if "lora_" not in name or ".decoder.layers." not in name or ".self_attn." not in name or ".encoder." in name
    ]
    if invalid:
        raise RuntimeError(f"parameters outside decoder attention LoRA became trainable: {invalid}")
    return adapted


def decoder_lora_parameter_partition(adapted_model: nn.Module) -> LoRAParameterPartition:
    """Classify LoRA factors: colwise B / rowwise A are TP shards; the opposite factors are replicas."""

    try:
        from peft.tuners.lora.layer import LoraLayer
    except ImportError as exc:  # pragma: no cover - minimal installs intentionally omit PEFT
        raise ImportError("install duo-vla's train dependencies to inspect LoRA parameters") from exc

    names_by_id = {
        id(parameter): name for name, parameter in adapted_model.named_parameters() if parameter.requires_grad
    }
    replicated: list[tuple[str, nn.Parameter]] = []
    sharded: list[tuple[str, nn.Parameter]] = []
    for module in adapted_model.modules():
        if not isinstance(module, LoraLayer):
            continue
        plan = getattr(module.get_base_layer(), "_hf_tp_plan", None)
        if plan not in {None, "colwise", "rowwise"}:
            raise RuntimeError(f"unsupported TP plan on decoder LoRA layer: {plan!r}")
        for factor_name, factor_modules in (("A", module.lora_A), ("B", module.lora_B)):
            is_sharded = (plan == "colwise" and factor_name == "B") or (plan == "rowwise" and factor_name == "A")
            destination = sharded if is_sharded else replicated
            for factor_module in factor_modules.values():
                for parameter in factor_module.parameters():
                    try:
                        name = names_by_id[id(parameter)]
                    except KeyError as exc:
                        raise RuntimeError("LoRA factor is not a declared trainable parameter") from exc
                    destination.append((name, parameter))
    observed_ids = {id(parameter) for _, parameter in [*replicated, *sharded]}
    if observed_ids != set(names_by_id) or len(observed_ids) != len(replicated) + len(sharded):
        raise RuntimeError("LoRA TP partition did not cover every trainable adapter parameter exactly once")
    return LoRAParameterPartition(tuple(replicated), tuple(sharded))
