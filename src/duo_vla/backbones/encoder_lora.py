"""Opt-in prefix-encoder adapters and cache-safe activation recomputation.

Decoder PEFT state stays separate: encoder factors deliberately do not use PEFT's
``lora_`` key namespace. Base tensors, tied decoder weights, and vision stay frozen.
"""

from __future__ import annotations

import math
from types import MethodType

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint, set_checkpoint_early_stop


class EncoderLoRALinear(nn.Module):
    """An independent zero-initialized low-rank residual on a frozen projection."""

    def __init__(self, base: nn.Linear, *, rank: int = 16, alpha: int = 32):
        super().__init__()
        if not isinstance(base, nn.Linear) or rank <= 0 or alpha <= 0:
            raise ValueError("encoder adapter requires a Linear and positive rank/alpha")
        self.base = base
        self.base.requires_grad_(False)
        self.adapter_a = nn.Parameter(
            torch.empty(rank, base.in_features, device=base.weight.device, dtype=torch.float32)
        )
        self.adapter_b = nn.Parameter(
            torch.zeros(base.out_features, rank, device=base.weight.device, dtype=torch.float32)
        )
        nn.init.kaiming_uniform_(self.adapter_a, a=math.sqrt(5))
        self.scale = alpha / rank
        self.rank, self.alpha = rank, alpha

    @property
    def weight(self):
        return self.base.weight

    def forward(self, hidden):
        original = self.base(hidden)
        residual = F.linear(F.linear(hidden.to(self.adapter_a.dtype), self.adapter_a), self.adapter_b)
        return original + (residual * self.scale).to(original.dtype)


def encoder_attention_targets(model: nn.Module) -> tuple[str, ...]:
    """Only projections reachable from the action loss via layer-wise prefix KV.

    Final-layer q/o affect only the discarded encoder last_hidden_state, not any
    KV consumed by the action decoder. Do not allocate dead trainable adapters.
    """
    layers = model.model.encoder.language_model.layers
    prefix = "model.encoder.language_model.layers"
    targets = []
    for index, layer in enumerate(layers):
        for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
            if index == len(layers) - 1 and name in {"q_proj", "o_proj"}:
                continue
            projection = getattr(layer.self_attn, name, None)
            if projection is not None:
                if not isinstance(projection, nn.Linear):
                    raise ValueError("encoder attention already adapted or incompatible")
                targets.append(f"{prefix}.{index}.self_attn.{name}")
    if not targets:
        raise ValueError("no action-reachable encoder attention projections")
    return tuple(targets)


def install_encoder_lora(model: nn.Module, *, rank: int = 16, alpha: int = 32) -> tuple[str, ...]:
    targets = encoder_attention_targets(model)
    for target in targets:
        parent, _, name = target.rpartition(".")
        module = model.get_submodule(parent)
        setattr(module, name, EncoderLoRALinear(getattr(module, name), rank=rank, alpha=alpha))
    return targets


def encoder_adapter_parameters(model: nn.Module) -> list[tuple[str, nn.Parameter]]:
    values = []
    for name, module in model.named_modules():
        if isinstance(module, EncoderLoRALinear):
            values.extend([(f"{name}.adapter_a", module.adapter_a), (f"{name}.adapter_b", module.adapter_b)])
    if not values:
        raise ValueError("encoder adapters are not installed")
    return values


def encoder_adapter_state(model: nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().contiguous() for name, value in encoder_adapter_parameters(model)}


def load_encoder_adapter_state(model: nn.Module, state: dict[str, torch.Tensor]) -> None:
    parameters = dict(encoder_adapter_parameters(model))
    if set(state) != set(parameters):
        raise ValueError("encoder adapter checkpoint key mismatch")
    for name, value in state.items():
        if value.shape != parameters[name].shape or value.dtype != torch.float32 or not value.isfinite().all():
            raise ValueError(f"invalid encoder adapter tensor: {name}")
    with torch.no_grad():
        for name, value in state.items():
            parameters[name].copy_(value)


class _OneLayerCache:
    """Recompute-local cache: never mutate or append to the shared conditioning cache."""

    def __init__(self, layer_index: int):
        self.layer_index = layer_index
        self.pair = None

    def update(self, key, value, layer_index, *args, **kwargs):
        if layer_index != self.layer_index or self.pair is not None:
            raise RuntimeError("checkpointed prefix layer must write its KV exactly once")
        self.pair = (key, value)
        return key, value


def _checkpointed_language_forward(
    self, input_ids=None, attention_mask=None, position_ids=None, past_key_values=None, inputs_embeds=None, **kwargs
):
    from transformers.cache_utils import DynamicCache
    from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
    from transformers.modeling_outputs import BaseModelOutputWithPast

    if not torch.is_grad_enabled():
        return self._duovla_original_forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )
    if (input_ids is None) == (inputs_embeds is None):
        raise ValueError("specify exactly one of input_ids and inputs_embeds")
    if past_key_values is not None:
        raise ValueError("trainable prefix recomputation supports fresh prefill only")
    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)
    cache = DynamicCache(config=self.config)
    if position_ids is None:
        position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device).unsqueeze(0)
    masks = attention_mask
    if not isinstance(masks, dict):
        mask_args = dict(
            config=self.config,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            past_key_values=cache,
            position_ids=position_ids,
        )
        masks = {
            "full_attention": create_causal_mask(**mask_args),
            "sliding_attention": create_sliding_window_causal_mask(**mask_args),
        }
    positions = {kind: self.rotary_emb(inputs_embeds, position_ids, kind) for kind in self.unique_layer_types}
    hidden = inputs_embeds
    for index, layer in enumerate(self.layers):
        kind = self.config.layer_types[index]

        def apply_layer(states, *, current=layer, layer_index=index, layer_kind=kind):
            local = _OneLayerCache(layer_index)
            # Call the module, not .forward: fused-MoE validation hooks must also run on recomputation.
            output = current(
                states,
                position_embeddings=positions[layer_kind],
                attention_mask=masks[layer_kind],
                position_ids=position_ids,
                past_key_values=local,
                **kwargs,
            )
            if local.pair is None:
                raise RuntimeError("encoder layer did not produce KV")
            return output, *local.pair

        # Complete recomputation also keeps stateful forward hooks balanced.
        with set_checkpoint_early_stop(False):
            hidden, key, value = checkpoint(apply_layer, hidden, use_reentrant=False, preserve_rng_state=True)
        cache.update(key, value, index)
    return BaseModelOutputWithPast(last_hidden_state=self.norm(hidden), past_key_values=cache)


def install_checkpointed_prefix(model: nn.Module) -> None:
    language = model.model.encoder.language_model
    if hasattr(language, "_duovla_original_forward"):
        raise ValueError("prefix recomputation is already installed")
    language._duovla_original_forward = language.forward
    language.forward = MethodType(_checkpointed_language_forward, language)
