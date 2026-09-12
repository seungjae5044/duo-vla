from __future__ import annotations

import copy

import pytest
import torch

pytest.importorskip("transformers")
from test_diffusion_gemma_adapter import _tiny_model

from duo_vla.backbones.diffusion_gemma import (
    DiffusionGemmaActionDecoder,
    encode_diffusion_gemma_prefix,
    encode_diffusion_gemma_prefix_trainable,
)
from duo_vla.backbones.encoder_lora import (
    EncoderLoRALinear,
    encoder_adapter_parameters,
    encoder_adapter_state,
    install_checkpointed_prefix,
    install_encoder_lora,
    load_encoder_adapter_state,
)
from duo_vla.backbones.sample_isolated_experts import (
    _InvocationState,
    _make_layer_post_hook,
    _make_layer_pre_hook,
)


def inputs():
    return {
        "input_ids": torch.tensor([[0, 0, 3, 4, 5], [6, 7, 8, 9, 10]]),
        "attention_mask": torch.tensor([[0, 0, 1, 1, 1], [1, 1, 1, 1, 1]]),
    }


def decode(model, prefix, actions):
    return DiffusionGemmaActionDecoder.from_block_diffusion_model(model).decode_actions(
        actions,
        prefix_cache=prefix.past_key_values,
        prefix_attention_mask=prefix.attention_mask,
        action_valid_mask=torch.ones(actions.shape[:2], dtype=torch.bool),
    )


def test_zero_init_preserves_frozen_model_and_tied_weights():
    torch.manual_seed(33)
    model = _tiny_model().requires_grad_(False)
    actions = torch.randn(2, 8, 32)
    before = decode(model, encode_diffusion_gemma_prefix(model, inputs()), actions)
    base_weight = model.model.encoder.language_model.layers[0].self_attn.k_proj.weight
    targets = install_encoder_lora(model, rank=2, alpha=4)
    after = decode(model, encode_diffusion_gemma_prefix_trainable(model, inputs()), actions)
    torch.testing.assert_close(before, after, rtol=0, atol=0)
    assert model.model.encoder.language_model.layers[0].self_attn.k_proj.weight is base_weight
    assert not any("vision" in name or "decoder" in name for name in targets)
    assert not any(name.endswith(("1.self_attn.q_proj", "1.self_attn.o_proj")) for name in targets)
    assert all(not p.requires_grad for p in model.model.encoder.vision_tower.parameters())
    assert not base_weight.requires_grad


@pytest.mark.parametrize("recompute", [False, True])
def test_action_loss_reaches_every_encoder_adapter_and_not_base(recompute):
    torch.manual_seed(24)
    model = _tiny_model(attention_implementation="sdpa").requires_grad_(False)
    install_encoder_lora(model, rank=2, alpha=4)
    if recompute:
        install_checkpointed_prefix(model)
    prefix = encode_diffusion_gemma_prefix_trainable(model, inputs())
    assert prefix.past_key_values.layers[0].keys.requires_grad
    actions = torch.randn(2, 8, 32)
    (decode(model, prefix, actions) * torch.randn_like(actions)).sum().backward()
    for name, parameter in encoder_adapter_parameters(model):
        assert parameter.grad is not None, name
        assert parameter.grad.isfinite().all(), name
        if name.endswith("adapter_b"):
            assert parameter.grad.abs().sum() > 0, name
    assert all(p.grad is None for p in model.parameters() if not p.requires_grad)
    assert all(prefix.past_key_values.get_seq_length(i) == 5 for i in range(2))
    frozen = encode_diffusion_gemma_prefix(model, inputs())
    assert not frozen.past_key_values.layers[0].keys.requires_grad


@pytest.mark.parametrize("attention", ["eager", "sdpa"])
def test_checkpointed_prefix_matches_native_values_and_gradients(attention):
    torch.manual_seed(66)
    reference = _tiny_model(attention_implementation=attention).requires_grad_(False)
    install_encoder_lora(reference, rank=2, alpha=4)
    with torch.no_grad():
        for name, parameter in encoder_adapter_parameters(reference):
            if name.endswith("adapter_b"):
                parameter.normal_(std=0.03)
    recomputed = copy.deepcopy(reference)
    install_checkpointed_prefix(recomputed)
    # Emulate fused-expert per-layer invocation hooks, including backward recomputation.
    for index, layer in enumerate(recomputed.model.encoder.language_model.layers):
        state = _InvocationState(target_name=f"test-{index}", physical_batch_size=2)
        layer.register_forward_pre_hook(_make_layer_pre_hook(state), with_kwargs=True)
        layer.register_forward_hook(_make_layer_post_hook(state), with_kwargs=True, always_call=True)

        def consumed(module, args, output, invocation=state):
            assert invocation.expected_flat_token_count == 10
            invocation.consumed = True

        layer.experts.register_forward_hook(consumed)
    actions = torch.randn(2, 8, 32)
    target = torch.randn_like(actions)
    outputs = []
    for model in (reference, recomputed):
        prefix = encode_diffusion_gemma_prefix_trainable(model, inputs())
        output = decode(model, prefix, actions)
        (output * target).sum().backward()
        assert all(prefix.past_key_values.get_seq_length(i) == 5 for i in range(2))
        outputs.append(output)
    torch.testing.assert_close(*outputs, rtol=0, atol=0)
    for (name, p), (other, q) in zip(
        encoder_adapter_parameters(reference), encoder_adapter_parameters(recomputed), strict=True
    ):
        assert name == other
        torch.testing.assert_close(p.grad, q.grad, rtol=1e-5, atol=1e-6)


def test_encoder_checkpoint_roundtrip_and_schema_rejection():
    model = _tiny_model().requires_grad_(False)
    install_encoder_lora(model, rank=2, alpha=4)
    original = encoder_adapter_state(model)
    modified = {name: value + 0.1 for name, value in original.items()}
    load_encoder_adapter_state(model, modified)
    for name, value in encoder_adapter_state(model).items():
        torch.testing.assert_close(value, modified[name])
    with pytest.raises(ValueError, match="key mismatch"):
        load_encoder_adapter_state(model, {})
    invalid = {name: value.clone() for name, value in modified.items()}
    next(iter(invalid.values())).fill_(float("nan"))
    with pytest.raises(ValueError, match="invalid encoder"):
        load_encoder_adapter_state(model, invalid)
    with pytest.raises(ValueError, match="already adapted"):
        install_encoder_lora(model)


def test_encoder_weights_are_excluded_from_decoder_peft_serialization():
    pytest.importorskip("peft")
    from peft import get_peft_model_state_dict

    from duo_vla.backbones.diffusion_gemma import apply_decoder_attention_lora

    model = _tiny_model()
    adapted = apply_decoder_attention_lora(model, rank=2, alpha=4)
    before = get_peft_model_state_dict(adapted)
    install_encoder_lora(model, rank=2, alpha=4)
    after = get_peft_model_state_dict(adapted)
    assert before.keys() == after.keys()
    for name in before:
        torch.testing.assert_close(before[name], after[name], rtol=0, atol=0)
    assert any(isinstance(module, EncoderLoRALinear) for module in model.modules())
