from __future__ import annotations

import pytest
import torch

transformers = pytest.importorskip("transformers", minversion="5.15.0")

from transformers import DiffusionGemmaConfig, DiffusionGemmaForBlockDiffusion  # noqa: E402

from duo_vla.backbones.diffusion_gemma import (  # noqa: E402
    DiffusionGemmaActionDecoder,
    apply_decoder_attention_lora,
    decoder_attention_lora_targets,
    decoder_lora_parameter_partition,
    encode_diffusion_gemma_prefix,
)
from duo_vla.backbones.loading import DEFAULT_DIFFUSION_GEMMA_SPEC  # noqa: E402


def _tiny_model(
    *,
    layer_types: tuple[str, ...] = ("sliding_attention", "full_attention"),
    attention_implementation: str = "eager",
) -> DiffusionGemmaForBlockDiffusion:
    config = DiffusionGemmaConfig(
        text_config={
            "vocab_size": 128,
            "hidden_size": 32,
            "intermediate_size": 48,
            "num_hidden_layers": len(layer_types),
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 8,
            "global_head_dim": 8,
            "num_global_key_value_heads": 2,
            "max_position_embeddings": 128,
            "sliding_window": 16,
            "layer_types": list(layer_types),
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
    config._attn_implementation = attention_implementation
    config.text_config._attn_implementation = attention_implementation
    return DiffusionGemmaForBlockDiffusion(config).eval()


def _prefix(
    model: DiffusionGemmaForBlockDiffusion,
    batch_size: int = 2,
    *,
    input_ids: torch.Tensor | None = None,
):
    if input_ids is None:
        input_ids = torch.randint(3, 128, (batch_size, 5))
    elif input_ids.shape != (batch_size, 5):
        raise ValueError("input_ids must have shape [batch_size, 5]")
    attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
    return encode_diffusion_gemma_prefix(
        model,
        {"input_ids": input_ids, "attention_mask": attention_mask},
    )


@pytest.mark.parametrize("attention_implementation", ["eager", "sdpa"])
def test_continuous_adapter_matches_native_decoder_for_token_embeddings(
    attention_implementation: str,
) -> None:
    torch.manual_seed(0)
    model = _tiny_model(attention_implementation=attention_implementation)
    prefix = _prefix(model)
    decoder_ids = torch.randint(3, 128, (2, 8))
    action_valid = torch.ones(2, 8, dtype=torch.bool)
    embeddings = model.model.decoder.embed_tokens(decoder_ids)

    with torch.no_grad():
        native = model.model.decoder(
            decoder_input_ids=decoder_ids,
            past_key_values=prefix.past_key_values,
            decoder_attention_mask=torch.cat((prefix.attention_mask, action_valid), dim=1),
        ).last_hidden_state
        adapted = DiffusionGemmaActionDecoder.from_block_diffusion_model(model).decode_actions(
            embeddings,
            prefix_cache=prefix.past_key_values,
            prefix_attention_mask=prefix.attention_mask,
            action_valid_mask=action_valid,
        )
    torch.testing.assert_close(adapted, native, atol=1e-6, rtol=1e-5)


def test_padding_cannot_change_valid_diffusion_gemma_outputs() -> None:
    torch.manual_seed(1)
    model = _tiny_model()
    prefix = _prefix(model, batch_size=1)
    adapter = DiffusionGemmaActionDecoder.from_block_diffusion_model(model)
    embeddings = torch.randn(1, 8, 32)
    perturbed = embeddings.clone()
    perturbed[:, 5:] = 1e5
    valid = torch.tensor([[True] * 5 + [False] * 3])

    with torch.no_grad():
        expected = adapter.decode_actions(
            embeddings,
            prefix_cache=prefix.past_key_values,
            prefix_attention_mask=prefix.attention_mask,
            action_valid_mask=valid,
        )
        actual = adapter.decode_actions(
            perturbed,
            prefix_cache=prefix.past_key_values,
            prefix_attention_mask=prefix.attention_mask,
            action_valid_mask=valid,
        )
    torch.testing.assert_close(actual[:, :5], expected[:, :5])
    torch.testing.assert_close(actual[:, 5:], torch.zeros_like(actual[:, 5:]))


@pytest.mark.parametrize("attention_implementation", ["eager", "sdpa"])
def test_left_padded_training_prefix_matches_unpadded_serving_prefix(
    attention_implementation: str,
) -> None:
    torch.manual_seed(11)
    model = _tiny_model(attention_implementation=attention_implementation)
    adapter = DiffusionGemmaActionDecoder.from_block_diffusion_model(model)
    short_ids = torch.tensor([[3, 5, 7, 11, 13]])
    short_mask = torch.ones_like(short_ids, dtype=torch.bool)
    padded_ids = torch.tensor(
        [
            [0, 0, 0, 3, 5, 7, 11, 13],
            [17, 19, 23, 29, 31, 37, 41, 43],
        ]
    )
    padded_mask = torch.tensor(
        [
            [False, False, False, True, True, True, True, True],
            [True, True, True, True, True, True, True, True],
        ]
    )
    unpadded = encode_diffusion_gemma_prefix(
        model,
        {"input_ids": short_ids, "attention_mask": short_mask},
    )
    batched = encode_diffusion_gemma_prefix(
        model,
        {"input_ids": padded_ids, "attention_mask": padded_mask},
    )
    action_embeddings = torch.randn(1, 8, 32)
    action_valid = torch.ones(1, 8, dtype=torch.bool)

    with torch.no_grad():
        serving = adapter.decode_actions(
            action_embeddings,
            prefix_cache=unpadded.past_key_values,
            prefix_attention_mask=unpadded.attention_mask,
            action_valid_mask=action_valid,
        )
        training = adapter.decode_actions(
            action_embeddings.expand(2, -1, -1),
            prefix_cache=batched.past_key_values,
            prefix_attention_mask=batched.attention_mask,
            action_valid_mask=action_valid.expand(2, -1),
        )[:1]

    torch.testing.assert_close(training, serving, atol=1e-6, rtol=1e-5)


def test_prefix_encoder_rejects_noncanonical_supplied_position_ids() -> None:
    model = _tiny_model()
    input_ids = torch.tensor([[0, 3, 5, 7, 11]])
    attention_mask = torch.tensor([[False, True, True, True, True]])

    with pytest.raises(ValueError, match="canonical padding-invariant"):
        encode_diffusion_gemma_prefix(
            model,
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "position_ids": torch.arange(5).unsqueeze(0),
            },
        )


def test_action_decode_reuses_prefix_cache_without_mutating_it() -> None:
    torch.manual_seed(2)
    model = _tiny_model()
    prefix = _prefix(model, batch_size=1)
    adapter = DiffusionGemmaActionDecoder.from_block_diffusion_model(model)
    embeddings = torch.randn(1, 8, 32)
    valid = torch.ones(1, 8, dtype=torch.bool)
    length_before = prefix.past_key_values.get_seq_length(layer_idx=0)

    with torch.no_grad():
        first = adapter.decode_actions(
            embeddings,
            prefix_cache=prefix.past_key_values,
            prefix_attention_mask=prefix.attention_mask,
            action_valid_mask=valid,
        )
        second = adapter.decode_actions(
            embeddings,
            prefix_cache=prefix.past_key_values,
            prefix_attention_mask=prefix.attention_mask,
            action_valid_mask=valid,
        )

    assert prefix.past_key_values.get_seq_length(layer_idx=0) == length_before
    torch.testing.assert_close(second, first, atol=0, rtol=0)


def test_sdpa_fresh_prefix_matches_reused_prefix_and_cache_is_read_only() -> None:
    torch.manual_seed(3)
    model = _tiny_model(attention_implementation="sdpa")
    input_ids = torch.tensor([[3, 5, 7, 11, 13]])
    reused = _prefix(model, batch_size=1, input_ids=input_ids)
    fresh = _prefix(model, batch_size=1, input_ids=input_ids)
    adapter = DiffusionGemmaActionDecoder.from_block_diffusion_model(model)
    embeddings = torch.randn(1, 8, 32)
    valid = torch.tensor([[True] * 5 + [False] * 3])

    def snapshot(prefix) -> tuple[tuple[torch.Tensor, torch.Tensor, int], ...]:
        return tuple(
            (
                layer.keys.detach().clone(),
                layer.values.detach().clone(),
                int(layer.get_seq_length()),
            )
            for layer in prefix.past_key_values.layers
        )

    reused_before = snapshot(reused)
    fresh_before = snapshot(fresh)
    for reused_layer, fresh_layer in zip(reused_before, fresh_before, strict=True):
        torch.testing.assert_close(reused_layer[0], fresh_layer[0], atol=0, rtol=0)
        torch.testing.assert_close(reused_layer[1], fresh_layer[1], atol=0, rtol=0)
        assert reused_layer[2] == fresh_layer[2]

    with torch.no_grad():
        reused_output = adapter.decode_actions(
            embeddings,
            prefix_cache=reused.past_key_values,
            prefix_attention_mask=reused.attention_mask,
            action_valid_mask=valid,
        )
        fresh_output = adapter.decode_actions(
            embeddings,
            prefix_cache=fresh.past_key_values,
            prefix_attention_mask=fresh.attention_mask,
            action_valid_mask=valid,
        )

    torch.testing.assert_close(reused_output, fresh_output, atol=0, rtol=0)
    for before, after in (
        (reused_before, snapshot(reused)),
        (fresh_before, snapshot(fresh)),
    ):
        for before_layer, after_layer in zip(before, after, strict=True):
            torch.testing.assert_close(before_layer[0], after_layer[0], atol=0, rtol=0)
            torch.testing.assert_close(before_layer[1], after_layer[1], atol=0, rtol=0)
            assert before_layer[2] == after_layer[2]


def test_action_decode_rejects_prefix_longer_than_native_sliding_window() -> None:
    model = _tiny_model(attention_implementation="sdpa")
    input_ids = torch.arange(3, 20).unsqueeze(0)
    attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
    prefix = encode_diffusion_gemma_prefix(
        model,
        {"input_ids": input_ids, "attention_mask": attention_mask},
    )
    adapter = DiffusionGemmaActionDecoder.from_block_diffusion_model(model)

    with pytest.raises(ValueError, match=r"prefix length 17.*sliding window 16"):
        adapter.decode_actions(
            torch.randn(1, 8, 32),
            prefix_cache=prefix.past_key_values,
            prefix_attention_mask=prefix.attention_mask,
            action_valid_mask=torch.ones(1, 8, dtype=torch.bool),
        )


def test_lora_targets_and_trainable_scope_are_decoder_attention_only() -> None:
    model = _tiny_model()
    targets = decoder_attention_lora_targets(
        model,
        expected_target_count=7,
        expected_projection_histogram={"q_proj": 2, "k_proj": 2, "v_proj": 1, "o_proj": 2},
        expected_v_projection_layers=(0,),
    )
    assert len(targets) == 7  # q/k/v/o on sliding layer; q/k/o on full-attention layer.
    assert all(target.startswith("model.decoder.layers.") for target in targets)
    assert not any("encoder" in target for target in targets)

    adapted = apply_decoder_attention_lora(model, rank=2, alpha=4)
    trainable = [name for name, parameter in adapted.named_parameters() if parameter.requires_grad]
    assert len(trainable) == 2 * len(targets)
    assert all("lora_" in name and ".decoder.layers." in name for name in trainable)
    assert not any(".encoder." in name for name in trainable)
    partition = decoder_lora_parameter_partition(adapted)
    assert len(partition.replicated) == len(trainable)
    assert not partition.sharded


def test_lora_topology_guard_rejects_changed_count_and_histogram() -> None:
    model = _tiny_model()

    with pytest.raises(RuntimeError, match="target count expected 8, observed 7"):
        decoder_attention_lora_targets(
            model,
            expected_target_count=8,
            expected_projection_histogram={"q_proj": 2, "k_proj": 2, "v_proj": 2, "o_proj": 2},
            expected_v_projection_layers=(0, 1),
        )


def test_lora_topology_guard_rejects_changed_v_projection_layers_with_same_histogram() -> None:
    model = _tiny_model()
    sliding_attention = model.model.decoder.layers[0].self_attn
    full_attention = model.model.decoder.layers[1].self_attn
    full_attention.v_proj = sliding_attention.v_proj
    sliding_attention.v_proj = None

    with pytest.raises(RuntimeError, match=r"v-projection layers expected \(0,\), observed \(1,\)"):
        decoder_attention_lora_targets(
            model,
            expected_target_count=7,
            expected_projection_histogram={"q_proj": 2, "k_proj": 2, "v_proj": 1, "o_proj": 2},
            expected_v_projection_layers=(0,),
        )


def test_lora_topology_guard_requires_a_complete_self_consistent_contract() -> None:
    model = _tiny_model()

    with pytest.raises(ValueError, match="must be provided together"):
        decoder_attention_lora_targets(model, expected_target_count=7)
    with pytest.raises(ValueError, match="does not equal the projection histogram total"):
        decoder_attention_lora_targets(
            model,
            expected_target_count=8,
            expected_projection_histogram={"q_proj": 2, "k_proj": 2, "v_proj": 1, "o_proj": 2},
            expected_v_projection_layers=(0,),
        )


def test_pinned_revision_decoder_lora_topology_has_115_targets() -> None:
    spec = DEFAULT_DIFFUSION_GEMMA_SPEC
    v_projection_layers = set(spec.expected_decoder_attention_lora_v_projection_layers)
    layer_types = tuple(
        "sliding_attention" if index in v_projection_layers else "full_attention"
        for index in range(spec.expected_num_layers)
    )
    model = _tiny_model(layer_types=layer_types)

    targets = decoder_attention_lora_targets(
        model,
        expected_target_count=spec.expected_decoder_attention_lora_target_count,
        expected_projection_histogram=dict(spec.expected_decoder_attention_lora_projection_histogram),
        expected_v_projection_layers=spec.expected_decoder_attention_lora_v_projection_layers,
    )

    assert len(targets) == 115
    assert (
        tuple(
            (projection_name, sum(target.endswith(f".{projection_name}") for target in targets))
            for projection_name in ("q_proj", "k_proj", "v_proj", "o_proj")
        )
        == spec.expected_decoder_attention_lora_projection_histogram
    )
