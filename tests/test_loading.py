from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file
from torch import nn

transformers = pytest.importorskip("transformers", minversion="5.15.0")

from transformers import DiffusionGemmaConfig  # noqa: E402

from duo_vla.backbones.loading import (  # noqa: E402
    DiffusionGemmaModelSpec,
    expected_decoder_attention_lora_adapter_config,
    expected_decoder_attention_lora_weight_schema,
    load_diffusion_gemma_bf16_tp,
    symmetric_diffusion_gemma_tp_plan,
    validate_decoder_attention_lora_adapter_config,
    validate_decoder_attention_lora_weights,
)


def _fake_loaded_model(*, v_projection_layers: tuple[int, ...] = (0,)) -> nn.Module:
    text_config = SimpleNamespace(
        base_model_tp_plan={"layers.*.self_attn.q_proj": "colwise"},
        hidden_size=4,
        num_hidden_layers=2,
        _experts_implementation="grouped_mm",
    )

    class Experts(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.config = text_config
            self.weight = nn.Parameter(torch.ones(1))

        def forward(
            self,
            hidden_states: torch.Tensor,
            _top_k_index: torch.Tensor,
            _top_k_weights: torch.Tensor,
        ) -> torch.Tensor:
            return hidden_states

    class Attention(nn.Module):
        def __init__(self, *, has_v_projection: bool) -> None:
            super().__init__()
            self.q_proj = nn.Linear(4, 4, bias=False)
            self.k_proj = nn.Linear(4, 4, bias=False)
            self.v_proj = nn.Linear(4, 4, bias=False) if has_v_projection else None
            self.o_proj = nn.Linear(4, 4, bias=False)

    class Layer(nn.Module):
        def __init__(self, *, has_v_projection: bool) -> None:
            super().__init__()
            self.self_attn = Attention(has_v_projection=has_v_projection)
            self.experts = Experts()

    class TextStack(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layers = nn.ModuleList([Layer(has_v_projection=index in v_projection_layers) for index in range(2)])

    class Encoder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.language_model = TextStack()

    class InnerModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = Encoder()
            self.decoder = TextStack()

    class LoadedModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model = InnerModel()
            self.config = SimpleNamespace(
                text_config=text_config,
                canvas_length=8,
            )

    return LoadedModel()


def _tiny_spec() -> DiffusionGemmaModelSpec:
    return DiffusionGemmaModelSpec(
        model_id="test/diffusion-gemma",
        revision="test-revision",
        expected_hidden_size=4,
        expected_num_layers=2,
        expected_canvas_length=8,
        expected_decoder_attention_lora_target_count=7,
        expected_decoder_attention_lora_projection_histogram=(
            ("q_proj", 2),
            ("k_proj", 2),
            ("v_proj", 1),
            ("o_proj", 2),
        ),
        expected_decoder_attention_lora_v_projection_layers=(0,),
    )


def _patch_loader_dependencies(monkeypatch: pytest.MonkeyPatch, model: nn.Module) -> None:
    # Resolve both lazy exports before replacing them; resolving the second export can otherwise replace the lazy
    # top-level module object and discard a patch already installed on the first one.
    _ = transformers.DiffusionGemmaConfig
    _ = transformers.DiffusionGemmaForBlockDiffusion
    active_transformers = sys.modules["transformers"]
    config_factory = SimpleNamespace(from_pretrained=lambda *args, **kwargs: model.config)

    def load_model(*_args, **kwargs):
        model._from_pretrained_kwargs = dict(kwargs)
        return model

    model_factory = SimpleNamespace(from_pretrained=load_model)
    monkeypatch.setattr(active_transformers, "DiffusionGemmaConfig", config_factory)
    monkeypatch.setattr(active_transformers, "DiffusionGemmaForBlockDiffusion", model_factory)
    monkeypatch.setattr(active_transformers, "DistributedConfig", lambda **kwargs: kwargs, raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)
    monkeypatch.setenv("WORLD_SIZE", "1")


def test_tp_plan_covers_both_tied_text_stacks_and_embeddings() -> None:
    config = DiffusionGemmaConfig(
        text_config={
            "num_hidden_layers": 2,
            "num_experts": 4,
            "top_k_experts": 2,
            "moe_intermediate_size": 16,
        },
        vision_config={"num_hidden_layers": 1},
    )
    plan = symmetric_diffusion_gemma_tp_plan(config)
    for suffix in ("embed_tokens", "layers.*.self_attn.q_proj", "layers.*.experts"):
        assert f"model.encoder.language_model.{suffix}" in plan
        assert f"model.decoder.{suffix}" in plan
    assert not any("vision_tower" in target for target in plan)


def test_loader_enforces_lora_topology_before_returning_model(monkeypatch: pytest.MonkeyPatch) -> None:
    model = _fake_loaded_model()
    _patch_loader_dependencies(monkeypatch, model)

    loaded = load_diffusion_gemma_bf16_tp(_tiny_spec(), tp_size=1, local_files_only=True)

    assert loaded is model
    assert all(not parameter.requires_grad for parameter in loaded.parameters())
    assert loaded._from_pretrained_kwargs["experts_implementation"] == "grouped_mm"
    assert _tiny_spec().expected_experts_implementation == "grouped_mm"


def test_loader_rejects_backend_override_and_loaded_backend_mismatch(monkeypatch: pytest.MonkeyPatch) -> None:
    model = _fake_loaded_model()
    _patch_loader_dependencies(monkeypatch, model)

    with pytest.raises(ValueError, match="experts_implementation is pinned"):
        load_diffusion_gemma_bf16_tp(
            _tiny_spec(),
            tp_size=1,
            local_files_only=True,
            experts_implementation="eager",
        )

    model.config.text_config._experts_implementation = "eager"
    with pytest.raises(RuntimeError, match="incompatible grouped-MM expert topology"):
        load_diffusion_gemma_bf16_tp(_tiny_spec(), tp_size=1, local_files_only=True)


def test_loader_rejects_same_histogram_on_wrong_v_projection_layer(monkeypatch: pytest.MonkeyPatch) -> None:
    model = _fake_loaded_model(v_projection_layers=(1,))
    _patch_loader_dependencies(monkeypatch, model)

    with pytest.raises(
        RuntimeError,
        match=r"pinned model revision test-revision.*v-projection layers expected \(0,\), observed \(1,\)",
    ):
        load_diffusion_gemma_bf16_tp(_tiny_spec(), tp_size=1, local_files_only=True)

    assert all(parameter.requires_grad for parameter in model.parameters())


def test_saved_lora_config_is_checked_against_exact_semantics_and_targets(tmp_path: Path) -> None:
    spec = _tiny_spec()
    payload = expected_decoder_attention_lora_adapter_config(spec=spec)
    path = tmp_path / "adapter_config.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    assert validate_decoder_attention_lora_adapter_config(path, spec=spec) == payload
    payload["r"] = 8
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="decoder-attention contract"):
        validate_decoder_attention_lora_adapter_config(path, spec=spec)

    payload = expected_decoder_attention_lora_adapter_config(spec=spec)
    payload["trainable_token_indices"] = [1]
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="decoder-attention contract"):
        validate_decoder_attention_lora_adapter_config(path, spec=spec)


def test_pinned_lora_weight_schema_has_exact_names_shapes_and_count() -> None:
    schema = expected_decoder_attention_lora_weight_schema()

    assert len(schema) == 230
    assert schema["base_model.model.model.decoder.layers.0.self_attn.q_proj.lora_A.weight"] == (16, 2816)
    assert schema["base_model.model.model.decoder.layers.0.self_attn.q_proj.lora_B.weight"] == (4096, 16)
    assert schema["base_model.model.model.decoder.layers.5.self_attn.q_proj.lora_B.weight"] == (8192, 16)
    assert schema["base_model.model.model.decoder.layers.5.self_attn.k_proj.lora_B.weight"] == (1024, 16)
    assert not any("layers.5.self_attn.v_proj" in name for name in schema)


def test_lora_weight_validator_rejects_missing_extra_shape_dtype_and_nonfinite(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import duo_vla.backbones.loading as loading

    schema = {"adapter.a": (2, 3), "adapter.b": (3, 2)}
    monkeypatch.setattr(loading, "expected_decoder_attention_lora_weight_schema", lambda **_kwargs: schema)
    path = tmp_path / "adapter_model.safetensors"
    good = {"adapter.a": torch.ones(2, 3), "adapter.b": torch.ones(3, 2)}
    save_file(good, path)
    assert validate_decoder_attention_lora_weights(path) == schema

    for name, tensors, message in (
        ("missing", {"adapter.a": good["adapter.a"]}, "tensor names"),
        ("shape", {**good, "adapter.a": torch.ones(3, 2)}, "schema mismatch"),
        ("dtype", {**good, "adapter.a": torch.ones(2, 3, dtype=torch.bfloat16)}, "schema mismatch"),
        ("nonfinite", {**good, "adapter.a": torch.full((2, 3), float("nan"))}, "non-finite"),
    ):
        candidate = tmp_path / f"{name}.safetensors"
        save_file(tensors, candidate)
        with pytest.raises(ValueError, match=message):
            validate_decoder_attention_lora_weights(candidate)
