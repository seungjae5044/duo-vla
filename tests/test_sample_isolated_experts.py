from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

import duo_vla.backbones.sample_isolated_experts as isolation
from duo_vla.backbones.sample_isolated_experts import (
    install_sample_isolated_grouped_mm_experts,
    validate_diffusion_gemma_grouped_mm_expert_topology,
    verify_sample_isolated_grouped_mm_experts,
)


def _fake_grouped_mm(
    experts: nn.Module,
    hidden_states: torch.Tensor,
    _top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
) -> torch.Tensor:
    projected = hidden_states @ experts.weight.T
    # Deliberately make one grouped call depend on all tokens passed to it.
    # The installed path must limit that dependence to one sample.
    return (projected + projected.mean(dim=0, keepdim=True)) * top_k_weights.sum(dim=-1, keepdim=True)


class _FakeExperts(nn.Module):
    def __init__(self, config: SimpleNamespace, hidden_size: int) -> None:
        super().__init__()
        self.config = config
        self.weight = nn.Parameter(
            torch.arange(1, hidden_size * hidden_size + 1, dtype=torch.float32).reshape(hidden_size, hidden_size)
            / hidden_size
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        return _fake_grouped_mm(self, hidden_states, top_k_index, top_k_weights)


class _FakeLayer(nn.Module):
    def __init__(self, config: SimpleNamespace, hidden_size: int, *, drop_flat_tokens: int = 0) -> None:
        super().__init__()
        self.experts = _FakeExperts(config, hidden_size)
        self.drop_flat_tokens = drop_flat_tokens

    @staticmethod
    def routes(flat_hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        token_count = int(flat_hidden_states.shape[0])
        top_k_index = torch.zeros(token_count, 2, dtype=torch.long, device=flat_hidden_states.device)
        top_k_weights = torch.full(
            (token_count, 2),
            0.5,
            dtype=flat_hidden_states.dtype,
            device=flat_hidden_states.device,
        )
        return top_k_index, top_k_weights

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        flat_hidden_states = hidden_states.reshape(-1, hidden_states.shape[-1])
        if self.drop_flat_tokens:
            flat_hidden_states = flat_hidden_states[: -self.drop_flat_tokens]
        top_k_index, top_k_weights = self.routes(flat_hidden_states)
        output = self.experts(flat_hidden_states, top_k_index, top_k_weights)
        return output.reshape(hidden_states.shape)


class _FakeTextStack(nn.Module):
    def __init__(
        self,
        config: SimpleNamespace,
        hidden_size: int,
        *,
        drop_first_layer_flat_tokens: int = 0,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                _FakeLayer(
                    config,
                    hidden_size,
                    drop_flat_tokens=drop_first_layer_flat_tokens if layer_index == 0 else 0,
                )
                for layer_index in range(config.num_hidden_layers)
            ]
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        return hidden_states


class _FakeEncoder(nn.Module):
    def __init__(
        self,
        config: SimpleNamespace,
        hidden_size: int,
        *,
        drop_first_layer_flat_tokens: int = 0,
    ) -> None:
        super().__init__()
        self.language_model = _FakeTextStack(
            config,
            hidden_size,
            drop_first_layer_flat_tokens=drop_first_layer_flat_tokens,
        )


class _FakeInnerModel(nn.Module):
    def __init__(
        self,
        config: SimpleNamespace,
        hidden_size: int,
        *,
        drop_first_encoder_layer_flat_tokens: int = 0,
    ) -> None:
        super().__init__()
        self.encoder = _FakeEncoder(
            config,
            hidden_size,
            drop_first_layer_flat_tokens=drop_first_encoder_layer_flat_tokens,
        )
        self.decoder = _FakeTextStack(config, hidden_size)


class _FakeModel(nn.Module):
    def __init__(
        self,
        *,
        num_layers: int = 2,
        hidden_size: int = 3,
        backend: str = "grouped_mm",
        drop_first_encoder_layer_flat_tokens: int = 0,
    ) -> None:
        super().__init__()
        text_config = SimpleNamespace(
            num_hidden_layers=num_layers,
            _experts_implementation=backend,
        )
        self.config = SimpleNamespace(text_config=text_config)
        self.model = _FakeInnerModel(
            text_config,
            hidden_size,
            drop_first_encoder_layer_flat_tokens=drop_first_encoder_layer_flat_tokens,
        )


def _stack(model: _FakeModel, name: str) -> _FakeTextStack:
    if name == "encoder":
        return model.model.encoder.language_model
    if name == "decoder":
        return model.model.decoder
    raise AssertionError(name)


def _per_sample_baseline(stack: _FakeTextStack, hidden_states: torch.Tensor) -> torch.Tensor:
    for layer in stack.layers:
        sample_outputs = []
        for sample_hidden_states in hidden_states:
            top_k_index, top_k_weights = layer.routes(sample_hidden_states)
            sample_outputs.append(
                _fake_grouped_mm(
                    layer.experts,
                    sample_hidden_states,
                    top_k_index,
                    top_k_weights,
                )
            )
        hidden_states = torch.stack(sample_outputs)
    return hidden_states


@pytest.fixture(autouse=True)
def _use_fake_grouped_mm(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(isolation, "_grouped_mm_experts_forward", _fake_grouped_mm)


def test_install_matches_per_sample_forward_and_backward_baseline() -> None:
    model = _FakeModel()
    stack = model.model.decoder
    expected_input = torch.randn(2, 4, 3, generator=torch.Generator().manual_seed(0), requires_grad=True)
    actual_input = expected_input.detach().clone().requires_grad_(True)

    expected = _per_sample_baseline(stack, expected_input)
    expected.square().sum().backward()
    expected_input_gradient = expected_input.grad.detach().clone()
    expected_weight_gradients = [layer.experts.weight.grad.detach().clone() for layer in stack.layers]
    model.zero_grad(set_to_none=True)

    contract = install_sample_isolated_grouped_mm_experts(model, physical_batch_size=2)
    actual = stack(actual_input)
    actual.square().sum().backward()

    torch.testing.assert_close(actual, expected.detach(), atol=0, rtol=0)
    torch.testing.assert_close(actual_input.grad, expected_input_gradient, atol=0, rtol=0)
    for layer, expected_gradient in zip(stack.layers, expected_weight_gradients, strict=True):
        torch.testing.assert_close(layer.experts.weight.grad, expected_gradient, atol=0, rtol=0)
    assert contract.target_count == 4
    assert verify_sample_isolated_grouped_mm_experts(model, physical_batch_size=2) == contract


@pytest.mark.parametrize("stack_name", ["encoder", "decoder"])
def test_output_is_bitwise_invariant_to_companions_and_row(stack_name: str) -> None:
    model = _FakeModel()
    install_sample_isolated_grouped_mm_experts(model, physical_batch_size=3)
    stack = _stack(model, stack_name)
    target = torch.tensor([[1.0, -2.0, 3.0], [4.0, 5.0, -6.0]])
    first_batch = torch.stack((target, torch.full_like(target, 7), torch.full_like(target, -9)))
    moved_batch = torch.stack((torch.full_like(target, 101), torch.full_like(target, -37), target))
    replicated_batch = target.unsqueeze(0).expand(3, -1, -1).clone()

    first_output = stack(first_batch)[0]
    moved_output = stack(moved_batch)[2]
    replicated_output = stack(replicated_batch)[1]

    assert torch.equal(first_output, moved_output)
    assert torch.equal(first_output, replicated_output)


def test_install_preserves_state_dict_parameters_and_existing_expert_hooks() -> None:
    model = _FakeModel(num_layers=1)
    experts = model.model.encoder.language_model.layers[0].experts
    events: list[str] = []

    def backend(*args, **kwargs):
        events.append("backend")
        return _fake_grouped_mm(*args, **kwargs)

    isolation._grouped_mm_experts_forward = backend
    experts.register_forward_pre_hook(lambda *_args: events.append("tp_pre"))
    experts.register_forward_hook(lambda *_args: events.append("tp_post"))
    state_keys_before = tuple(model.state_dict())
    parameter_ids_before = {name: id(parameter) for name, parameter in model.named_parameters()}

    contract = install_sample_isolated_grouped_mm_experts(model, physical_batch_size=2)
    model.model.encoder.language_model.layers[0](torch.randn(2, 3, 3))

    assert events == ["tp_pre", "backend", "backend", "tp_post"]
    assert tuple(model.state_dict()) == state_keys_before
    assert {name: id(parameter) for name, parameter in model.named_parameters()} == parameter_ids_before
    assert contract.target_names == (
        "model.encoder.language_model.layers.0.experts",
        "model.decoder.layers.0.experts",
    )


@pytest.mark.parametrize("physical_batch_size", [0, -1, True, 1.5])
def test_install_rejects_invalid_physical_batch_size(physical_batch_size: object) -> None:
    with pytest.raises(ValueError, match="physical_batch_size must be a positive integer"):
        install_sample_isolated_grouped_mm_experts(
            _FakeModel(),
            physical_batch_size=physical_batch_size,
        )


def test_runtime_rejects_wrong_physical_batch_and_direct_expert_call() -> None:
    model = _FakeModel(num_layers=1)
    install_sample_isolated_grouped_mm_experts(model, physical_batch_size=2)
    layer = model.model.encoder.language_model.layers[0]

    with pytest.raises(RuntimeError, match=r"physical batch mismatch.*expected 2, observed 1"):
        layer(torch.randn(1, 3, 3))
    with pytest.raises(RuntimeError, match="outside its validated layer"):
        layer.experts(
            torch.randn(6, 3),
            torch.zeros(6, 2, dtype=torch.long),
            torch.full((6, 2), 0.5),
        )

    assert layer(torch.randn(2, 3, 3)).shape == (2, 3, 3)


def test_runtime_rejects_reentrant_layer_and_recovers_after_failure() -> None:
    model = _FakeModel(num_layers=1)
    install_sample_isolated_grouped_mm_experts(model, physical_batch_size=2)
    layer = model.model.encoder.language_model.layers[0]

    recursive_hook = layer.experts.register_forward_pre_hook(lambda *_args: layer(torch.randn(2, 3, 3)))
    with pytest.raises(RuntimeError, match="concurrent or re-entrant"):
        layer(torch.randn(2, 3, 3))
    recursive_hook.remove()

    assert layer(torch.randn(2, 3, 3)).shape == (2, 3, 3)


@pytest.mark.parametrize(
    ("drop_flat_tokens", "message"),
    [
        (1, "not divisible by physical batch"),
        (2, "flattened token count mismatch"),
    ],
)
def test_runtime_rejects_malformed_flattened_token_count(drop_flat_tokens: int, message: str) -> None:
    model = _FakeModel(num_layers=1, drop_first_encoder_layer_flat_tokens=drop_flat_tokens)
    install_sample_isolated_grouped_mm_experts(model, physical_batch_size=2)

    with pytest.raises(RuntimeError, match=message):
        model.model.encoder.language_model.layers[0](torch.randn(2, 3, 3))

    model.model.encoder.language_model.layers[0].drop_flat_tokens = 0
    assert model.model.encoder.language_model.layers[0](torch.randn(2, 3, 3)).shape == (2, 3, 3)


def test_verify_rejects_replaced_hook_callable() -> None:
    model = _FakeModel(num_layers=1)
    install_sample_isolated_grouped_mm_experts(model, physical_batch_size=2)
    layer = model.model.encoder.language_model.layers[0]
    installed_hook_id = next(iter(layer._forward_pre_hooks))
    layer._forward_pre_hooks[installed_hook_id] = lambda *_args: None

    with pytest.raises(RuntimeError, match="physical-batch pre-hook is missing"):
        verify_sample_isolated_grouped_mm_experts(model, physical_batch_size=2)


def test_install_rejects_wrong_backend_duplicate_install_and_incomplete_coverage() -> None:
    with pytest.raises(RuntimeError, match="experts backend mismatch"):
        install_sample_isolated_grouped_mm_experts(_FakeModel(backend="eager"), physical_batch_size=2)

    target_backend_mismatch = _FakeModel()
    target_backend_mismatch.model.decoder.layers[0].experts.config = SimpleNamespace(_experts_implementation="eager")
    with pytest.raises(RuntimeError, match=r"expert backend mismatch at model\.decoder\.layers\.0\.experts"):
        install_sample_isolated_grouped_mm_experts(target_backend_mismatch, physical_batch_size=2)

    installed = _FakeModel()
    install_sample_isolated_grouped_mm_experts(installed, physical_batch_size=2)
    with pytest.raises(RuntimeError, match="already installed"):
        install_sample_isolated_grouped_mm_experts(installed, physical_batch_size=2)
    with pytest.raises(RuntimeError, match="installed physical batch mismatch"):
        verify_sample_isolated_grouped_mm_experts(installed, physical_batch_size=3)

    incomplete = _FakeModel()
    incomplete.model.decoder.layers = nn.ModuleList([incomplete.model.decoder.layers[0]])
    with pytest.raises(RuntimeError, match="expert coverage is incomplete"):
        validate_diffusion_gemma_grouped_mm_expert_topology(incomplete)

    aliased = _FakeModel(num_layers=1)
    aliased.model.decoder.layers[0].experts = aliased.model.encoder.language_model.layers[0].experts
    with pytest.raises(RuntimeError, match="must be distinct module instances"):
        install_sample_isolated_grouped_mm_experts(aliased, physical_batch_size=2)
