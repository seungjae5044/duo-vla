from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from duo_vla.backbones.sample_isolated_experts_v2 import (
    EXPECTED_NUM_EXPERTS,
    EXPECTED_TOP_K,
    SUPPORTED_PHYSICAL_BATCH_SIZES,
    _composite_routing_plan,
    _fused_shared_weight_experts,
    _require_tensor_route_range,
    _shared_weight_grouped_linear,
    install_sample_isolated_grouped_mm_experts_v2,
    verify_sample_isolated_grouped_mm_experts_v2,
)


class _FakeExperts(nn.Module):
    def __init__(self, config: SimpleNamespace, *, trainable: bool = False) -> None:
        super().__init__()
        self.config = config
        self.num_experts = EXPECTED_NUM_EXPERTS
        self.hidden_dim = 4
        self.intermediate_dim = 2
        self.has_gate = True
        self.has_bias = False
        self.is_transposed = False
        self.is_concatenated = True
        self.gate_up_proj = nn.Parameter(
            torch.zeros(self.num_experts, 2 * self.intermediate_dim, self.hidden_dim),
            requires_grad=trainable,
        )
        self.down_proj = nn.Parameter(
            torch.zeros(self.num_experts, self.hidden_dim, self.intermediate_dim),
            requires_grad=trainable,
        )

    def _apply_gate(self, projected: torch.Tensor) -> torch.Tensor:
        gate, up = projected.chunk(2, dim=-1)
        return torch.nn.functional.gelu(gate, approximate="tanh") * up

    def forward(
        self,
        hidden_states: torch.Tensor,
        _top_k_index: torch.Tensor,
        _top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        return hidden_states


class _FakeLayer(nn.Module):
    def __init__(self, config: SimpleNamespace, *, trainable: bool = False) -> None:
        super().__init__()
        self.experts = _FakeExperts(config, trainable=trainable)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        flat = hidden_states.flatten(0, 1)
        routes = torch.zeros(flat.shape[0], EXPECTED_TOP_K, dtype=torch.int64, device=flat.device)
        weights = torch.full_like(routes, 1 / EXPECTED_TOP_K, dtype=flat.dtype)
        return self.experts(flat, routes, weights).view_as(hidden_states)


class _FakeStack(nn.Module):
    def __init__(self, config: SimpleNamespace, *, trainable: bool = False) -> None:
        super().__init__()
        self.layers = nn.ModuleList([_FakeLayer(config, trainable=trainable)])


class _FakeModel(nn.Module):
    def __init__(self, *, trainable: bool = False) -> None:
        super().__init__()
        text_config = SimpleNamespace(num_hidden_layers=1, _experts_implementation="grouped_mm")
        self.config = SimpleNamespace(text_config=text_config)
        encoder = nn.Module()
        encoder.language_model = _FakeStack(text_config, trainable=trainable)
        inner = nn.Module()
        inner.encoder = encoder
        inner.decoder = _FakeStack(text_config, trainable=trainable)
        self.model = inner


@pytest.mark.parametrize("physical_batch_size", sorted(SUPPORTED_PHYSICAL_BATCH_SIZES))
def test_v2_install_supports_candidate_physical_batches_without_changing_registration(
    physical_batch_size: int,
) -> None:
    model = _FakeModel()
    state_keys = tuple(model.state_dict())
    parameter_ids = {name: id(parameter) for name, parameter in model.named_parameters()}

    contract = install_sample_isolated_grouped_mm_experts_v2(
        model,
        physical_batch_size=physical_batch_size,
    )

    assert contract.physical_batch_size == physical_batch_size
    assert contract.target_count == 2
    assert verify_sample_isolated_grouped_mm_experts_v2(
        model,
        physical_batch_size=physical_batch_size,
    ) == contract
    assert tuple(model.state_dict()) == state_keys
    assert {name: id(parameter) for name, parameter in model.named_parameters()} == parameter_ids


@pytest.mark.parametrize("physical_batch_size", [1, 2, 4, 7, 9, 128])
def test_v2_install_rejects_unqualified_physical_batches(physical_batch_size: int) -> None:
    with pytest.raises(ValueError, match=r"must be one of \{8, 16, 32, 64\}"):
        install_sample_isolated_grouped_mm_experts_v2(
            _FakeModel(),
            physical_batch_size=physical_batch_size,
        )


def test_v2_install_and_verify_fail_closed_on_trainable_expert_weights() -> None:
    with pytest.raises(RuntimeError, match="forbids trainable expert weights"):
        install_sample_isolated_grouped_mm_experts_v2(_FakeModel(trainable=True), physical_batch_size=8)

    model = _FakeModel()
    install_sample_isolated_grouped_mm_experts_v2(model, physical_batch_size=8)
    model.model.decoder.layers[0].experts.down_proj.requires_grad_(True)
    with pytest.raises(RuntimeError, match="forbids trainable expert weights"):
        verify_sample_isolated_grouped_mm_experts_v2(model, physical_batch_size=8)


def test_composite_routing_uses_global_tail_for_sentinels() -> None:
    physical_batch_size = 8
    tokens_per_sample = 2
    route_ids = torch.arange(
        physical_batch_size * tokens_per_sample * EXPECTED_TOP_K,
        dtype=torch.int64,
    ).remainder(EXPECTED_NUM_EXPERTS)
    route_ids[1] = EXPECTED_NUM_EXPERTS
    route_ids[-2] = EXPECTED_NUM_EXPERTS
    route_ids = route_ids.view(physical_batch_size * tokens_per_sample, EXPECTED_TOP_K)

    permutation, offsets, sentinel_mask = _composite_routing_plan(
        route_ids,
        physical_batch_size=physical_batch_size,
        tokens_per_sample=tokens_per_sample,
        num_experts=EXPECTED_NUM_EXPERTS,
    )

    sorted_route_ids = route_ids.reshape(-1)[permutation]
    assert offsets.shape == (physical_batch_size * EXPECTED_NUM_EXPERTS,)
    assert int(offsets[-1]) == route_ids.numel() - 2
    assert torch.equal(sorted_route_ids[-2:], torch.full((2,), EXPECTED_NUM_EXPERTS))
    assert torch.equal(sentinel_mask.squeeze(-1), sorted_route_ids == EXPECTED_NUM_EXPERTS)


@pytest.mark.parametrize("invalid_id", [-1, EXPECTED_NUM_EXPERTS + 1])
def test_route_range_rejects_every_non_sentinel_out_of_range_id(invalid_id: int) -> None:
    route_ids = torch.zeros(2, EXPECTED_TOP_K, dtype=torch.int64)
    route_ids[0, 0] = invalid_id
    with pytest.raises(RuntimeError, match=r"ids must be in \[0, E\]"):
        _require_tensor_route_range(route_ids, num_experts=EXPECTED_NUM_EXPERTS, target_name="test.experts")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="v2 Triton kernel requires CUDA")
@pytest.mark.parametrize("physical_batch_size", sorted(SUPPORTED_PHYSICAL_BATCH_SIZES))
def test_shared_weight_kernel_reuses_experts_across_composite_groups(physical_batch_size: int) -> None:
    device = torch.device("cuda", 0)
    torch.manual_seed(2026)
    input_size = 64
    output_size = 64
    group_count = physical_batch_size * EXPECTED_NUM_EXPERTS
    counts = torch.zeros(group_count, dtype=torch.int32, device=device)
    active_groups = torch.arange(physical_batch_size, device=device) * EXPECTED_NUM_EXPERTS
    active_groups += torch.arange(physical_batch_size, device=device).remainder(EXPECTED_NUM_EXPERTS)
    counts[active_groups] = 3
    offsets = counts.cumsum(dim=0, dtype=torch.int32)
    route_count = physical_batch_size * 3
    weight = torch.randn(
        EXPECTED_NUM_EXPERTS,
        output_size,
        input_size,
        dtype=torch.bfloat16,
        device=device,
    )
    actual_input = torch.randn(route_count, input_size, dtype=torch.bfloat16, device=device, requires_grad=True)
    expected_input = actual_input.detach().clone().requires_grad_(True)

    actual = _shared_weight_grouped_linear(actual_input, weight, offsets)
    expected_groups = []
    start = 0
    for group_id, end in enumerate(offsets.tolist()):
        if end != start:
            expected_groups.append(expected_input[start:end] @ weight[group_id % EXPECTED_NUM_EXPERTS].T)
        start = end
    expected = torch.cat(expected_groups)

    gradient = torch.randn_like(actual)
    actual.backward(gradient)
    expected.backward(gradient)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    torch.testing.assert_close(actual_input.grad, expected_input.grad, atol=0, rtol=0)
    assert weight.grad is None
    assert weight.untyped_storage().nbytes() == weight.numel() * weight.element_size()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="v2 Triton kernel requires CUDA")
def test_shared_weight_autograd_rejects_trainable_weight() -> None:
    weight = torch.randn(
        EXPECTED_NUM_EXPERTS,
        64,
        64,
        dtype=torch.bfloat16,
        device="cuda",
        requires_grad=True,
    )
    input = torch.randn(3, 64, dtype=torch.bfloat16, device="cuda")
    offsets = torch.zeros(8 * EXPECTED_NUM_EXPERTS, dtype=torch.int32, device="cuda")
    offsets[0:] = 3
    with pytest.raises(RuntimeError, match="forbids trainable expert weights"):
        _shared_weight_grouped_linear(input, weight, offsets)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="v2 Triton kernel requires CUDA")
def test_fused_experts_preserve_routes_sentinels_and_companion_isolation() -> None:
    moe = pytest.importorskip("transformers.integrations.moe")
    device = torch.device("cuda", 0)
    torch.manual_seed(2026)
    physical_batch_size = 8
    tokens_per_sample = 4
    hidden_size = 64
    intermediate_size = 32

    experts = _FakeExperts(SimpleNamespace(_experts_implementation="grouped_mm"))
    experts.hidden_dim = hidden_size
    experts.intermediate_dim = intermediate_size
    experts.gate_up_proj = nn.Parameter(
        torch.randn(
            EXPECTED_NUM_EXPERTS,
            2 * intermediate_size,
            hidden_size,
            dtype=torch.bfloat16,
            device=device,
        )
        * 0.03,
        requires_grad=False,
    )
    experts.down_proj = nn.Parameter(
        torch.randn(
            EXPECTED_NUM_EXPERTS,
            hidden_size,
            intermediate_size,
            dtype=torch.bfloat16,
            device=device,
        )
        * 0.04,
        requires_grad=False,
    )

    expected_input = torch.randn(
        physical_batch_size * tokens_per_sample,
        hidden_size,
        dtype=torch.bfloat16,
        device=device,
        requires_grad=True,
    )
    actual_input = expected_input.detach().clone().requires_grad_(True)
    route_ids = torch.randint(
        EXPECTED_NUM_EXPERTS,
        (physical_batch_size * tokens_per_sample, EXPECTED_TOP_K),
        dtype=torch.int64,
        device=device,
    )
    route_ids[0, -1] = EXPECTED_NUM_EXPERTS
    expected_route_weights = torch.softmax(
        torch.randn(route_ids.shape, dtype=torch.float32, device=device),
        dim=-1,
    ).requires_grad_(True)
    with torch.no_grad():
        expected_route_weights[0, -1] = 0.0
    actual_route_weights = expected_route_weights.detach().clone().requires_grad_(True)

    expected = torch.cat(
        [
            moe.grouped_mm_experts_forward(
                experts,
                expected_input[start : start + tokens_per_sample],
                route_ids[start : start + tokens_per_sample],
                expected_route_weights[start : start + tokens_per_sample],
            )
            for start in range(0, expected_input.shape[0], tokens_per_sample)
        ],
        dim=0,
    )
    actual = _fused_shared_weight_experts(
        experts,
        actual_input,
        route_ids,
        actual_route_weights,
        physical_batch_size=physical_batch_size,
        tokens_per_sample=tokens_per_sample,
        target_name="test.experts",
    )
    repeated = _fused_shared_weight_experts(
        experts,
        actual_input,
        route_ids,
        actual_route_weights,
        physical_batch_size=physical_batch_size,
        tokens_per_sample=tokens_per_sample,
        target_name="test.experts",
    )
    assert torch.equal(actual, repeated)
    assert bool(torch.isfinite(actual).all())

    gradient = torch.randn_like(actual)
    actual.backward(gradient, retain_graph=True)
    actual_input_gradient = actual_input.grad.detach().clone()
    actual_route_gradient = actual_route_weights.grad.detach().clone()
    actual_input.grad = None
    actual_route_weights.grad = None
    repeated.backward(gradient)
    assert torch.equal(actual_input.grad, actual_input_gradient)
    assert torch.equal(actual_route_weights.grad, actual_route_gradient)

    expected.backward(gradient)
    torch.testing.assert_close(actual, expected, atol=0.01, rtol=0.01)
    torch.testing.assert_close(actual_input_gradient, expected_input.grad, atol=0.02, rtol=0.02)
    torch.testing.assert_close(actual_route_gradient, expected_route_weights.grad, atol=0.25, rtol=0.01)
    assert actual_route_gradient[0, -1].item() == 0.0

    # A target-only loss must not leak through either projection or routing
    # weights into companion rows.
    isolation_input = actual_input.detach().clone().requires_grad_(True)
    isolation_weights = actual_route_weights.detach().clone().requires_grad_(True)
    isolated = _fused_shared_weight_experts(
        experts,
        isolation_input,
        route_ids,
        isolation_weights,
        physical_batch_size=physical_batch_size,
        tokens_per_sample=tokens_per_sample,
        target_name="test.experts",
    )
    isolated[:tokens_per_sample].float().square().mean().backward()
    assert int(torch.count_nonzero(isolation_input.grad[tokens_per_sample:]).item()) == 0
    assert int(torch.count_nonzero(isolation_weights.grad[tokens_per_sample:]).item()) == 0
