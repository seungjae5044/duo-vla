"""Fused sample-isolated DiffusionGemma experts with shared weights.

Version 1 obtains exact singleton semantics by invoking Transformers'
grouped-MM expert forward once per physical sample.  This experimental v2
keeps the same ``(sample_id, expert_id)`` routing groups, but submits every
composite group to one Triton kernel per projection.  The kernel maps each
composite group back to one of the original expert matrices, so no expert
weight is repeated or materialized.

The implementation is deliberately narrow and fail-closed.  It supports the
single-GPU DiffusionGemma contract used by Duo-VLA: E=128, top-k=8, BF16 CUDA,
concatenated gated experts without bias, frozen expert matrices, and physical
batches 8/16/32/64.  Expert-parallel sentinel id E is preserved as a global
tail group and masked exactly once before/after the two projections.  Other
out-of-range route ids are rejected asynchronously on CUDA.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MethodType
from typing import Any
from weakref import WeakKeyDictionary

import torch
from torch import nn
from torch.utils.hooks import RemovableHandle

from duo_vla.backbones.sample_isolated_experts import (
    GROUPED_MM_EXPERTS_IMPLEMENTATION,
    SampleIsolatedGroupedMMContract,
    _diffusion_gemma_targets,
    _InvocationState,
    _make_layer_post_hook,
    _make_layer_pre_hook,
    _positive_integer,
)

SAMPLE_ISOLATED_GROUPED_MM_V2 = "sample_isolated_grouped_mm_v2"
SUPPORTED_PHYSICAL_BATCH_SIZES = frozenset({8, 16, 32, 64, 72, 80})
EXPECTED_NUM_EXPERTS = 128
EXPECTED_TOP_K = 8


@dataclass(frozen=True, slots=True)
class _InstalledTargetV2:
    name: str
    layer: nn.Module
    experts: nn.Module
    invocation: _InvocationState
    layer_pre_hook_callable: Any
    layer_post_hook_callable: Any
    layer_pre_hook: RemovableHandle
    layer_post_hook: RemovableHandle


@dataclass(frozen=True, slots=True)
class _InstallationV2:
    contract: SampleIsolatedGroupedMMContract
    targets: tuple[_InstalledTargetV2, ...]


_MODEL_INSTALLATIONS_V2: WeakKeyDictionary[nn.Module, _InstallationV2] = WeakKeyDictionary()
_EXPERT_INVOCATIONS_V2: WeakKeyDictionary[nn.Module, _InvocationState] = WeakKeyDictionary()


def _run_shared_weight_grouped_mm(
    input: torch.Tensor,
    weight: torch.Tensor,
    group_offsets: torch.Tensor,
    *,
    weight_k_stride: int,
    weight_n_stride: int,
    output_size: int,
) -> torch.Tensor:
    """Import the pinned Triton implementation only in the training runtime."""

    try:
        from duo_vla.backbones.shared_weight_grouped_mm_triton import shared_weight_grouped_mm
    except ImportError as exc:  # pragma: no cover - the pinned train venv includes Triton
        raise ImportError("sample-isolated grouped-MM v2 requires the pinned Triton training runtime") from exc
    return shared_weight_grouped_mm(
        input,
        weight,
        group_offsets,
        weight_k_stride=weight_k_stride,
        weight_n_stride=weight_n_stride,
        output_size=output_size,
    )


class _FrozenSharedWeightGroupedLinear(torch.autograd.Function):
    """Autograd wrapper that computes only the input gradient.

    Expert matrices are frozen by the Duo-VLA adapter contract.  Refusing a
    trainable matrix is important: silently dropping its gradient would create
    a plausible-looking but incorrect training run.
    """

    @staticmethod
    def forward(
        ctx: Any,
        input: torch.Tensor,
        weight: torch.Tensor,
        group_offsets: torch.Tensor,
    ) -> torch.Tensor:
        if weight.requires_grad:
            raise RuntimeError("sample-isolated grouped-MM v2 forbids trainable expert weights")
        if weight.ndim != 3 or input.ndim != 2 or int(input.shape[1]) != int(weight.shape[2]):
            raise RuntimeError("shared-weight grouped linear input/weight shape mismatch")
        ctx.save_for_backward(weight, group_offsets)
        return _run_shared_weight_grouped_mm(
            input,
            weight,
            group_offsets,
            weight_k_stride=int(weight.stride(2)),
            weight_n_stride=int(weight.stride(1)),
            output_size=int(weight.shape[1]),
        )

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[torch.Tensor, None, None]:
        weight, group_offsets = ctx.saved_tensors
        if weight.requires_grad:
            raise RuntimeError("sample-isolated grouped-MM v2 expert weights became trainable during backward")
        grad_input = _run_shared_weight_grouped_mm(
            grad_output.contiguous(),
            weight,
            group_offsets,
            weight_k_stride=int(weight.stride(1)),
            weight_n_stride=int(weight.stride(2)),
            output_size=int(weight.shape[2]),
        )
        return grad_input, None, None


def _shared_weight_grouped_linear(
    input: torch.Tensor,
    weight: torch.Tensor,
    group_offsets: torch.Tensor,
) -> torch.Tensor:
    return _FrozenSharedWeightGroupedLinear.apply(input, weight, group_offsets)


def _require_supported_physical_batch_size(value: object) -> int:
    physical_batch_size = _positive_integer(value, name="physical_batch_size")
    if physical_batch_size not in SUPPORTED_PHYSICAL_BATCH_SIZES:
        supported = ", ".join(str(item) for item in sorted(SUPPORTED_PHYSICAL_BATCH_SIZES))
        raise ValueError(f"sample-isolated grouped-MM v2 physical_batch_size must be one of {{{supported}}}")
    return physical_batch_size


def _require_tensor_route_range(top_k_index: torch.Tensor, *, num_experts: int, target_name: str) -> None:
    """Allow only real expert ids and the documented EP sentinel id E."""

    valid = (top_k_index >= 0) & (top_k_index <= num_experts)
    message = f"expert route ids must be in [0, E] (E is the only sentinel) at {target_name}"
    if top_k_index.device.type == "cuda":
        # A host .item() here would synchronize every expert layer.  The
        # asynchronous assertion leaves the valid hot path asynchronous and
        # intentionally poisons a process that violates this fail-closed
        # routing contract.
        torch._assert_async(valid.all(), message)
    elif not bool(valid.all().item()):
        raise RuntimeError(message)


def _validate_expert_contract(
    experts: nn.Module,
    hidden_states: torch.Tensor,
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
    *,
    target_name: str,
) -> tuple[torch.Tensor, torch.Tensor, int, int]:
    if hidden_states.device.type != "cuda":
        raise RuntimeError(f"sample-isolated grouped-MM v2 requires CUDA hidden states at {target_name}")
    if hidden_states.dtype != torch.bfloat16:
        raise RuntimeError(f"sample-isolated grouped-MM v2 requires bfloat16 hidden states at {target_name}")
    if top_k_index.dtype != torch.int64:
        raise RuntimeError(f"sample-isolated grouped-MM v2 requires int64 route ids at {target_name}")
    if top_k_weights.dtype not in (torch.bfloat16, torch.float32):
        raise RuntimeError(f"sample-isolated grouped-MM v2 requires bfloat16/float32 route weights at {target_name}")
    if hidden_states.device != top_k_index.device or hidden_states.device != top_k_weights.device:
        raise RuntimeError(f"sample-isolated grouped-MM v2 routing tensors must share one device at {target_name}")

    if getattr(experts, "has_gate", None) is not True:
        raise RuntimeError(f"sample-isolated grouped-MM v2 requires gated experts at {target_name}")
    if getattr(experts, "has_bias", None) is not False:
        raise RuntimeError(f"sample-isolated grouped-MM v2 does not support expert bias at {target_name}")
    if getattr(experts, "is_transposed", None) is not False:
        raise RuntimeError(f"sample-isolated grouped-MM v2 requires non-transposed expert weights at {target_name}")
    if getattr(experts, "is_concatenated", None) is not True:
        raise RuntimeError(f"sample-isolated grouped-MM v2 requires concatenated gate/up weights at {target_name}")

    num_experts = getattr(experts, "num_experts", None)
    if num_experts != EXPECTED_NUM_EXPERTS:
        raise RuntimeError(
            f"sample-isolated grouped-MM v2 requires E={EXPECTED_NUM_EXPERTS}, "
            f"observed {num_experts!r} at {target_name}"
        )
    if int(top_k_index.shape[1]) != EXPECTED_TOP_K:
        raise RuntimeError(
            f"sample-isolated grouped-MM v2 requires top-k={EXPECTED_TOP_K}, "
            f"observed {int(top_k_index.shape[1])} at {target_name}"
        )

    gate_up_proj = getattr(experts, "gate_up_proj", None)
    down_proj = getattr(experts, "down_proj", None)
    if not isinstance(gate_up_proj, torch.Tensor) or not isinstance(down_proj, torch.Tensor):
        raise RuntimeError(f"sample-isolated grouped-MM v2 expert matrices are missing at {target_name}")
    if gate_up_proj.requires_grad or down_proj.requires_grad:
        raise RuntimeError(f"sample-isolated grouped-MM v2 forbids trainable expert weights at {target_name}")
    if gate_up_proj.device != hidden_states.device or down_proj.device != hidden_states.device:
        raise RuntimeError(
            f"sample-isolated grouped-MM v2 expert matrices must share the hidden-state device at {target_name}"
        )
    if gate_up_proj.dtype != torch.bfloat16 or down_proj.dtype != torch.bfloat16:
        raise RuntimeError(f"sample-isolated grouped-MM v2 expert matrices must be bfloat16 at {target_name}")
    if gate_up_proj.ndim != 3 or down_proj.ndim != 3:
        raise RuntimeError(f"sample-isolated grouped-MM v2 expert matrices must be rank 3 at {target_name}")

    hidden_size = int(hidden_states.shape[1])
    if tuple(gate_up_proj.shape[:1]) != (num_experts,) or int(gate_up_proj.shape[2]) != hidden_size:
        raise RuntimeError(f"sample-isolated grouped-MM v2 gate/up projection shape mismatch at {target_name}")
    if int(gate_up_proj.shape[1]) % 2 != 0:
        raise RuntimeError(f"sample-isolated grouped-MM v2 gate/up projection must split evenly at {target_name}")
    intermediate_size = int(gate_up_proj.shape[1]) // 2
    if tuple(down_proj.shape) != (num_experts, hidden_size, intermediate_size):
        raise RuntimeError(f"sample-isolated grouped-MM v2 down projection shape mismatch at {target_name}")
    if not gate_up_proj.is_contiguous() or not down_proj.is_contiguous():
        raise RuntimeError(f"sample-isolated grouped-MM v2 requires contiguous expert matrices at {target_name}")
    if not callable(getattr(experts, "_apply_gate", None)):
        raise RuntimeError(f"sample-isolated grouped-MM v2 gate function is missing at {target_name}")

    _require_tensor_route_range(top_k_index, num_experts=num_experts, target_name=target_name)
    return gate_up_proj, down_proj, num_experts, intermediate_size


def _composite_routing_plan(
    top_k_index: torch.Tensor,
    *,
    physical_batch_size: int,
    tokens_per_sample: int,
    num_experts: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sort routes by (sample, expert), putting all sentinels at one tail."""

    flat_expert_ids = top_k_index.reshape(-1)
    routes_per_sample = tokens_per_sample * int(top_k_index.shape[1])
    sample_ids = torch.arange(physical_batch_size, dtype=flat_expert_ids.dtype, device=flat_expert_ids.device)
    sample_ids = sample_ids.repeat_interleave(routes_per_sample)
    group_count = physical_batch_size * num_experts
    composite_ids = sample_ids * num_experts + flat_expert_ids
    composite_ids = composite_ids.masked_fill(flat_expert_ids == num_experts, group_count)
    composite_ids, permutation = torch.sort(composite_ids)

    histc_input = composite_ids.int() if composite_ids.device.type == "cuda" else composite_ids.float()
    tokens_per_group = torch.histc(histc_input, bins=group_count, min=0, max=group_count - 1)
    group_offsets = torch.cumsum(tokens_per_group, dim=0, dtype=torch.int32)
    sentinel_mask = (composite_ids == group_count).unsqueeze(-1)
    return permutation, group_offsets, sentinel_mask


def _fused_shared_weight_experts(
    experts: nn.Module,
    hidden_states: torch.Tensor,
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
    *,
    physical_batch_size: int,
    tokens_per_sample: int,
    target_name: str,
) -> torch.Tensor:
    """Execute the v2 numerical core after layer/invocation validation."""

    gate_up_proj, down_proj, num_experts, _ = _validate_expert_contract(
        experts,
        hidden_states,
        top_k_index,
        top_k_weights,
        target_name=target_name,
    )
    permutation, group_offsets, sentinel_mask = _composite_routing_plan(
        top_k_index,
        physical_batch_size=physical_batch_size,
        tokens_per_sample=tokens_per_sample,
        num_experts=num_experts,
    )

    num_top_k = int(top_k_index.shape[1])
    token_indices = permutation // num_top_k
    selected_hidden_states = hidden_states[token_indices]
    # The pre-mask is required for backward: custom grouped-MM deliberately
    # leaves sentinel-tail rows unwritten, and this mask's backward prevents
    # those rows from reaching the gather/scatter-add into hidden_states.
    selected_hidden_states.masked_fill_(sentinel_mask, 0.0)

    projected = _shared_weight_grouped_linear(selected_hidden_states, gate_up_proj, group_offsets)
    projected = experts._apply_gate(projected)
    projected = _shared_weight_grouped_linear(projected, down_proj, group_offsets)

    sorted_route_weights = top_k_weights.reshape(-1)[permutation]
    # A zero upstream gradient multiplied by an uninitialized sentinel value
    # can still produce NaN in d(top_k_weights).  Clear the projection itself
    # before the routing-weight multiply, not merely its final output.
    projected.masked_fill_(sentinel_mask, 0.0)
    weighted_output = projected * sorted_route_weights.unsqueeze(-1)
    # Never use multiplication-by-zero to remove an uninitialized sentinel
    # row: NaN * 0 remains NaN.  A post-mask is the exact safe operation.
    weighted_output.masked_fill_(sentinel_mask, 0.0)

    inverse_permutation = torch.empty_like(permutation)
    inverse_permutation[permutation] = torch.arange(permutation.numel(), device=permutation.device)
    weighted_output = weighted_output[inverse_permutation]
    output = weighted_output.view(hidden_states.shape[0], num_top_k, hidden_states.shape[1]).sum(dim=1)
    return output.to(hidden_states.dtype)


def _sample_isolated_grouped_mm_v2_forward(
    experts: nn.Module,
    hidden_states: torch.Tensor,
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
) -> torch.Tensor:
    invocation = _EXPERT_INVOCATIONS_V2.get(experts)
    if invocation is None:
        raise RuntimeError("sample-isolated grouped-MM v2 forward is not associated with an active installation")
    if invocation.expected_flat_token_count is None or invocation.tokens_per_sample is None:
        raise RuntimeError(f"expert module was invoked outside its validated layer at {invocation.target_name}")
    if invocation.consumed:
        invocation.failed = True
        raise RuntimeError(f"expert module was invoked more than once at {invocation.target_name}")

    try:
        if not isinstance(hidden_states, torch.Tensor) or hidden_states.ndim != 2:
            raise RuntimeError(f"expert hidden states must have shape [B*S, D] at {invocation.target_name}")
        if not isinstance(top_k_index, torch.Tensor) or not isinstance(top_k_weights, torch.Tensor):
            raise RuntimeError(f"expert routing inputs must be tensors at {invocation.target_name}")
        if top_k_index.ndim != 2 or top_k_weights.ndim != 2 or top_k_index.shape != top_k_weights.shape:
            raise RuntimeError(f"expert routing inputs must have matching shape [B*S, K] at {invocation.target_name}")

        flat_token_count = int(hidden_states.shape[0])
        physical_batch_size = invocation.physical_batch_size
        if flat_token_count != invocation.expected_flat_token_count:
            raise RuntimeError(
                f"flattened token count mismatch at {invocation.target_name}: "
                f"expected {invocation.expected_flat_token_count}, observed {flat_token_count}"
            )
        if int(hidden_states.shape[1]) != invocation.hidden_size:
            raise RuntimeError(
                f"expert hidden size mismatch at {invocation.target_name}: "
                f"expected {invocation.hidden_size}, observed {int(hidden_states.shape[1])}"
            )
        if int(top_k_index.shape[0]) != flat_token_count or int(top_k_index.shape[1]) <= 0:
            raise RuntimeError(f"expert routing token/top-k dimensions are invalid at {invocation.target_name}")

        tokens_per_sample = flat_token_count // physical_batch_size
        if tokens_per_sample != invocation.tokens_per_sample:
            raise RuntimeError(
                f"tokens per sample mismatch at {invocation.target_name}: "
                f"expected {invocation.tokens_per_sample}, observed {tokens_per_sample}"
            )

        output = _fused_shared_weight_experts(
            experts,
            hidden_states,
            top_k_index,
            top_k_weights,
            physical_batch_size=physical_batch_size,
            tokens_per_sample=tokens_per_sample,
            target_name=invocation.target_name,
        )
        if output.shape != hidden_states.shape:
            raise RuntimeError(
                f"grouped-MM expert output shape mismatch at {invocation.target_name}: "
                f"expected {tuple(hidden_states.shape)}, observed {tuple(output.shape)}"
            )
        invocation.consumed = True
        return output
    except Exception:
        invocation.failed = True
        raise


def _validate_target_weights_for_install(experts: nn.Module, *, target_name: str) -> None:
    for parameter_name in ("gate_up_proj", "down_proj"):
        parameter = getattr(experts, parameter_name, None)
        if not isinstance(parameter, torch.Tensor):
            raise RuntimeError(f"sample-isolated grouped-MM v2 missing {parameter_name} at {target_name}")
        if parameter.requires_grad:
            raise RuntimeError(f"sample-isolated grouped-MM v2 forbids trainable expert weights at {target_name}")


def install_sample_isolated_grouped_mm_experts_v2(
    model: nn.Module,
    *,
    physical_batch_size: int,
) -> SampleIsolatedGroupedMMContract:
    """Install the fused shared-weight v2 execution path on every expert layer."""

    physical_batch_size = _require_supported_physical_batch_size(physical_batch_size)
    if model in _MODEL_INSTALLATIONS_V2:
        raise RuntimeError("sample-isolated grouped-MM v2 experts are already installed on this model")

    topology, targets = _diffusion_gemma_targets(
        model,
        expected_num_layers=None,
        expected_experts_implementation=GROUPED_MM_EXPERTS_IMPLEMENTATION,
    )
    for target in targets:
        if target.experts in _EXPERT_INVOCATIONS_V2 or "forward" in target.experts.__dict__:
            raise RuntimeError(f"expert forward is already patched at {target.name}")
        _validate_target_weights_for_install(target.experts, target_name=target.name)

    contract = SampleIsolatedGroupedMMContract(
        physical_batch_size=physical_batch_size,
        experts_implementation=topology.experts_implementation,
        encoder_layer_count=topology.encoder_layer_count,
        decoder_layer_count=topology.decoder_layer_count,
        target_names=topology.target_names,
    )
    installed: list[_InstalledTargetV2] = []
    patched_experts: list[nn.Module] = []
    created_hooks: list[RemovableHandle] = []
    try:
        for target in targets:
            invocation = _InvocationState(target_name=target.name, physical_batch_size=physical_batch_size)
            target.experts.forward = MethodType(_sample_isolated_grouped_mm_v2_forward, target.experts)
            patched_experts.append(target.experts)
            _EXPERT_INVOCATIONS_V2[target.experts] = invocation
            pre_hook_callable = _make_layer_pre_hook(invocation)
            post_hook_callable = _make_layer_post_hook(invocation)
            pre_hook = target.layer.register_forward_pre_hook(pre_hook_callable, with_kwargs=True)
            created_hooks.append(pre_hook)
            post_hook = target.layer.register_forward_hook(
                post_hook_callable,
                with_kwargs=True,
                always_call=True,
            )
            created_hooks.append(post_hook)
            installed.append(
                _InstalledTargetV2(
                    name=target.name,
                    layer=target.layer,
                    experts=target.experts,
                    invocation=invocation,
                    layer_pre_hook_callable=pre_hook_callable,
                    layer_post_hook_callable=post_hook_callable,
                    layer_pre_hook=pre_hook,
                    layer_post_hook=post_hook,
                )
            )
    except Exception:
        for hook in reversed(created_hooks):
            hook.remove()
        for experts in reversed(patched_experts):
            _EXPERT_INVOCATIONS_V2.pop(experts, None)
            if "forward" in experts.__dict__:
                delattr(experts, "forward")
        raise

    _MODEL_INSTALLATIONS_V2[model] = _InstallationV2(contract=contract, targets=tuple(installed))
    return verify_sample_isolated_grouped_mm_experts_v2(model, physical_batch_size=physical_batch_size)


def verify_sample_isolated_grouped_mm_experts_v2(
    model: nn.Module,
    *,
    physical_batch_size: int,
) -> SampleIsolatedGroupedMMContract:
    """Verify exact v2 method/hook identities and frozen expert matrices."""

    physical_batch_size = _require_supported_physical_batch_size(physical_batch_size)
    installation = _MODEL_INSTALLATIONS_V2.get(model)
    if installation is None:
        raise RuntimeError("sample-isolated grouped-MM v2 experts are not installed on this model")
    if installation.contract.physical_batch_size != physical_batch_size:
        raise RuntimeError(
            "installed physical batch mismatch: "
            f"expected {physical_batch_size}, observed {installation.contract.physical_batch_size}"
        )

    topology, targets = _diffusion_gemma_targets(
        model,
        expected_num_layers=installation.contract.encoder_layer_count,
        expected_experts_implementation=installation.contract.experts_implementation,
    )
    if topology.target_names != installation.contract.target_names or len(targets) != len(installation.targets):
        raise RuntimeError("installed v2 expert coverage no longer matches the model topology")

    for observed, installed in zip(targets, installation.targets, strict=True):
        if (
            observed.name != installed.name
            or observed.layer is not installed.layer
            or observed.experts is not installed.experts
        ):
            raise RuntimeError(f"expert target identity changed at {observed.name}")
        bound_forward = observed.experts.__dict__.get("forward")
        if (
            not isinstance(bound_forward, MethodType)
            or bound_forward.__func__ is not _sample_isolated_grouped_mm_v2_forward
        ):
            raise RuntimeError(f"sample-isolated grouped-MM v2 forward is missing at {observed.name}")
        if _EXPERT_INVOCATIONS_V2.get(observed.experts) is not installed.invocation:
            raise RuntimeError(f"sample-isolated grouped-MM v2 invocation guard is missing at {observed.name}")
        if observed.layer._forward_pre_hooks.get(installed.layer_pre_hook.id) is not installed.layer_pre_hook_callable:
            raise RuntimeError(f"physical-batch pre-hook is missing at {observed.name}")
        if installed.layer_pre_hook.id not in observed.layer._forward_pre_hooks_with_kwargs:
            raise RuntimeError(f"physical-batch pre-hook kwargs contract is missing at {observed.name}")
        if observed.layer._forward_hooks.get(installed.layer_post_hook.id) is not installed.layer_post_hook_callable:
            raise RuntimeError(f"expert-consumption post-hook is missing at {observed.name}")
        if installed.layer_post_hook.id not in observed.layer._forward_hooks_with_kwargs:
            raise RuntimeError(f"expert-consumption post-hook kwargs contract is missing at {observed.name}")
        if installed.layer_post_hook.id not in observed.layer._forward_hooks_always_called:
            raise RuntimeError(f"expert-consumption always-call contract is missing at {observed.name}")
        _validate_target_weights_for_install(observed.experts, target_name=observed.name)

    return installation.contract


__all__ = [
    "EXPECTED_NUM_EXPERTS",
    "EXPECTED_TOP_K",
    "SAMPLE_ISOLATED_GROUPED_MM_V2",
    "SUPPORTED_PHYSICAL_BATCH_SIZES",
    "install_sample_isolated_grouped_mm_experts_v2",
    "verify_sample_isolated_grouped_mm_experts_v2",
]
