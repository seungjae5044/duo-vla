"""Exact per-sample execution for DiffusionGemma's grouped-MM experts.

The Transformers grouped-MM implementation flattens every token in a physical
batch before sorting routes by expert. Consequently, the arithmetic for one
sample can depend on its batch companions. This module retains a fixed physical
batch at the surrounding layer while executing each sample's expert work
independently, which reproduces singleton grouped-MM semantics.

Installation deliberately happens on the expert forward method, inside any
existing module forward hooks (including native tensor-parallel hooks). It does
not replace modules or parameters and therefore does not change state_dict keys.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MethodType
from typing import Any
from weakref import WeakKeyDictionary

import torch
from torch import nn
from torch.utils.hooks import RemovableHandle

GROUPED_MM_EXPERTS_IMPLEMENTATION = "grouped_mm"


@dataclass(frozen=True, slots=True)
class GroupedMMExpertTopology:
    """Validated encoder/decoder expert coverage for DiffusionGemma."""

    experts_implementation: str
    encoder_layer_count: int
    decoder_layer_count: int
    target_names: tuple[str, ...]

    @property
    def target_count(self) -> int:
        return len(self.target_names)


@dataclass(frozen=True, slots=True)
class SampleIsolatedGroupedMMContract:
    """The installed fixed-physical-batch expert execution contract."""

    physical_batch_size: int
    experts_implementation: str
    encoder_layer_count: int
    decoder_layer_count: int
    target_names: tuple[str, ...]

    @property
    def target_count(self) -> int:
        return len(self.target_names)


@dataclass(slots=True)
class _InvocationState:
    target_name: str
    physical_batch_size: int
    expected_flat_token_count: int | None = None
    tokens_per_sample: int | None = None
    hidden_size: int | None = None
    consumed: bool = False
    failed: bool = False

    def clear(self) -> None:
        self.expected_flat_token_count = None
        self.tokens_per_sample = None
        self.hidden_size = None
        self.consumed = False
        self.failed = False


@dataclass(frozen=True, slots=True)
class _TopologyTarget:
    name: str
    layer: nn.Module
    experts: nn.Module


@dataclass(frozen=True, slots=True)
class _InstalledTarget:
    topology: _TopologyTarget
    invocation: _InvocationState
    layer_pre_hook_callable: Any
    layer_post_hook_callable: Any
    layer_pre_hook: RemovableHandle
    layer_post_hook: RemovableHandle


@dataclass(frozen=True, slots=True)
class _Installation:
    contract: SampleIsolatedGroupedMMContract
    targets: tuple[_InstalledTarget, ...]


_MODEL_INSTALLATIONS: WeakKeyDictionary[nn.Module, _Installation] = WeakKeyDictionary()
_EXPERT_INVOCATIONS: WeakKeyDictionary[nn.Module, _InvocationState] = WeakKeyDictionary()


def _grouped_mm_experts_forward(
    experts: nn.Module,
    hidden_states: torch.Tensor,
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
) -> torch.Tensor:
    """Resolve the pinned Transformers implementation lazily for minimal installs."""

    try:
        from transformers.integrations.moe import grouped_mm_experts_forward
    except ImportError as exc:  # pragma: no cover - the train environment pins Transformers
        raise ImportError("sample-isolated grouped-MM experts require Duo-VLA's train dependencies") from exc
    return grouped_mm_experts_forward(experts, hidden_states, top_k_index, top_k_weights)


def _positive_integer(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _diffusion_gemma_targets(
    model: nn.Module,
    *,
    expected_num_layers: int | None,
    expected_experts_implementation: str,
) -> tuple[GroupedMMExpertTopology, tuple[_TopologyTarget, ...]]:
    if expected_experts_implementation != GROUPED_MM_EXPERTS_IMPLEMENTATION:
        raise ValueError(f"sample isolation supports only experts_implementation={GROUPED_MM_EXPERTS_IMPLEMENTATION!r}")

    try:
        text_config = model.config.text_config
        configured_num_layers = text_config.num_hidden_layers
        encoder_layers = model.model.encoder.language_model.layers
        decoder_layers = model.model.decoder.layers
    except AttributeError as exc:
        raise RuntimeError("model does not expose the required DiffusionGemma encoder/decoder topology") from exc

    configured_num_layers = _positive_integer(configured_num_layers, name="config.text_config.num_hidden_layers")
    if expected_num_layers is not None:
        expected_num_layers = _positive_integer(expected_num_layers, name="expected_num_layers")
        if configured_num_layers != expected_num_layers:
            raise RuntimeError(
                "DiffusionGemma expert layer count changed: "
                f"expected {expected_num_layers}, config declares {configured_num_layers}"
            )
    if not isinstance(encoder_layers, nn.ModuleList) or not isinstance(decoder_layers, nn.ModuleList):
        raise RuntimeError("DiffusionGemma encoder and decoder layers must be registered ModuleList instances")
    if len(encoder_layers) != configured_num_layers or len(decoder_layers) != configured_num_layers:
        raise RuntimeError(
            "DiffusionGemma expert coverage is incomplete: "
            f"config={configured_num_layers}, encoder={len(encoder_layers)}, decoder={len(decoder_layers)}"
        )

    observed_backend = getattr(text_config, "_experts_implementation", None)
    if observed_backend != expected_experts_implementation:
        raise RuntimeError(
            "DiffusionGemma experts backend mismatch: "
            f"expected {expected_experts_implementation!r}, observed {observed_backend!r}"
        )

    targets: list[_TopologyTarget] = []
    for stack_name, layers in (
        ("model.encoder.language_model.layers", encoder_layers),
        ("model.decoder.layers", decoder_layers),
    ):
        for layer_index, layer in enumerate(layers):
            target_name = f"{stack_name}.{layer_index}.experts"
            experts = getattr(layer, "experts", None)
            if not isinstance(experts, nn.Module) or not callable(getattr(experts, "forward", None)):
                raise RuntimeError(f"missing callable expert module at {target_name}")
            target_backend = getattr(getattr(experts, "config", None), "_experts_implementation", None)
            if target_backend != expected_experts_implementation:
                raise RuntimeError(
                    f"expert backend mismatch at {target_name}: "
                    f"expected {expected_experts_implementation!r}, observed {target_backend!r}"
                )
            targets.append(_TopologyTarget(name=target_name, layer=layer, experts=experts))

    layer_ids = [id(target.layer) for target in targets]
    expert_ids = [id(target.experts) for target in targets]
    if len(layer_ids) != len(set(layer_ids)) or len(expert_ids) != len(set(expert_ids)):
        raise RuntimeError("DiffusionGemma encoder/decoder expert targets must be distinct module instances")

    registered_modules = dict(model.named_modules(remove_duplicate=False))
    missing_or_aliased = [
        target.name for target in targets if registered_modules.get(target.name) is not target.experts
    ]
    if missing_or_aliased:
        raise RuntimeError(
            f"DiffusionGemma expert modules are not registered at their exact target paths: {missing_or_aliased}"
        )

    topology = GroupedMMExpertTopology(
        experts_implementation=expected_experts_implementation,
        encoder_layer_count=len(encoder_layers),
        decoder_layer_count=len(decoder_layers),
        target_names=tuple(target.name for target in targets),
    )
    return topology, tuple(targets)


def validate_diffusion_gemma_grouped_mm_expert_topology(
    model: nn.Module,
    *,
    expected_num_layers: int | None = None,
    expected_experts_implementation: str = GROUPED_MM_EXPERTS_IMPLEMENTATION,
) -> GroupedMMExpertTopology:
    """Validate exact encoder and decoder coverage by the grouped-MM backend."""

    topology, _ = _diffusion_gemma_targets(
        model,
        expected_num_layers=expected_num_layers,
        expected_experts_implementation=expected_experts_implementation,
    )
    return topology


def _hidden_states_from_layer_call(args: tuple[Any, ...], kwargs: dict[str, Any]) -> torch.Tensor:
    hidden_states = kwargs.get("hidden_states", args[0] if args else None)
    if not isinstance(hidden_states, torch.Tensor) or hidden_states.ndim != 3:
        raise RuntimeError("DiffusionGemma expert layer input must be a rank-3 hidden-state tensor [B, S, D]")
    return hidden_states


def _make_layer_pre_hook(invocation: _InvocationState):
    def hook(_module: nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
        if invocation.expected_flat_token_count is not None:
            raise RuntimeError(f"concurrent or re-entrant expert layer call is forbidden at {invocation.target_name}")
        hidden_states = _hidden_states_from_layer_call(args, kwargs)
        observed_batch = int(hidden_states.shape[0])
        if observed_batch != invocation.physical_batch_size:
            raise RuntimeError(
                f"physical batch mismatch at {invocation.target_name}: "
                f"expected {invocation.physical_batch_size}, observed {observed_batch}"
            )
        tokens_per_sample = int(hidden_states.shape[1])
        hidden_size = int(hidden_states.shape[2])
        if tokens_per_sample <= 0 or hidden_size <= 0:
            raise RuntimeError(f"empty token or hidden dimension at {invocation.target_name}")
        invocation.expected_flat_token_count = invocation.physical_batch_size * tokens_per_sample
        invocation.tokens_per_sample = tokens_per_sample
        invocation.hidden_size = hidden_size
        invocation.consumed = False
        invocation.failed = False

    return hook


def _make_layer_post_hook(invocation: _InvocationState):
    def hook(
        _module: nn.Module,
        _args: tuple[Any, ...],
        _kwargs: dict[str, Any],
        output: Any,
    ) -> None:
        try:
            # always_call=True also invokes this hook while another part of the
            # layer is raising. A normal DiffusionGemma layer always returns a
            # tensor, so do not replace an in-flight exception when output is None.
            if (
                invocation.expected_flat_token_count is not None
                and not invocation.consumed
                and not invocation.failed
                and output is not None
            ):
                raise RuntimeError(f"expert module was not called exactly once at {invocation.target_name}")
        finally:
            invocation.clear()

    return hook


def _sample_isolated_grouped_mm_forward(
    experts: nn.Module,
    hidden_states: torch.Tensor,
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
) -> torch.Tensor:
    invocation = _EXPERT_INVOCATIONS.get(experts)
    if invocation is None:
        raise RuntimeError("sample-isolated expert forward is not associated with an active installation")
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
        if flat_token_count % physical_batch_size != 0:
            raise RuntimeError(
                f"flattened token count {flat_token_count} is not divisible by physical batch "
                f"{physical_batch_size} at {invocation.target_name}"
            )
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

        sample_outputs = []
        for sample_index in range(physical_batch_size):
            start = sample_index * tokens_per_sample
            sample_outputs.append(
                _grouped_mm_experts_forward(
                    experts,
                    hidden_states.narrow(0, start, tokens_per_sample),
                    top_k_index.narrow(0, start, tokens_per_sample),
                    top_k_weights.narrow(0, start, tokens_per_sample),
                )
            )
        output = torch.cat(sample_outputs, dim=0)
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


def install_sample_isolated_grouped_mm_experts(
    model: nn.Module,
    *,
    physical_batch_size: int,
) -> SampleIsolatedGroupedMMContract:
    """Install exact singleton-style grouped-MM execution for a fixed physical batch.

    The installation is intentionally one-shot. Calls with an unexpected batch,
    malformed flattened token count, incomplete topology, or another expert
    backend fail closed.
    """

    physical_batch_size = _positive_integer(physical_batch_size, name="physical_batch_size")
    if model in _MODEL_INSTALLATIONS:
        raise RuntimeError("sample-isolated grouped-MM experts are already installed on this model")

    topology, targets = _diffusion_gemma_targets(
        model,
        expected_num_layers=None,
        expected_experts_implementation=GROUPED_MM_EXPERTS_IMPLEMENTATION,
    )
    for target in targets:
        if target.experts in _EXPERT_INVOCATIONS or "forward" in target.experts.__dict__:
            raise RuntimeError(f"expert forward is already patched at {target.name}")

    contract = SampleIsolatedGroupedMMContract(
        physical_batch_size=physical_batch_size,
        experts_implementation=topology.experts_implementation,
        encoder_layer_count=topology.encoder_layer_count,
        decoder_layer_count=topology.decoder_layer_count,
        target_names=topology.target_names,
    )
    installed: list[_InstalledTarget] = []
    patched_experts: list[nn.Module] = []
    created_hooks: list[RemovableHandle] = []
    try:
        for target in targets:
            invocation = _InvocationState(
                target_name=target.name,
                physical_batch_size=physical_batch_size,
            )
            target.experts.forward = MethodType(_sample_isolated_grouped_mm_forward, target.experts)
            patched_experts.append(target.experts)
            _EXPERT_INVOCATIONS[target.experts] = invocation
            pre_hook_callable = _make_layer_pre_hook(invocation)
            post_hook_callable = _make_layer_post_hook(invocation)
            pre_hook = target.layer.register_forward_pre_hook(
                pre_hook_callable,
                with_kwargs=True,
            )
            created_hooks.append(pre_hook)
            post_hook = target.layer.register_forward_hook(
                post_hook_callable,
                with_kwargs=True,
                always_call=True,
            )
            created_hooks.append(post_hook)
            installed.append(
                _InstalledTarget(
                    topology=target,
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
            _EXPERT_INVOCATIONS.pop(experts, None)
            if "forward" in experts.__dict__:
                delattr(experts, "forward")
        raise

    installation = _Installation(contract=contract, targets=tuple(installed))
    _MODEL_INSTALLATIONS[model] = installation
    return verify_sample_isolated_grouped_mm_experts(
        model,
        physical_batch_size=physical_batch_size,
    )


def verify_sample_isolated_grouped_mm_experts(
    model: nn.Module,
    *,
    physical_batch_size: int,
) -> SampleIsolatedGroupedMMContract:
    """Verify that the exact installation and all layer guards remain intact."""

    physical_batch_size = _positive_integer(physical_batch_size, name="physical_batch_size")
    installation = _MODEL_INSTALLATIONS.get(model)
    if installation is None:
        raise RuntimeError("sample-isolated grouped-MM experts are not installed on this model")
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
    if topology.target_names != installation.contract.target_names:
        raise RuntimeError("installed expert target coverage no longer matches the model topology")
    if len(targets) != len(installation.targets):
        raise RuntimeError("installed expert target count no longer matches the model topology")

    for observed, installed in zip(targets, installation.targets, strict=True):
        if observed.layer is not installed.topology.layer or observed.experts is not installed.topology.experts:
            raise RuntimeError(f"expert target identity changed at {observed.name}")
        bound_forward = observed.experts.__dict__.get("forward")
        if (
            not isinstance(bound_forward, MethodType)
            or bound_forward.__func__ is not _sample_isolated_grouped_mm_forward
        ):
            raise RuntimeError(f"sample-isolated expert forward is missing at {observed.name}")
        if _EXPERT_INVOCATIONS.get(observed.experts) is not installed.invocation:
            raise RuntimeError(f"sample-isolated expert invocation guard is missing at {observed.name}")
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

    return installation.contract


__all__ = [
    "GROUPED_MM_EXPERTS_IMPLEMENTATION",
    "GroupedMMExpertTopology",
    "SampleIsolatedGroupedMMContract",
    "install_sample_isolated_grouped_mm_experts",
    "validate_diffusion_gemma_grouped_mm_expert_topology",
    "verify_sample_isolated_grouped_mm_experts",
]
