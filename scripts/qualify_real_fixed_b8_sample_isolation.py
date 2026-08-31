#!/usr/bin/env python3
"""Qualify exact fixed-B=8 sample-isolated MoE execution on real DiffusionGemma.

This is a deliberately expensive, fail-closed hardware gate.  It exercises the
frozen vision-language prefix encoder and continuous action decoder under native
TP=2, then requires one target sample to be bitwise invariant to batch
companions and row placement.  A target-only backward pass also verifies that
the target input gradient is invariant and every companion input gradient is
exactly zero.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import secrets
import time
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from PIL import Image
from torch import Tensor, nn
from transformers import AutoProcessor
from transformers import __version__ as transformers_version

from duo_vla.backbones.diffusion_gemma import (
    DiffusionGemmaActionDecoder,
    DiffusionGemmaPrefix,
    encode_diffusion_gemma_prefix,
)
from duo_vla.backbones.loading import DEFAULT_DIFFUSION_GEMMA_SPEC, load_diffusion_gemma_bf16_tp
from duo_vla.backbones.sample_isolated_experts import (
    install_sample_isolated_grouped_mm_experts,
    verify_sample_isolated_grouped_mm_experts,
)
from duo_vla.prefix_geometry import apply_fixed_prefix_chat_template

REPORT_SCHEMA = "duo-vla-real-fixed-b8-sample-isolation-v1"
EXPECTED_TRANSFORMERS_VERSION = "5.15.0"
EXPECTED_PADDING_SIDE = "left"
EXPECTED_CUBLAS_WORKSPACE_CONFIG = ":4096:8"
PHYSICAL_BATCH_SIZE = 8
ACTION_HORIZON = 8
DEFAULT_PREFIX_WIDTH = 545
EXPECTED_TARGET_VALID_PREFIX_LENGTH = 544
MOVED_TARGET_ROW = 5
IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256


@dataclass(frozen=True, slots=True)
class SampleSpec:
    identifier: str
    instruction: str


TARGET = SampleSpec(
    identifier="target",
    instruction="pick up the black bowl in the top drawer of the wooden cabinet and place it on the plate",
)
COMPANIONS = (
    SampleSpec("companion-0", "turn on the stove"),
    SampleSpec("companion-1", "open the middle drawer of the cabinet"),
    SampleSpec("companion-2", "pick up the milk and place it in the basket"),
    SampleSpec("companion-3", "put the moka pot on the stove"),
    SampleSpec("companion-4", "pick up the book and place it in the back compartment of the caddy"),
    SampleSpec("companion-5", "pick up the black bowl next to the plate and place it on the plate"),
    SampleSpec("companion-6", "turn on the stove and put the moka pot on it"),
)


@dataclass(frozen=True, slots=True)
class CasePlan:
    name: str
    samples: tuple[SampleSpec, ...]
    target_row: int
    require_replicated_rows: bool = False


@dataclass(frozen=True, slots=True)
class RegistrationSnapshot:
    state_dict_keys: tuple[str, ...]
    parameter_identities: tuple[tuple[str, int, int, bool], ...]
    buffer_identities: tuple[tuple[str, int, int], ...]
    schema_sha256: str


@dataclass(frozen=True, slots=True)
class CaseArtifacts:
    target_input_sha256: str
    target_output: Tensor
    target_gradient: Tensor
    prefix_digest_by_rank: tuple[str, ...]
    prefix_layer_digests_by_rank: tuple[tuple[str, ...], ...]
    encoder_layer_zero_component_digests_by_rank: tuple[dict[str, str], ...]
    loss_bytes: bytes


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def finalized_report(value: dict[str, Any]) -> dict[str, Any]:
    require("report_sha256" not in value, "unfinalized report unexpectedly contains report_sha256")
    report = dict(value)
    report["report_sha256"] = hashlib.sha256(canonical_json_bytes(value)).hexdigest()
    return report


def write_canonical_json_exclusive(path: Path, value: dict[str, Any]) -> Path:
    """Publish one canonical report without following or replacing the target."""

    parent = path.parent.resolve(strict=True)
    require(parent.is_dir(), f"report parent is not a directory: {parent}")
    filename = path.name
    require(filename not in {"", ".", ".."}, "report filename is invalid")
    directory_fd = os.open(parent, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
    temporary = f".{filename}.tmp-{secrets.token_hex(12)}"
    raw = canonical_json_bytes(value)
    temporary_created = False
    try:
        file_fd = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC,
            0o600,
            dir_fd=directory_fd,
        )
        temporary_created = True
        try:
            view = memoryview(raw)
            while view:
                written = os.write(file_fd, view)
                require(written > 0, "zero-byte write while publishing qualification report")
                view = view[written:]
            os.fsync(file_fd)
        finally:
            os.close(file_fd)
        os.link(
            temporary,
            filename,
            src_dir_fd=directory_fd,
            dst_dir_fd=directory_fd,
            follow_symlinks=False,
        )
        os.unlink(temporary, dir_fd=directory_fd)
        temporary_created = False
        os.fsync(directory_fd)
    finally:
        if temporary_created:
            with suppress(FileNotFoundError):
                os.unlink(temporary, dir_fd=directory_fd)
        os.close(directory_fd)
    return parent / filename


def case_plans() -> tuple[CasePlan, ...]:
    mixed = (TARGET, *COMPANIONS)
    moved = (*COMPANIONS[:MOVED_TARGET_ROW], TARGET, *COMPANIONS[MOVED_TARGET_ROW:])
    plans = (
        CasePlan("replicated", (TARGET,) * PHYSICAL_BATCH_SIZE, 0, True),
        CasePlan("mixed", mixed, 0),
        CasePlan("target_moved", moved, MOVED_TARGET_ROW),
    )
    for plan in plans:
        require(len(plan.samples) == PHYSICAL_BATCH_SIZE, f"case {plan.name} does not contain exactly B=8 samples")
        require(plan.samples[plan.target_row] == TARGET, f"case {plan.name} target row does not contain the target")
    require(
        sorted(sample.identifier for sample in mixed) == sorted(sample.identifier for sample in moved),
        "mixed and moved cases must contain the same sample multiset",
    )
    return plans


def deterministic_rgb_image(sample: SampleSpec, *, camera_index: int) -> Image.Image:
    require(camera_index in {0, 1}, "each sample must use exactly camera indices zero and one")
    seed = hashlib.sha256(f"{sample.identifier}:camera-{camera_index}".encode()).digest()
    y, x = np.indices((IMAGE_HEIGHT, IMAGE_WIDTH), dtype=np.uint32)
    channels = [
        (int(seed[index]) + (index + 3) * x + (index + 5) * y + (x * y) % (19 + index)) % 256 for index in range(3)
    ]
    return Image.fromarray(np.stack(channels, axis=-1).astype(np.uint8), mode="RGB")


def conversation(sample: SampleSpec) -> list[dict[str, Any]]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": deterministic_rgb_image(sample, camera_index=0)},
                {"type": "image", "image": deterministic_rgb_image(sample, camera_index=1)},
                {"type": "text", "text": sample.instruction},
            ],
        }
    ]


def _processor_inputs(
    processor: Any,
    plan: CasePlan,
    *,
    prefix_width: int,
    device: torch.device,
) -> tuple[dict[str, Any], dict[str, Any]]:
    conversations = [conversation(sample) for sample in plan.samples]
    output = apply_fixed_prefix_chat_template(
        processor,
        conversations,
        fixed_physical_prefix_width=prefix_width,
        padding_side=EXPECTED_PADDING_SIDE,
        expected_batch_size=PHYSICAL_BATCH_SIZE,
        images_per_prefix=2,
    )
    values = dict(output)
    input_ids = values.get("input_ids")
    attention_mask = values.get("attention_mask")
    pixel_values = values.get("pixel_values")
    require(
        isinstance(input_ids, Tensor) and input_ids.shape == (PHYSICAL_BATCH_SIZE, prefix_width),
        f"case {plan.name} input_ids do not have exact shape B=8 x P={prefix_width}",
    )
    require(
        isinstance(attention_mask, Tensor) and attention_mask.shape == input_ids.shape,
        f"case {plan.name} attention mask does not match the exact input shape",
    )
    require(
        isinstance(pixel_values, Tensor) and pixel_values.ndim >= 1 and pixel_values.shape[0] == 16,
        f"case {plan.name} must produce exactly two encoded images for each of eight samples",
    )
    valid_lengths = tuple(int(value) for value in attention_mask.sum(dim=1).tolist())
    require(
        valid_lengths[plan.target_row] == EXPECTED_TARGET_VALID_PREFIX_LENGTH,
        f"target valid prefix length changed: expected {EXPECTED_TARGET_VALID_PREFIX_LENGTH}, "
        f"observed {valid_lengths[plan.target_row]}",
    )
    if plan.require_replicated_rows:
        require(len(set(valid_lengths)) == 1, "replicated case has unequal valid prefix lengths")
        for name in ("input_ids", "attention_mask", "mm_token_type_ids"):
            tensor = values.get(name)
            if isinstance(tensor, Tensor):
                require(
                    all(torch.equal(tensor[0], tensor[row]) for row in range(1, PHYSICAL_BATCH_SIZE)),
                    f"replicated case processor tensor {name} differs across rows",
                )
        for row in range(1, PHYSICAL_BATCH_SIZE):
            start = 2 * row
            require(
                torch.equal(pixel_values[0:2], pixel_values[start : start + 2]),
                "replicated case encoded image tensors differ across rows",
            )
    geometry = {
        "input_ids_shape": list(input_ids.shape),
        "pixel_values_shape": list(pixel_values.shape),
        "target_valid_prefix_length": valid_lengths[plan.target_row],
        "valid_prefix_lengths": list(valid_lengths),
    }
    moved = {name: value.to(device) if isinstance(value, Tensor) else value for name, value in values.items()}
    return moved, geometry


def deterministic_action_embeddings(samples: tuple[SampleSpec, ...], *, hidden_size: int) -> Tensor:
    require(len(samples) == PHYSICAL_BATCH_SIZE, "action embedding batch must have physical B=8")
    positions = torch.arange(ACTION_HORIZON * hidden_size, dtype=torch.float32).reshape(
        ACTION_HORIZON,
        hidden_size,
    )
    rows: list[Tensor] = []
    for sample in samples:
        digest = hashlib.sha256(sample.identifier.encode()).digest()
        phase = int.from_bytes(digest[:4], "big") / float(2**32)
        row = 0.625 * torch.sin(positions / 97.0 + phase) + 0.25 * torch.cos(positions / 193.0 - phase)
        rows.append(row.to(torch.bfloat16))
    return torch.stack(rows)


def _local_tensor(value: Tensor) -> Tensor:
    local = value.to_local() if hasattr(value, "to_local") else value
    require(isinstance(local, Tensor), "tensor-parallel value did not materialize as a local Tensor")
    return local


def tensor_sha256(value: Tensor) -> str:
    local = value.detach().contiguous().cpu()
    digest = hashlib.sha256()
    digest.update(str(local.dtype).encode())
    digest.update(str(tuple(local.shape)).encode())
    digest.update(local.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _gather_object(value: Any) -> tuple[Any, ...]:
    if not (dist.is_available() and dist.is_initialized()):
        return (value,)
    observed: list[Any] = [None] * dist.get_world_size()
    dist.all_gather_object(observed, value)
    return tuple(observed)


def _require_on_all_ranks(condition: bool, message: str) -> None:
    rank = dist.get_rank() if dist.is_initialized() else 0
    failures = _gather_object(None if condition else f"rank={rank}: {message}")
    observed = [failure for failure in failures if failure is not None]
    if observed:
        raise RuntimeError("; ".join(observed))


def assert_replicated_tensor(name: str, value: Tensor) -> str:
    placements = getattr(value, "placements", ())
    if placements:
        replicated = [
            callable(getattr(placement, "is_replicate", None)) and placement.is_replicate() for placement in placements
        ]
        require(all(replicated), f"output {name} has non-replicated DTensor placements: {placements}")
    materialized = value.full_tensor() if hasattr(value, "full_tensor") else value
    require(isinstance(materialized, Tensor), f"replicated output {name} did not materialize as a Tensor")
    require(bool(torch.isfinite(materialized).all()), f"replicated output {name} contains non-finite values")
    digest = tensor_sha256(materialized)
    observed = _gather_object(digest)
    require(all(item == digest for item in observed), f"replicated output {name} differs across TP ranks: {observed}")
    return digest


def _schema_sha256(state_dict: dict[str, Tensor]) -> str:
    entries = [
        {
            "dtype": str(value.dtype),
            "name": name,
            "requires_grad": bool(value.requires_grad),
            "shape": list(value.shape),
        }
        for name, value in state_dict.items()
    ]
    return hashlib.sha256(canonical_json_bytes(entries)).hexdigest()


def registration_snapshot(model: nn.Module) -> RegistrationSnapshot:
    state_dict = model.state_dict(keep_vars=True)
    return RegistrationSnapshot(
        state_dict_keys=tuple(state_dict),
        parameter_identities=tuple(
            (name, id(parameter), int(parameter._version), bool(parameter.requires_grad))
            for name, parameter in model.named_parameters(remove_duplicate=False)
        ),
        buffer_identities=tuple(
            (name, id(buffer), int(buffer._version)) for name, buffer in model.named_buffers(remove_duplicate=False)
        ),
        schema_sha256=_schema_sha256(state_dict),
    )


def assert_registration_unchanged(
    expected: RegistrationSnapshot,
    model: nn.Module,
    *,
    context: str,
) -> RegistrationSnapshot:
    observed = registration_snapshot(model)
    require(observed.state_dict_keys == expected.state_dict_keys, f"{context} changed state_dict keys or ordering")
    require(
        observed.parameter_identities == expected.parameter_identities,
        f"{context} changed parameter identities, versions, ordering, or frozen status",
    )
    require(
        observed.buffer_identities == expected.buffer_identities,
        f"{context} changed buffer identities, versions, or ordering",
    )
    require(observed.schema_sha256 == expected.schema_sha256, f"{context} changed the state_dict schema")
    return observed


def _scalar_int(value: Any) -> int:
    if isinstance(value, Tensor):
        require(value.numel() == 1, "cache length metadata must be scalar")
        return int(_local_tensor(value).item())
    require(isinstance(value, int) and not isinstance(value, bool), "cache length metadata must be an integer")
    return value


def _cache_structure(prefix: DiffusionGemmaPrefix) -> tuple[tuple[Any, ...], ...]:
    layers = getattr(prefix.past_key_values, "layers", None)
    require(isinstance(layers, list) and len(layers) == 30, "prefix cache must contain exactly 30 layers")
    structure: list[tuple[Any, ...]] = []
    for index, layer in enumerate(layers):
        keys = getattr(layer, "keys", None)
        values = getattr(layer, "values", None)
        require(isinstance(keys, Tensor) and isinstance(values, Tensor), f"cache layer {index} has no K/V tensors")
        local_keys = _local_tensor(keys)
        local_values = _local_tensor(values)
        structure.append(
            (
                index,
                id(layer),
                id(keys),
                id(values),
                local_keys.data_ptr(),
                local_values.data_ptr(),
                local_keys._version,
                local_values._version,
                tuple(local_keys.shape),
                tuple(local_values.shape),
            )
        )
    return tuple(structure)


def _target_component_sha256(output: Any, *, target_row: int, tokens_per_sample: int) -> str | None:
    digest = hashlib.sha256()
    selected = 0

    def visit(value: Any, path: str) -> None:
        nonlocal selected
        if isinstance(value, Tensor):
            local = _local_tensor(value).detach()
            if local.ndim == 0:
                return
            if int(local.shape[0]) == PHYSICAL_BATCH_SIZE:
                target = local[target_row]
            elif int(local.shape[0]) == PHYSICAL_BATCH_SIZE * tokens_per_sample:
                target = local.narrow(0, target_row * tokens_per_sample, tokens_per_sample)
            else:
                return
            target = target.contiguous().cpu()
            digest.update(path.encode())
            digest.update(str(target.dtype).encode())
            digest.update(str(tuple(target.shape)).encode())
            digest.update(target.reshape(-1).view(torch.uint8).numpy().tobytes())
            selected += 1
        elif isinstance(value, (tuple, list)):
            for index, item in enumerate(value):
                visit(item, f"{path}.{index}")

    visit(output, "output")
    return digest.hexdigest() if selected else None


def _install_encoder_layer_zero_diagnostics(
    model: nn.Module,
    *,
    target_row: int,
    tokens_per_sample: int,
) -> tuple[dict[str, str], list[Any]]:
    layer = model.model.encoder.language_model.layers[0]
    digests: dict[str, str] = {}
    handles: list[Any] = []
    modules = dict(layer.named_modules())
    component_names = (
        "",
        "input_layernorm",
        "self_attn.q_proj",
        "self_attn.q_norm",
        "self_attn.k_proj",
        "self_attn.k_norm",
        "self_attn.v_proj",
        "self_attn.v_norm",
        "self_attn.o_proj",
        "self_attn",
        "post_attention_layernorm",
        "pre_feedforward_layernorm",
        "mlp.gate_proj",
        "mlp.up_proj",
        "mlp.down_proj",
        "mlp",
        "post_feedforward_layernorm_1",
        "pre_feedforward_layernorm_2",
        "router.norm",
        "router.proj",
        "router",
        "experts",
        "post_feedforward_layernorm_2",
        "post_feedforward_layernorm",
    )
    for raw_name in component_names:
        module = modules.get(raw_name)
        if module is None:
            continue
        name = raw_name or "<layer>"

        def capture(_module: nn.Module, _args: tuple[Any, ...], output: Any, *, component: str = name) -> None:
            component_digest = _target_component_sha256(
                output,
                target_row=target_row,
                tokens_per_sample=tokens_per_sample,
            )
            if component_digest is not None:
                require(component not in digests, f"encoder layer-zero component {component} ran more than once")
                digests[component] = component_digest

        handles.append(module.register_forward_hook(capture))
    return digests, handles


def prefix_target_layer_sha256s(
    prefix: DiffusionGemmaPrefix,
    *,
    target_row: int,
    prefix_width: int,
    require_replicated_rows: bool,
) -> tuple[str, ...]:
    require(
        prefix.attention_mask.shape == (PHYSICAL_BATCH_SIZE, prefix_width),
        "encoded prefix attention mask does not have exact B=8 fixed width",
    )
    require(prefix.attention_mask.dtype == torch.bool, "encoded prefix attention mask must be boolean")
    layers = getattr(prefix.past_key_values, "layers", None)
    require(isinstance(layers, list) and len(layers) == 30, "encoded prefix cache must contain 30 layers")
    layer_digests: list[str] = []
    for index, layer in enumerate(layers):
        require(
            _scalar_int(layer.get_seq_length()) == prefix_width,
            f"cache layer {index} sequence length does not equal fixed prefix width {prefix_width}",
        )
        digest = hashlib.sha256()
        for tensor_name in ("keys", "values"):
            local = _local_tensor(getattr(layer, tensor_name))
            require(local.shape[0] == PHYSICAL_BATCH_SIZE, f"cache layer {index} {tensor_name} is not B=8")
            require(local.dtype == torch.bfloat16, f"cache layer {index} {tensor_name} is not BF16")
            require(bool(torch.isfinite(local).all()), f"cache layer {index} {tensor_name} is non-finite")
            if require_replicated_rows:
                require(
                    all(torch.equal(local[0], local[row]) for row in range(1, PHYSICAL_BATCH_SIZE)),
                    f"replicated prefix cache differs across rows at layer {index} {tensor_name}",
                )
            target = local[target_row].detach().contiguous().cpu()
            digest.update(f"{index}:{tensor_name}".encode())
            digest.update(str(target.dtype).encode())
            digest.update(str(tuple(target.shape)).encode())
            digest.update(target.view(torch.uint8).numpy().tobytes())
        layer_digests.append(digest.hexdigest())
    return tuple(layer_digests)


def validate_runtime(model: nn.Module, *, prefix_width: int) -> dict[str, Any]:
    require(transformers_version == EXPECTED_TRANSFORMERS_VERSION, "Transformers must be pinned to 5.15.0")
    require(dist.is_initialized() and dist.get_world_size() == 2, "qualification requires exactly TP=2 ranks")
    require(int(getattr(model, "_tp_size", 0)) == 2, "loaded model does not report native TP size two")
    require(torch.cuda.is_bf16_supported(), "selected CUDA device does not support BF16")
    require(not any(parameter.requires_grad for parameter in model.parameters()), "base model is not fully frozen")
    floating_dtypes = {str(parameter.dtype) for parameter in model.parameters() if parameter.is_floating_point()}
    require(floating_dtypes == {"torch.bfloat16"}, f"base parameters are not uniformly BF16: {floating_dtypes}")
    text_config = model.config.text_config
    require(text_config.hidden_size == 2816, "pinned hidden size changed")
    require(text_config.num_hidden_layers == 30, "pinned layer count changed")
    require(text_config.num_experts == 128, "pinned expert count changed")
    require(text_config.top_k_experts == 8, "pinned expert top-k changed")
    require(
        isinstance(text_config.sliding_window, int) and prefix_width <= text_config.sliding_window,
        "fixed prefix width exceeds the pinned sliding-attention window",
    )
    encoder_layers = model.model.encoder.language_model.layers
    decoder_layers = model.model.decoder.layers
    implementations = {
        layer.self_attn.config._attn_implementation for layer in (*tuple(encoder_layers), *tuple(decoder_layers))
    }
    require(implementations == {"sdpa"}, f"attention implementation is not uniformly SDPA: {implementations}")
    return {
        "attention_implementation": "sdpa",
        "backbone_dtype": "torch.bfloat16",
        "hidden_size": int(text_config.hidden_size),
        "num_experts": int(text_config.num_experts),
        "num_hidden_layers": int(text_config.num_hidden_layers),
        "sliding_window": int(text_config.sliding_window),
        "top_k_experts": int(text_config.top_k_experts),
    }


def _run_case(
    plan: CasePlan,
    *,
    processor: Any,
    model: nn.Module,
    backend: DiffusionGemmaActionDecoder,
    prefix_width: int,
    device: torch.device,
) -> tuple[CaseArtifacts, dict[str, Any]]:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    case_started = time.perf_counter()
    processor_inputs, geometry = _processor_inputs(
        processor,
        plan,
        prefix_width=prefix_width,
        device=device,
    )
    torch.cuda.synchronize(device)
    preprocessing_seconds = time.perf_counter() - case_started

    component_digests, diagnostic_handles = _install_encoder_layer_zero_diagnostics(
        model,
        target_row=plan.target_row,
        tokens_per_sample=prefix_width,
    )
    prefix_started = time.perf_counter()
    try:
        prefix = encode_diffusion_gemma_prefix(model, processor_inputs)
    finally:
        for handle in diagnostic_handles:
            handle.remove()
    torch.cuda.synchronize(device)
    prefix_seconds = time.perf_counter() - prefix_started
    require("<layer>" in component_digests and "experts" in component_digests, "layer-zero diagnostics are incomplete")
    component_digests_by_rank = tuple(dict(values) for values in _gather_object(component_digests))
    del processor_inputs

    prefix_layer_digests = prefix_target_layer_sha256s(
        prefix,
        target_row=plan.target_row,
        prefix_width=prefix_width,
        require_replicated_rows=plan.require_replicated_rows,
    )
    prefix_digest = hashlib.sha256(canonical_json_bytes(list(prefix_layer_digests))).hexdigest()
    prefix_digest_by_rank = tuple(str(value) for value in _gather_object(prefix_digest))
    prefix_layer_digests_by_rank = tuple(
        tuple(str(digest) for digest in values) for values in _gather_object(prefix_layer_digests)
    )
    cache_before = _cache_structure(prefix)

    embeddings = deterministic_action_embeddings(plan.samples, hidden_size=2816).to(device).requires_grad_(True)
    input_sha256 = assert_replicated_tensor(f"{plan.name}.action_embeddings", embeddings)
    target_input_sha256 = assert_replicated_tensor(
        f"{plan.name}.target_action_embeddings",
        embeddings[plan.target_row],
    )
    valid = torch.ones((PHYSICAL_BATCH_SIZE, ACTION_HORIZON), device=device, dtype=torch.bool)
    forward_started = time.perf_counter()
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        output = backend.decode_actions(
            embeddings,
            prefix_cache=prefix.past_key_values,
            prefix_attention_mask=prefix.attention_mask,
            action_valid_mask=valid,
        )
    torch.cuda.synchronize(device)
    forward_seconds = time.perf_counter() - forward_started
    require(
        output.shape == (PHYSICAL_BATCH_SIZE, ACTION_HORIZON, 2816),
        f"case {plan.name} decoder output shape changed: {tuple(output.shape)}",
    )
    require(output.dtype == torch.bfloat16, f"case {plan.name} decoder output is not BF16")
    output_sha256 = assert_replicated_tensor(f"{plan.name}.decoder_output", output)
    if plan.require_replicated_rows:
        require(
            all(torch.equal(output[0], output[row]) for row in range(1, PHYSICAL_BATCH_SIZE)),
            "replicated case decoder output differs across rows",
        )

    target_output = output[plan.target_row]
    target_output_sha256 = assert_replicated_tensor(f"{plan.name}.target_output", target_output)
    loss = target_output.float().square().mean()
    require(bool(torch.isfinite(loss)), f"case {plan.name} target-only loss is non-finite")
    loss_sha256 = assert_replicated_tensor(f"{plan.name}.target_loss", loss)
    backward_started = time.perf_counter()
    loss.backward()
    torch.cuda.synchronize(device)
    backward_seconds = time.perf_counter() - backward_started

    gradient = embeddings.grad
    require(isinstance(gradient, Tensor), f"case {plan.name} action embeddings have no gradient")
    require(bool(torch.isfinite(gradient).all()), f"case {plan.name} action embedding gradient is non-finite")
    require(
        int(torch.count_nonzero(gradient[plan.target_row]).item()) > 0,
        f"case {plan.name} target action embedding gradient is identically zero",
    )
    companion_rows = [row for row in range(PHYSICAL_BATCH_SIZE) if row != plan.target_row]
    companion_nonzero = sum(int(torch.count_nonzero(gradient[row]).item()) for row in companion_rows)
    require(companion_nonzero == 0, f"case {plan.name} companion gradients are not exactly zero")
    gradient_sha256 = assert_replicated_tensor(f"{plan.name}.embedding_gradient", gradient)
    target_gradient_sha256 = assert_replicated_tensor(
        f"{plan.name}.target_embedding_gradient",
        gradient[plan.target_row],
    )
    require(_cache_structure(prefix) == cache_before, f"case {plan.name} mutated prefix cache identity or version")
    parameters_with_grad = sum(parameter.grad is not None for parameter in model.parameters())
    require(parameters_with_grad == 0, f"case {plan.name} materialized gradients on frozen base parameters")

    torch.cuda.synchronize(device)
    local_runtime = {
        "backward_seconds": backward_seconds,
        "forward_seconds": forward_seconds,
        "peak_memory_gib": torch.cuda.max_memory_allocated(device) / 2**30,
        "prefix_seconds": prefix_seconds,
        "preprocessing_seconds": preprocessing_seconds,
        "rank": dist.get_rank(),
    }
    report = {
        "companion_gradient_nonzero_count": companion_nonzero,
        "encoder_layer_zero_component_sha256_by_rank": list(component_digests_by_rank),
        "full_action_embedding_sha256": input_sha256,
        "full_embedding_gradient_sha256": gradient_sha256,
        "full_output_sha256": output_sha256,
        "loss_sha256": loss_sha256,
        "prefix_target_sha256_by_rank": list(prefix_digest_by_rank),
        "prefix_target_layer_sha256_by_rank": [list(values) for values in prefix_layer_digests_by_rank],
        "processor_geometry": geometry,
        "runtime_by_rank": list(_gather_object(local_runtime)),
        "sample_identifiers": [sample.identifier for sample in plan.samples],
        "target_action_embedding_sha256": target_input_sha256,
        "target_embedding_gradient_sha256": target_gradient_sha256,
        "target_output_sha256": target_output_sha256,
        "target_row": plan.target_row,
    }
    artifacts = CaseArtifacts(
        target_input_sha256=target_input_sha256,
        target_output=target_output.detach().cpu().clone(),
        target_gradient=gradient[plan.target_row].detach().cpu().clone(),
        prefix_digest_by_rank=prefix_digest_by_rank,
        prefix_layer_digests_by_rank=prefix_layer_digests_by_rank,
        encoder_layer_zero_component_digests_by_rank=component_digests_by_rank,
        loss_bytes=loss.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes(),
    )
    del embeddings, gradient, loss, output, prefix, target_output, valid
    torch.cuda.empty_cache()
    return artifacts, report


def assert_case_matches(reference: CaseArtifacts, candidate: CaseArtifacts, *, name: str) -> dict[str, bool]:
    checks = {
        "loss_bitwise_equal": candidate.loss_bytes == reference.loss_bytes,
        "prefix_target_bitwise_equal": candidate.prefix_digest_by_rank == reference.prefix_digest_by_rank,
        "target_input_bitwise_equal": candidate.target_input_sha256 == reference.target_input_sha256,
        "target_gradient_bitwise_equal": torch.equal(candidate.target_gradient, reference.target_gradient),
        "target_output_bitwise_equal": torch.equal(candidate.target_output, reference.target_output),
    }
    prefix_mismatches = [
        [
            layer_index
            for layer_index, (expected, observed) in enumerate(zip(reference_rank, candidate_rank, strict=True))
            if expected != observed
        ]
        for reference_rank, candidate_rank in zip(
            reference.prefix_layer_digests_by_rank,
            candidate.prefix_layer_digests_by_rank,
            strict=True,
        )
    ]
    component_mismatches = [
        [
            component
            for component in sorted(set(reference_rank) | set(candidate_rank))
            if reference_rank.get(component) != candidate_rank.get(component)
        ]
        for reference_rank, candidate_rank in zip(
            reference.encoder_layer_zero_component_digests_by_rank,
            candidate.encoder_layer_zero_component_digests_by_rank,
            strict=True,
        )
    ]
    _require_on_all_ranks(
        all(checks.values()),
        f"case {name} target parity failed: {checks}; prefix_mismatched_layers_by_rank={prefix_mismatches}; "
        f"encoder_layer_zero_component_mismatches_by_rank={component_mismatches}",
    )
    return checks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-json",
        required=True,
        type=Path,
        help="new path for the canonical qualification report",
    )
    parser.add_argument(
        "--prefix-width",
        type=int,
        default=DEFAULT_PREFIX_WIDTH,
        help="fixed physical prefix width (default: LIBERO maximum 544 plus one mask sentinel)",
    )
    args = parser.parse_args()
    if isinstance(args.prefix_width, bool) or args.prefix_width <= 0:
        parser.error("--prefix-width must be a positive integer")
    if args.prefix_width <= EXPECTED_TARGET_VALID_PREFIX_LENGTH:
        parser.error("--prefix-width must leave a padding sentinel beyond the target valid length")
    return args


def main() -> None:
    args = parse_args()
    require(
        os.environ.get("CUBLAS_WORKSPACE_CONFIG") == EXPECTED_CUBLAS_WORKSPACE_CONFIG,
        f"set CUBLAS_WORKSPACE_CONFIG={EXPECTED_CUBLAS_WORKSPACE_CONFIG} before torchrun",
    )
    require(os.environ.get("PYTHONHASHSEED") == "0", "set PYTHONHASHSEED=0 before torchrun")
    require("LOCAL_RANK" in os.environ, "launch the qualification gate with torchrun")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", device_id=device)
    started = time.perf_counter()
    try:
        require(dist.get_world_size() == 2, "qualification requires torchrun --nproc-per-node=2")
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")

        processor = AutoProcessor.from_pretrained(
            DEFAULT_DIFFUSION_GEMMA_SPEC.model_id,
            revision=DEFAULT_DIFFUSION_GEMMA_SPEC.revision,
            local_files_only=True,
        )
        require(
            getattr(getattr(processor, "tokenizer", None), "padding_side", None) == EXPECTED_PADDING_SIDE,
            f"pinned processor padding side must be {EXPECTED_PADDING_SIDE!r}",
        )
        model = load_diffusion_gemma_bf16_tp(local_files_only=True, tp_size=dist.get_world_size())
        model.eval()
        model.model.encoder.eval()
        runtime_contract = validate_runtime(model, prefix_width=args.prefix_width)

        before_install = registration_snapshot(model)
        installation = install_sample_isolated_grouped_mm_experts(
            model,
            physical_batch_size=PHYSICAL_BATCH_SIZE,
        )
        verified_installation = verify_sample_isolated_grouped_mm_experts(
            model,
            physical_batch_size=PHYSICAL_BATCH_SIZE,
        )
        require(installation == verified_installation, "installed and verified expert contracts differ")
        require(
            installation.encoder_layer_count == installation.decoder_layer_count == 30
            and installation.target_count == 60,
            "sample-isolated expert installation does not cover all 60 encoder/decoder layers",
        )
        assert_registration_unchanged(before_install, model, context="sample-isolated expert installation")

        backend = DiffusionGemmaActionDecoder.from_block_diffusion_model(model)
        artifacts_by_name: dict[str, CaseArtifacts] = {}
        reports_by_name: dict[str, dict[str, Any]] = {}
        for plan in case_plans():
            verify_sample_isolated_grouped_mm_experts(
                model,
                physical_batch_size=PHYSICAL_BATCH_SIZE,
            )
            artifacts, report = _run_case(
                plan,
                processor=processor,
                model=model,
                backend=backend,
                prefix_width=args.prefix_width,
                device=device,
            )
            artifacts_by_name[plan.name] = artifacts
            reports_by_name[plan.name] = report

        reference = artifacts_by_name["replicated"]
        comparisons = {
            "mixed_vs_replicated": assert_case_matches(reference, artifacts_by_name["mixed"], name="mixed"),
            "moved_vs_replicated": assert_case_matches(
                reference,
                artifacts_by_name["target_moved"],
                name="target_moved",
            ),
        }
        final_installation = verify_sample_isolated_grouped_mm_experts(
            model,
            physical_batch_size=PHYSICAL_BATCH_SIZE,
        )
        require(final_installation == installation, "expert installation contract changed during qualification")
        assert_registration_unchanged(before_install, model, context="qualification forwards/backwards")
        torch.cuda.synchronize(device)

        rank = dist.get_rank()
        maximum_case_peak_memory = max(
            float(case_report["runtime_by_rank"][rank]["peak_memory_gib"]) for case_report in reports_by_name.values()
        )
        local_total_runtime = {
            "elapsed_seconds": time.perf_counter() - started,
            "maximum_case_peak_memory_gib": maximum_case_peak_memory,
            "rank": rank,
        }
        report = finalized_report(
            {
                "cases": reports_by_name,
                "comparisons": comparisons,
                "determinism": {
                    "cublas_workspace_config": EXPECTED_CUBLAS_WORKSPACE_CONFIG,
                    "deterministic_algorithms": True,
                    "float32_matmul_precision": "highest",
                    "python_hash_seed": 0,
                    "tf32_allowed": False,
                },
                "expert_installation": {
                    "decoder_layer_count": installation.decoder_layer_count,
                    "encoder_layer_count": installation.encoder_layer_count,
                    "experts_implementation": installation.experts_implementation,
                    "physical_batch_size": installation.physical_batch_size,
                    "target_count": installation.target_count,
                    "target_names_sha256": hashlib.sha256(
                        canonical_json_bytes(list(installation.target_names))
                    ).hexdigest(),
                },
                "image_contract": {
                    "dtype": "uint8",
                    "height": IMAGE_HEIGHT,
                    "ordered_cameras_per_sample": 2,
                    "width": IMAGE_WIDTH,
                },
                "model": {
                    "id": DEFAULT_DIFFUSION_GEMMA_SPEC.model_id,
                    "revision": DEFAULT_DIFFUSION_GEMMA_SPEC.revision,
                    **runtime_contract,
                },
                "physical_batch_size": PHYSICAL_BATCH_SIZE,
                "prefix_width": args.prefix_width,
                "registration": {
                    "buffer_entries": len(before_install.buffer_identities),
                    "parameter_entries_including_aliases": len(before_install.parameter_identities),
                    "preserved_across_install_and_execution": True,
                    "state_dict_entries": len(before_install.state_dict_keys),
                    "state_dict_schema_sha256": before_install.schema_sha256,
                },
                "runtime_by_rank": list(_gather_object(local_total_runtime)),
                "schema": REPORT_SCHEMA,
                "status": "ok",
                "torch_version": torch.__version__,
                "transformers_version": transformers_version,
                "world_size": dist.get_world_size(),
            }
        )

        write_error: str | None = None
        published_path: Path | None = None
        if dist.get_rank() == 0:
            try:
                published_path = write_canonical_json_exclusive(args.output_json, report)
            except Exception as exc:
                write_error = f"cannot publish qualification report: {type(exc).__name__}: {exc}"
        messages = [write_error]
        dist.broadcast_object_list(messages, src=0)
        require(messages[0] is None, str(messages[0]))
        dist.barrier()
        if dist.get_rank() == 0:
            assert published_path is not None
            print(canonical_json_bytes(report).decode("utf-8"), end="")
            print(f"report_path={published_path}", file=os.sys.stderr)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
