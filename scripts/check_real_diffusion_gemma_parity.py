#!/usr/bin/env python3
"""Fail-closed real DiffusionGemma prefix-cache and continuous-adapter parity gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from collections.abc import Callable
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
from duo_vla.checkpointing import load_lora_checkpoint
from duo_vla.optimization import assert_replicated_tensor

EXPECTED_PREFIX_LENGTH = 532
ACTION_HORIZON = 8
EXPECTED_TRANSFORMERS_VERSION = "5.15.0"
BF16_ATOL = 8e-3
BF16_RTOL = 8e-3
MIN_COSINE_SIMILARITY = 0.99999


@dataclass(frozen=True, slots=True)
class TensorSnapshot:
    object_id: int
    data_ptr: int
    version: int
    shape: tuple[int, ...]
    stride: tuple[int, ...]
    dtype: str
    sha256: str


@dataclass(frozen=True, slots=True)
class CacheLayerSnapshot:
    index: int
    layer_type: str
    is_sliding: bool
    sequence_length: int
    cumulative_length: int | None
    keys: TensorSnapshot
    values: TensorSnapshot


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def local_tensor(value: Tensor) -> Tensor:
    local = value.to_local() if hasattr(value, "to_local") else value
    require(isinstance(local, Tensor), "tensor-parallel value did not materialize as a local Tensor")
    return local


def tensor_sha256(value: Tensor) -> str:
    local = local_tensor(value).detach().contiguous()
    raw = local.view(torch.uint8).cpu().numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def scalar_int(value: Any) -> int:
    if isinstance(value, Tensor):
        require(value.numel() == 1, "cache length metadata must be scalar")
        return int(local_tensor(value).item())
    require(isinstance(value, int) and not isinstance(value, bool), "cache length metadata must be an integer")
    return value


def snapshot_tensor(value: Tensor) -> TensorSnapshot:
    local = local_tensor(value)
    require(bool(torch.isfinite(local).all()), "cache tensor contains non-finite values")
    return TensorSnapshot(
        object_id=id(value),
        data_ptr=local.data_ptr(),
        version=local._version,
        shape=tuple(local.shape),
        stride=tuple(local.stride()),
        dtype=str(local.dtype),
        sha256=tensor_sha256(local),
    )


def snapshot_cache(cache: Any) -> tuple[CacheLayerSnapshot, ...]:
    layers = getattr(cache, "layers", None)
    require(isinstance(layers, list) and len(layers) == 30, "prefix cache must contain exactly 30 layers")
    snapshots: list[CacheLayerSnapshot] = []
    for index, layer in enumerate(layers):
        require(bool(getattr(layer, "is_initialized", False)), f"prefix cache layer {index} is not initialized")
        keys = getattr(layer, "keys", None)
        values = getattr(layer, "values", None)
        require(
            isinstance(keys, Tensor) and isinstance(values, Tensor),
            f"prefix cache layer {index} has no K/V tensors",
        )
        cumulative = getattr(layer, "cumulative_length", None)
        snapshots.append(
            CacheLayerSnapshot(
                index=index,
                layer_type=type(layer).__name__,
                is_sliding=bool(getattr(layer, "is_sliding", False)),
                sequence_length=scalar_int(layer.get_seq_length()),
                cumulative_length=None if cumulative is None else scalar_int(cumulative),
                keys=snapshot_tensor(keys),
                values=snapshot_tensor(values),
            )
        )
    return tuple(snapshots)


def assert_cache_unchanged(before: tuple[CacheLayerSnapshot, ...], cache: Any, *, name: str) -> None:
    after = snapshot_cache(cache)
    if after != before:
        differing = [
            index for index, (expected, actual) in enumerate(zip(before, after, strict=True)) if expected != actual
        ]
        raise RuntimeError(f"{name} mutated prefix cache layers {differing}")


def tensor_metrics(actual: Tensor, expected: Tensor) -> dict[str, float | bool]:
    actual_local = local_tensor(actual).detach()
    expected_local = local_tensor(expected).detach()
    require(actual_local.shape == expected_local.shape, "parity tensors have different shapes")
    require(bool(torch.isfinite(actual_local).all()), "actual parity tensor contains non-finite values")
    require(bool(torch.isfinite(expected_local).all()), "expected parity tensor contains non-finite values")
    actual_float = actual_local.float()
    expected_float = expected_local.float()
    difference = (actual_float - expected_float).abs()
    denominator = expected_float.abs().clamp_min(BF16_ATOL)
    actual_flat = actual_float.reshape(-1)
    expected_flat = expected_float.reshape(-1)
    norm_product = torch.linalg.vector_norm(actual_flat) * torch.linalg.vector_norm(expected_flat)
    cosine = (
        float(torch.dot(actual_flat, expected_flat).div(norm_product).item())
        if float(norm_product.item()) > 0.0
        else float(torch.equal(actual_local, expected_local))
    )
    return {
        "bitwise_equal": bool(torch.equal(actual_local, expected_local)),
        "cosine_similarity": cosine,
        "max_absolute_error": float(difference.max().item()) if difference.numel() else 0.0,
        "max_relative_error": float((difference / denominator).max().item()) if difference.numel() else 0.0,
        "mismatch_fraction": (
            float((actual_local != expected_local).float().mean().item()) if difference.numel() else 0.0
        ),
    }


def assert_tensor_close(
    actual: Tensor,
    expected: Tensor,
    *,
    name: str,
    require_cosine: bool,
) -> dict[str, float | bool]:
    metrics = tensor_metrics(actual, expected)
    try:
        torch.testing.assert_close(
            local_tensor(actual).float(),
            local_tensor(expected).float(),
            atol=BF16_ATOL,
            rtol=BF16_RTOL,
        )
    except AssertionError as exc:
        raise RuntimeError(f"{name} exceeded BF16 parity tolerance: {metrics}") from exc
    if require_cosine and metrics["cosine_similarity"] < MIN_COSINE_SIMILARITY:
        raise RuntimeError(f"{name} cosine similarity is below {MIN_COSINE_SIMILARITY}: {metrics}")
    return metrics


def assert_fresh_cache_close(reference: Any, fresh: Any, *, name: str) -> dict[str, Any]:
    reference_layers = getattr(reference, "layers", None)
    fresh_layers = getattr(fresh, "layers", None)
    require(
        isinstance(reference_layers, list)
        and isinstance(fresh_layers, list)
        and len(reference_layers) == len(fresh_layers) == 30,
        "fresh/reference caches must both contain 30 layers",
    )
    maximum_absolute_error = 0.0
    maximum_relative_error = 0.0
    all_bitwise_equal = True
    for index, (expected_layer, actual_layer) in enumerate(zip(reference_layers, fresh_layers, strict=True)):
        require(type(actual_layer) is type(expected_layer), f"{name} cache layer {index} type changed")
        require(
            bool(getattr(actual_layer, "is_sliding", False)) == bool(getattr(expected_layer, "is_sliding", False)),
            f"{name} cache layer {index} sliding metadata changed",
        )
        require(
            scalar_int(actual_layer.get_seq_length()) == scalar_int(expected_layer.get_seq_length()),
            f"{name} cache layer {index} sequence length changed",
        )
        for tensor_name in ("keys", "values"):
            expected = getattr(expected_layer, tensor_name)
            actual = getattr(actual_layer, tensor_name)
            require(
                local_tensor(actual).shape == local_tensor(expected).shape,
                f"{name} cache layer {index} {tensor_name} shape changed",
            )
            require(
                local_tensor(actual).dtype == local_tensor(expected).dtype == torch.bfloat16,
                f"{name} cache layer {index} {tensor_name} is not BF16",
            )
            metrics = assert_tensor_close(
                actual,
                expected,
                name=f"{name}.layer_{index}.{tensor_name}",
                require_cosine=False,
            )
            maximum_absolute_error = max(maximum_absolute_error, float(metrics["max_absolute_error"]))
            maximum_relative_error = max(maximum_relative_error, float(metrics["max_relative_error"]))
            all_bitwise_equal = all_bitwise_equal and bool(metrics["bitwise_equal"])
    return {
        "all_bitwise_equal": all_bitwise_equal,
        "maximum_absolute_error": maximum_absolute_error,
        "maximum_relative_error": maximum_relative_error,
    }


def processor_inputs(processor: Any, *, device: torch.device) -> dict[str, Tensor]:
    instruction = "move the red object to the left"
    agentview = Image.fromarray(np.full((64, 64, 3), (180, 30, 30), dtype=np.uint8))
    wrist = Image.fromarray(np.full((64, 64, 3), (30, 180, 30), dtype=np.uint8))
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": agentview},
                {"type": "image", "image": wrist},
                {"type": "text", "text": instruction},
            ],
        }
    ]
    values = processor.apply_chat_template(
        conversation,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    return dict(values.to(device))


def validate_prefix(prefix: DiffusionGemmaPrefix, *, name: str) -> None:
    require(
        prefix.attention_mask.shape == (1, EXPECTED_PREFIX_LENGTH),
        f"{name} prefix mask must have shape (1, {EXPECTED_PREFIX_LENGTH}), got {tuple(prefix.attention_mask.shape)}",
    )
    require(prefix.attention_mask.dtype == torch.bool, f"{name} prefix mask must be boolean")
    require(bool(prefix.attention_mask.all()), f"{name} production prefix unexpectedly contains padding")
    snapshots = snapshot_cache(prefix.past_key_values)
    for layer in snapshots:
        require(
            layer.sequence_length == EXPECTED_PREFIX_LENGTH,
            f"{name} cache layer {layer.index} has sequence length {layer.sequence_length}",
        )


def deterministic_canvases(*, device: torch.device, hidden_size: int) -> tuple[Tensor, Tensor]:
    positions = torch.arange(ACTION_HORIZON * hidden_size, dtype=torch.float32)
    first = torch.sin(positions / 97.0).reshape(1, ACTION_HORIZON, hidden_size)
    second = (0.5 * torch.cos(positions / 89.0)).reshape(1, ACTION_HORIZON, hidden_size)
    return first.to(device=device, dtype=torch.bfloat16), second.to(device=device, dtype=torch.bfloat16)


def decode(
    backend: DiffusionGemmaActionDecoder,
    embeddings: Tensor,
    prefix: DiffusionGemmaPrefix,
    valid: Tensor,
) -> Tensor:
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        return backend.decode_actions(
            embeddings,
            prefix_cache=prefix.past_key_values,
            prefix_attention_mask=prefix.attention_mask,
            action_valid_mask=valid,
        )


def run_prefix_reuse_parity(
    model: nn.Module,
    backend: DiffusionGemmaActionDecoder,
    inputs: dict[str, Tensor],
    *,
    device: torch.device,
) -> tuple[DiffusionGemmaPrefix, dict[str, Any]]:
    reused = encode_diffusion_gemma_prefix(model, inputs)
    validate_prefix(reused, name="reused")
    reused_initial = snapshot_cache(reused.past_key_values)
    hidden_size = int(backend.decoder.text_config.hidden_size)
    valid = torch.ones((1, ACTION_HORIZON), device=device, dtype=torch.bool)
    steps: list[dict[str, Any]] = []
    for index, canvas in enumerate(deterministic_canvases(device=device, hidden_size=hidden_size)):
        before_reuse = snapshot_cache(reused.past_key_values)
        reused_output = decode(backend, canvas, reused, valid)
        assert_cache_unchanged(before_reuse, reused.past_key_values, name=f"reused.step_{index}")

        fresh = encode_diffusion_gemma_prefix(model, inputs)
        validate_prefix(fresh, name=f"fresh.step_{index}")
        require(
            torch.equal(fresh.attention_mask, reused.attention_mask),
            f"fresh step {index} prefix attention mask differs from the reused prefix",
        )
        cache_metrics = assert_fresh_cache_close(
            reused.past_key_values,
            fresh.past_key_values,
            name=f"fresh.step_{index}",
        )
        before_fresh = snapshot_cache(fresh.past_key_values)
        fresh_output = decode(backend, canvas, fresh, valid)
        assert_cache_unchanged(before_fresh, fresh.past_key_values, name=f"fresh.step_{index}")
        output_metrics = assert_tensor_close(
            fresh_output,
            reused_output,
            name=f"fresh_vs_reused.step_{index}",
            require_cosine=True,
        )
        steps.append(
            {
                "cache": cache_metrics,
                "fresh_output_sha256": assert_replicated_tensor(f"prefix_output.{index}", fresh_output),
                "output": output_metrics,
                "reused_output_sha256": assert_replicated_tensor(f"prefix_output.{index}", reused_output),
            }
        )
        del fresh, fresh_output, reused_output
    assert_cache_unchanged(reused_initial, reused.past_key_values, name="reused.final")
    return reused, {"steps": steps}


def extract_hook_tensor(output: Any, *, name: str) -> Tensor:
    value = output[0] if isinstance(output, tuple) else output
    require(isinstance(value, Tensor), f"decoder hook {name} did not return a Tensor")
    return local_tensor(value).detach().clone()


def capture_decoder_stack(
    decoder: nn.Module,
    call: Callable[[], Tensor],
) -> tuple[Tensor, dict[str, Tensor]]:
    captures: dict[str, Tensor] = {}
    handles: list[Any] = []

    def make_hook(name: str) -> Callable[[nn.Module, tuple[Any, ...], Any], None]:
        def hook(_: nn.Module, __: tuple[Any, ...], output: Any) -> None:
            require(name not in captures, f"decoder hook {name} ran more than once")
            captures[name] = extract_hook_tensor(output, name=name)

        return hook

    modules: list[tuple[str, nn.Module]] = [("self_conditioning", decoder.self_conditioning)]
    modules.extend((f"layer_{index:02d}", layer) for index, layer in enumerate(decoder.layers))
    modules.append(("final_norm", decoder.norm))
    try:
        handles = [module.register_forward_hook(make_hook(name)) for name, module in modules]
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output = call()
    finally:
        for handle in handles:
            handle.remove()
    expected_names = {name for name, _ in modules}
    require(set(captures) == expected_names, "decoder stack capture did not cover every expected module")
    return output, captures


def run_native_adapter_parity(
    backend: DiffusionGemmaActionDecoder,
    prefix: DiffusionGemmaPrefix,
    *,
    device: torch.device,
) -> dict[str, Any]:
    decoder = backend.decoder
    decoder_ids = torch.arange(3, 3 + ACTION_HORIZON, device=device, dtype=torch.long).unsqueeze(0)
    valid = torch.ones((1, ACTION_HORIZON), device=device, dtype=torch.bool)
    combined_mask = torch.cat((prefix.attention_mask.to(device), valid), dim=1)
    with torch.inference_mode():
        token_embeddings = decoder.embed_tokens(decoder_ids)
    require(token_embeddings.dtype == torch.bfloat16, "native decoder token embeddings must be BF16")
    before = snapshot_cache(prefix.past_key_values)

    native_output, native_captures = capture_decoder_stack(
        decoder,
        lambda: (
            decoder(
                decoder_input_ids=decoder_ids,
                past_key_values=prefix.past_key_values,
                decoder_attention_mask=combined_mask,
            ).last_hidden_state
        ),
    )
    assert_cache_unchanged(before, prefix.past_key_values, name="native_token_path")
    adapted_output, adapted_captures = capture_decoder_stack(
        decoder,
        lambda: backend.decode_actions(
            token_embeddings,
            prefix_cache=prefix.past_key_values,
            prefix_attention_mask=prefix.attention_mask,
            action_valid_mask=valid,
        ),
    )
    assert_cache_unchanged(before, prefix.past_key_values, name="continuous_adapter_path")

    require(native_captures.keys() == adapted_captures.keys(), "native/adapter stack capture names differ")
    layer_metrics = {
        name: assert_tensor_close(
            adapted_captures[name],
            native_captures[name],
            name=f"native_vs_adapter.{name}",
            require_cosine=True,
        )
        for name in native_captures
    }
    final_metrics = assert_tensor_close(
        adapted_output,
        native_output,
        name="native_vs_adapter.final_output",
        require_cosine=True,
    )
    return {
        "adapted_output_sha256": assert_replicated_tensor("native_adapter.output", adapted_output),
        "final_output": final_metrics,
        "layers": layer_metrics,
        "native_output_sha256": assert_replicated_tensor("native_adapter.output", native_output),
    }


def cache_bytes_per_rank(cache: Any) -> int:
    total = 0
    for layer in cache.layers:
        for name in ("keys", "values"):
            value = local_tensor(getattr(layer, name))
            total += value.numel() * value.element_size()
    return total


def validate_runtime(model: nn.Module) -> None:
    require(
        transformers_version == EXPECTED_TRANSFORMERS_VERSION,
        f"real parity gate requires Transformers {EXPECTED_TRANSFORMERS_VERSION}, observed {transformers_version}",
    )
    require(dist.get_world_size() == 2, "real parity gate requires exactly two tensor-parallel ranks")
    require(int(getattr(model, "_tp_size", 0)) == 2, "loaded model does not report tensor parallel size 2")
    decoder = model.model.decoder
    require(len(decoder.layers) == 30, "pinned DiffusionGemma decoder must contain 30 layers")
    require(decoder.text_config.hidden_size == 2816, "pinned DiffusionGemma hidden size must be 2816")
    require(decoder.embed_tokens.weight.dtype == torch.bfloat16, "pinned decoder token embeddings are not BF16")
    implementations = {layer.self_attn.config._attn_implementation for layer in decoder.layers}
    require(implementations == {"sdpa"}, f"decoder attention implementation is not uniformly SDPA: {implementations}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-files-only", action="store_true", help="forbid Hugging Face network access")
    parser.add_argument(
        "--load-checkpoint",
        type=Path,
        help="optionally attach a verified trained LoRA checkpoint before checking parity",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.load_checkpoint is not None:
        require(args.load_checkpoint.is_dir(), f"checkpoint directory does not exist: {args.load_checkpoint}")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", device_id=device)
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    try:
        require(torch.cuda.is_bf16_supported(), "selected CUDA device does not support BF16")
        processor = AutoProcessor.from_pretrained(
            DEFAULT_DIFFUSION_GEMMA_SPEC.model_id,
            revision=DEFAULT_DIFFUSION_GEMMA_SPEC.revision,
            local_files_only=args.local_files_only,
        )
        model = load_diffusion_gemma_bf16_tp(
            local_files_only=args.local_files_only,
            tp_size=dist.get_world_size(),
        )
        adapted: nn.Module | None = None
        loaded_manifest: dict[str, Any] | None = None
        if args.load_checkpoint is not None:
            adapted, loaded_manifest = load_lora_checkpoint(args.load_checkpoint, model, is_trainable=False)
            adapted.eval()
        model.eval()
        model.model.encoder.eval()
        validate_runtime(model)
        backend = DiffusionGemmaActionDecoder.from_block_diffusion_model(model)
        inputs = processor_inputs(processor, device=device)

        reused_prefix, cache_result = run_prefix_reuse_parity(
            model,
            backend,
            inputs,
            device=device,
        )
        adapter_result = run_native_adapter_parity(backend, reused_prefix, device=device)
        torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - started
        local_runtime = {
            "elapsed_seconds": elapsed,
            "peak_memory_gib": torch.cuda.max_memory_allocated(device) / 2**30,
            "prefix_cache_bytes": cache_bytes_per_rank(reused_prefix.past_key_values),
            "rank": dist.get_rank(),
        }
        runtimes: list[dict[str, Any] | None] = [None] * dist.get_world_size()
        dist.all_gather_object(runtimes, local_runtime)
        dist.barrier()
        if dist.get_rank() == 0:
            checkpoint = None
            if args.load_checkpoint is not None:
                assert loaded_manifest is not None
                checkpoint = {
                    "manifest_sha256": sha256_file(args.load_checkpoint / "manifest.json"),
                    "path": str(args.load_checkpoint.resolve()),
                    "policy_contract_sha256": loaded_manifest.get("policy_contract_sha256"),
                }
            print(
                json.dumps(
                    {
                        "adapter_stack_parity": adapter_result,
                        "attention_implementation": "sdpa",
                        "backbone_dtype": "torch.bfloat16",
                        "checkpoint": checkpoint,
                        "model_id": DEFAULT_DIFFUSION_GEMMA_SPEC.model_id,
                        "model_revision": DEFAULT_DIFFUSION_GEMMA_SPEC.revision,
                        "prefix_cache_parity": cache_result,
                        "prefix_length": EXPECTED_PREFIX_LENGTH,
                        "runtime_by_rank": runtimes,
                        "status": "ok",
                        "tolerance": {
                            "atol": BF16_ATOL,
                            "minimum_cosine_similarity": MIN_COSINE_SIMILARITY,
                            "rtol": BF16_RTOL,
                        },
                        "transformers_version": transformers_version,
                        "world_size": dist.get_world_size(),
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
