"""Trainable-only Duo-VLA checkpoints with distributed-safe LoRA gathering."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import warnings
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch import Tensor, nn
from torch.distributed.tensor import DTensor

SCHEMA_VERSION = "duo-vla-checkpoint-v1"
_MODULE_NAME = re.compile(r"^[a-z][a-z0-9_]*$")


def _rank() -> int:
    return dist.get_rank() if dist.is_available() and dist.is_initialized() else 0


def _barrier() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def _fsync_file(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _broadcast_error(error: str | None) -> str | None:
    if not dist.is_available() or not dist.is_initialized():
        return error
    value = [error]
    dist.broadcast_object_list(value, src=0)
    return value[0]


def _validate_modules(modules: Mapping[str, nn.Module]) -> None:
    if not modules:
        raise ValueError("at least one interface module is required")
    invalid = [name for name in modules if not _MODULE_NAME.fullmatch(name)]
    if invalid:
        raise ValueError(f"invalid interface module names: {invalid}")


def interface_state_dict(modules: Mapping[str, nn.Module]) -> dict[str, Tensor]:
    """Return a flat CPU state dict with stable module-name prefixes."""

    _validate_modules(modules)
    flattened: dict[str, Tensor] = {}
    for module_name in sorted(modules):
        for parameter_name, value in modules[module_name].state_dict().items():
            if not isinstance(value, Tensor):
                raise TypeError(f"non-tensor state at {module_name}.{parameter_name}")
            if isinstance(value, DTensor):
                raise TypeError("action interface state must be replicated, not a DTensor")
            flattened[f"{module_name}.{parameter_name}"] = value.detach().cpu().contiguous()
    if not flattened:
        raise ValueError("interface modules have no state")
    return flattened


def load_interface_state_dict(
    path: str | Path,
    modules: Mapping[str, nn.Module],
) -> None:
    """Authenticate and load every declared interface module from safetensors."""

    from safetensors.torch import load_file

    _validate_modules(modules)
    state = load_file(str(path), device="cpu")

    expected_shapes: dict[str, tuple[int, ...]] = {}
    for module_name in sorted(modules):
        for parameter_name, value in modules[module_name].state_dict().items():
            if not isinstance(value, Tensor):
                raise TypeError(f"non-tensor state at {module_name}.{parameter_name}")
            if isinstance(value, DTensor):
                raise TypeError("action interface state must be replicated, not a DTensor")
            expected_shapes[f"{module_name}.{parameter_name}"] = tuple(value.shape)
    if not expected_shapes:
        raise ValueError("interface modules have no state")

    observed_keys = set(state)
    expected_keys = set(expected_shapes)
    if observed_keys != expected_keys:
        raise ValueError(
            "interface checkpoint tensor keys differ from the declared modules: "
            f"missing={sorted(expected_keys - observed_keys)}, "
            f"unexpected={sorted(observed_keys - expected_keys)}"
        )
    wrong_shapes = {
        key: {"expected": expected_shapes[key], "observed": tuple(state[key].shape)}
        for key in sorted(expected_keys)
        if tuple(state[key].shape) != expected_shapes[key]
    }
    if wrong_shapes:
        raise ValueError(f"interface checkpoint tensor shapes differ: {wrong_shapes}")
    wrong_dtypes = {key: str(state[key].dtype) for key in sorted(expected_keys) if state[key].dtype != torch.float32}
    if wrong_dtypes:
        raise ValueError(f"interface checkpoint tensors must all have dtype torch.float32: {wrong_dtypes}")
    nonfinite = [key for key in sorted(expected_keys) if not bool(torch.isfinite(state[key]).all())]
    if nonfinite:
        raise ValueError(f"interface checkpoint tensors contain non-finite values: {nonfinite}")

    # All tensors are authenticated before mutating any destination module.
    for module_name, module in modules.items():
        prefix = f"{module_name}."
        module_state = {key.removeprefix(prefix): value for key, value in state.items() if key.startswith(prefix)}
        module.load_state_dict(module_state, strict=True, assign=False)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def save_trainable_checkpoint(
    checkpoint_dir: str | Path,
    *,
    adapted_model: Any,
    interface_modules: Mapping[str, nn.Module],
    manifest: Mapping[str, Any],
    additional_artifacts: Mapping[str, str | Path] | None = None,
) -> dict[str, Any] | None:
    """Save LoRA + interface state; every TP rank must call this function."""

    from safetensors.torch import save_file

    _validate_modules(interface_modules)
    extras = dict(additional_artifacts or {})
    invalid_artifact_names = [name for name in extras if not _MODULE_NAME.fullmatch(name)]
    if invalid_artifact_names:
        raise ValueError(f"invalid additional artifact names: {invalid_artifact_names}")
    checkpoint_path = Path(checkpoint_dir)
    rank = _rank()
    creation_error: str | None = None
    if rank == 0:
        try:
            checkpoint_path.mkdir(parents=True, exist_ok=False)
            incomplete_path = checkpoint_path / "INCOMPLETE"
            with incomplete_path.open("w", encoding="utf-8") as handle:
                handle.write("checkpoint write in progress\n")
                handle.flush()
                os.fsync(handle.fileno())
            _fsync_directory(checkpoint_path)
            _fsync_directory(checkpoint_path.parent)
        except OSError as exc:
            creation_error = f"cannot create new checkpoint directory {checkpoint_path}: {exc}"
    creation_error = _broadcast_error(creation_error)
    if creation_error is not None:
        raise FileExistsError(creation_error)
    _barrier()

    lora_path = checkpoint_path / "lora"
    # PEFT's TP-aware state extraction contains collectives, so every rank participates.
    serialization_error: str | None = None
    try:
        adapted_model.save_pretrained(
            str(lora_path),
            safe_serialization=True,
            is_main_process=rank == 0,
            save_embedding_layers=False,
        )
    except Exception as exc:
        serialization_error = f"rank {rank}: {type(exc).__name__}: {exc}"
    if dist.is_available() and dist.is_initialized():
        serialization_errors: list[str | None] = [None] * dist.get_world_size()
        dist.all_gather_object(serialization_errors, serialization_error)
    else:
        serialization_errors = [serialization_error]
    if any(error is not None for error in serialization_errors):
        raise RuntimeError(f"LoRA checkpoint serialization failed: {serialization_errors}")
    _barrier()

    completed_manifest: dict[str, Any] | None = None
    finalization_error: str | None = None
    if rank == 0:
        try:
            interface_path = checkpoint_path / "interface.safetensors"
            save_file(interface_state_dict(interface_modules), str(interface_path), metadata={"format": "pt"})
            required_lora_files = (lora_path / "adapter_config.json", lora_path / "adapter_model.safetensors")
            missing = [str(path) for path in required_lora_files if not path.is_file()]
            if missing:
                raise RuntimeError(f"PEFT checkpoint is incomplete: {missing}")
            for path in required_lora_files:
                _fsync_file(path)
            _fsync_directory(lora_path)
            _fsync_file(interface_path)
            artifacts = {}
            for name, path in (
                ("interface", interface_path),
                ("lora_config", required_lora_files[0]),
                ("lora_weights", required_lora_files[1]),
            ):
                artifacts[name] = {
                    "path": str(path.relative_to(checkpoint_path)),
                    "bytes": path.stat().st_size,
                    "sha256": _sha256(path),
                }
            if extras:
                extras_path = checkpoint_path / "artifacts"
                extras_path.mkdir()
                for name, source_value in sorted(extras.items()):
                    source = Path(source_value)
                    if not source.is_file():
                        raise FileNotFoundError(f"additional checkpoint artifact is missing: {source}")
                    suffix = "".join(source.suffixes) or ".bin"
                    destination = extras_path / f"{name}{suffix}"
                    shutil.copyfile(source, destination)
                    _fsync_file(destination)
                    artifacts[name] = {
                        "path": str(destination.relative_to(checkpoint_path)),
                        "bytes": destination.stat().st_size,
                        "sha256": _sha256(destination),
                    }
                _fsync_directory(extras_path)
            completed_manifest = dict(manifest)
            completed_manifest["schema"] = SCHEMA_VERSION
            completed_manifest["artifacts"] = artifacts
            manifest_path = checkpoint_path / "manifest.json"
            temporary_manifest = checkpoint_path / "manifest.json.tmp"
            with temporary_manifest.open("w", encoding="utf-8") as handle:
                handle.write(json.dumps(completed_manifest, allow_nan=False, indent=2, sort_keys=True) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_manifest, manifest_path)
            _fsync_directory(checkpoint_path)
            incomplete_path.unlink()
            _fsync_directory(checkpoint_path)
        except Exception as exc:  # keep peers out of a final barrier if rank 0 cannot finalize
            finalization_error = f"{type(exc).__name__}: {exc}"
    finalization_error = _broadcast_error(finalization_error)
    if finalization_error is not None:
        raise RuntimeError(f"checkpoint finalization failed: {finalization_error}")
    _barrier()
    return completed_manifest


def load_checkpoint_manifest(checkpoint_dir: str | Path, *, verify_hashes: bool = True) -> dict[str, Any]:
    checkpoint_path = Path(checkpoint_dir).resolve()
    if (checkpoint_path / "INCOMPLETE").exists():
        raise RuntimeError(f"checkpoint is marked incomplete: {checkpoint_path}")
    manifest_path = checkpoint_path / "manifest.json"
    with manifest_path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    if not isinstance(manifest, dict) or manifest.get("schema") != SCHEMA_VERSION:
        raise ValueError("unsupported or missing Duo-VLA checkpoint schema")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("checkpoint manifest has no artifact table")
    missing_required = {"interface", "lora_config", "lora_weights"} - set(artifacts)
    if missing_required:
        raise ValueError(f"checkpoint manifest is missing required artifacts: {sorted(missing_required)}")
    for name, entry in artifacts.items():
        if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
            raise ValueError(f"checkpoint manifest has no valid {name!r} artifact")
        relative = Path(entry["path"])
        if relative.is_absolute() or relative == Path(".") or ".." in relative.parts:
            raise ValueError(f"checkpoint artifact path is not a contained relative path: {relative}")
        path = (checkpoint_path / relative).resolve()
        try:
            path.relative_to(checkpoint_path)
        except ValueError as exc:
            raise ValueError(f"checkpoint artifact escapes the checkpoint directory: {relative}") from exc
        if not path.is_file() or path.stat().st_size != entry.get("bytes"):
            raise ValueError(f"checkpoint artifact is missing or has the wrong size: {path}")
        if verify_hashes and _sha256(path) != entry.get("sha256"):
            raise ValueError(f"checkpoint artifact hash mismatch: {path}")
    return manifest


def load_lora_checkpoint(
    checkpoint_dir: str | Path,
    base_model: nn.Module,
    *,
    is_trainable: bool = False,
    validate_decoder_contract: bool = False,
    expected_rank: int = 16,
    allow_encoder_adapter: bool = False,
    allow_action_self_conditioning: bool = False,
) -> tuple[nn.Module, dict[str, Any]]:
    """Verify and attach a saved PEFT adapter to an already TP-sharded base model."""

    from peft import PeftModel

    checkpoint_path = Path(checkpoint_dir).resolve()
    manifest = load_checkpoint_manifest(checkpoint_path)
    artifacts = manifest["artifacts"]
    if "encoder_adapter" in artifacts and not allow_encoder_adapter:
        raise ValueError("encoder-adapted checkpoint requires an explicit encoder-aware loader")
    if manifest.get("action_self_conditioning", "none") != "none" and not allow_action_self_conditioning:
        raise ValueError("self-conditioned checkpoint requires an explicit action-SC-aware loader")
    canonical_lora_artifacts = {
        "lora_config": "lora/adapter_config.json",
        "lora_weights": "lora/adapter_model.safetensors",
    }
    mismatches = {
        name: artifacts[name].get("path")
        for name, expected_path in canonical_lora_artifacts.items()
        if artifacts[name].get("path") != expected_path
    }
    if mismatches:
        raise ValueError(
            "checkpoint LoRA artifact paths do not match the fixed PEFT consumer: "
            f"expected={canonical_lora_artifacts}, observed={mismatches}"
        )
    expected_saved_schema: dict[str, tuple[int, int]] | None = None
    if validate_decoder_contract:
        from duo_vla.backbones.loading import (
            validate_decoder_attention_lora_adapter_config,
            validate_decoder_attention_lora_weights,
        )

        validate_decoder_attention_lora_adapter_config(
            checkpoint_path / canonical_lora_artifacts["lora_config"],
            rank=expected_rank,
        )
        expected_saved_schema = validate_decoder_attention_lora_weights(
            checkpoint_path / canonical_lora_artifacts["lora_weights"],
            rank=expected_rank,
        )
    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")
        adapted = PeftModel.from_pretrained(
            base_model,
            str(checkpoint_path / "lora"),
            is_trainable=is_trainable,
            # PEFT otherwise infers generic "cuda". Safetensors resolves that
            # to GPU 0, creating an unwanted context on nonzero DP/TP ranks.
            # CPU staging copies into the already rank-local adapter parameters.
            torch_device="cpu",
        )
    if validate_decoder_contract:
        if caught_warnings:
            messages = [str(item.message) for item in caught_warnings]
            raise RuntimeError(f"PEFT emitted warnings while loading the strict LoRA checkpoint: {messages}")
        assert expected_saved_schema is not None
        expected_loaded_names = {
            name.replace(".lora_A.weight", ".lora_A.default.weight").replace(".lora_B.weight", ".lora_B.default.weight")
            for name in expected_saved_schema
        }
        observed_loaded = {
            name: parameter
            for name, parameter in adapted.named_parameters()
            if ".lora_A." in name or ".lora_B." in name
        }
        if set(observed_loaded) != expected_loaded_names:
            raise RuntimeError(
                "loaded PEFT parameter coverage differs from the authenticated adapter: "
                f"missing={sorted(expected_loaded_names - set(observed_loaded))}, "
                f"extra={sorted(set(observed_loaded) - expected_loaded_names)}"
            )
        nonfinite = [
            name
            for name, parameter in observed_loaded.items()
            if not bool((parameter.to_local() if isinstance(parameter, DTensor) else parameter).isfinite().all())
        ]
        if nonfinite:
            raise RuntimeError(f"loaded PEFT parameters contain non-finite values: {nonfinite}")
    return adapted, manifest
