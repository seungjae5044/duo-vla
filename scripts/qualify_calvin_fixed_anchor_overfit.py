#!/usr/bin/env python3
"""G3 qualification: overfit one immutable CALVIN A/B/C anchor inventory.

This is deliberately not a production trainer and never emits a serving
checkpoint.  It reuses the canonical CALVIN trainer's authenticated data,
prefix-geometry, model, LoRA, optimizer, TP, and deterministic-runtime
contracts, while replacing update-time data sampling with one materialized
inventory of 32--128 distinct action anchors.  Every optimizer update visits
that inventory exactly once, in the same physical-B=8 microbatch order, with
the same rectified-flow tensors.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import platform
import random
import secrets
import sys
import time
from contextlib import suppress
from dataclasses import asdict, dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from torch import Tensor, nn
from transformers import AutoProcessor

from duo_vla.action_interface import ActionInputProjector, VelocityHead
from duo_vla.backbones.diffusion_gemma import (
    DiffusionGemmaActionDecoder,
    DiffusionGemmaPrefix,
    apply_decoder_attention_lora,
    decoder_lora_parameter_partition,
    encode_diffusion_gemma_prefix,
)
from duo_vla.backbones.loading import DEFAULT_DIFFUSION_GEMMA_SPEC, load_diffusion_gemma_bf16_tp
from duo_vla.backbones.sample_isolated_experts import (
    GROUPED_MM_EXPERTS_IMPLEMENTATION,
    install_sample_isolated_grouped_mm_experts,
    verify_sample_isolated_grouped_mm_experts,
)
from duo_vla.data.calvin import CalvinAnchor, CalvinNpzDataset, CalvinTaskUniformAnchorSampler
from duo_vla.data.calvin_batching import CalvinBatch, collate_calvin_samples
from duo_vla.data.calvin_stats import (
    CALVIN_STORAGE_MODE_ARCHIVE_DIRECT,
    AuthenticatedCalvinDatasetGeneration,
    load_calvin_state_normalizer,
)
from duo_vla.modeling import DuoVLADenoiser
from duo_vla.objectives import PolicyTrainingPair, make_seeded_policy_training_pair
from duo_vla.optimization import (
    OptimizationConfig,
    assert_replicated_parameter_values,
    assert_replicated_tensor,
    clip_tensor_parallel_grad_norm_,
    create_optimizer_and_scheduler,
)
from duo_vla.policy_contract import RECTIFIED_FLOW
from duo_vla.prefix_geometry import SnapshotTreeIdentity, load_prefix_geometry_contract
from duo_vla.run_config import canonical_config_sha256
from duo_vla.training import masked_element_count, masked_sse
from duo_vla.training_checkpoint import file_sha256

REPORT_SCHEMA = "duovla-calvin-fixed-anchor-overfit-qualification-v2"
QUALIFICATION_PROTOCOL = "duovla-calvin-fixed-anchor-overfit-g3-v2"
QUALIFICATION_KIND = "calvin-abc-training-fixed-anchor-overfit-qualification"
MINIMUM_ANCHOR_COUNT = 32
MAXIMUM_ANCHOR_COUNT = 128
DEFAULT_ANCHOR_COUNT = 32
DEFAULT_ANCHOR_SEED = 314159
DEFAULT_PAIR_SEED = 271828
DEFAULT_UPDATES = 200
DEFAULT_WARMUP_UPDATES = 10
REQUIRED_MINIMUM_REDUCTION = 20.0


def _load_canonical_trainer() -> ModuleType:
    path = Path(__file__).resolve().with_name("train_calvin.py")
    spec = importlib.util.spec_from_file_location("_duo_vla_canonical_calvin_trainer", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load canonical CALVIN trainer contracts: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(spec.name, module)
    spec.loader.exec_module(module)
    return module


TRAIN = _load_canonical_trainer()
PHYSICAL_BATCH_SIZE = TRAIN.PHYSICAL_BATCH_SIZE


@dataclass(frozen=True, slots=True)
class QualificationSettings:
    anchor_count: int = DEFAULT_ANCHOR_COUNT
    anchor_seed: int = DEFAULT_ANCHOR_SEED
    pair_seed: int = DEFAULT_PAIR_SEED
    updates: int = DEFAULT_UPDATES
    warmup_updates: int = DEFAULT_WARMUP_UPDATES
    minimum_reduction: float = REQUIRED_MINIMUM_REDUCTION
    max_cached_frames: int = 512

    @property
    def microbatches_per_update(self) -> int:
        return self.anchor_count // PHYSICAL_BATCH_SIZE


@dataclass(frozen=True, slots=True)
class FrozenRegistrationSnapshot:
    state_schema_sha256: str
    parameter_structure: tuple[tuple[str, int, bool], ...]
    frozen_parameter_versions: tuple[tuple[str, int, int], ...]
    buffer_versions: tuple[tuple[str, int, int], ...]


@dataclass(frozen=True, slots=True)
class PreparedFixedMicrobatch:
    anchor_sha256: str
    static_input_sha256: str
    batch: CalvinBatch
    state: Tensor
    clean: Tensor
    valid: Tensor
    pair: PolicyTrainingPair
    prefix: DiffusionGemmaPrefix
    pair_seed: int


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


def canonical_object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def calvin_storage_report(generation: AuthenticatedCalvinDatasetGeneration) -> dict[str, Any]:
    if generation.storage.mode != CALVIN_STORAGE_MODE_ARCHIVE_DIRECT:
        raise ValueError("CALVIN G3 requires archive-direct v4 production storage")
    return {
        "identity": generation.storage.to_dict(),
        "identity_sha256": generation.storage.content_sha256,
    }


def finalized_report(value: dict[str, Any]) -> dict[str, Any]:
    require("report_sha256" not in value, "unfinalized report unexpectedly contains report_sha256")
    require(value.get("schema") == REPORT_SCHEMA, "qualification report schema is not canonical")
    require(value.get("protocol") == QUALIFICATION_PROTOCOL, "qualification report protocol is not canonical")
    require(value.get("kind") == QUALIFICATION_KIND, "qualification report kind is not canonical")
    require(value.get("status") == "passed", "only a passed qualification may be finalized")
    checks = value.get("pass_criteria")
    require(isinstance(checks, dict) and checks, "qualification report has no measurable pass criteria")
    require(
        all(type(passed) is bool and passed for passed in checks.values()), "qualification report has failed checks"
    )
    scope = value.get("scope")
    require(isinstance(scope, dict), "qualification report has no scope boundary")
    require(scope.get("qualification_only") is True, "qualification report must be qualification-only")
    require(scope.get("official_benchmark_claim") is False, "qualification report cannot claim an official benchmark")
    require(scope.get("benchmark_environment_accessed") is False, "qualification report cannot access CALVIN D")
    require(scope.get("checkpoint_emitted") is False, "qualification report cannot emit a serving checkpoint")
    report = dict(value)
    report["report_sha256"] = hashlib.sha256(canonical_json_bytes(value)).hexdigest()
    return report


def write_canonical_json_exclusive(path: Path, value: dict[str, Any]) -> Path:
    """Publish one fsynced report without following or replacing the target."""

    recorded_sha256 = value.get("report_sha256")
    unsigned = {key: field for key, field in value.items() if key != "report_sha256"}
    require(
        isinstance(recorded_sha256, str)
        and len(recorded_sha256) == 64
        and hashlib.sha256(canonical_json_bytes(unsigned)).hexdigest() == recorded_sha256,
        "qualification report self-hash is invalid",
    )
    parent = path.parent.resolve(strict=True)
    require(parent.is_dir(), f"report parent is not a directory: {parent}")
    require(path.name not in {"", ".", ".."}, "report filename is invalid")
    directory_fd = os.open(parent, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
    temporary = f".{path.name}.tmp-{secrets.token_hex(12)}"
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
            path.name,
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
    return parent / path.name


def validate_settings(settings: QualificationSettings) -> QualificationSettings:
    integer_fields = {
        "anchor_count": settings.anchor_count,
        "anchor_seed": settings.anchor_seed,
        "pair_seed": settings.pair_seed,
        "updates": settings.updates,
        "warmup_updates": settings.warmup_updates,
        "max_cached_frames": settings.max_cached_frames,
    }
    invalid_types = []
    for name, value in integer_fields.items():
        if isinstance(value, bool) or not isinstance(value, int):
            invalid_types.append(name)
    if invalid_types:
        raise ValueError(f"qualification integer settings have invalid types: {invalid_types}")
    if not MINIMUM_ANCHOR_COUNT <= settings.anchor_count <= MAXIMUM_ANCHOR_COUNT:
        raise ValueError(f"fixed-anchor qualification requires {MINIMUM_ANCHOR_COUNT}--{MAXIMUM_ANCHOR_COUNT} chunks")
    if settings.anchor_count % PHYSICAL_BATCH_SIZE:
        raise ValueError(f"anchor_count must be divisible by physical B={PHYSICAL_BATCH_SIZE}")
    if not 0 <= settings.anchor_seed < 2**63 or not 0 <= settings.pair_seed < 2**63:
        raise ValueError("anchor_seed and pair_seed must be in [0, 2^63)")
    if settings.updates <= 0:
        raise ValueError("updates must be positive")
    if settings.updates == 1:
        valid_warmup = settings.warmup_updates == 0
    else:
        valid_warmup = 0 <= settings.warmup_updates < settings.updates - 1
    if not valid_warmup:
        raise ValueError("warmup_updates must satisfy the canonical optimizer schedule")
    if not math.isfinite(settings.minimum_reduction) or settings.minimum_reduction < REQUIRED_MINIMUM_REDUCTION:
        raise ValueError(f"minimum_reduction cannot be below the G3 threshold {REQUIRED_MINIMUM_REDUCTION}x")
    if settings.max_cached_frames <= 0:
        raise ValueError("max_cached_frames must be positive")
    return settings


def draw_fixed_action_anchors(
    sampler: CalvinTaskUniformAnchorSampler,
    *,
    count: int,
    seed: int,
) -> tuple[CalvinAnchor, ...]:
    """Draw a deterministic inventory with distinct physical action anchors."""

    if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
        raise ValueError("count must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**63:
        raise ValueError("seed must be an integer in [0, 2^63)")
    if count > sampler.population_size:
        raise ValueError(f"cannot draw {count} anchors from a population of {sampler.population_size}")
    generator = torch.Generator().manual_seed(seed)
    anchors: list[CalvinAnchor] = []
    seen_global_indices: set[int] = set()
    maximum_draws = max(10_000, count * 1_024)
    draws = 0
    while len(anchors) < count and draws < maximum_draws:
        anchor = sampler.draw(generator)
        draws += 1
        if anchor.global_index not in seen_global_indices:
            anchors.append(anchor)
            seen_global_indices.add(anchor.global_index)
    if len(anchors) != count:
        raise RuntimeError(
            f"could not construct {count} distinct physical action anchors after {maximum_draws} deterministic draws"
        )
    return tuple(anchors)


def _update_array_digest(digest: Any, name: str, value: np.ndarray | Tensor) -> None:
    if isinstance(value, Tensor):
        tensor = value.detach()
        if hasattr(tensor, "to_local"):
            tensor = tensor.to_local()
        array = tensor.cpu().contiguous().numpy()
    else:
        array = np.ascontiguousarray(value)
    digest.update(name.encode("utf-8"))
    digest.update(str(array.dtype).encode("utf-8"))
    digest.update(str(tuple(array.shape)).encode("utf-8"))
    digest.update(array.tobytes(order="C"))


def sample_content_sha256(sample: Any) -> str:
    digest = hashlib.sha256()
    digest.update(sample.instruction.encode("utf-8"))
    digest.update(sample.task.encode("utf-8"))
    digest.update(str(sample.annotation_index).encode("ascii"))
    digest.update(str(sample.episode_index).encode("ascii"))
    digest.update(str(sample.global_index).encode("ascii"))
    _update_array_digest(digest, "rgb_static", sample.observation.third_person)
    _update_array_digest(digest, "rgb_gripper", sample.observation.wrist)
    _update_array_digest(digest, "state", sample.observation.state)
    _update_array_digest(digest, "actions", sample.action_chunk.actions)
    _update_array_digest(digest, "valid", sample.action_chunk.valid_mask)
    return digest.hexdigest()


def build_anchor_inventory(
    anchors: tuple[CalvinAnchor, ...],
    samples: tuple[Any, ...],
) -> tuple[tuple[dict[str, Any], ...], str]:
    if not anchors or len(anchors) != len(samples):
        raise ValueError("anchors and samples must be non-empty and have equal length")
    records: list[dict[str, Any]] = []
    global_indices: set[int] = set()
    for ordinal, (anchor, sample) in enumerate(zip(anchors, samples, strict=True)):
        expected = (anchor.annotation_index, anchor.global_index, anchor.task)
        observed = (sample.annotation_index, sample.global_index, sample.task)
        if observed != expected:
            raise ValueError(f"materialized sample {ordinal} differs from its fixed anchor")
        if anchor.global_index in global_indices:
            raise ValueError("fixed inventory contains duplicate physical action anchors")
        global_indices.add(anchor.global_index)
        records.append(
            {
                "annotation_index": anchor.annotation_index,
                "episode_index": sample.episode_index,
                "global_index": anchor.global_index,
                "instruction_sha256": hashlib.sha256(sample.instruction.encode("utf-8")).hexdigest(),
                "ordinal": ordinal,
                "sample_content_sha256": sample_content_sha256(sample),
                "task": anchor.task,
            }
        )
    frozen = tuple(records)
    return frozen, canonical_object_sha256(frozen)


def fixed_update_inventory_sha256(microbatches: tuple[PreparedFixedMicrobatch, ...] | tuple[Any, ...]) -> str:
    records = [
        {
            "anchor_sha256": batch.anchor_sha256,
            "microstep": microstep,
            "static_input_sha256": batch.static_input_sha256,
        }
        for microstep, batch in enumerate(microbatches)
    ]
    if not records:
        raise ValueError("fixed update inventory must contain at least one microbatch")
    return canonical_object_sha256(records)


def _assert_replicated_object(name: str, value: Any) -> str:
    digest = canonical_object_sha256(value)
    if dist.is_available() and dist.is_initialized():
        observed: list[str | None] = [None] * dist.get_world_size()
        dist.all_gather_object(observed, digest)
        if any(item != digest for item in observed):
            raise RuntimeError(f"replicated object {name!r} differs across TP ranks: {observed}")
    return digest


def _assert_finite_replicated_tensor(name: str, value: Tensor) -> str:
    materialized = value.full_tensor() if hasattr(value, "full_tensor") else value
    if materialized.is_floating_point() and not bool(torch.isfinite(materialized).all()):
        raise RuntimeError(f"replicated tensor {name!r} contains non-finite values")
    return assert_replicated_tensor(name, value)


def _tensor_bundle_sha256(name: str, values: dict[str, Any]) -> tuple[dict[str, str], str]:
    hashes: dict[str, str] = {}
    non_tensors = [key for key, value in values.items() if not isinstance(value, Tensor)]
    if non_tensors:
        raise TypeError(f"processor output contains non-tensor fields: {non_tensors}")
    for key, value in sorted(values.items()):
        hashes[key] = _assert_finite_replicated_tensor(f"{name}.{key}", value)
    return hashes, canonical_object_sha256(hashes)


def _state_schema(modules: tuple[tuple[str, nn.Module], ...]) -> str:
    records: list[dict[str, Any]] = []
    for prefix, module in modules:
        for name, value in module.state_dict(keep_vars=True).items():
            records.append(
                {
                    "dtype": str(value.dtype),
                    "name": f"{prefix}.{name}",
                    "shape": list(value.shape),
                }
            )
    return canonical_object_sha256(records)


def frozen_registration_snapshot(
    modules: tuple[tuple[str, nn.Module], ...],
) -> FrozenRegistrationSnapshot:
    parameter_structure: list[tuple[str, int, bool]] = []
    frozen_versions: list[tuple[str, int, int]] = []
    buffer_versions: list[tuple[str, int, int]] = []
    for prefix, module in modules:
        for name, parameter in module.named_parameters(remove_duplicate=False):
            qualified = f"{prefix}.{name}"
            parameter_structure.append((qualified, id(parameter), bool(parameter.requires_grad)))
            if not parameter.requires_grad:
                frozen_versions.append((qualified, id(parameter), int(parameter._version)))
        for name, buffer in module.named_buffers(remove_duplicate=False):
            buffer_versions.append((f"{prefix}.{name}", id(buffer), int(buffer._version)))
    return FrozenRegistrationSnapshot(
        state_schema_sha256=_state_schema(modules),
        parameter_structure=tuple(parameter_structure),
        frozen_parameter_versions=tuple(frozen_versions),
        buffer_versions=tuple(buffer_versions),
    )


def assert_only_declared_tensors_may_change(
    expected: FrozenRegistrationSnapshot,
    modules: tuple[tuple[str, nn.Module], ...],
) -> FrozenRegistrationSnapshot:
    observed = frozen_registration_snapshot(modules)
    require(
        observed.state_schema_sha256 == expected.state_schema_sha256,
        "qualification changed module state schema",
    )
    require(
        observed.parameter_structure == expected.parameter_structure,
        "qualification changed parameter identities, ordering, or trainable declarations",
    )
    require(
        observed.frozen_parameter_versions == expected.frozen_parameter_versions,
        "qualification modified or replaced a frozen parameter tensor",
    )
    require(
        observed.buffer_versions == expected.buffer_versions,
        "qualification modified or replaced a module buffer tensor",
    )
    return observed


def _local_tensor(value: Tensor) -> Tensor:
    local = value.to_local() if hasattr(value, "to_local") else value
    require(isinstance(local, Tensor), "tensor-parallel value did not expose a local tensor")
    return local


def prefix_cache_structure(prefix: DiffusionGemmaPrefix) -> tuple[tuple[Any, ...], ...]:
    layers = getattr(prefix.past_key_values, "layers", None)
    require(isinstance(layers, list) and len(layers) == 30, "fixed prefix cache must contain exactly 30 layers")
    structure: list[tuple[Any, ...]] = []
    for index, layer in enumerate(layers):
        keys = getattr(layer, "keys", None)
        values = getattr(layer, "values", None)
        require(isinstance(keys, Tensor) and isinstance(values, Tensor), f"prefix cache layer {index} has no K/V")
        local_keys = _local_tensor(keys)
        local_values = _local_tensor(values)
        length = layer.get_seq_length()
        length = int(_local_tensor(length).item()) if isinstance(length, Tensor) else int(length)
        structure.append(
            (
                index,
                type(layer).__qualname__,
                bool(getattr(layer, "is_sliding", False)),
                length,
                id(keys),
                id(values),
                local_keys.data_ptr(),
                local_values.data_ptr(),
                int(local_keys._version),
                int(local_values._version),
                tuple(local_keys.shape),
                tuple(local_values.shape),
                str(local_keys.dtype),
                str(local_values.dtype),
            )
        )
    return tuple(structure)


def fixed_input_tensor_structure(
    fixed_batches: tuple[PreparedFixedMicrobatch, ...],
) -> tuple[tuple[Any, ...], ...]:
    """Snapshot every non-cache tensor reused by all optimizer updates."""

    records: list[tuple[Any, ...]] = []
    for microstep, fixed in enumerate(fixed_batches):
        tensors = {
            "batch.clean": fixed.batch.clean_actions,
            "batch.state": fixed.batch.states,
            "batch.valid": fixed.batch.action_valid_mask,
            "device.clean": fixed.clean,
            "device.state": fixed.state,
            "device.valid": fixed.valid,
            "pair.input": fixed.pair.input_actions,
            "pair.target": fixed.pair.target,
            "pair.timesteps": fixed.pair.timesteps,
            "prefix.attention_mask": fixed.prefix.attention_mask,
        }
        for name, value in sorted(tensors.items()):
            local = _local_tensor(value)
            records.append(
                (
                    microstep,
                    name,
                    id(value),
                    local.data_ptr(),
                    int(local._version),
                    tuple(local.shape),
                    str(local.dtype),
                    str(local.device),
                )
            )
    if not records:
        raise ValueError("fixed input tensor structure cannot be empty")
    return tuple(records)


def _trainable_snapshot(named_parameters: list[tuple[str, nn.Parameter]]) -> dict[str, Tensor]:
    return {name: _local_tensor(parameter.detach()).cpu().clone() for name, parameter in named_parameters}


def _changed_trainable_names(
    named_parameters: list[tuple[str, nn.Parameter]],
    before: dict[str, Tensor],
) -> tuple[str, ...]:
    return tuple(
        name
        for name, parameter in named_parameters
        if not torch.equal(_local_tensor(parameter.detach()).cpu(), before[name])
    )


def optimizer_state_dtypes(optimizer: torch.optim.Optimizer) -> tuple[str, ...]:
    dtypes: set[str] = set()
    for state in optimizer.state.values():
        for value in state.values():
            if isinstance(value, Tensor) and value.is_floating_point():
                if value.dtype != torch.float32:
                    raise RuntimeError(f"optimizer state must be FP32, found {value.dtype}")
                dtypes.add(str(value.dtype))
    if not dtypes:
        raise RuntimeError("optimizer has no materialized floating-point state")
    return tuple(sorted(dtypes))


def qualification_checks(
    *,
    settings: QualificationSettings,
    observed_anchor_count: int,
    distinct_global_indices: int,
    update_inventory_verifications: int,
    update_inventory_hash_count: int,
    initial_loss: float,
    final_loss: float,
    reduction: float,
    changed_lora_tensors: int,
    lora_tensors: int,
    changed_interface_tensors: int,
    interface_tensors: int,
    optimizer_inventory_exact: bool,
    frozen_registration_unchanged: bool,
    prefix_cache_unchanged: bool,
    benchmark_environment_accessed: bool,
) -> dict[str, bool]:
    return {
        "anchor_count_in_declared_range": (
            MINIMUM_ANCHOR_COUNT <= observed_anchor_count <= MAXIMUM_ANCHOR_COUNT
            and observed_anchor_count == settings.anchor_count
        ),
        "all_action_anchors_distinct": distinct_global_indices == observed_anchor_count,
        "all_interface_tensors_changed": interface_tensors > 0 and changed_interface_tensors == interface_tensors,
        "all_lora_tensors_changed": lora_tensors > 0 and changed_lora_tensors == lora_tensors,
        "fixed_inventory_identical_every_update": (
            update_inventory_verifications == settings.updates and update_inventory_hash_count == 1
        ),
        "frozen_parameters_and_buffers_unchanged": frozen_registration_unchanged,
        "losses_finite_and_nonnegative": (
            math.isfinite(initial_loss) and math.isfinite(final_loss) and initial_loss > 0.0 and final_loss >= 0.0
        ),
        "minimum_fixed_loss_reduction_met": math.isfinite(reduction) and reduction >= settings.minimum_reduction,
        "no_calvin_d_policy_access": not benchmark_environment_accessed,
        "optimizer_contains_only_declared_trainables": optimizer_inventory_exact,
        "prefix_cache_remained_read_only": prefix_cache_unchanged,
    }


def require_all_checks(checks: dict[str, bool]) -> None:
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError(f"CALVIN fixed-anchor G3 qualification failed: {failed}")


def _evaluate_fixed_loss(
    fixed_batches: tuple[PreparedFixedMicrobatch, ...],
    *,
    denoiser: DuoVLADenoiser,
    adapted: nn.Module,
    encoder: nn.Module,
    device: torch.device,
) -> float:
    denoiser.eval()
    adapted.eval()
    encoder.eval()
    numerator = 0.0
    element_count = 0
    with torch.no_grad():
        for fixed in fixed_batches:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                prediction = denoiser(
                    fixed.pair.input_actions,
                    fixed.pair.timesteps,
                    fixed.state,
                    prefix_cache=fixed.prefix.past_key_values,
                    prefix_attention_mask=fixed.prefix.attention_mask,
                    action_valid_mask=fixed.valid,
                )
            component = masked_sse(prediction, fixed.pair.target, fixed.valid)
            numerator += float(component.squared_error_sum)
            element_count += component.element_count
    adapted.train()
    denoiser.train()
    encoder.eval()
    loss = numerator / element_count
    _assert_finite_replicated_tensor("fixed_probe_loss", torch.tensor(loss, device=device, dtype=torch.float64))
    return loss


def _qualification_source_identity(project_root: Path) -> dict[str, str]:
    single_gpu = os.environ.get("CUDA_VISIBLE_DEVICES") == "0"
    launcher = (
        "scripts/calvin/run_fixed_anchor_overfit_single_gpu.sh"
        if single_gpu
        else "scripts/calvin/run_fixed_anchor_overfit.sh"
    )
    files = {
        "qualification_launcher_sha256": file_sha256(project_root / launcher),
        "qualification_script_sha256": file_sha256(Path(__file__).resolve()),
        "production_source_tree_sha256": TRAIN._source_tree_sha256(project_root, single_gpu=single_gpu),
    }
    return {**files, "qualification_source_sha256": canonical_config_sha256(files)}


def require_qualification_source_unchanged(
    expected: dict[str, str],
    observed: dict[str, str],
    *,
    context: str,
) -> dict[str, str]:
    """Reject source mutation between import, startup, and report publication."""

    require(set(expected) == set(observed), f"{context} qualification source identity fields differ")
    require(
        canonical_json_bytes(expected) == canonical_json_bytes(observed),
        f"{context} qualification source identity changed",
    )
    return observed


_IMPORTED_QUALIFICATION_SOURCE_IDENTITY = _qualification_source_identity(Path(__file__).resolve().parents[1])


def _validate_report_target(path: Path) -> Path:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"qualification report already exists: {path}")
    parent = path.parent.resolve(strict=True)
    if not parent.is_dir():
        raise NotADirectoryError(f"qualification report parent is not a directory: {parent}")
    return parent / path.name


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("training_root", type=Path)
    parser.add_argument("normalization_artifact", type=Path)
    parser.add_argument("prefix_geometry_artifact", type=Path)
    parser.add_argument("report", type=Path)
    parser.add_argument("--anchor-count", type=int, default=DEFAULT_ANCHOR_COUNT)
    parser.add_argument("--anchor-seed", type=int, default=DEFAULT_ANCHOR_SEED)
    parser.add_argument("--pair-seed", type=int, default=DEFAULT_PAIR_SEED)
    parser.add_argument("--updates", type=int, default=DEFAULT_UPDATES)
    parser.add_argument("--warmup-updates", type=int, default=DEFAULT_WARMUP_UPDATES)
    parser.add_argument("--minimum-reduction", type=float, default=REQUIRED_MINIMUM_REDUCTION)
    parser.add_argument("--max-cached-frames", type=int, default=512)
    parser.add_argument(
        "--runtime-preflight-only",
        action="store_true",
        help="Validate the canonical pinned train runtime without CUDA, data, model, or report access.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    settings = validate_settings(
        QualificationSettings(
            anchor_count=args.anchor_count,
            anchor_seed=args.anchor_seed,
            pair_seed=args.pair_seed,
            updates=args.updates,
            warmup_updates=args.warmup_updates,
            minimum_reduction=args.minimum_reduction,
            max_cached_frames=args.max_cached_frames,
        )
    )
    project_root = Path(__file__).resolve().parents[1]
    source_identity_start = require_qualification_source_unchanged(
        _IMPORTED_QUALIFICATION_SOURCE_IDENTITY,
        _qualification_source_identity(project_root),
        context="between module import and main startup",
    )
    runtime_preflight = TRAIN._configure_and_validate_training_runtime(project_root)
    if args.runtime_preflight_only:
        print(
            json.dumps(
                {
                    "qualification_settings": asdict(settings),
                    "runtime_preflight": runtime_preflight,
                    "source_identity": source_identity_start,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return

    training_root = args.training_root.resolve(strict=True)
    normalization_artifact = args.normalization_artifact.resolve(strict=True)
    prefix_geometry_artifact = args.prefix_geometry_artifact.resolve(strict=True)
    report_path = _validate_report_target(args.report.absolute())
    if training_root.name != "training" or training_root.parent.name != "task_ABC_D":
        raise ValueError("qualification data must be the canonical task_ABC_D/training A/B/C split")

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", device_id=device)
    if dist.get_world_size() not in {1, 2}:
        raise RuntimeError("CALVIN fixed-anchor qualification requires TP world size one or two")
    rank = dist.get_rank()
    dataset: CalvinNpzDataset | None = None
    try:
        config_name = "calvin_abc_to_d_single_gpu.toml" if dist.get_world_size() == 1 else "calvin_abc_to_d.toml"
        config_path = (project_root / "configs" / config_name).resolve(strict=True)
        config = TRAIN.load_resolved_toml(config_path)
        interface_config = TRAIN._validate_and_build_interface_config(config)
        policy_contract = TRAIN.policy_contract_from_config(config)
        if policy_contract.objective != RECTIFIED_FLOW:
            raise ValueError("the canonical CALVIN G3 qualification requires the rectified-flow production config")

        generation = TRAIN._authenticate_calvin_dataset_generation_distributed(training_root)
        storage_report = calvin_storage_report(generation)

        dataset = CalvinNpzDataset(
            training_root,
            max_cached_frames=settings.max_cached_frames,
            expected_scenes=TRAIN.CALVIN_EXPECTED_SCENES,
            authenticated_generation=generation,
        )
        state_normalizer, stats_manifest = load_calvin_state_normalizer(
            normalization_artifact,
            expected_archive_sha256=TRAIN.CALVIN_ABC_D_ARCHIVE_SHA256,
            training_root=training_root,
            authenticated_generation=generation,
        )
        train_indices, validation_indices = TRAIN._validate_calvin_split_recipe(
            stats_manifest,
            dataset,
            config["training"],
        )
        sampler = CalvinTaskUniformAnchorSampler(dataset.annotations, train_indices)
        anchors = draw_fixed_action_anchors(sampler, count=settings.anchor_count, seed=settings.anchor_seed)
        samples = dataset.sample_many(anchors)
        anchor_records, anchor_inventory_sha256 = build_anchor_inventory(anchors, samples)
        if _assert_replicated_object("fixed_anchor_inventory", anchor_records) != anchor_inventory_sha256:
            raise RuntimeError("fixed anchor inventory hash implementation disagrees")

        calvin_source_revisions = TRAIN._calvin_source_revisions(project_root)
        calvin_identity = TRAIN._calvin_data_identity(
            stats_manifest,
            source_revisions=calvin_source_revisions,
        )
        model_snapshot = (
            Path(os.environ.get("HF_HOME", "/root/.cache/huggingface"))
            / "hub/models--google--diffusiongemma-26B-A4B-it/snapshots"
            / DEFAULT_DIFFUSION_GEMMA_SPEC.revision
        )
        model_identity: list[dict[str, Any] | None] = [None]
        if rank == 0:
            try:
                model_identity[0] = TRAIN._snapshot_identity(
                    model_snapshot,
                    expected_revision=DEFAULT_DIFFUSION_GEMMA_SPEC.revision,
                )
            except Exception as exc:
                model_identity[0] = {"error": f"{type(exc).__name__}: {exc}"}
        dist.broadcast_object_list(model_identity, src=0)
        model_identity_payload = model_identity[0]
        if not isinstance(model_identity_payload, dict):
            raise RuntimeError("rank zero did not broadcast a model snapshot identity")
        if "error" in model_identity_payload:
            raise RuntimeError(f"model snapshot authentication failed: {model_identity_payload['error']}")
        snapshot_identity = SnapshotTreeIdentity.from_huggingface_report(
            DEFAULT_DIFFUSION_GEMMA_SPEC.model_id,
            model_identity_payload,
        )
        prefix_geometry = load_prefix_geometry_contract(
            prefix_geometry_artifact,
            expected_content_sha256=config["benchmark"]["prefix_geometry_content_sha256"],
            expected_model_identity=snapshot_identity,
            expected_processor_identity=snapshot_identity,
            expected_ordered_cameras=TRAIN._calvin_prefix_cameras(),
            expected_fixed_physical_prefix_width=config["benchmark"]["fixed_physical_prefix_width"],
        )
        training_instruction_inventory_sha256 = TRAIN._validate_training_instruction_coverage(
            prefix_geometry,
            dataset.annotations,
        )

        random.seed(2026)
        np.random.seed(2026)
        torch.manual_seed(2026)
        torch.cuda.reset_peak_memory_stats(device)
        processor = AutoProcessor.from_pretrained(
            DEFAULT_DIFFUSION_GEMMA_SPEC.model_id,
            revision=DEFAULT_DIFFUSION_GEMMA_SPEC.revision,
            local_files_only=True,
        )
        if (
            getattr(getattr(processor, "tokenizer", None), "padding_side", None)
            != prefix_geometry["tokenization"]["padding_side"]
        ):
            raise RuntimeError("processor padding side differs from the authenticated CALVIN prefix geometry")
        model = load_diffusion_gemma_bf16_tp(local_files_only=True, tp_size=dist.get_world_size())
        installed_experts = install_sample_isolated_grouped_mm_experts(
            model,
            physical_batch_size=PHYSICAL_BATCH_SIZE,
        )
        if (
            installed_experts.experts_implementation != GROUPED_MM_EXPERTS_IMPLEMENTATION
            or installed_experts.physical_batch_size != PHYSICAL_BATCH_SIZE
        ):
            raise RuntimeError("installed expert execution differs from the canonical B=8 contract")
        verify_sample_isolated_grouped_mm_experts(model, physical_batch_size=PHYSICAL_BATCH_SIZE)
        backend = DiffusionGemmaActionDecoder.from_block_diffusion_model(model)
        torch.manual_seed(2027)
        adapted = apply_decoder_attention_lora(
            model,
            rank=int(config["lora"]["rank"]),
            alpha=int(config["lora"]["alpha"]),
            dropout=float(config["lora"]["dropout"]),
        )
        verify_sample_isolated_grouped_mm_experts(model, physical_batch_size=PHYSICAL_BATCH_SIZE)
        projector = ActionInputProjector(interface_config).to(device)
        head = VelocityHead(
            interface_config.hidden_size,
            interface_config.action_dim,
            init_std=interface_config.output_init_std,
        ).to(device)
        denoiser = DuoVLADenoiser(projector, backend, head).train()
        adapted.train()
        model.model.encoder.eval()

        fixed_batches_list: list[PreparedFixedMicrobatch] = []
        static_input_records: list[dict[str, Any]] = []
        for microstep in range(settings.microbatches_per_update):
            start = microstep * PHYSICAL_BATCH_SIZE
            stop = start + PHYSICAL_BATCH_SIZE
            batch_samples = samples[start:stop]
            batch = collate_calvin_samples(batch_samples, state_normalizer=state_normalizer)
            state = batch.states.to(device)
            clean = batch.clean_actions.to(device)
            valid = batch.action_valid_mask.to(device)
            processor_inputs = dict(
                TRAIN._processor_inputs(
                    processor,
                    batch.samples,
                    device,
                    prefix_geometry=prefix_geometry,
                )
            )
            processor_hashes, processor_sha256 = _tensor_bundle_sha256(
                f"fixed_microbatch.{microstep}.processor",
                processor_inputs,
            )
            prefix = encode_diffusion_gemma_prefix(model, processor_inputs)
            plan = TRAIN.make_microbatch_plan(settings.pair_seed, 0, microstep)
            pair = make_seeded_policy_training_pair(clean, policy_contract, seed=plan.flow_seed)
            tensor_hashes = {
                "clean": _assert_finite_replicated_tensor(f"fixed_microbatch.{microstep}.clean", clean),
                "pair_input": _assert_finite_replicated_tensor(
                    f"fixed_microbatch.{microstep}.pair_input",
                    pair.input_actions,
                ),
                "pair_target": _assert_finite_replicated_tensor(
                    f"fixed_microbatch.{microstep}.pair_target",
                    pair.target,
                ),
                "state": _assert_finite_replicated_tensor(f"fixed_microbatch.{microstep}.state", state),
                "timesteps": _assert_finite_replicated_tensor(
                    f"fixed_microbatch.{microstep}.timesteps",
                    pair.timesteps,
                ),
                "valid": _assert_finite_replicated_tensor(f"fixed_microbatch.{microstep}.valid", valid),
            }
            anchor_sha256 = canonical_object_sha256(anchor_records[start:stop])
            static_input_sha256 = canonical_object_sha256(
                {
                    "anchor_sha256": anchor_sha256,
                    "pair_seed": plan.flow_seed,
                    "processor_sha256": processor_sha256,
                    "tensor_sha256": tensor_hashes,
                }
            )
            fixed_batches_list.append(
                PreparedFixedMicrobatch(
                    anchor_sha256=anchor_sha256,
                    static_input_sha256=static_input_sha256,
                    batch=batch,
                    state=state,
                    clean=clean,
                    valid=valid,
                    pair=pair,
                    prefix=prefix,
                    pair_seed=plan.flow_seed,
                )
            )
            static_input_records.append(
                {
                    "anchor_sha256": anchor_sha256,
                    "microstep": microstep,
                    "pair_seed": plan.flow_seed,
                    "processor_sha256": processor_sha256,
                    "processor_tensor_sha256": processor_hashes,
                    "static_input_sha256": static_input_sha256,
                    "tensor_sha256": tensor_hashes,
                }
            )
        fixed_batches = tuple(fixed_batches_list)
        fixed_update_sha256 = fixed_update_inventory_sha256(fixed_batches)
        _assert_replicated_object("fixed_update_inventory", static_input_records)
        total_elements = sum(
            masked_element_count(fixed.valid, action_dim=interface_config.action_dim) for fixed in fixed_batches
        )

        lora_partition = decoder_lora_parameter_partition(adapted)
        lora_named_parameters = [
            (name, parameter) for name, parameter in adapted.named_parameters() if parameter.requires_grad
        ]
        interface_named_parameters = [
            *((f"action_projector.{name}", parameter) for name, parameter in projector.named_parameters()),
            *((f"velocity_head.{name}", parameter) for name, parameter in head.named_parameters()),
        ]
        lora_parameters = [parameter for _, parameter in lora_named_parameters]
        interface_parameters = [parameter for _, parameter in interface_named_parameters]
        TRAIN._assert_fp32_trainables(lora_parameters, interface_parameters)
        declared_ids = {id(parameter) for parameter in [*lora_parameters, *interface_parameters]}
        if len(declared_ids) != len(lora_parameters) + len(interface_parameters):
            raise RuntimeError("declared LoRA and interface trainable inventories overlap")

        optimization = config["optimization"]
        optimization_config = OptimizationConfig(
            lora_learning_rate=float(optimization["lora_learning_rate"]),
            interface_learning_rate=float(optimization["interface_learning_rate"]),
            beta1=float(optimization["adam_beta1"]),
            beta2=float(optimization["adam_beta2"]),
            epsilon=float(optimization["adam_epsilon"]),
            weight_decay=float(optimization["weight_decay"]),
            gradient_clip_norm=float(optimization["gradient_clip_norm"]),
            total_updates=settings.updates,
            warmup_updates=settings.warmup_updates,
            final_learning_rate_scale=float(optimization["final_learning_rate_scale"]),
        )
        optimizer, scheduler = create_optimizer_and_scheduler(
            lora_parameters,
            interface_parameters,
            optimization_config,
        )
        optimizer_ids = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}
        optimizer_inventory_exact = optimizer_ids == declared_ids
        if not optimizer_inventory_exact:
            raise RuntimeError("qualification optimizer inventory differs from declared trainables")

        guarded_modules = (
            ("adapted", adapted),
            ("action_projector", projector),
            ("velocity_head", head),
        )
        frozen_before = frozen_registration_snapshot(guarded_modules)
        cache_before = tuple(prefix_cache_structure(fixed.prefix) for fixed in fixed_batches)
        fixed_inputs_before = fixed_input_tensor_structure(fixed_batches)
        lora_before = _trainable_snapshot(lora_named_parameters)
        interface_before = _trainable_snapshot(interface_named_parameters)

        initial_loss = _evaluate_fixed_loss(
            fixed_batches,
            denoiser=denoiser,
            adapted=adapted,
            encoder=model.model.encoder,
            device=device,
        )
        best_pre_update_loss = initial_loss
        maximum_gradient_norm = 0.0
        update_inventory_hashes: list[str] = []
        started = time.perf_counter()
        for update in range(settings.updates):
            if fixed_input_tensor_structure(fixed_batches) != fixed_inputs_before:
                raise RuntimeError(
                    f"fixed input tensor identity/content version changed before optimizer update {update}"
                )
            if tuple(prefix_cache_structure(fixed.prefix) for fixed in fixed_batches) != cache_before:
                raise RuntimeError(f"fixed prefix cache changed before optimizer update {update}")
            observed_update_sha256 = fixed_update_inventory_sha256(fixed_batches)
            if observed_update_sha256 != fixed_update_sha256:
                raise RuntimeError(f"fixed update inventory changed at optimizer update {update}")
            update_inventory_hashes.append(observed_update_sha256)
            optimizer.zero_grad(set_to_none=True)
            numerator = 0.0
            for fixed in fixed_batches:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    prediction = denoiser(
                        fixed.pair.input_actions,
                        fixed.pair.timesteps,
                        fixed.state,
                        prefix_cache=fixed.prefix.past_key_values,
                        prefix_attention_mask=fixed.prefix.attention_mask,
                        action_valid_mask=fixed.valid,
                    )
                component = masked_sse(prediction, fixed.pair.target, fixed.valid)
                component.loss_for_total(total_elements).backward()
                numerator += float(component.squared_error_sum.detach())
            TRAIN._assert_distributed_gradient_health([*lora_parameters, *interface_parameters])
            if update == 0:
                TRAIN._assert_initial_gradients_present(lora_partition, interface_named_parameters)
            gradient_norm = clip_tensor_parallel_grad_norm_(
                [*(parameter for _, parameter in lora_partition.replicated), *interface_parameters],
                [parameter for _, parameter in lora_partition.sharded],
                max_norm=optimization_config.gradient_clip_norm,
                error_if_nonfinite=True,
            )
            maximum_gradient_norm = max(maximum_gradient_norm, float(gradient_norm))
            optimizer.step()
            if update == 0:
                TRAIN._assert_fp32_optimizer_state(optimizer)
                assert_replicated_parameter_values(interface_named_parameters)
            scheduler.step()
            train_loss = numerator / total_elements
            _assert_finite_replicated_tensor(
                f"fixed_train_loss.{update + 1}",
                torch.tensor(train_loss, device=device, dtype=torch.float64),
            )
            best_pre_update_loss = min(best_pre_update_loss, train_loss)
            if rank == 0 and (update == 0 or (update + 1) % 25 == 0):
                print(json.dumps({"fixed_train_loss": train_loss, "update": update + 1}, sort_keys=True), flush=True)
        elapsed_seconds = time.perf_counter() - started

        final_loss = _evaluate_fixed_loss(
            fixed_batches,
            denoiser=denoiser,
            adapted=adapted,
            encoder=model.model.encoder,
            device=device,
        )
        reduction = initial_loss / max(final_loss, torch.finfo(torch.float32).tiny)
        changed_lora = _changed_trainable_names(lora_named_parameters, lora_before)
        changed_interface = _changed_trainable_names(interface_named_parameters, interface_before)
        changes_by_rank: list[dict[str, Any] | None] = [None] * dist.get_world_size()
        dist.all_gather_object(
            changes_by_rank,
            {
                "changed_interface_tensors": len(changed_interface),
                "changed_lora_tensors": len(changed_lora),
                "interface_tensors": len(interface_named_parameters),
                "lora_tensors": len(lora_named_parameters),
                "rank": rank,
            },
        )

        local_audit_error: str | None = None
        try:
            assert_only_declared_tensors_may_change(frozen_before, guarded_modules)
            if fixed_input_tensor_structure(fixed_batches) != fixed_inputs_before:
                raise RuntimeError("fixed input tensors were mutated during qualification")
            if tuple(prefix_cache_structure(fixed.prefix) for fixed in fixed_batches) != cache_before:
                raise RuntimeError("fixed prefix cache was mutated during qualification")
        except Exception as exc:
            local_audit_error = f"rank {rank}: {type(exc).__name__}: {exc}"
        TRAIN._raise_if_rank_errors("fixed-anchor frozen/cache audit", local_audit_error)
        replicated_parameter_sha256 = assert_replicated_parameter_values(
            [*lora_partition.replicated, *interface_named_parameters]
        )
        state_dtypes = optimizer_state_dtypes(optimizer)
        checks = qualification_checks(
            settings=settings,
            observed_anchor_count=len(anchor_records),
            distinct_global_indices=len({record["global_index"] for record in anchor_records}),
            update_inventory_verifications=len(update_inventory_hashes),
            update_inventory_hash_count=len(set(update_inventory_hashes)),
            initial_loss=initial_loss,
            final_loss=final_loss,
            reduction=reduction,
            changed_lora_tensors=len(changed_lora),
            lora_tensors=len(lora_named_parameters),
            changed_interface_tensors=len(changed_interface),
            interface_tensors=len(interface_named_parameters),
            optimizer_inventory_exact=optimizer_inventory_exact,
            frozen_registration_unchanged=True,
            prefix_cache_unchanged=True,
            benchmark_environment_accessed=False,
        )
        local_check_error: str | None = None
        try:
            require_all_checks(checks)
        except Exception as exc:
            local_check_error = f"rank {rank}: {type(exc).__name__}: {exc}"
        TRAIN._raise_if_rank_errors("fixed-anchor pass criteria", local_check_error)

        source_identity = require_qualification_source_unchanged(
            source_identity_start,
            _qualification_source_identity(project_root),
            context="during fixed-anchor GPU qualification",
        )
        qualification_contract = {
            "anchor_inventory_sha256": anchor_inventory_sha256,
            "calvin_data_identity_sha256": canonical_config_sha256(calvin_identity),
            "calvin_storage_identity_sha256": storage_report["identity_sha256"],
            "fixed_update_sha256": fixed_update_sha256,
            "model_tree_sha256": model_identity_payload["tree_metadata_sha256"],
            "normalization_sha256": stats_manifest["content_sha256"],
            "policy_contract_sha256": canonical_config_sha256(policy_contract.to_dict()),
            "prefix_geometry_content_sha256": prefix_geometry["content_sha256"],
            "production_config_sha256": canonical_config_sha256(config),
            "protocol": QUALIFICATION_PROTOCOL,
            "settings": asdict(settings),
            "source_sha256": source_identity["qualification_source_sha256"],
        }
        report = finalized_report(
            {
                "anchor_inventory": {
                    "count": len(anchor_records),
                    "records": list(anchor_records),
                    "sha256": anchor_inventory_sha256,
                },
                "artifacts": {
                    "normalization_path": str(normalization_artifact),
                    "normalization_sha256": file_sha256(normalization_artifact),
                    "prefix_geometry_path": str(prefix_geometry_artifact),
                    "prefix_geometry_sha256": file_sha256(prefix_geometry_artifact),
                },
                "audits": {
                    "changes_by_rank": changes_by_rank,
                    "fixed_update_inventory_sha256": fixed_update_sha256,
                    "fixed_update_inventory_verifications": len(update_inventory_hashes),
                    "frozen_buffer_tensors": len(frozen_before.buffer_versions),
                    "frozen_parameter_tensors": len(frozen_before.frozen_parameter_versions),
                    "optimizer_inventory_exact": optimizer_inventory_exact,
                    "optimizer_state_dtypes": list(state_dtypes),
                    "prefix_cache_unchanged": True,
                    "replicated_parameter_sha256": replicated_parameter_sha256,
                    "static_microbatches": static_input_records,
                },
                "calvin_data_identity": calvin_identity,
                "calvin_storage": storage_report,
                "calvin_source_revisions": calvin_source_revisions,
                "execution_environment": TRAIN._execution_environment(runtime_preflight),
                "execution_profile": ("duovla-single-gpu-tp1-v1" if dist.get_world_size() == 1 else "duovla-tp2-v1"),
                "kind": QUALIFICATION_KIND,
                "metrics": {
                    "best_pre_update_loss": best_pre_update_loss,
                    "elapsed_seconds": elapsed_seconds,
                    "final_fixed_loss": final_loss,
                    "initial_fixed_loss": initial_loss,
                    "maximum_gradient_norm_before_clip": maximum_gradient_norm,
                    "peak_memory_gib_per_rank": torch.cuda.max_memory_allocated(device) / 2**30,
                    "reduction": reduction,
                    "total_valid_action_scalars": total_elements,
                },
                "model_identity": model_identity_payload,
                "pass_criteria": checks,
                "policy_contract": policy_contract.to_dict(),
                "prefix_geometry": {
                    "content_sha256": prefix_geometry["content_sha256"],
                    "fixed_physical_prefix_width": prefix_geometry["geometry"]["fixed_physical_prefix_width"],
                    "training_instruction_inventory_sha256": training_instruction_inventory_sha256,
                },
                "production_config_sha256": canonical_config_sha256(config),
                "protocol": QUALIFICATION_PROTOCOL,
                "qualification_contract": qualification_contract,
                "qualification_contract_sha256": canonical_config_sha256(qualification_contract),
                "schema": REPORT_SCHEMA,
                "scope": {
                    "benchmark_environment": "CALVIN D",
                    "benchmark_environment_accessed": False,
                    "checkpoint_emitted": False,
                    "data": "authenticated task_ABC_D/training scenes A/B/C only",
                    "official_benchmark_claim": False,
                    "qualification_only": True,
                    "validation_episode_count": len(validation_indices),
                },
                "settings": asdict(settings),
                "source_identity": source_identity,
                "status": "passed",
                "system": {"platform": platform.platform(), "python": sys.version.split()[0]},
            }
        )
        report_error: str | None = None
        if rank == 0:
            try:
                write_canonical_json_exclusive(report_path, report)
            except Exception as exc:
                report_error = f"cannot publish qualification report: {type(exc).__name__}: {exc}"
        TRAIN._broadcast_rank0_error(report_error)
        dist.barrier()
        if rank == 0:
            print(
                json.dumps(
                    {
                        "anchor_inventory_sha256": anchor_inventory_sha256,
                        "final_fixed_loss": final_loss,
                        "initial_fixed_loss": initial_loss,
                        "official_benchmark_claim": False,
                        "reduction": reduction,
                        "report": str(report_path),
                        "report_sha256": report["report_sha256"],
                        "status": "passed",
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
    finally:
        if dataset is not None:
            dataset.close()
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
