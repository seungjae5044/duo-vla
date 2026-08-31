"""Per-TP-rank optimizer, scheduler, progress, and RNG checkpoint state."""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import random
import stat
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

from duo_vla.training import TrainerState

TRAINING_RANK_STATE_SCHEMA = "duo-vla-training-rank-state-v2"
OPTIMIZER_PARAMETER_SCHEMA = "duo-vla-optimizer-parameter-schema-v1"
_STABLE_FILE_IDENTITY_FIELDS = ("st_dev", "st_ino", "st_mode", "st_size", "st_mtime_ns", "st_ctime_ns", "st_nlink")


def _same_file_identity(left: os.stat_result, right: os.stat_result) -> bool:
    return all(getattr(left, field) == getattr(right, field) for field in _STABLE_FILE_IDENTITY_FIELDS)


def _open_stable_regular_file_bytes(
    path: Path,
    *,
    require_single_link: bool,
) -> tuple[int, bytes, os.stat_result]:
    before = os.stat(path, follow_symlinks=False)
    if not stat.S_ISREG(before.st_mode):
        raise ValueError(f"checkpoint input must be a regular non-symlink file: {path}")
    if require_single_link and before.st_nlink != 1:
        raise ValueError(f"checkpoint input must have exactly one link: {path}")
    descriptor = os.open(path, os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW | os.O_CLOEXEC)
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode) or not _same_file_identity(before, opened):
            raise ValueError(f"checkpoint input changed while opening: {path}")
        blocks: list[bytes] = []
        while block := os.read(descriptor, 1024 * 1024):
            blocks.append(block)
        after_descriptor = os.fstat(descriptor)
        after_path = os.stat(path, follow_symlinks=False)
        if not _same_file_identity(opened, after_descriptor) or not _same_file_identity(opened, after_path):
            raise ValueError(f"checkpoint input changed while reading: {path}")
        raw = b"".join(blocks)
        if len(raw) != opened.st_size:
            raise ValueError(f"checkpoint input size changed while reading: {path}")
        return descriptor, raw, opened
    except BaseException:
        os.close(descriptor)
        raise


def _require_open_file_still_bound(descriptor: int, path: Path, expected: os.stat_result) -> None:
    after_descriptor = os.fstat(descriptor)
    after_path = os.stat(path, follow_symlinks=False)
    if not _same_file_identity(expected, after_descriptor) or not _same_file_identity(expected, after_path):
        raise ValueError(f"checkpoint input identity changed during deserialization: {path}")


def file_sha256(path: str | Path) -> str:
    descriptor, raw, identity = _open_stable_regular_file_bytes(Path(path), require_single_link=False)
    try:
        _require_open_file_still_bound(descriptor, Path(path), identity)
        return hashlib.sha256(raw).hexdigest()
    finally:
        os.close(descriptor)


def optimizer_parameter_inventory(
    optimizer: torch.optim.Optimizer,
    named_parameters: Iterable[tuple[str, torch.nn.Parameter]],
) -> dict[str, Any]:
    """Return the canonical positional parameter mapping consumed by optimizer restore."""

    entries = tuple(named_parameters)
    if not entries:
        raise ValueError("named_parameters must not be empty")
    names = [name for name, _ in entries]
    identities = [id(parameter) for _, parameter in entries]
    if any(not isinstance(name, str) or not name for name in names):
        raise ValueError("named_parameters must have nonempty string names")
    if len(set(names)) != len(names) or len(set(identities)) != len(identities):
        raise ValueError("named_parameters must have unique names and tensors")
    name_by_id = {id(parameter): name for name, parameter in entries}
    seen: set[int] = set()
    groups: list[dict[str, Any]] = []
    for group_index, group in enumerate(optimizer.param_groups):
        group_name = group.get("name")
        if not isinstance(group_name, str) or not group_name:
            group_name = f"group-{group_index}"
        parameters: list[dict[str, Any]] = []
        for parameter in group["params"]:
            identity = id(parameter)
            if identity in seen:
                raise ValueError("optimizer contains a duplicate parameter")
            try:
                name = name_by_id[identity]
            except KeyError as exc:
                raise ValueError("optimizer contains an unnamed parameter") from exc
            seen.add(identity)
            parameters.append(
                {
                    "dtype": str(parameter.dtype),
                    "name": name,
                    "requires_grad": parameter.requires_grad,
                    "shape": list(parameter.shape),
                }
            )
        if not parameters:
            raise ValueError("optimizer parameter groups must not be empty")
        groups.append({"name": group_name, "parameters": parameters})
    if not groups:
        raise ValueError("optimizer must have parameter groups")
    if seen != set(identities):
        missing = sorted(name_by_id[identity] for identity in set(identities) - seen)
        raise ValueError(f"named parameters are absent from the optimizer: {missing}")
    return _validated_optimizer_parameter_inventory({"groups": groups, "schema": OPTIMIZER_PARAMETER_SCHEMA})


def _validated_optimizer_parameter_inventory(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {"groups", "schema"}:
        raise ValueError("optimizer parameter inventory schema differs")
    if value["schema"] != OPTIMIZER_PARAMETER_SCHEMA:
        raise ValueError("unsupported optimizer parameter inventory schema")
    groups = value["groups"]
    if not isinstance(groups, list) or not groups:
        raise ValueError("optimizer parameter inventory groups are invalid")
    canonical_groups: list[dict[str, Any]] = []
    observed_names: set[str] = set()
    observed_group_names: set[str] = set()
    for group in groups:
        if not isinstance(group, Mapping) or set(group) != {"name", "parameters"}:
            raise ValueError("optimizer parameter inventory group schema differs")
        group_name = group["name"]
        parameters = group["parameters"]
        if (
            not isinstance(group_name, str)
            or not group_name
            or group_name in observed_group_names
            or not isinstance(parameters, list)
            or not parameters
        ):
            raise ValueError("optimizer parameter inventory group is invalid")
        observed_group_names.add(group_name)
        canonical_parameters: list[dict[str, Any]] = []
        for parameter in parameters:
            if not isinstance(parameter, Mapping) or set(parameter) != {"dtype", "name", "requires_grad", "shape"}:
                raise ValueError("optimizer parameter inventory entry schema differs")
            name = parameter["name"]
            shape = parameter["shape"]
            dtype = parameter["dtype"]
            requires_grad = parameter["requires_grad"]
            if (
                not isinstance(name, str)
                or not name
                or name in observed_names
                or not isinstance(dtype, str)
                or not dtype.startswith("torch.")
                or type(requires_grad) is not bool
                or not isinstance(shape, list)
                or not all(type(dimension) is int and dimension >= 0 for dimension in shape)
            ):
                raise ValueError("optimizer parameter inventory entry is invalid")
            observed_names.add(name)
            canonical_parameters.append(
                {"dtype": dtype, "name": name, "requires_grad": requires_grad, "shape": list(shape)}
            )
        canonical_groups.append({"name": group_name, "parameters": canonical_parameters})
    return {"groups": canonical_groups, "schema": OPTIMIZER_PARAMETER_SCHEMA}


def optimizer_parameter_inventory_sha256(value: Any) -> str:
    canonical = _validated_optimizer_parameter_inventory(value)
    encoded = json.dumps(canonical, allow_nan=False, separators=(",", ":"), sort_keys=True).encode()
    return hashlib.sha256(encoded).hexdigest()


def optimizer_parameter_schema_sha256(
    optimizer: torch.optim.Optimizer,
    named_parameters: Iterable[tuple[str, torch.nn.Parameter]],
) -> str:
    """Hash the exact positional parameter mapping consumed by ``Optimizer.load_state_dict``."""

    return optimizer_parameter_inventory_sha256(optimizer_parameter_inventory(optimizer, named_parameters))


def validate_optimizer_state_dict(
    value: Any,
    parameter_inventory: Any,
    *,
    expected_update: int,
) -> None:
    """Bind positional optimizer state coverage and tensor shapes to a named inventory."""

    inventory = _validated_optimizer_parameter_inventory(parameter_inventory)
    if not isinstance(value, Mapping) or set(value) != {"param_groups", "state"}:
        raise ValueError("optimizer state-dict schema differs")
    state_groups = value["param_groups"]
    state = value["state"]
    if not isinstance(state_groups, list) or len(state_groups) != len(inventory["groups"]):
        raise ValueError("optimizer state-dict group inventory differs")
    expected_parameters: dict[int, dict[str, Any]] = {}
    next_identifier = 0
    for group_index, (state_group, inventory_group) in enumerate(zip(state_groups, inventory["groups"], strict=True)):
        if not isinstance(state_group, Mapping):
            raise ValueError("optimizer state-dict group is invalid")
        state_group_name = state_group.get("name")
        if not isinstance(state_group_name, str) or not state_group_name:
            state_group_name = f"group-{group_index}"
        if state_group_name != inventory_group["name"]:
            raise ValueError("optimizer state-dict group name differs from its parameter inventory")
        parameters = state_group.get("params")
        count = len(inventory_group["parameters"])
        expected_ids = list(range(next_identifier, next_identifier + count))
        if parameters != expected_ids:
            raise ValueError("optimizer state-dict positional parameter ordering differs")
        for identifier, metadata in zip(expected_ids, inventory_group["parameters"], strict=True):
            expected_parameters[identifier] = metadata
        next_identifier += count
    if not isinstance(state, Mapping):
        raise ValueError("optimizer parameter state is invalid")
    expected_state_ids = set(expected_parameters) if expected_update > 0 else set()
    if set(state) != expected_state_ids:
        raise ValueError("optimizer state does not cover the canonical parameter inventory")
    for identifier, parameter_state in state.items():
        if not isinstance(parameter_state, Mapping) or set(parameter_state) != {"exp_avg", "exp_avg_sq", "step"}:
            raise ValueError("optimizer parameter state schema differs")
        metadata = expected_parameters[identifier]
        step = parameter_state["step"]
        first_moment = parameter_state["exp_avg"]
        second_moment = parameter_state["exp_avg_sq"]
        if (
            not isinstance(step, torch.Tensor)
            or step.shape != torch.Size([])
            or step.numel() != 1
            or step.item() != expected_update
        ):
            raise ValueError("optimizer parameter step differs from the checkpoint update")
        expected_shape = torch.Size(metadata["shape"])
        if (
            not isinstance(first_moment, torch.Tensor)
            or not isinstance(second_moment, torch.Tensor)
            or first_moment.layout != torch.strided
            or second_moment.layout != torch.strided
            or first_moment.shape != expected_shape
            or second_moment.shape != expected_shape
            or str(first_moment.dtype) != metadata["dtype"]
            or str(second_moment.dtype) != metadata["dtype"]
        ):
            raise ValueError("optimizer parameter moment tensors differ from the canonical inventory")


def validate_training_progress(
    trainer_state: TrainerState,
    *,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    examples_per_update: int,
    expected_learning_rates: Sequence[float],
) -> None:
    """Reject positional or off-by-one resume state before another optimizer update."""

    if isinstance(examples_per_update, bool) or not isinstance(examples_per_update, int) or examples_per_update <= 0:
        raise ValueError("examples_per_update must be a positive integer")
    if trainer_state.examples_seen != trainer_state.next_update * examples_per_update:
        raise ValueError("trainer examples_seen is inconsistent with next_update and global batch size")
    if scheduler.last_epoch != trainer_state.next_update:
        raise ValueError("scheduler last_epoch is inconsistent with trainer next_update")
    if len(expected_learning_rates) != len(optimizer.param_groups):
        raise ValueError("expected learning-rate count does not match optimizer groups")
    for index, (group, expected) in enumerate(zip(optimizer.param_groups, expected_learning_rates, strict=True)):
        observed = float(group["lr"])
        if not math.isclose(observed, float(expected), rel_tol=1e-12, abs_tol=0.0):
            raise ValueError(f"optimizer group {index} learning rate is inconsistent with the schedule")

    expected_step = trainer_state.next_update
    parameters = [parameter for group in optimizer.param_groups for parameter in group["params"]]
    if expected_step == 0:
        if optimizer.state:
            raise ValueError("zero-update trainer unexpectedly has optimizer state")
        return
    if len(optimizer.state) != len(parameters):
        raise ValueError("optimizer state does not cover every parameter")
    for parameter in parameters:
        state = optimizer.state.get(parameter)
        if not isinstance(state, dict) or "step" not in state:
            raise ValueError("optimizer parameter state is missing its step counter")
        step = state["step"]
        if isinstance(step, torch.Tensor):
            if step.numel() != 1:
                raise ValueError("optimizer step counter must be scalar")
            step = step.item()
        if int(step) != expected_step or float(step) != expected_step:
            raise ValueError("optimizer step counter is inconsistent with trainer next_update")


def capture_rng_state(device: torch.device | str | None = None) -> dict[str, Any]:
    numpy_state = np.random.get_state()
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": {
            "bit_generator": numpy_state[0],
            "keys": torch.from_numpy(numpy_state[1].copy()),
            "position": int(numpy_state[2]),
            "has_gauss": int(numpy_state[3]),
            "cached_gaussian": float(numpy_state[4]),
        },
        "torch_cpu": torch.get_rng_state(),
    }
    if device is not None:
        resolved = torch.device(device)
        if resolved.type != "cuda":
            raise ValueError("device must be CUDA when supplied")
        state["torch_cuda"] = torch.cuda.get_rng_state(resolved).cpu()
        state["cuda_device_index"] = resolved.index
    return state


def restore_rng_state(state: Mapping[str, Any], device: torch.device | str | None = None) -> None:
    required = {"python", "numpy", "torch_cpu"}
    if not required.issubset(state):
        raise ValueError("RNG state is missing required generators")
    random.setstate(state["python"])
    numpy_state = state["numpy"]
    if not isinstance(numpy_state, Mapping):
        raise ValueError("NumPy RNG state must be a mapping")
    keys = numpy_state.get("keys")
    if not isinstance(keys, torch.Tensor) or keys.dtype != torch.uint32:
        raise ValueError("NumPy RNG keys must be a uint32 tensor")
    np.random.set_state(
        (
            str(numpy_state["bit_generator"]),
            keys.cpu().numpy(),
            int(numpy_state["position"]),
            int(numpy_state["has_gauss"]),
            float(numpy_state["cached_gaussian"]),
        )
    )
    torch_cpu = state["torch_cpu"]
    if not isinstance(torch_cpu, torch.Tensor) or torch_cpu.dtype != torch.uint8:
        raise ValueError("Torch CPU RNG state must be a uint8 tensor")
    torch.set_rng_state(torch_cpu.cpu())
    if device is not None:
        resolved = torch.device(device)
        torch_cuda = state.get("torch_cuda")
        if resolved.type != "cuda" or not isinstance(torch_cuda, torch.Tensor):
            raise ValueError("checkpoint has no valid CUDA RNG state")
        if state.get("cuda_device_index") != resolved.index:
            raise ValueError("CUDA RNG device index differs from the current TP rank")
        torch.cuda.set_rng_state(torch_cuda.cpu(), resolved)


def save_training_rank_state(
    path: str | Path,
    *,
    rank: int,
    world_size: int,
    trainer_state: TrainerState,
    optimizer: torch.optim.Optimizer,
    named_parameters: Iterable[tuple[str, torch.nn.Parameter]],
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    run_contract: Mapping[str, str],
    device: torch.device | str | None = None,
) -> str:
    if not 0 <= rank < world_size:
        raise ValueError("rank must be in [0, world_size)")
    if not run_contract or any(
        not isinstance(key, str) or not isinstance(value, str) for key, value in run_contract.items()
    ):
        raise ValueError("run_contract must be a nonempty string mapping")
    parameter_inventory = optimizer_parameter_inventory(optimizer, named_parameters)
    parameter_inventory_sha256 = optimizer_parameter_inventory_sha256(parameter_inventory)
    if run_contract.get("optimizer_parameter_schema_sha256") != parameter_inventory_sha256:
        raise ValueError("run_contract optimizer parameter schema hash mismatch")
    optimizer_state = optimizer.state_dict()
    validate_optimizer_state_dict(
        optimizer_state,
        parameter_inventory,
        expected_update=trainer_state.next_update,
    )
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": TRAINING_RANK_STATE_SCHEMA,
        "rank": rank,
        "world_size": world_size,
        "trainer_state": trainer_state.to_dict(),
        "optimizer": optimizer_state,
        "optimizer_parameter_inventory": parameter_inventory,
        "scheduler": scheduler.state_dict(),
        "rng": capture_rng_state(device),
        "run_contract": dict(sorted(run_contract.items())),
    }
    temporary = output.with_name(f".{output.name}.tmp-{os.getpid()}")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return file_sha256(output)


def load_training_rank_state(
    path: str | Path,
    *,
    expected_sha256: str,
    rank: int,
    world_size: int,
    optimizer: torch.optim.Optimizer,
    named_parameters: Iterable[tuple[str, torch.nn.Parameter]],
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    run_contract: Mapping[str, str],
    device: torch.device | str | None = None,
) -> TrainerState:
    source = Path(path)
    descriptor, raw, identity = _open_stable_regular_file_bytes(source, require_single_link=True)
    try:
        if hashlib.sha256(raw).hexdigest() != expected_sha256:
            raise ValueError("training rank-state hash mismatch")
        payload = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
        _require_open_file_still_bound(descriptor, source, identity)
    finally:
        os.close(descriptor)
    if not isinstance(payload, dict) or payload.get("schema") != TRAINING_RANK_STATE_SCHEMA:
        raise ValueError("unsupported training rank-state schema")
    required_fields = {
        "optimizer",
        "optimizer_parameter_inventory",
        "rank",
        "rng",
        "run_contract",
        "scheduler",
        "schema",
        "trainer_state",
        "world_size",
    }
    if set(payload) != required_fields:
        raise ValueError("training rank-state field inventory differs")
    if payload.get("rank") != rank or payload.get("world_size") != world_size:
        raise ValueError("training rank-state topology mismatch")
    if payload.get("run_contract") != dict(sorted(run_contract.items())):
        raise ValueError("training rank-state run contract mismatch")
    trainer_state = TrainerState.from_dict(payload.get("trainer_state"))
    current_inventory = optimizer_parameter_inventory(optimizer, named_parameters)
    checkpoint_inventory = _validated_optimizer_parameter_inventory(payload["optimizer_parameter_inventory"])
    if checkpoint_inventory != current_inventory:
        raise ValueError("training rank-state optimizer parameter inventory mismatch")
    inventory_sha256 = optimizer_parameter_inventory_sha256(checkpoint_inventory)
    if run_contract.get("optimizer_parameter_schema_sha256") != inventory_sha256:
        raise ValueError("training rank-state optimizer parameter inventory hash mismatch")
    validate_optimizer_state_dict(
        payload["optimizer"],
        checkpoint_inventory,
        expected_update=trainer_state.next_update,
    )
    optimizer.load_state_dict(payload["optimizer"])
    scheduler.load_state_dict(payload["scheduler"])
    restore_rng_state(payload["rng"], device)
    return trainer_state
