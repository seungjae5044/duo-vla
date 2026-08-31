"""Optimizer parameter-group and learning-rate schedule contracts."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Iterable
from dataclasses import dataclass

import torch
import torch.distributed as dist
from torch import Tensor, nn


@dataclass(frozen=True, slots=True)
class OptimizationConfig:
    lora_learning_rate: float = 1e-4
    interface_learning_rate: float = 1e-3
    beta1: float = 0.9
    beta2: float = 0.95
    epsilon: float = 1e-8
    weight_decay: float = 1e-10
    gradient_clip_norm: float = 1.0
    total_updates: int = 30_000
    warmup_updates: int = 1_000
    final_learning_rate_scale: float = 0.1

    def __post_init__(self) -> None:
        if self.lora_learning_rate <= 0 or self.interface_learning_rate <= 0:
            raise ValueError("learning rates must be positive")
        if not 0 < self.beta1 < 1 or not 0 < self.beta2 < 1:
            raise ValueError("Adam betas must be strictly between zero and one")
        if self.epsilon <= 0 or self.weight_decay < 0 or self.gradient_clip_norm <= 0:
            raise ValueError("epsilon/gradient clip must be positive and weight decay nonnegative")
        if self.total_updates <= 0 or self.warmup_updates < 0:
            raise ValueError("total updates must be positive and warmup nonnegative")
        if self.total_updates > 1 and self.warmup_updates >= self.total_updates - 1:
            raise ValueError("the warmup peak must precede the final optimizer update")
        if self.total_updates == 1 and self.warmup_updates != 0:
            raise ValueError("a single-update schedule cannot have warmup")
        if not 0 <= self.final_learning_rate_scale <= 1:
            raise ValueError("final learning-rate scale must be in [0, 1]")


def learning_rate_scale(update: int, config: OptimizationConfig) -> float:
    """Return the scale applied at optimizer update index ``update``.

    Real optimizer updates are indexed from ``0`` through ``total_updates - 1``. With warmup, update zero starts at
    ``1 / (warmup_updates + 1)`` and update ``warmup_updates`` is the peak. The final real update always receives
    ``final_learning_rate_scale``. A one-update run is the degenerate case and applies the configured final scale.
    """

    if isinstance(update, bool) or not isinstance(update, int) or update < 0:
        raise ValueError("update must be a nonnegative integer")
    if config.total_updates == 1:
        return config.final_learning_rate_scale
    if config.warmup_updates and update < config.warmup_updates:
        initial = 1.0 / (config.warmup_updates + 1)
        return initial + (1.0 - initial) * update / config.warmup_updates
    progress = min(
        1.0,
        (update - config.warmup_updates) / (config.total_updates - 1 - config.warmup_updates),
    )
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return config.final_learning_rate_scale + (1.0 - config.final_learning_rate_scale) * cosine


def _materialize_trainable(parameters: Iterable[nn.Parameter], name: str) -> list[nn.Parameter]:
    values = list(parameters)
    if not values:
        raise ValueError(f"{name} parameter group is empty")
    if any(not parameter.requires_grad for parameter in values):
        raise ValueError(f"{name} parameter group contains frozen parameters")
    if len({id(parameter) for parameter in values}) != len(values):
        raise ValueError(f"{name} parameter group contains duplicate tensors")
    return values


def create_optimizer_and_scheduler(
    lora_parameters: Iterable[nn.Parameter],
    interface_parameters: Iterable[nn.Parameter],
    config: OptimizationConfig,
) -> tuple[torch.optim.AdamW, torch.optim.lr_scheduler.LambdaLR]:
    lora = _materialize_trainable(lora_parameters, "lora")
    interface = _materialize_trainable(interface_parameters, "interface")
    overlap = {id(parameter) for parameter in lora} & {id(parameter) for parameter in interface}
    if overlap:
        raise ValueError("LoRA and interface parameter groups overlap")
    optimizer = torch.optim.AdamW(
        [
            {"name": "lora", "params": lora, "lr": config.lora_learning_rate},
            {"name": "interface", "params": interface, "lr": config.interface_learning_rate},
        ],
        betas=(config.beta1, config.beta2),
        eps=config.epsilon,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda update: learning_rate_scale(update, config),
    )
    return optimizer, scheduler


def finite_gradient_audit(parameters: Iterable[nn.Parameter]) -> tuple[int, int]:
    """Return counts of missing and non-finite gradients without altering them."""

    missing = 0
    nonfinite = 0
    for parameter in parameters:
        if parameter.grad is None:
            missing += 1
        elif not _isfinite(parameter.grad):
            nonfinite += 1
    return missing, nonfinite


def assert_replicated_parameter_values(
    named_parameters: Iterable[tuple[str, nn.Parameter]],
) -> str:
    """Hash trainable values and require identical replicas on every initialized rank."""

    digest = hashlib.sha256()
    entries = sorted(named_parameters, key=lambda item: item[0])
    if not entries:
        raise ValueError("named_parameters must not be empty")
    if len({name for name, _ in entries}) != len(entries):
        raise ValueError("parameter names must be unique")
    for name, parameter in entries:
        value = parameter.detach()
        if hasattr(value, "full_tensor"):
            value = value.full_tensor()
        value = value.cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    local_hash = digest.hexdigest()
    if dist.is_available() and dist.is_initialized():
        hashes: list[str | None] = [None] * dist.get_world_size()
        dist.all_gather_object(hashes, local_hash)
        if any(value != local_hash for value in hashes):
            raise RuntimeError(f"replicated trainable parameters diverged across ranks: {hashes}")
    return local_hash


def assert_replicated_tensor(name: str, value: Tensor) -> str:
    """Require a replicated output tensor to be byte-identical on every rank."""

    local = value.detach()
    if hasattr(local, "full_tensor"):
        local = local.full_tensor()
    local = local.cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(name.encode())
    digest.update(str(local.dtype).encode())
    digest.update(str(tuple(local.shape)).encode())
    digest.update(local.reshape(-1).view(torch.uint8).numpy().tobytes())
    local_hash = digest.hexdigest()
    if dist.is_available() and dist.is_initialized():
        hashes: list[str | None] = [None] * dist.get_world_size()
        dist.all_gather_object(hashes, local_hash)
        if any(item != local_hash for item in hashes):
            raise RuntimeError(f"replicated tensor {name!r} diverged across ranks: {hashes}")
    return local_hash


def clip_tensor_parallel_grad_norm_(
    replicated_parameters: Iterable[nn.Parameter],
    sharded_parameters: Iterable[nn.Parameter],
    *,
    max_norm: float,
    error_if_nonfinite: bool = True,
) -> Tensor:
    """Clip one TP model using one copy of replicated norms plus all local shard norms."""

    if max_norm <= 0:
        raise ValueError("max_norm must be positive")
    replicated = list(replicated_parameters)
    sharded = list(sharded_parameters)
    all_parameters = [*replicated, *sharded]
    if not all_parameters:
        raise ValueError("at least one parameter is required")
    if len({id(parameter) for parameter in all_parameters}) != len(all_parameters):
        raise ValueError("replicated and sharded parameter lists overlap or contain duplicates")
    gradients = [parameter.grad for parameter in all_parameters if parameter.grad is not None]
    if not gradients:
        raise ValueError("no parameter has a gradient")
    device = gradients[0].device
    if any(gradient.device != device for gradient in gradients):
        raise ValueError("all gradients must reside on the same device")

    replicated_squared = torch.zeros((), device=device, dtype=torch.float32)
    for parameter in replicated:
        if parameter.grad is not None:
            replicated_squared += parameter.grad.detach().float().square().sum()
    sharded_squared = torch.zeros((), device=device, dtype=torch.float32)
    for parameter in sharded:
        if parameter.grad is not None:
            sharded_squared += parameter.grad.detach().float().square().sum()
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(sharded_squared, op=dist.ReduceOp.SUM)
    elif sharded and int(getattr(sharded[0], "_tp_size", 1)) > 1:
        raise RuntimeError("sharded gradient clipping requires an initialized process group")

    local_total_norm = torch.sqrt(replicated_squared + sharded_squared)
    total_norm = _validated_shared_tp_norm(replicated_squared, local_total_norm)
    if error_if_nonfinite and not bool(torch.isfinite(total_norm)):
        raise RuntimeError("the total TP gradient norm is non-finite")
    coefficient = torch.clamp(max_norm / (total_norm + 1e-6), max=1.0)
    for gradient in gradients:
        gradient.mul_(coefficient.to(dtype=gradient.dtype))
    return total_norm


def _validated_shared_tp_norm(replicated_squared: Tensor, total_norm: Tensor) -> Tensor:
    """Require TP ranks to agree on replica accounting and return one shared total norm."""

    if not (dist.is_available() and dist.is_initialized()):
        return total_norm
    local = torch.stack((replicated_squared, total_norm))
    observed = [torch.empty_like(local) for _ in range(dist.get_world_size())]
    dist.all_gather(observed, local)
    values = torch.stack(observed)
    replicated_values = values[:, 0]
    if not torch.allclose(
        replicated_values,
        replicated_values[0].expand_as(replicated_values),
        rtol=0.0,
        atol=0.0,
        equal_nan=True,
    ):
        raise RuntimeError(f"replicated gradient squared norm differs across TP ranks: {replicated_values.tolist()}")
    total_values = values[:, 1]
    if not torch.allclose(
        total_values,
        total_values[0].expand_as(total_values),
        rtol=0.0,
        atol=0.0,
        equal_nan=True,
    ):
        raise RuntimeError(f"total gradient norm differs across TP ranks: {total_values.tolist()}")
    return total_values[0]


def _isfinite(value: Tensor) -> bool:
    result = torch.isfinite(value).all()
    if hasattr(result, "full_tensor"):
        result = result.full_tensor()
    return bool(result.item())
