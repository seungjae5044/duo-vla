"""Opt-in CALVIN execution profiles; benchmark semantics remain unchanged.

Training keeps the canonical eight B8 RNG/data plans per global B64 update.
Physical coalescing and DP partitioning must never resample those plans.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

LEGACY_BACKEND = "sample_isolated_grouped_mm_v1"
FUSED_BACKEND = "sample_isolated_grouped_mm_v2"
CANONICAL_BATCH = 8
GLOBAL_BATCH = 64
SERVING_BATCH = 8
SUPPORTED_BATCHES = (8, 16, 32, 64)
DP2_PROFILE = "duovla-calvin-dp2-tp1-fused-v2-train-b32-serve-b8-v1"
EXTENDED_GEOMETRY_FIELDS = {"serving_batch_size", "data_parallel_size", "global_batch_size"}
OPTIMIZED_SOURCE_PATHS = (
    "scripts/run_calvin_train_dp2.sh",
    *(
        f"configs/calvin_abc_to_d{direct}_fused_v2_b{batch}.toml"
        for direct in ("", "_direct")
        for batch in SUPPORTED_BATCHES
    ),
    *(f"configs/calvin_abc_to_d{direct}_dp2_fused_v2_b32.toml" for direct in ("", "_direct")),
)


def single_profile(batch: int) -> str:
    if type(batch) is not int or batch not in SUPPORTED_BATCHES:
        raise ValueError("unsupported CALVIN physical batch")
    return f"duovla-calvin-tp1-fused-v2-train-b{batch}-serve-b8-v1"


@dataclass(frozen=True)
class CalvinExecution:
    backend: str
    physical_batch: int
    accumulation: int
    tensor_parallel_size: int
    data_parallel_size: int
    profile: str | None

    @property
    def optimized(self) -> bool:
        return self.backend == FUSED_BACKEND

    @property
    def world_size(self) -> int:
        return self.tensor_parallel_size * self.data_parallel_size


def execution_from_config(config: Mapping[str, Any], *, world_size: int | None = None) -> CalvinExecution:
    model, opt = config["model"], config["optimization"]
    backend = model.get("expert_batch_isolation")
    batch = opt.get("physical_batch_size")
    tp = model.get("tensor_parallel_size")
    if type(tp) is not int or tp not in (1, 2):
        raise ValueError("CALVIN tensor_parallel_size must be one or two")
    if type(batch) is not int or batch not in SUPPORTED_BATCHES:
        raise ValueError("CALVIN physical batch must be one of 8,16,32,64")
    profile = config.get("execution_profile")
    distributed = config.get("distributed")
    dp = 2 if profile == DP2_PROFILE else 1
    if backend == LEGACY_BACKEND:
        expected = "duovla-single-gpu-tp1-v1" if tp == 1 else None
        if batch != 8 or profile != expected or distributed is not None:
            raise ValueError("legacy CALVIN requires its original B8 topology/profile")
        if "serving_batch_size" in opt:
            raise ValueError("legacy CALVIN cannot declare optimized serving geometry")
    elif backend == FUSED_BACKEND:
        if tp != 1 or profile != (DP2_PROFILE if dp == 2 else single_profile(batch)):
            raise ValueError("fused CALVIN requires an explicit TP1 execution profile")
        if type(opt.get("serving_batch_size")) is not int or opt["serving_batch_size"] != SERVING_BATCH:
            raise ValueError("CALVIN optimized serving remains physical B8")
        if dp == 2:
            expected_dp = {
                "strategy": "data_parallel",
                "world_size": 2,
                "data_parallel_size": 2,
                "tensor_parallel_size": 1,
                "rank_physical_batch_size": 32,
                "canonical_plan_partition": "contiguous-b8-chunks-by-rank",
                "gradient_reduction": "sum_globally_normalized_sse_gradients",
            }
            if batch != 32 or distributed != expected_dp:
                raise ValueError("CALVIN DP2 requires rank B32, global B64 and explicit SUM reduction")
            for key in ("world_size", "data_parallel_size", "tensor_parallel_size", "rank_physical_batch_size"):
                if type(distributed[key]) is not int:
                    raise ValueError("CALVIN distributed sizes must be plain integers")
        elif distributed is not None:
            raise ValueError("single-GPU CALVIN must not declare a distributed strategy")
    else:
        raise ValueError("unsupported CALVIN expert isolation backend")
    accumulation = GLOBAL_BATCH // (batch * dp)
    for key, expected in {
        "microbatch_size": batch,
        "global_batch_size": GLOBAL_BATCH,
        "gradient_accumulation_steps": accumulation,
    }.items():
        if type(opt.get(key)) is not int or opt[key] != expected:
            raise ValueError(f"CALVIN optimization.{key} must equal {expected}")
    result = CalvinExecution(backend, batch, accumulation, tp, dp, profile)
    if world_size is not None and world_size != result.world_size:
        raise ValueError("CALVIN launched world size differs from configured TP/DP topology")
    return result


def execution_geometry(config: Mapping[str, Any]) -> dict[str, Any]:
    execution = execution_from_config(config)
    benchmark = config["benchmark"]
    geometry = {
        "expert_batch_isolation": execution.backend,
        "experts_implementation": config["model"]["experts_implementation"],
        "fixed_physical_prefix_width": benchmark["fixed_physical_prefix_width"],
        "physical_batch_size": execution.physical_batch,
        "prefix_geometry_content_sha256": benchmark["prefix_geometry_content_sha256"],
    }
    if execution.profile is not None:
        geometry.update(execution_profile=execution.profile, tensor_parallel_size=execution.tensor_parallel_size)
    if execution.optimized:
        geometry.update(
            serving_batch_size=SERVING_BATCH,
            data_parallel_size=execution.data_parallel_size,
            global_batch_size=GLOBAL_BATCH,
        )
    return geometry


def canonical_recipe_name(config: Mapping[str, Any], objective: str) -> str:
    execution = execution_from_config(config)
    if objective not in ("rectified_flow", "direct_regression"):
        raise ValueError("unsupported CALVIN objective")
    stem = "calvin_abc_to_d" + ("_direct" if objective == "direct_regression" else "")
    if execution.optimized:
        suffix = "_dp2_fused_v2_b32" if execution.data_parallel_size == 2 else f"_fused_v2_b{execution.physical_batch}"
    else:
        suffix = "_single_gpu" if execution.tensor_parallel_size == 1 else ""
    return stem + suffix + ".toml"


def expert_backend_functions(backend: str):
    if backend == LEGACY_BACKEND:
        from duo_vla.backbones.sample_isolated_experts import (
            install_sample_isolated_grouped_mm_experts,
            verify_sample_isolated_grouped_mm_experts,
        )

        return install_sample_isolated_grouped_mm_experts, verify_sample_isolated_grouped_mm_experts
    if backend == FUSED_BACKEND:
        from duo_vla.backbones.sample_isolated_experts_v2 import (
            install_sample_isolated_grouped_mm_experts_v2,
            verify_sample_isolated_grouped_mm_experts_v2,
        )

        return install_sample_isolated_grouped_mm_experts_v2, verify_sample_isolated_grouped_mm_experts_v2
    raise ValueError("unsupported CALVIN expert isolation backend")


def group_plans(plans, *, physical_batch: int, rank: int = 0, data_parallel_size: int = 1):
    """Partition before coalescing; rank order reconstructs the canonical stream."""
    values = tuple(plans)
    if type(physical_batch) is not int or physical_batch not in SUPPORTED_BATCHES:
        raise ValueError("unsupported physical batch")
    if type(data_parallel_size) is not int or data_parallel_size not in (1, 2):
        raise ValueError("unsupported data parallel size")
    if type(rank) is not int or not 0 <= rank < data_parallel_size:
        raise ValueError("invalid data-parallel rank")
    if not values or len(values) % data_parallel_size:
        raise ValueError("canonical plans do not partition evenly")
    local_count = len(values) // data_parallel_size
    local = values[rank * local_count : (rank + 1) * local_count]
    chunks = physical_batch // CANONICAL_BATCH
    if len(local) % chunks:
        raise ValueError("canonical plans do not form complete physical batches")
    return tuple(local[start : start + chunks] for start in range(0, len(local), chunks))


def canonical_training_pair(clean_actions, contract, plans):
    import torch

    from duo_vla.objectives import PolicyTrainingPair, make_seeded_policy_training_pair

    values = tuple(plans)
    if not values or clean_actions.shape[0] != len(values) * CANONICAL_BATCH:
        raise ValueError("objective inputs must match canonical B8 plans")
    pairs = [
        make_seeded_policy_training_pair(clean_actions[i * 8 : (i + 1) * 8], contract, seed=p.flow_seed)
        for i, p in enumerate(values)
    ]
    return PolicyTrainingPair(
        **{
            key: torch.cat([getattr(pair, key) for pair in pairs], dim=0)
            for key in ("input_actions", "timesteps", "target")
        }
    )


def sum_gradients_(named_parameters):
    """One SUM bucket, not DDP's average: loss already used the global valid count."""
    import torch
    import torch.distributed as dist

    entries = sorted(named_parameters, key=lambda entry: entry[0])
    if dist.get_world_size() != 2 or not entries:
        raise ValueError("CALVIN DP gradient reduction requires two ranks and trainables")
    if len({name for name, _ in entries}) != len(entries) or len({id(p) for _, p in entries}) != len(entries):
        raise ValueError("CALVIN DP trainables must be unique")
    gradients = [p.grad for _, p in entries]
    if any(g is None or g.is_sparse or g.layout != torch.strided for g in gradients):
        raise RuntimeError("CALVIN DP requires a dense gradient for every trainable")
    if any(g.dtype != torch.float32 or g.device != gradients[0].device for g in gradients):
        raise RuntimeError("CALVIN DP gradients must be FP32 on one local device")
    flat = torch.cat([g.reshape(-1) for g in gradients])
    dist.all_reduce(flat, op=dist.ReduceOp.SUM)
    offset = 0
    for gradient in gradients:
        gradient.copy_(flat.narrow(0, offset, gradient.numel()).view_as(gradient))
        offset += gradient.numel()


def reduce_scalar(value, *, device, dtype, maximum=False):
    import torch
    import torch.distributed as dist

    if dist.get_world_size() != 2:
        raise ValueError("CALVIN DP scalar reduction requires two ranks")
    tensor = torch.tensor(value, device=device, dtype=dtype)
    dist.all_reduce(tensor, op=dist.ReduceOp.MAX if maximum else dist.ReduceOp.SUM)
    return tensor.item()
