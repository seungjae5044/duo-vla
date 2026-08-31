import math
from pathlib import Path
from typing import Any

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn

from duo_vla.optimization import (
    OptimizationConfig,
    assert_replicated_parameter_values,
    assert_replicated_tensor,
    clip_tensor_parallel_grad_norm_,
    create_optimizer_and_scheduler,
    finite_gradient_audit,
    learning_rate_scale,
)


def test_schedule_has_exact_warmup_peak_and_cosine_endpoint() -> None:
    config = OptimizationConfig(total_updates=101, warmup_updates=10, final_learning_rate_scale=0.1)
    assert learning_rate_scale(0, config) == pytest.approx(1 / 11)
    assert learning_rate_scale(10, config) == pytest.approx(1.0)
    assert learning_rate_scale(55, config) == pytest.approx(0.55)
    assert learning_rate_scale(100, config) == pytest.approx(0.1)
    assert learning_rate_scale(1_000, config) == pytest.approx(0.1)


def test_scheduler_applies_final_scale_on_the_last_optimizer_step() -> None:
    lora = nn.Parameter(torch.ones(()))
    interface = nn.Parameter(torch.ones(()))
    config = OptimizationConfig(total_updates=4, warmup_updates=1, final_learning_rate_scale=0.1)
    optimizer, scheduler = create_optimizer_and_scheduler([lora], [interface], config)
    applied_scales: list[float] = []

    for _ in range(config.total_updates):
        applied_scales.append(optimizer.param_groups[0]["lr"] / config.lora_learning_rate)
        optimizer.step()
        scheduler.step()

    assert applied_scales == pytest.approx([0.5, 1.0, 0.55, 0.1])
    assert scheduler.last_epoch == config.total_updates
    assert scheduler.get_last_lr()[0] == pytest.approx(config.lora_learning_rate * 0.1)


def test_schedule_handles_one_update_and_rejects_a_peak_at_the_final_index() -> None:
    one_update = OptimizationConfig(total_updates=1, warmup_updates=0, final_learning_rate_scale=0.2)
    assert learning_rate_scale(0, one_update) == pytest.approx(0.2)
    assert learning_rate_scale(10, one_update) == pytest.approx(0.2)
    with pytest.raises(ValueError, match="peak must precede"):
        OptimizationConfig(total_updates=2, warmup_updates=1)
    with pytest.raises(ValueError, match="nonnegative integer"):
        learning_rate_scale(True, one_update)


def test_optimizer_keeps_named_lora_and_interface_learning_rates() -> None:
    lora = nn.Parameter(torch.ones(()))
    interface = nn.Parameter(torch.ones(()))
    config = OptimizationConfig(total_updates=10, warmup_updates=0)
    optimizer, scheduler = create_optimizer_and_scheduler([lora], [interface], config)
    assert [group["name"] for group in optimizer.param_groups] == ["lora", "interface"]
    assert optimizer.param_groups[0]["lr"] == pytest.approx(config.lora_learning_rate)
    assert optimizer.param_groups[1]["lr"] == pytest.approx(config.interface_learning_rate)
    optimizer.step()
    scheduler.step()
    expected_scale = learning_rate_scale(1, config)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(config.lora_learning_rate * expected_scale)
    assert optimizer.param_groups[1]["lr"] == pytest.approx(config.interface_learning_rate * expected_scale)


def test_optimizer_rejects_frozen_duplicate_and_overlapping_groups() -> None:
    parameter = nn.Parameter(torch.ones(()))
    frozen = nn.Parameter(torch.ones(()), requires_grad=False)
    config = OptimizationConfig()
    with pytest.raises(ValueError, match="frozen"):
        create_optimizer_and_scheduler([frozen], [parameter], config)
    with pytest.raises(ValueError, match="duplicate"):
        create_optimizer_and_scheduler([parameter, parameter], [nn.Parameter(torch.ones(()))], config)
    with pytest.raises(ValueError, match="overlap"):
        create_optimizer_and_scheduler([parameter], [parameter], config)


def test_gradient_audit_counts_missing_and_nonfinite() -> None:
    good = nn.Parameter(torch.ones(()))
    bad = nn.Parameter(torch.ones(()))
    missing = nn.Parameter(torch.ones(()))
    good.grad = torch.tensor(2.0)
    bad.grad = torch.tensor(math.inf)
    assert finite_gradient_audit([good, bad, missing]) == (1, 1)


def test_parameter_hash_is_order_independent_and_value_sensitive() -> None:
    first = nn.Parameter(torch.tensor([1.0, 2.0]))
    second = nn.Parameter(torch.tensor([3.0]))
    expected = assert_replicated_parameter_values([("first", first), ("second", second)])
    assert assert_replicated_parameter_values([("second", second), ("first", first)]) == expected
    second.data.add_(1)
    assert assert_replicated_parameter_values([("first", first), ("second", second)]) != expected
    assert len(assert_replicated_tensor("output", first)) == 64


def test_replicated_hashes_accept_scalar_tensors() -> None:
    scalar_parameter = nn.Parameter(torch.tensor(1.0, dtype=torch.float64))
    assert len(assert_replicated_parameter_values([("scalar", scalar_parameter)])) == 64
    assert len(assert_replicated_tensor("scalar", torch.tensor(2.0, dtype=torch.float64))) == 64


def test_tp_gradient_clip_counts_replicas_once_and_local_shards_once() -> None:
    replicated = nn.Parameter(torch.zeros(2))
    sharded = nn.Parameter(torch.zeros(1))
    replicated.grad = torch.tensor([3.0, 4.0])
    sharded.grad = torch.tensor([12.0])
    norm = clip_tensor_parallel_grad_norm_([replicated], [sharded], max_norm=1.0)
    assert float(norm) == pytest.approx(13.0)
    torch.testing.assert_close(replicated.grad, torch.tensor([3.0 / 13.0, 4.0 / 13.0]))
    torch.testing.assert_close(sharded.grad, torch.tensor([12.0 / 13.0]))


def _distributed_clip_worker(rank: int, world_size: int, init_path: str, queue: Any) -> None:
    dist.init_process_group(
        "gloo",
        init_method=Path(init_path).as_uri(),
        rank=rank,
        world_size=world_size,
    )
    try:
        replicated = nn.Parameter(torch.zeros(2))
        sharded = nn.Parameter(torch.zeros(1))
        replicated.grad = torch.tensor([3.0, 4.0])
        sharded.grad = torch.tensor([0.0 if rank == 0 else 12.0])
        norm = clip_tensor_parallel_grad_norm_([replicated], [sharded], max_norm=1.0)

        mismatched = nn.Parameter(torch.zeros(2))
        mismatched.grad = torch.tensor([3.0, 4.0]) if rank == 0 else torch.zeros(2)
        before = mismatched.grad.clone()
        error: str | None = None
        try:
            clip_tensor_parallel_grad_norm_([mismatched], [], max_norm=1.0)
        except RuntimeError as exc:
            error = str(exc)

        queue.put(
            {
                "rank": rank,
                "norm": float(norm),
                "replicated": replicated.grad.tolist(),
                "sharded": sharded.grad.tolist(),
                "mismatch_error": error,
                "mismatch_unchanged": torch.equal(mismatched.grad, before),
            }
        )
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(
    not dist.is_available() or not dist.is_gloo_available(),
    reason="the distributed clipping regression requires Gloo",
)
def test_tp_gradient_clip_uses_one_global_norm_and_rejects_replica_disagreement(tmp_path: Path) -> None:
    world_size = 2
    context = mp.get_context("spawn")
    queue = context.SimpleQueue()
    mp.spawn(
        _distributed_clip_worker,
        args=(world_size, str(tmp_path / "gradient-clip-init"), queue),
        nprocs=world_size,
        join=True,
    )
    results = sorted((queue.get() for _ in range(world_size)), key=lambda value: value["rank"])

    for result in results:
        assert result["norm"] == pytest.approx(13.0)
        assert result["replicated"] == pytest.approx([3.0 / 13.0, 4.0 / 13.0])
        expected_shard = [0.0] if result["rank"] == 0 else [12.0 / 13.0]
        assert result["sharded"] == pytest.approx(expected_shard)
        assert "replicated gradient squared norm differs" in result["mismatch_error"]
        assert result["mismatch_unchanged"]
