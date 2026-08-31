#!/usr/bin/env python3
"""Validate two-GPU BF16 NCCL collectives and report a simple all-reduce timing."""

from __future__ import annotations

import json
import os
import time

import torch
import torch.distributed as dist


def main() -> None:
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    device = torch.device("cuda", local_rank)
    try:
        values = torch.full((32 * 1024 * 1024,), local_rank + 1, device=device, dtype=torch.bfloat16)
        for _ in range(3):
            dist.all_reduce(values)
            values.fill_(local_rank + 1)
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        iterations = 10
        for _ in range(iterations):
            values.fill_(local_rank + 1)
            dist.all_reduce(values)
        torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - started
        if not bool((values == 3).all()):
            raise AssertionError("NCCL all-reduce produced an incorrect result")
        dist.barrier()
        if dist.get_rank() == 0:
            print(
                json.dumps(
                    {
                        "bf16_supported": torch.cuda.is_bf16_supported(),
                        "bytes_per_tensor": values.numel() * values.element_size(),
                        "elapsed_seconds": elapsed,
                        "iterations": iterations,
                        "milliseconds_per_all_reduce": 1000 * elapsed / iterations,
                        "topology": "NODE (no NVLink)",
                        "world_size": dist.get_world_size(),
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
