"""Triton grouped linear with composite groups and shared expert weights.

This module is intentionally imported lazily by the sample-isolation v2
backend.  Keeping the Triton dependency here lets lightweight Duo-VLA installs
continue to import the rest of the package without the training stack.

For an input whose rows have already been sorted by ``(sample_id,
expert_id)``, ``group_offsets`` describes ``B * E`` composite groups.  The
kernel maps composite group ``g`` back to expert ``g % E`` and reads that
expert's original weight tensor directly.  In particular, it never constructs
or expands a ``B * E`` weight tensor.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _shared_weight_grouped_mm_kernel(
    input_ptr,
    weight_ptr,
    group_offsets_ptr,
    output_ptr,
    input_row_stride: tl.constexpr,
    input_k_stride: tl.constexpr,
    weight_expert_stride: tl.constexpr,
    weight_k_stride: tl.constexpr,
    weight_n_stride: tl.constexpr,
    output_row_stride: tl.constexpr,
    output_n_stride: tl.constexpr,
    NUM_EXPERTS: tl.constexpr,
    K: tl.constexpr,
    N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Compute all row blocks for one composite group/output-column tile."""

    group_id = tl.program_id(axis=0)
    n_block = tl.program_id(axis=1)
    group_start = tl.load(group_offsets_ptr + group_id - 1, mask=group_id > 0, other=0)
    group_end = tl.load(group_offsets_ptr + group_id)
    expert_id = group_id % NUM_EXPERTS

    n_offsets = n_block * BLOCK_N + tl.arange(0, BLOCK_N)
    k_offsets = tl.arange(0, BLOCK_K)
    row_start = group_start
    while row_start < group_end:
        row_offsets = row_start + tl.arange(0, BLOCK_M)
        accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k_start in range(0, K, BLOCK_K):
            current_k = k_start + k_offsets
            input_block = tl.load(
                input_ptr
                + row_offsets[:, None] * input_row_stride
                + current_k[None, :] * input_k_stride,
                mask=(row_offsets[:, None] < group_end) & (current_k[None, :] < K),
                other=0.0,
            )
            weight_block = tl.load(
                weight_ptr
                + expert_id * weight_expert_stride
                + current_k[:, None] * weight_k_stride
                + n_offsets[None, :] * weight_n_stride,
                mask=(current_k[:, None] < K) & (n_offsets[None, :] < N),
                other=0.0,
            )
            accumulator += tl.dot(input_block, weight_block)

        tl.store(
            output_ptr
            + row_offsets[:, None] * output_row_stride
            + n_offsets[None, :] * output_n_stride,
            accumulator,
            mask=(row_offsets[:, None] < group_end) & (n_offsets[None, :] < N),
        )
        row_start += BLOCK_M


def _kernel_config(*, input_size: int, output_size: int) -> tuple[int, int, int, int]:
    """Return a conservative tensor-core configuration for the pinned shapes."""

    # The DiffusionGemma projections are wide enough to benefit from a 128-wide
    # output tile.  M=32 covers the usual per-(sample, expert) prefix occupancy
    # without wasting much work on the short decoder sequence.
    block_m = 32
    block_n = 128 if output_size >= 128 else triton.next_power_of_2(output_size)
    block_k = 32 if input_size >= 32 else triton.next_power_of_2(input_size)
    num_warps = 8 if block_n >= 128 else 4
    return block_m, block_n, block_k, num_warps


def shared_weight_grouped_mm(
    input: torch.Tensor,
    weight: torch.Tensor,
    group_offsets: torch.Tensor,
    *,
    weight_k_stride: int,
    weight_n_stride: int,
    output_size: int,
) -> torch.Tensor:
    """Run one shared-weight grouped matrix multiplication.

    ``weight`` owns only ``E`` matrices.  ``group_offsets`` may own ``B * E``
    groups; the kernel performs the group-to-expert modulo indirection.  The
    stride arguments make the same kernel usable for the forward projection
    and its input-gradient projection without transposing or copying weights.
    """

    if input.device.type != "cuda" or weight.device.type != "cuda" or group_offsets.device.type != "cuda":
        raise RuntimeError("shared-weight grouped-MM requires CUDA tensors")
    if input.device != weight.device or input.device != group_offsets.device:
        raise RuntimeError("shared-weight grouped-MM tensors must be on one CUDA device")
    if input.ndim != 2 or weight.ndim != 3 or group_offsets.ndim != 1:
        raise RuntimeError("shared-weight grouped-MM expects rank-2 input, rank-3 weight, and rank-1 offsets")
    if input.dtype != torch.bfloat16 or weight.dtype != torch.bfloat16:
        raise RuntimeError("shared-weight grouped-MM supports only bfloat16 input and weights")
    if group_offsets.dtype != torch.int32:
        raise RuntimeError("shared-weight grouped-MM offsets must be int32")
    if input.shape[0] <= 0 or input.shape[1] <= 0 or weight.shape[0] <= 0 or output_size <= 0:
        raise RuntimeError("shared-weight grouped-MM dimensions must be positive")
    if int(weight_k_stride) <= 0 or int(weight_n_stride) <= 0:
        raise RuntimeError("shared-weight grouped-MM weight strides must be positive")
    if int(group_offsets.shape[0]) % int(weight.shape[0]) != 0:
        raise RuntimeError("composite group count must be a multiple of the expert count")

    input_size = int(input.shape[1])
    output = torch.empty((int(input.shape[0]), int(output_size)), dtype=input.dtype, device=input.device)
    block_m, block_n, block_k, num_warps = _kernel_config(
        input_size=input_size,
        output_size=int(output_size),
    )
    grid = (int(group_offsets.shape[0]), triton.cdiv(int(output_size), block_n))
    _shared_weight_grouped_mm_kernel[grid](
        input,
        weight,
        group_offsets,
        output,
        input_row_stride=int(input.stride(0)),
        input_k_stride=int(input.stride(1)),
        weight_expert_stride=int(weight.stride(0)),
        weight_k_stride=int(weight_k_stride),
        weight_n_stride=int(weight_n_stride),
        output_row_stride=int(output.stride(0)),
        output_n_stride=int(output.stride(1)),
        NUM_EXPERTS=int(weight.shape[0]),
        K=input_size,
        N=int(output_size),
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        num_warps=num_warps,
        num_stages=3,
    )
    return output


__all__ = ["shared_weight_grouped_mm"]
