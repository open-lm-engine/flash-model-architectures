# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import torch
import triton
import triton.language as tl

from ....constants import LIBRARY_NAME
from ....math import ceil_divide
from ....utils import cute_op
from .group_kernel import _get_autotune_configs


@triton.autotune(configs=_get_autotune_configs(), key=["H"], reset_to_zero=["y_ptr"])
@triton.jit
def ungroup_with_padding_triton_kernel(
    x_ptr,
    expert_padding_offset_ptr,
    sorted_idxs_ptr,
    scattered_idxs_ptr,
    y_ptr,
    T,
    H,
    K,
    ATOMIC_ADD: tl.constexpr,
    BLOCK_SIZE_B: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
):
    BLOCK_ID = tl.program_id(axis=0)
    B = T * K

    indices_b = BLOCK_ID * BLOCK_SIZE_B + tl.arange(0, BLOCK_SIZE_B)
    mask_b = indices_b < B

    scattered_idxs = tl.load(scattered_idxs_ptr + indices_b, mask=mask_b)

    x_ptrs = x_ptr + indices_b[:, None] * H

    if ATOMIC_ADD:
        y_ptrs = y_ptr + (scattered_idxs // K)[:, None] * H
    else:
        y_ptrs = y_ptr + scattered_idxs[:, None] * H

    if expert_padding_offset_ptr is not None:
        sorted_idxs = tl.load(sorted_idxs_ptr + indices_b, mask=mask_b)
        expert_padding_offset = tl.load(expert_padding_offset_ptr + sorted_idxs)

        x_ptrs += expert_padding_offset[:, None] * H

    NUM_BLOCKS_H = tl.cdiv(H, BLOCK_SIZE_H)
    for h in range(NUM_BLOCKS_H):
        indices_h = h * BLOCK_SIZE_H + tl.arange(0, BLOCK_SIZE_H)

        if h < NUM_BLOCKS_H - 1:
            x = tl.load(x_ptrs + indices_h[None, :], mask=mask_b[:, None])

            if ATOMIC_ADD:
                tl.atomic_add(y_ptrs + indices_h[None, :], x, mask=mask_b[:, None])
            else:
                tl.store(y_ptrs + indices_h[None, :], x, mask=mask_b[:, None])
        else:
            mask_h = indices_h < H
            mask_bh = mask_b[:, None] & mask_h[None, :]

            x = tl.load(x_ptrs + indices_h[None, :], mask=mask_bh)

            if ATOMIC_ADD:
                tl.atomic_add(y_ptrs + indices_h[None, :], x, mask=mask_bh)
            else:
                tl.store(y_ptrs + indices_h[None, :], x, mask=mask_bh)


@cute_op(f"{LIBRARY_NAME}::ungroup_with_padding_triton", mutates_args={"output"})
def ungroup_with_padding_triton(
    x: torch.Tensor,
    expert_padding_offset: torch.Tensor,
    sorted_idxs: torch.Tensor,
    scattered_idxs: torch.Tensor,
    output: torch.Tensor,
    T: int,
    H: int,
    K: int,
    ATOMIC_ADD: bool,
) -> None:
    GRID = lambda meta: (ceil_divide(T * K, meta["BLOCK_SIZE_B"]),)

    with torch.device(x.device):
        ungroup_with_padding_triton_kernel[GRID](
            x_ptr=x,
            expert_padding_offset_ptr=expert_padding_offset,
            sorted_idxs_ptr=sorted_idxs,
            scattered_idxs_ptr=scattered_idxs,
            y_ptr=output,
            T=T,
            H=H,
            K=K,
            ATOMIC_ADD=ATOMIC_ADD,
        )
