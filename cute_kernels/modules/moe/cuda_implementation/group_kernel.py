# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import torch
import triton
import triton.language as tl
from torch.library import custom_op

from ....constants import LIBRARY_NAME
from ....math import ceil_divide, get_powers_of_2


def _get_autotune_configs() -> list[triton.Config]:
    configs = []
    for BLOCK_SIZE_B in get_powers_of_2(16, 128):
        for BLOCK_SIZE_H in get_powers_of_2(16, 128):
            for num_warps in get_powers_of_2(4, 8):
                configs.append(
                    triton.Config({"BLOCK_SIZE_B": BLOCK_SIZE_B, "BLOCK_SIZE_H": BLOCK_SIZE_H}, num_warps=num_warps)
                )

    return configs


@triton.autotune(configs=_get_autotune_configs(), key=["H"])
@triton.jit
def group_with_padding_triton_kernel(
    x_ptr,
    expert_padding_offset_ptr,
    sorted_idxs_ptr,
    scattered_idxs_ptr,
    y_ptr,
    T,
    H,
    K,
    NEEDS_DUPLICATION: tl.constexpr,
    BLOCK_SIZE_B: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
):
    BLOCK_ID = tl.program_id(axis=0)
    B = T * K

    indices_b = BLOCK_ID * BLOCK_SIZE_B + tl.arange(0, BLOCK_SIZE_B)
    mask_b = indices_b < B

    scattered_idxs = tl.load(scattered_idxs_ptr + indices_b, mask=mask_b)

    if NEEDS_DUPLICATION:
        x_ptrs = x_ptr + (scattered_idxs // K)[:, None] * H
    else:
        x_ptrs = x_ptr + scattered_idxs[:, None] * H

    y_ptrs = y_ptr + indices_b[:, None] * H

    if expert_padding_offset_ptr is not None:
        sorted_idxs = tl.load(sorted_idxs_ptr + indices_b, mask=mask_b)
        expert_padding_offset = tl.load(expert_padding_offset_ptr + sorted_idxs)

        y_ptrs += expert_padding_offset[:, None] * H

    NUM_BLOCKS_H = tl.cdiv(H, BLOCK_SIZE_H)
    for h in range(NUM_BLOCKS_H):
        indices_h = h * BLOCK_SIZE_H + tl.arange(0, BLOCK_SIZE_H)

        if h < NUM_BLOCKS_H - 1:
            x = tl.load(x_ptrs + indices_h[None, :], mask=mask_b[:, None])
            tl.store(y_ptrs + indices_h[None, :], x, mask=mask_b[:, None])
        else:
            mask_h = indices_h < H
            mask_bh = mask_b[:, None] & mask_h[None, :]

            x = tl.load(x_ptrs + indices_h[None, :], mask=mask_bh)
            tl.store(y_ptrs + indices_h[None, :], x, mask=mask_bh)


@custom_op(f"{LIBRARY_NAME}::group_with_padding_triton", mutates_args={"output"})
def group_with_padding_triton(
    x: torch.Tensor,
    expert_padding_offset: torch.Tensor,
    sorted_idxs: torch.Tensor,
    scattered_idxs: torch.Tensor,
    output: torch.Tensor,
    T: int,
    H: int,
    K: int,
    NEEDS_DUPLICATION: bool,
) -> None:
    GRID = lambda meta: (ceil_divide(T * K, meta["BLOCK_SIZE_B"]),)

    with torch.device(x.device):
        group_with_padding_triton_kernel[GRID](
            x_ptr=x,
            expert_padding_offset_ptr=expert_padding_offset,
            sorted_idxs_ptr=sorted_idxs,
            scattered_idxs_ptr=scattered_idxs,
            y_ptr=output,
            T=T,
            H=H,
            K=K,
            NEEDS_DUPLICATION=NEEDS_DUPLICATION,
        )
