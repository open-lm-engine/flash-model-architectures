# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import triton
import triton.language as tl


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
    BLOCK_SIZE_B: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
):
    BLOCK_ID = tl.program_id(axis=0)
    B = T * K

    indices_b = BLOCK_ID * BLOCK_SIZE_B + tl.arange(0, BLOCK_SIZE_B)
    mask_b = indices_b < B

    scattered_idxs = tl.load(scattered_idxs_ptr + indices_b, mask=mask_b)

    x_ptrs = x_ptr + indices_b[:, None] * H
    y_ptrs = y_ptr + scattered_idxs[:, None] * H

    if expert_padding_offset_ptr is not None:
        sorted_idxs = tl.load(sorted_idxs_ptr + indices_b, mask=mask_b)
        expert_padding_offset = tl.load(expert_padding_offset_ptr + sorted_idxs)

        x_ptrs += expert_padding_offset * H

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
