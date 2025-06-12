# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import torch
import triton
import triton.language as tl

from ....constants import LIBRARY_NAME
from ....math import ceil_divide, get_powers_of_2
from ....utils import cute_op
from .group_kernel import _get_autotune_configs


@triton.autotune(configs=_get_autotune_configs(), key=["H"])
@triton.jit
def group_with_padding_backward_triton_kernel(
    dy_ptr,
    expert_padding_offset_ptr,
    sorted_idxs_ptr,
    scattered_idxs_ptr,
    router_weights_ptr,
    router_weights_grad_ptr,
    dx_ptr,
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

    sorted_idxs = tl.load(sorted_idxs_ptr + indices_b, mask=mask_b)
    expert_padding_offset = tl.load(expert_padding_offset_ptr + sorted_idxs)

    scattered_idxs = tl.load(scattered_idxs_ptr + indices_b, mask=mask_b)

    dy_ptrs = dy_ptr + scattered_idxs[:, None] * H
    dx_ptrs = dx_ptr + (indices_b + expert_padding_offset)[:, None] * H

    NUM_BLOCKS_H = tl.cdiv(H, BLOCK_SIZE_H)
    for h in range(NUM_BLOCKS_H):
        indices_h = h * BLOCK_SIZE_H + tl.arange(0, BLOCK_SIZE_H)

        if h < NUM_BLOCKS_H - 1:
            x = tl.load(dy_ptrs + indices_h[None, :], mask=mask_b[:, None])
            tl.store(dx_ptrs + indices_h[None, :], x, mask=mask_b[:, None])
        else:
            mask_h = indices_h < H
            mask_bh = mask_b[:, None] & mask_h[None, :]

            x = tl.load(dy_ptrs + indices_h[None, :], mask=mask_bh)
            tl.store(dx_ptrs + indices_h[None, :], x, mask=mask_bh)


@cute_op(f"{LIBRARY_NAME}::group_with_padding_backward_triton", mutates_args={"router_weights_grad", "x_grad"})
def group_with_padding_backward_triton(
    output_grad: torch.Tensor,
    expert_padding_offset: torch.Tensor,
    sorted_idxs: torch.Tensor,
    scattered_idxs: torch.Tensor,
    router_weights: torch.Tensor,
    router_weights_grad: torch.Tensor,
    x_grad: torch.Tensor,
    T: int,
    H: int,
    K: int,
) -> None:
    GRID = lambda meta: (ceil_divide(T * K, meta["BLOCK_SIZE_B"]),)

    with torch.device(output_grad.device):
        group_with_padding_backward_triton_kernel[GRID](
            dy_ptr=output_grad,
            expert_padding_offset_ptr=expert_padding_offset,
            sorted_idxs_ptr=sorted_idxs,
            scattered_idxs_ptr=scattered_idxs,
            router_weights_ptr=router_weights,
            router_weights_grad_ptr=router_weights_grad,
            dx_ptr=x_grad,
            T=T,
            H=H,
            K=K,
        )
