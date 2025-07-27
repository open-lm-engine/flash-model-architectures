# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import torch
import triton
import triton.language as tl
from torch.library import custom_op

from ....constants import LIBRARY_NAME, MAX_TRITON_BLOCK_SIZE
from ....math import ceil_divide, get_next_power_of_2
from ....utils import get_num_elements_and_hidden_size


@triton.jit
def norm_2_forward_triton_kernel(
    x_ptr,
    weight_ptr,
    output_ptr,
    eps,
    p_norm_denominator_ptr,
    B,
    H,
    BLOCK_SIZE_B: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
):
    pid_b = tl.program_id(axis=0)

    indices_b = pid_b * BLOCK_SIZE_B + tl.arange(0, BLOCK_SIZE_B)
    indices_h = tl.arange(0, BLOCK_SIZE_H)
    indices_bh = indices_b[:, None] * H + indices_h[None, :]

    mask_b = indices_b < B
    mask_h = indices_h < H

    mask_bh = mask_b[:, None] & mask_h[None, :]

    x = tl.load(x_ptr + indices_bh, mask=mask_bh).to(tl.float32)

    r = x * x
    r = tl.sum(r, axis=1)
    # can't use rqsrt since we need to do a max(r, eps)
    r = tl.sqrt(r)
    r = max(r, eps)
    r = 1 / r

    if p_norm_denominator_ptr is not None:
        tl.store(p_norm_denominator_ptr + indices_b, r, mask=mask_b)

    x *= r[:, None]

    if weight_ptr is not None:
        weight = tl.load(weight_ptr + indices_h, mask=mask_h)
        x = x.to(x_ptr.dtype.element_ty) * weight[None, :]

    tl.store(output_ptr + indices_bh, x, mask=mask_bh)


@custom_op(f"{LIBRARY_NAME}::norm_2_forward_triton", mutates_args={"output", "p_norm_denominator"})
def norm_2_forward_triton(
    x: torch.Tensor,
    weight: torch.Tensor | None,
    output: torch.Tensor,
    eps: float,
    p_norm_denominator: torch.Tensor | None,
) -> None:
    B, H = get_num_elements_and_hidden_size(x)

    BLOCK_SIZE_B = 1
    BLOCK_SIZE_H = get_next_power_of_2(H)
    assert BLOCK_SIZE_H <= MAX_TRITON_BLOCK_SIZE
    NUM_WARPS = 8

    with torch.device(x.device):
        norm_2_forward_triton_kernel[ceil_divide(B, BLOCK_SIZE_B),](
            x_ptr=x,
            weight_ptr=weight,
            output_ptr=output,
            eps=eps,
            p_norm_denominator_ptr=p_norm_denominator,
            B=B,
            H=H,
            BLOCK_SIZE_B=BLOCK_SIZE_B,
            BLOCK_SIZE_H=BLOCK_SIZE_H,
            num_warps=NUM_WARPS,
        )
