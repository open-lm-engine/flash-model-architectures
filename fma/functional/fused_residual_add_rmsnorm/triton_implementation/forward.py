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
def fused_residual_add_rmsnorm_forward_triton_kernel(
    x_ptr,
    x_stride,
    r_ptr,
    r_stride,
    W_ptr,
    W_stride,
    y_ptr,
    y_stride,
    xr_ptr,
    xr_stride,
    s_ptr,
    s_stride,
    eps,
    multiplier,
    B,
    H,
    BLOCK_SIZE_B: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
):
    BLOCK_ID_B = tl.program_id(axis=0)

    BLOCK_B = BLOCK_ID_B * BLOCK_SIZE_B + tl.arange(0, BLOCK_SIZE_B)
    BLOCK_H = tl.arange(0, BLOCK_SIZE_H)

    MASK_B = BLOCK_B < B
    MASK_H = BLOCK_H < H

    MASK_BH = MASK_B[:, None] & MASK_H[None, :]

    x = tl.load(x_ptr + BLOCK_B[:, None] * x_stride[0] + BLOCK_H[None, :] * x_stride[1], mask=MASK_BH).to(tl.float32)

    if multiplier is not None:
        x *= multiplier

    if r_ptr is not None:
        r = tl.load(r_ptr + BLOCK_B[:, None] * r_stride[0] + BLOCK_H[None, :] * r_stride[1], mask=MASK_BH)
        x += r

    if xr_ptr is not None:
        tl.store(xr_ptr + BLOCK_B[:, None] * xr_stride[0] + BLOCK_H[None, :] * xr_stride[1], x, mask=MASK_BH)

    r = tl.sum(x * x, axis=1)
    r = tl.rsqrt((r / H) + eps)

    if s_ptr is not None:
        tl.store(s_ptr + BLOCK_B * r_stride[0], r, mask=MASK_B)

    x *= r[:, None]

    if W_ptr is not None:
        W = tl.load(W_ptr + BLOCK_H * W_stride[0], mask=MASK_H)
        x = x.to(x_ptr.dtype.element_ty) * W[None, :]

    tl.store(y_ptr + BLOCK_B[:, None] * y_stride[0] + BLOCK_H[None, :] * y_stride[1], x, mask=MASK_BH)


@custom_op(
    f"{LIBRARY_NAME}::fused_residual_add_rmsnorm_forward_triton",
    mutates_args={"output", "added_x_residual", "rmsnorm_denominator"},
)
def fused_residual_add_rmsnorm_forward_triton(
    x: torch.Tensor,
    residual: torch.Tensor | None,
    weight: torch.Tensor | None,
    output: torch.Tensor,
    eps: float,
    multiplier: float | None,
    added_x_residual: torch.Tensor | None,
    rmsnorm_denominator: torch.Tensor | None,
) -> None:
    B, H = get_num_elements_and_hidden_size(x)

    BLOCK_SIZE_B = 1
    BLOCK_SIZE_H = get_next_power_of_2(H)
    assert BLOCK_SIZE_H <= MAX_TRITON_BLOCK_SIZE
    NUM_WARPS = 8

    with torch.device(x.device):
        fused_residual_add_rmsnorm_forward_triton_kernel[ceil_divide(B, BLOCK_SIZE_B),](
            x_ptr=x,
            x_stride=x.stride(),
            r_ptr=residual,
            r_stride=residual.stride(),
            W_ptr=weight,
            W_stride=weight.stride(),
            y_ptr=output,
            y_stride=output.stride(),
            xr_ptr=added_x_residual,
            xr_stride=added_x_residual.stride(),
            s_ptr=rmsnorm_denominator,
            s_stride=rmsnorm_denominator.stride(),
            eps=eps,
            multiplier=multiplier,
            B=B,
            H=H,
            BLOCK_SIZE_B=BLOCK_SIZE_B,
            BLOCK_SIZE_H=BLOCK_SIZE_H,
            num_warps=NUM_WARPS,
        )
