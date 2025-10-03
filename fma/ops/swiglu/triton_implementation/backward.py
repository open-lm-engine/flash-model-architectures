# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import torch
import triton
import triton.language as tl
from torch.library import custom_op

from ....constants import LIBRARY_NAME
from ....math import ceil_divide
from ....triton_math import sigmoid
from ....utils import get_num_elements_and_hidden_size


@triton.jit
def swiglu_backward_triton_kernel(
    g_ptr,
    g_stride_b,
    up_ptr,
    dy_ptr,
    dy_stride_b,
    dg_ptr,
    du_ptr,
    B,
    H,
    BLOCK_SIZE_B: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
):
    BLOCK_ID_B = tl.program_id(axis=0)
    BLOCK_ID_H = tl.program_id(axis=1)

    BLOCK_B = BLOCK_ID_B * BLOCK_SIZE_B + tl.arange(0, BLOCK_SIZE_B)
    BLOCK_H = BLOCK_ID_H * BLOCK_SIZE_H + tl.arange(0, BLOCK_SIZE_H)

    MASK_B = BLOCK_B < B
    MASK_H = BLOCK_H < H
    MASK = MASK_B[:, None] & MASK_H[None, :]

    indices_g = BLOCK_B[:, None] * g_stride_b + BLOCK_H[None, :]
    indices_output = BLOCK_B[:, None] * dy_stride_b + BLOCK_H[None, :]

    g = tl.load(g_ptr + indices_g, mask=MASK).to(tl.float32)
    up = tl.load(up_ptr + indices_g, mask=MASK)

    dy = tl.load(dy_ptr + indices_output, mask=MASK)

    g_sigmoid = sigmoid(g)
    g_silu = g * g_sigmoid

    dg = dy * up * (g_sigmoid + g_silu * (1 - g_sigmoid))
    du = dy * g_silu

    tl.store(dg_ptr + indices_g, dg, mask=MASK)
    tl.store(du_ptr + indices_g, du, mask=MASK)


@custom_op(f"{LIBRARY_NAME}::swiglu_backward_triton", mutates_args={"dg", "du"})
def swiglu_backward_triton(
    g: torch.Tensor,
    up: torch.Tensor,
    dy: torch.Tensor,
    dg: torch.Tensor,
    du: torch.Tensor,
) -> None:
    B, H = get_num_elements_and_hidden_size(g)
    BLOCK_SIZE_B = 64
    BLOCK_SIZE_H = 64

    with torch.device(g.device):
        swiglu_backward_triton_kernel[ceil_divide(B, BLOCK_SIZE_B), ceil_divide(H, BLOCK_SIZE_H)](
            g_ptr=g,
            g_stride_b=g.stride(-2),
            up_ptr=up,
            dy_ptr=dy,
            dy_stride_b=dy.stride(-2),
            dg_ptr=dg,
            du_ptr=du,
            B=B,
            H=H,
            BLOCK_SIZE_B=BLOCK_SIZE_B,
            BLOCK_SIZE_H=BLOCK_SIZE_H,
        )
