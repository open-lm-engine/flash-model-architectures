# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import torch
import triton
import triton.language as tl
from torch.library import custom_op

from ....constants import LIBRARY_NAME
from ....math import ceil_divide
from ....triton_utils import sigmoid
from ....utils import get_num_elements_and_hidden_size


@triton.jit
def swiglu_backward_triton_kernel(
    g_ptr,
    g_stride,
    u_ptr,
    u_stride,
    dy_ptr,
    dy_stride,
    dg_ptr,
    dg_stride,
    du_ptr,
    du_stride,
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

    g = tl.load(g_ptr + BLOCK_B[:, None] * g_stride[0] + BLOCK_H[None, :] * g_stride[1], mask=MASK).to(tl.float32)
    u = tl.load(u_ptr + BLOCK_B[:, None] * u_stride[0] + BLOCK_H[None, :] * u_stride[1], mask=MASK)

    dy = tl.load(dy_ptr + BLOCK_B[:, None] * dy_stride[0] + BLOCK_H[None, :] * dy_stride[1], mask=MASK)

    g_sigmoid = sigmoid(g)
    g_silu = g * g_sigmoid

    dg = dy * u * (g_sigmoid + g_silu * (1 - g_sigmoid))
    du = dy * g_silu

    tl.store(dg_ptr + BLOCK_B[:, None] * dg_stride[0] + BLOCK_H[None, :] * dg_stride[1], dg, mask=MASK)
    tl.store(du_ptr + BLOCK_B[:, None] * du_stride[0] + BLOCK_H[None, :] * du_stride[1], du, mask=MASK)


@custom_op(f"{LIBRARY_NAME}::swiglu_backward_triton", mutates_args={"gate_grad", "up_grad"})
def swiglu_backward_triton(
    gate: torch.Tensor,
    up: torch.Tensor,
    output_grad: torch.Tensor,
    gate_grad: torch.Tensor,
    up_grad: torch.Tensor,
) -> None:
    B, H = get_num_elements_and_hidden_size(gate)
    BLOCK_SIZE_B = 64
    BLOCK_SIZE_H = 64

    with torch.device(gate.device):
        swiglu_backward_triton_kernel[ceil_divide(B, BLOCK_SIZE_B), ceil_divide(H, BLOCK_SIZE_H)](
            g_ptr=gate,
            g_stride=gate.stride(),
            u_ptr=up,
            u_stride=up.stride(),
            dy_ptr=output_grad,
            dy_stride=output_grad.stride(),
            dg_ptr=gate_grad,
            dg_stride=gate_grad.stride(),
            du_ptr=up_grad,
            du_stride=up_grad.stride(),
            B=B,
            H=H,
            BLOCK_SIZE_B=BLOCK_SIZE_B,
            BLOCK_SIZE_H=BLOCK_SIZE_H,
        )
