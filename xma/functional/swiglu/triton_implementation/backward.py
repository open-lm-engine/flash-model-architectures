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
    gate_grad_ptr,
    up_grad_ptr,
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

    output_grad = tl.load(dy_ptr + BLOCK_B[:, None] * dy_stride[0] + BLOCK_H[None, :] * dy_stride[1], mask=MASK)

    g_sigmoid = sigmoid(g)
    g_silu = g * g_sigmoid

    gate_grad = output_grad * u * (g_sigmoid + g_silu * (1 - g_sigmoid))
    up_grad = output_grad * g_silu

    tl.store(gate_grad_ptr + BLOCK_B[:, None] * g_stride[0] + BLOCK_H[None, :] * g_stride[1], gate_grad, mask=MASK)
    tl.store(up_grad_ptr + BLOCK_B[:, None] * g_stride[0] + BLOCK_H[None, :] * g_stride[1], up_grad, mask=MASK)


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
            output_grad_ptr=output_grad,
            output_grad_stride_b=output_grad.stride(-2),
            gate_grad_ptr=gate_grad,
            up_grad_ptr=up_grad,
            B=B,
            H=H,
            BLOCK_SIZE_B=BLOCK_SIZE_B,
            BLOCK_SIZE_H=BLOCK_SIZE_H,
        )
