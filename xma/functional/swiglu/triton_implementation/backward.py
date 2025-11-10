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
    up_ptr,
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

    indices_gate = BLOCK_B[:, None] * g_stride[0] + BLOCK_H[None, :] * g_stride[1]
    indices_output = BLOCK_B[:, None] * dy_stride[0] + BLOCK_H[None, :] * dy_stride[1]

    gate = tl.load(g_ptr + indices_gate, mask=MASK).to(tl.float32)
    up = tl.load(up_ptr + indices_gate, mask=MASK)

    output_grad = tl.load(dy_ptr + indices_output, mask=MASK)

    gate_sigmoid = sigmoid(gate)
    gate_silu = gate * gate_sigmoid

    gate_grad = output_grad * up * (gate_sigmoid + gate_silu * (1 - gate_sigmoid))
    up_grad = output_grad * gate_silu

    tl.store(gate_grad_ptr + indices_gate, gate_grad, mask=MASK)
    tl.store(up_grad_ptr + indices_gate, up_grad, mask=MASK)


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
            up_ptr=up,
            output_grad_ptr=output_grad,
            output_grad_stride_b=output_grad.stride(-2),
            gate_grad_ptr=gate_grad,
            up_grad_ptr=up_grad,
            B=B,
            H=H,
            BLOCK_SIZE_B=BLOCK_SIZE_B,
            BLOCK_SIZE_H=BLOCK_SIZE_H,
        )
