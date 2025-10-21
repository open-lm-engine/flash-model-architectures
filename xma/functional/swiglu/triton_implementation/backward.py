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
    gate_ptr,
    gate_stride_b,
    up_ptr,
    output_grad_ptr,
    output_grad_stride_b,
    gate_grad_ptr,
    up_grad_ptr,
    B,
    H,
    BLOCK_SIZE_B: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
):
    BLOCK_ID_B = tl.program_id(axis=0)
    BLOCK_ID_H = tl.program_id(axis=1)

    indices_b = BLOCK_ID_B * BLOCK_SIZE_B + tl.arange(0, BLOCK_SIZE_B)
    indices_h = BLOCK_ID_H * BLOCK_SIZE_H + tl.arange(0, BLOCK_SIZE_H)

    mask_b = indices_b < B
    mask_h = indices_h < H
    mask = mask_b[:, None] & mask_h[None, :]

    indices_gate = indices_b[:, None] * gate_stride_b + indices_h[None, :]
    indices_output = indices_b[:, None] * output_grad_stride_b + indices_h[None, :]

    gate = tl.load(gate_ptr + indices_gate, mask=mask).to(tl.float32)
    up = tl.load(up_ptr + indices_gate, mask=mask)

    output_grad = tl.load(output_grad_ptr + indices_output, mask=mask)

    gate_sigmoid = sigmoid(gate)
    gate_silu = gate * gate_sigmoid

    gate_grad = output_grad * up * (gate_sigmoid + gate_silu * (1 - gate_sigmoid))
    up_grad = output_grad * gate_silu

    tl.store(gate_grad_ptr + indices_gate, gate_grad, mask=mask)
    tl.store(up_grad_ptr + indices_gate, up_grad, mask=mask)


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
            gate_ptr=gate,
            gate_stride_b=gate.stride(-2),
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
