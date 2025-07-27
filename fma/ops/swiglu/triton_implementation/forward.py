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
def swiglu_forward_triton_kernel(
    gate_ptr,
    gate_stride_b,
    up_ptr,
    output_ptr,
    output_stride_b,
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

    indices = indices_b[:, None] * gate_stride_b + indices_h[None, :]

    gate = tl.load(gate_ptr + indices, mask=mask).to(tl.float32)
    up = tl.load(up_ptr + indices, mask=mask)

    output = up * gate * sigmoid(gate)

    indices = indices_b[:, None] * output_stride_b + indices_h[None, :]
    tl.store(output_ptr + indices, output, mask=mask)


@custom_op(f"{LIBRARY_NAME}::swiglu_forward_triton", mutates_args={"output"})
def swiglu_forward_triton(gate: torch.Tensor, up: torch.Tensor, output: torch.Tensor) -> None:
    B, H = get_num_elements_and_hidden_size(gate)
    BLOCK_SIZE_B = 64
    BLOCK_SIZE_H = 64

    # second last stride can be used to iterate the token dimension
    with torch.device(gate.device):
        swiglu_forward_triton_kernel[ceil_divide(B, BLOCK_SIZE_B), ceil_divide(H, BLOCK_SIZE_H)](
            gate_ptr=gate,
            gate_stride_b=gate.stride(-2),
            up_ptr=up,
            output_ptr=output,
            output_stride_b=output.stride(-2),
            B=B,
            H=H,
            BLOCK_SIZE_B=BLOCK_SIZE_B,
            BLOCK_SIZE_H=BLOCK_SIZE_H,
        )
