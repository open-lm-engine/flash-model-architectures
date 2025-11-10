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
def swiglu_forward_triton_kernel(
    g_ptr,
    g_stride,
    u_ptr,
    u_stride,
    y_ptr,
    y_stride,
    B,
    H,
    BLOCK_SIZE_B: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
):
    BLOCK_ID_B = tl.program_id(axis=0)
    BLOCK_ID_H = tl.program_id(axis=1)

    BLOCK_B = BLOCK_ID_B * BLOCK_SIZE_B + tl.arange(0, BLOCK_SIZE_B)
    BLOCK_H = BLOCK_ID_H * BLOCK_SIZE_H + tl.arange(0, BLOCK_SIZE_H)

    mask_b = BLOCK_B < B
    mask_h = BLOCK_H < H
    mask = mask_b[:, None] & mask_h[None, :]

    g = tl.load(g_ptr + BLOCK_B[:, None] * g_stride[0] + BLOCK_H[None, :] * g_stride[1], mask=mask).to(tl.float32)
    u = tl.load(u_ptr + BLOCK_B[:, None] * u_stride[0] + BLOCK_H[None, :] * u_stride[1], mask=mask)

    y = u * g * sigmoid(g)

    tl.store(y_ptr + BLOCK_B[:, None] * y_stride[0] + BLOCK_H[None, :] * y_stride[1], y, mask=mask)


@custom_op(f"{LIBRARY_NAME}::swiglu_forward_triton", mutates_args={"output"})
def swiglu_forward_triton(gate: torch.Tensor, up: torch.Tensor, output: torch.Tensor) -> None:
    B, H = get_num_elements_and_hidden_size(gate)
    BLOCK_SIZE_B = 64
    BLOCK_SIZE_H = 64

    # second last stride can be used to iterate the token dimension
    with torch.device(gate.device):
        swiglu_forward_triton_kernel[ceil_divide(B, BLOCK_SIZE_B), ceil_divide(H, BLOCK_SIZE_H)](
            g_ptr=gate,
            g_stride=gate.stride(),
            u_ptr=up,
            u_stride=up.stride(),
            y_ptr=output,
            y_stride=output.stride(),
            B=B,
            H=H,
            BLOCK_SIZE_B=BLOCK_SIZE_B,
            BLOCK_SIZE_H=BLOCK_SIZE_H,
        )
