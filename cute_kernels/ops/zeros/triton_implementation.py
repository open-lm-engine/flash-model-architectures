# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import torch
import triton
import triton.language as tl
from torch.library import custom_op

from ...constants import LIBRARY_NAME
from ...math import ceil_divide


@triton.jit
def fill_triton_kernel(x_ptr, fill_value, N, BLOCK_SIZE: tl.constexpr):
    NUM_BLOCKS = tl.num_programs(axis=0)
    BLOCK_ID = tl.program_id(axis=0)

    indices = BLOCK_ID * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    if BLOCK_ID < NUM_BLOCKS - 1:
        tl.store(x_ptr + indices, fill_value)
    else:
        tl.store(x_ptr + indices, fill_value, mask=indices < N)


@custom_op(f"{LIBRARY_NAME}::fill_triton", mutates_args={"x"})
def fill_triton(x: torch.Tensor, fill_value: float) -> None:
    BLOCK_SIZE = 4096
    NUM_WARPS = 32

    N = x.numel()

    with torch.device(x.device):
        fill_triton_kernel[ceil_divide(N, BLOCK_SIZE),](
            x_ptr=x, fill_value=fill_value, N=N, BLOCK_SIZE=BLOCK_SIZE, num_warps=NUM_WARPS
        )
