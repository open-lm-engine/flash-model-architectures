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
def matrix_transpose_triton_kernel(x_ptr, y_ptr, M, N, BLOCK_SIZE: tl.constexpr):
    BLOCK_ID_M = tl.program_id(axis=0)
    BLOCK_ID_N = tl.program_id(axis=1)

    indices_m = BLOCK_ID_M * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    indices_n = BLOCK_ID_N * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    mask_m = indices_m < M
    mask_n = indices_n < N

    indices = indices_m[:, None] * N + indices_n[None, :]
    mask = mask_m[:, None] & mask_n[None, :]

    x = tl.load(x_ptr + indices, mask=mask)

    indices = indices_n[:, None] * M + indices_m[None, :]
    tl.store(y_ptr + indices, x.T, mask=mask.T)


@custom_op(f"{LIBRARY_NAME}::matrix_transpose_triton", mutates_args={"output"})
def matrix_transpose_triton(x: torch.Tensor, output: torch.Tensor) -> None:
    M, N = x.size()
    NUM_WARPS = 32
    BLOCK_SIZE = 64

    with torch.device(x.device):
        matrix_transpose_triton_kernel[(ceil_divide(M, BLOCK_SIZE), ceil_divide(N, BLOCK_SIZE))](
            x_ptr=x, y_ptr=output, M=M, N=N, BLOCK_SIZE=BLOCK_SIZE, num_warps=NUM_WARPS
        )
