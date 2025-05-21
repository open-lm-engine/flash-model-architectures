# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import torch
import triton
import triton.language as tl

from ...constants import LIBRARY_NAME
from ...math import ceil_divide
from ...utils import cute_op


@triton.jit
def matrix_transpose_triton_kernel(x_ptr, y_ptr, M, N, BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr):
    BLOCK_ID_M = tl.program_id(axis=0)
    BLOCK_ID_N = tl.program_id(axis=1)

    indices_m = BLOCK_ID_M * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    indices_n = BLOCK_ID_N * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    mask_m = indices_m < M
    mask_n = indices_n < N

    indices = indices_m[:, None] * N & indices_n[None, :]
    mask = mask_m[:, None] & mask_n[None, :]

    x = tl.load(x_ptr + indices, mask=mask)

    indices = indices_n[:, None] * M + indices_m
    tl.store(y_ptr + indices, x, mask=mask.T)


@cute_op(f"{LIBRARY_NAME}::matrix_transpose_triton", mutates_args={"output"})
def matrix_transpose_triton(x: torch.Tensor, output: torch.Tensor) -> None:
    M, N = x.size()
    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 128

    with torch.device(x.device):
        matrix_transpose_triton_kernel[(ceil_divide(M, BLOCK_SIZE_M), ceil_divide(N, BLOCK_SIZE_N))](
            x_ptr=x, y_ptr=output, M=M, N=N, BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N
        )
