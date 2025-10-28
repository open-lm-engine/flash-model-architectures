# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import torch
from torch.library import custom_op

import cutlass.cute as cute

from ....constants import LIBRARY_NAME


@cute.kernel
def swiglu_forward_cuda_kernel(gG: cute.Tensor, gU: cute.Tensor, gY: cute.Tensor) -> None:
    BLOCK_ID = cute.arch.block_idx()
    _THREAD_ID = cute.arch.thread_idx()
    BLOCK_DIM = cute.arch.block_dim()

    THREAD_ID = BLOCK_ID * BLOCK_DIM + _THREAD_ID

    N = gG.shape[1]

    row = THREAD_ID // N
    col = THREAD_ID % N

    g = gG[row, col]
    u = gU[row, col]

    gY[row, col] = u * g * sigmoid(g)


@cute.jit
def swiglu_forward_cuda_jit(mG: cute.Tensor, mU: cute.Tensor, mY: cute.Tensor, BLOCK_SIZE: int) -> None:
    M, N = mG.shape
    NUM_BLOCKS = (M * N + BLOCK_SIZE - 1) / BLOCK_SIZE

    kernel = swiglu_forward_cuda_kernel(mG, mU, mY)
    kernel.launch(grid=(NUM_BLOCKS, 1, 1), block=(BLOCK_SIZE, 1, 1))


@custom_op(f"{LIBRARY_NAME}::swiglu_forward_cuda", mutates_args={"output"})
def swiglu_forward_cuda(gate: torch.Tensor, up: torch.Tensor, output: torch.Tensor, BLOCK_SIZE: int) -> None:
    function = cute.compile(swiglu_forward_cuda_jit, gate, up, output, BLOCK_SIZE)
    function(gate, up, output, BLOCK_SIZE)
