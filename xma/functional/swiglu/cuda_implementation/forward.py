# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import torch
from torch.library import custom_op

import cutlass.cute as cute

from ....constants import LIBRARY_NAME
from ....cute_dsl_math import sigmoid
from ....cute_dsl_utils import torch_tensor_to_cute_tensor
from ....math import ceil_divide


@cute.kernel
def swiglu_forward_cuda_kernel(gG: cute.Tensor, gU: cute.Tensor, gY: cute.Tensor) -> None:
    BLOCK_ID, _, _ = cute.arch.block_idx()
    _THREAD_ID, _, _ = cute.arch.thread_idx()
    BLOCK_DIM, _, _ = cute.arch.block_dim()

    THREAD_ID = BLOCK_ID * BLOCK_DIM + _THREAD_ID

    N = gG.shape[1]

    row = THREAD_ID // N
    col = THREAD_ID % N

    g = gG[row, col]
    u = gU[row, col]

    dtype = g.dtype
    g = g.to(cute.Float32)

    y = u * g * sigmoid(g)
    y = y.to(dtype)

    gY[row, col] = y


@cute.jit
def swiglu_forward_cuda_jit(mG: cute.Tensor, mU: cute.Tensor, mY: cute.Tensor) -> None:
    BLOCK_SIZE = 1024

    M, N = mG.shape
    NUM_BLOCKS = ceil_divide(M * N, BLOCK_SIZE)

    kernel = swiglu_forward_cuda_kernel(mG, mU, mY)
    kernel.launch(grid=(NUM_BLOCKS, 1, 1), block=(BLOCK_SIZE, 1, 1))


@custom_op(f"{LIBRARY_NAME}::swiglu_forward_cuda", mutates_args={"output"})
def swiglu_forward_cuda(gate: torch.Tensor, up: torch.Tensor, output: torch.Tensor, BLOCK_SIZE: int) -> None:
    gate = torch_tensor_to_cute_tensor(gate, leading_dim=1)
    up = torch_tensor_to_cute_tensor(up, leading_dim=1)
    output = torch_tensor_to_cute_tensor(output, leading_dim=1)

    key = gate.element_type
    function = swiglu_forward_cuda.cache.get(key, None)

    if function is None:
        function = cute.compile(swiglu_forward_cuda_jit, gate, up, output)
        swiglu_forward_cuda.cache[key] = function

    function(gate, up, output)


swiglu_forward_cuda.cache = {}
