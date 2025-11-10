# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import torch
from torch.library import custom_op

import cutlass.cute as cute
from cutlass import Float32

from ....constants import LIBRARY_NAME
from ....cute_dsl_utils import LOG_WARP_SIZE, WARP_SIZE, sigmoid, torch_tensor_to_cute_tensor


@cute.kernel
def swiglu_forward_cuda_kernel(
    gG: cute.Tensor, gU: cute.Tensor, gY: cute.Tensor, gID: cute.Tensor, tv_layout: cute.Layout
) -> None:
    BLOCK_ID, _, _ = cute.arch.block_idx()
    THREAD_ID, _, _ = cute.arch.thread_idx()

    bG = gG[None, BLOCK_ID]
    bU = gU[None, BLOCK_ID]
    bY = gY[None, BLOCK_ID]

    tidfrgG = cute.composition(bG, tv_layout)
    tidfrgU = cute.composition(bU, tv_layout)
    tidfrgY = cute.composition(bY, tv_layout)

    g = tidfrgG[THREAD_ID, None].load()
    u = tidfrgU[THREAD_ID, None].load()

    dtype = g.dtype
    y = u * g * sigmoid(g, output_dtype=Float32)
    y = y.to(dtype)

    tidfrgY[THREAD_ID, None] = y


@cute.jit
def swiglu_forward_cuda_jit(mG: cute.Tensor, mU: cute.Tensor, mY: cute.Tensor) -> None:
    BLOCK_SIZE = 768
    vector_size = 128 // mG.element_type.width

    thr_layout = cute.make_ordered_layout((BLOCK_SIZE >> LOG_WARP_SIZE, WARP_SIZE), order=(1, 0))
    val_layout = cute.make_ordered_layout((1, vector_size), order=(1, 0))
    tiler_mn, tv_layout = cute.make_layout_tv(thr_layout, val_layout)

    mID = cute.make_identity_tensor(mG.shape)

    gG = cute.zipped_divide(mG, tiler_mn)
    gU = cute.zipped_divide(mU, tiler_mn)
    gY = cute.zipped_divide(mY, tiler_mn)
    gID = cute.zipped_divide(mID, tiler_mn)

    NUM_BLOCKS = cute.size(gG, mode=[1])

    kernel = swiglu_forward_cuda_kernel(gG, gU, gY, gID, tv_layout)
    kernel.launch(grid=(NUM_BLOCKS, 1, 1), block=(BLOCK_SIZE, 1, 1))


@custom_op(f"{LIBRARY_NAME}::swiglu_forward_cuda", mutates_args={"output"})
def swiglu_forward_cuda(gate: torch.Tensor, up: torch.Tensor, output: torch.Tensor) -> None:
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
