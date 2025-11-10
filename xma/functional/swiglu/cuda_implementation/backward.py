# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import torch
from torch.library import custom_op

import cutlass.cute as cute
from cutlass import Boolean, Float32, range_constexpr

from ....constants import LIBRARY_NAME
from ....cute_dsl_utils import LOG_WARP_SIZE, WARP_SIZE, sigmoid, torch_tensor_to_cute_tensor


@cute.kernel
def swiglu_backward_cuda_kernel(
    gG: cute.Tensor,
    gU: cute.Tensor,
    gY: cute.Tensor,
    gID: cute.Tensor,
    copy_atom: cute.CopyAtom,
    tiled_copy: cute.TiledCopy,
    shape: cute.Shape,
) -> None:
    BLOCK_ID, _, _ = cute.arch.block_idx()
    THREAD_ID, _, _ = cute.arch.thread_idx()

    block_coord = ((None, None), BLOCK_ID)

    bG = gG[block_coord]
    bU = gU[block_coord]
    bY = gY[block_coord]
    bID = gID[block_coord]

    thr_copy = tiled_copy.get_slice(THREAD_ID)

    tG = thr_copy.partition_S(bG)
    tU = thr_copy.partition_S(bU)
    tY = thr_copy.partition_D(bY)
    tID = thr_copy.partition_S(bID)

    fragG = cute.make_fragment_like(tG)
    fragU = cute.make_fragment_like(tU)
    fragY = cute.make_fragment_like(tY)

    fragID = cute.make_fragment(tID.shape, Boolean)
    for i in range_constexpr(cute.size(fragID)):
        fragID[i] = cute.elem_less(tID[i], shape)

    cute.copy(copy_atom, tG, fragG, pred=fragID)
    cute.copy(copy_atom, tU, fragU, pred=fragID)

    # convert rmem Tensor to TensorSSA
    g = fragG.load()
    u = fragU.load()

    dtype = g.dtype
    y = u * g * sigmoid(g, output_dtype=Float32)
    y = y.to(dtype)

    fragY.store(y)
    cute.copy(copy_atom, fragY, tY, pred=fragID)


@cute.jit
def swiglu_backward_cuda_jit(mG: cute.Tensor, mU: cute.Tensor, mY: cute.Tensor) -> None:
    BLOCK_SIZE = 1024
    vector_size = 128 // mG.element_type.width

    thr_layout = cute.make_ordered_layout((BLOCK_SIZE >> LOG_WARP_SIZE, WARP_SIZE), order=(1, 0))
    val_layout = cute.make_ordered_layout((1, vector_size), order=(1, 0))
    tiler_mn, tv_layout = cute.make_layout_tv(thr_layout, val_layout)

    mID = cute.make_identity_tensor(mG.shape)

    gG = cute.zipped_divide(mG, tiler_mn)
    gU = cute.zipped_divide(mU, tiler_mn)
    gY = cute.zipped_divide(mY, tiler_mn)
    gID = cute.zipped_divide(mID, tiler_mn)

    copy_atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), gG.element_type)
    tiled_copy = cute.make_tiled_copy_tv(copy_atom, thr_layout, val_layout)

    NUM_BLOCKS = cute.size(gG, mode=[1])

    kernel = swiglu_backward_cuda_kernel(gG, gU, gY, gID, copy_atom, tiled_copy, mG.shape)
    kernel.launch(grid=(NUM_BLOCKS, 1, 1), block=(BLOCK_SIZE, 1, 1))


@custom_op(f"{LIBRARY_NAME}::swiglu_backward_cuda", mutates_args={"output"})
def swiglu_backward_cuda(
    gate: torch.Tensor, up: torch.Tensor, output_grad: torch.Tensor, gate_grad: torch.Tensor, up_grad: torch.Tensor
) -> None:
    gate = torch_tensor_to_cute_tensor(gate, leading_dim=1)
    up = torch_tensor_to_cute_tensor(up, leading_dim=1)
    output_grad = torch_tensor_to_cute_tensor(output_grad, leading_dim=1)
    gate_grad = torch_tensor_to_cute_tensor(gate_grad, leading_dim=1)
    up_grad = torch_tensor_to_cute_tensor(up_grad, leading_dim=1)

    key = gate.element_type
    function = swiglu_backward_cuda.cache.get(key, None)

    if function is None:
        function = cute.compile(swiglu_backward_cuda_jit, gate, up, output_grad, gate_grad, up_grad)
        swiglu_backward_cuda.cache[key] = function

    function(gate, up, output_grad, gate_grad, up_grad)


swiglu_backward_cuda.cache = {}
