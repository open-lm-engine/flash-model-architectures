# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import torch
from torch.library import custom_op

import cutlass.cute as cute
from cutlass import Boolean, Float32, range_constexpr

from ....constants import LIBRARY_NAME, LOG_WARP_SIZE, WARP_SIZE
from ....cute_dsl_utils import sigmoid, torch_tensor_to_cute_tensor


@cute.kernel
def packed_swiglu_forward_cuda_kernel(
    gX: cute.Tensor,
    gY: cute.Tensor,
    gID: cute.Tensor,
    copy_atom: cute.CopyAtom,
    tiled_copy: cute.TiledCopy,
    shape: cute.Shape,
) -> None:
    BLOCK_ID, _, _ = cute.arch.block_idx()
    THREAD_ID, _, _ = cute.arch.thread_idx()

    block_coord = ((None, None), BLOCK_ID)

    bX = gX[block_coord]
    bY = gY[block_coord]
    bID = gID[block_coord]

    thr_copy = tiled_copy.get_slice(THREAD_ID)

    tX = thr_copy.partition_S(bX)
    tY = thr_copy.partition_D(bY)
    tID = thr_copy.partition_S(bID)

    fragX = cute.make_fragment_like(tX)
    fragY = cute.make_fragment_like(tY)

    fragID = cute.make_fragment(tID.shape, Boolean)
    for i in range_constexpr(cute.size(fragID)):
        fragID[i] = cute.elem_less(tID[i], shape)

    cute.copy(copy_atom, tX, fragX, pred=fragID)

    # convert rmem Tensor to TensorSSA
    x = fragX.load()

    dtype = x.dtype
    y = u * g * sigmoid(g, output_dtype=Float32)
    y = y.to(dtype)

    fragY.store(y)
    cute.copy(copy_atom, fragY, tY, pred=fragID)


@cute.jit
def packed_swiglu_forward_cuda_jit(mX: cute.Tensor, mY: cute.Tensor) -> None:
    BLOCK_SIZE = 1024
    vector_size = 128 // mX.element_type.width

    thr_layout = cute.make_ordered_layout((BLOCK_SIZE >> LOG_WARP_SIZE, WARP_SIZE), order=(1, 0))
    val_layout = cute.make_ordered_layout((1, vector_size), order=(1, 0))
    tiler_mn, tv_layout = cute.make_layout_tv(thr_layout, val_layout)

    gX = cute.zipped_divide(mX, tiler_mn)
    gY = cute.zipped_divide(mY, tiler_mn)

    mID = cute.make_identity_tensor(mX.shape)
    gID = cute.zipped_divide(mID, tiler_mn)

    copy_atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), gX.element_type)
    tiled_copy = cute.make_tiled_copy_tv(copy_atom, thr_layout, val_layout)

    NUM_BLOCKS = cute.size(gX, mode=[1])

    kernel = packed_swiglu_forward_cuda_kernel(
        gX=gX, gY=gY, gID=gID, copy_atom=copy_atom, tiled_copy=tiled_copy, shape=mX.shape
    )

    kernel.launch(grid=(NUM_BLOCKS, 1, 1), block=(BLOCK_SIZE, 1, 1))


@custom_op(f"{LIBRARY_NAME}::packed_swiglu_forward_cuda", mutates_args={"output"})
def packed_swiglu_forward_cuda(x: torch.Tensor, output: torch.Tensor) -> None:
    x = torch_tensor_to_cute_tensor(x, leading_dim=1)
    output = torch_tensor_to_cute_tensor(output, leading_dim=1)

    key = x.element_type
    function = packed_swiglu_forward_cuda.cache.get(key, None)

    if function is None:
        function = cute.compile(packed_swiglu_forward_cuda_jit, x, output)
        packed_swiglu_forward_cuda.cache[key] = function

    function(x, output)


packed_swiglu_forward_cuda.cache = {}
