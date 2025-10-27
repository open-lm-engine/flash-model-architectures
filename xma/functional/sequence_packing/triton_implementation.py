# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import torch
import triton
import triton.language as tl
from torch.library import custom_op

from ...constants import LIBRARY_NAME


@triton.jit
def _copy_array(x_ptr, x_stride, y_ptr, y_stride, BLOCK_ID_B, BLOCK_ID_S, t, S, N, pack, BLOCK_SIZE):
    unpacked_offset = (BLOCK_ID_B * S + BLOCK_ID_S) * N
    packed_offset = t * N

    for i in range(tl.cdiv(N, BLOCK_SIZE)):
        indices = i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = indices < N

        if pack:
            source = tl.load(x_ptr + unpacked_offset + indices, mask=mask)
            tl.store(y_ptr + packed_offset + indices, source, mask=mask)
        else:
            source = tl.load(x_ptr + packed_offset + indices, mask=mask)
            tl.store(y_ptr + unpacked_offset + indices, source, mask=mask)


@triton.jit
def pack_unpack_sequence_triton_kernel(
    x_ptr,
    x_stride,
    y_ptr,
    y_stride,
    cu_seqlens_ptr,
    cu_seqlens_stride,
    S,
    N,
    PADDING_SIDE: tl.constexpr,
    PACK: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    BLOCK_ID_S = tl.program_id(axis=0)
    BLOCK_ID_B = tl.program_id(axis=1)

    cu_seqlens_ptrs = cu_seqlens_ptr + BLOCK_ID_B * cu_seqlens_stride[0]
    start = tl.load(cu_seqlens_ptrs)
    end = tl.load(cu_seqlens_ptrs + cu_seqlens_stride[0])
    seqlens = end - start

    pad_tokens = (S - seqlens) if PADDING_SIDE == "left" else 0

    if PADDING_SIDE == "left":
        if BLOCK_ID_S >= pad_tokens:
            _copy_array(
                x_ptr=x_ptr,
                x_stride=x_stride,
                y_ptr=y_ptr,
                y_stride=y_stride,
                BLOCK_ID_B=BLOCK_ID_B,
                BLOCK_ID_S=BLOCK_ID_S,
                t=start + BLOCK_ID_S - pad_tokens,
                S=S,
                N=N,
                pack=PACK,
                BLOCK_SIZE=BLOCK_SIZE,
            )
    else:
        if BLOCK_ID_S < seqlens:
            _copy_array(
                x_ptr=x_ptr,
                x_stride=x_stride,
                y_ptr=y_ptr,
                y_stride=y_stride,
                BLOCK_ID_B=BLOCK_ID_B,
                BLOCK_ID_S=BLOCK_ID_S,
                t=start + BLOCK_ID_S - pad_tokens,
                S=S,
                N=N,
                pack=PACK,
                BLOCK_SIZE=BLOCK_SIZE,
            )


@custom_op(f"{LIBRARY_NAME}::pack_unpack_sequence_triton", mutates_args={"output"})
def pack_unpack_sequence_triton(
    x: torch.Tensor, output: torch.Tensor, cu_seqlens: torch.Tensor, padding_side: str, pack: bool
) -> None:
    if pack:
        B, S = x.size()[:2]
        N = x.numel() // (B * S)
    else:
        B, S = output.size()[:2]
        N = output.numel() // (B * S)

    BLOCK_SIZE = 4096
    NUM_WARPS = 32

    with torch.device(x.device):
        pack_unpack_sequence_triton_kernel[S, B](
            x_ptr=x,
            x_stride=x.stride(),
            y_ptr=output,
            y_stride=output.stride(),
            cu_seqlens_ptr=cu_seqlens,
            cu_seqlens_stride=cu_seqlens.stride(),
            S=S,
            N=N,
            PADDING_SIDE=padding_side,
            PACK=pack,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=NUM_WARPS,
        )
