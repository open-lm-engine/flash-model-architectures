# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import torch
import triton
import triton.language as tl
from torch.library import custom_op

from ...constants import LIBRARY_NAME


@triton.jit
def _copy_array(source_ptr, destination_ptr, b, s, t, S, N, pack, BLOCK_SIZE):
    unpacked_offset = (b * S + s) * N
    packed_offset = t * N

    for i in range(tl.cdiv(N, BLOCK_SIZE)):
        indices = i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = indices < N

        if pack:
            source = tl.load(source_ptr + unpacked_offset + indices, mask=mask)
            tl.store(destination_ptr + packed_offset + indices, source, mask=mask)
        else:
            source = tl.load(source_ptr + packed_offset + indices, mask=mask)
            tl.store(destination_ptr + unpacked_offset + indices, source, mask=mask)


@triton.jit
def pack_unpack_sequence_triton_kernel(
    x_ptr,
    output_ptr,
    cu_seqlens_ptr,
    S,
    N,
    PADDING_SIDE: tl.constexpr,
    PACK: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    s = tl.program_id(axis=0)
    b = tl.program_id(axis=1)

    cu_seqlens_ptrs = cu_seqlens_ptr + b
    start = tl.load(cu_seqlens_ptrs)
    end = tl.load(cu_seqlens_ptrs + 1)
    seqlens = end - start

    if PADDING_SIDE == "left":
        pad_tokens = S - seqlens
        if s >= pad_tokens:
            _copy_array(x_ptr, output_ptr, b, s, start + s - pad_tokens, S, N, PACK, BLOCK_SIZE)
    else:
        if s < seqlens:
            _copy_array(x_ptr, output_ptr, b, s, start + s, S, N, PACK, BLOCK_SIZE)


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
            output_ptr=output,
            cu_seqlens_ptr=cu_seqlens,
            S=S,
            N=N,
            PADDING_SIDE=padding_side,
            PACK=pack,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=NUM_WARPS,
        )
