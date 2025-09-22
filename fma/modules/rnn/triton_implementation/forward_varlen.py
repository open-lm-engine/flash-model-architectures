# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import torch
import triton
import triton.language as tl
from torch.library import custom_op

from ....constants import LIBRARY_NAME
from ....math import ceil_divide, get_next_power_of_2
from ....triton_math import matmul, tanh
from .forward import _get_autotune_configs


@triton.autotune(configs=_get_autotune_configs(), key=["BLOCK_SIZE_H"])
@triton.jit
def rnn_varlen_forward_triton_kernel(
    x_ptr,
    x_stride,
    W_ptr,
    W_stride,
    h0_ptr,
    h0_stride,
    y_ptr,
    cu_seqlens_ptr,
    IS_MAX_SEQLEN_TENSOR: tl.constexpr,
    max_seqlen_ptr,
    B,
    H,
    BLOCK_SIZE_B: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
):
    BLOCK_ID_B = tl.program_id(axis=0)
    BLOCK_ID_N = tl.program_id(axis=1)

    BLOCK_B = BLOCK_ID_B * BLOCK_SIZE_B + tl.arange(0, BLOCK_SIZE_B)
    BLOCK_H = tl.arange(0, BLOCK_SIZE_H)

    mask_b = BLOCK_B < B
    mask_h = BLOCK_H < H

    mask_bh = mask_b[:, None] & mask_h[None, :]

    W = tl.load(
        W_ptr + BLOCK_ID_N * W_stride[0] + BLOCK_H[:, None] * W_stride[1] + BLOCK_H[None, :] * W_stride[2],
        mask=mask_h[:, None] & mask_h[None, :],
    )

    if h0_ptr is None:
        h = tl.zeros((BLOCK_SIZE_B, BLOCK_SIZE_H), dtype=x_ptr.dtype.element_ty)
    else:
        h = tl.load(
            h0_ptr + BLOCK_B[:, None] * h0_stride[0] + BLOCK_ID_N * h0_stride[1] + BLOCK_H[None, :] * h0_stride[2],
            mask=mask_bh,
        )

    cu_seqlens_ptrs = cu_seqlens_ptr + BLOCK_B[:, None]
    start = tl.load(cu_seqlens_ptrs, mask=mask_b[:, None])
    end = tl.load(cu_seqlens_ptrs + 1, mask=mask_b[:, None])

    if IS_MAX_SEQLEN_TENSOR:
        max_seqlen = tl.load(max_seqlen_ptr)
    else:
        max_seqlen = max_seqlen_ptr

    BLOCK = start * x_stride[0] + BLOCK_ID_N * x_stride[1] + BLOCK_H[None, :] * x_stride[2]

    for _ in range(max_seqlen):
        unfinished = start < end
        mask = unfinished & mask_h[None, :]

        x = tl.load(x_ptr + BLOCK, mask=mask_bh)
        h = matmul(A=h, B=W, C=x, output_dtype=tl.float32)
        h = tanh(h, output_dtype=x.dtype)
        tl.store(y_ptr + BLOCK, h, mask=mask)

        BLOCK += x_stride[0]
        start += 1


@custom_op(f"{LIBRARY_NAME}::rnn_varlen_forward_triton", mutates_args={"output"})
def rnn_varlen_forward_triton(
    input: torch.Tensor,
    weight: torch.Tensor,
    input_state: torch.Tensor | None,
    output: torch.Tensor,
    cu_seqlens: torch.Tensor,
    max_seqlen_tensor: torch.Tensor | None,
    max_seqlen: int | None,
) -> None:
    B = cu_seqlens.size(0) - 1
    _, N, H = input.size()

    BLOCK_SIZE_H = get_next_power_of_2(H)
    BLOCK_SIZE_H = max(16, BLOCK_SIZE_H)
    GRID = lambda meta: (ceil_divide(B, meta["BLOCK_SIZE_B"]), N)

    is_max_seqlen_tensor = max_seqlen_tensor is not None

    with torch.device(input.device):
        rnn_varlen_forward_triton_kernel[GRID](
            x_ptr=input,
            x_stride=input.stride(),
            W_ptr=weight,
            W_stride=weight.stride(),
            h0_ptr=input_state,
            h0_stride=None if input_state is None else input_state.stride(),
            y_ptr=output,
            cu_seqlens_ptr=cu_seqlens,
            IS_MAX_SEQLEN_TENSOR=is_max_seqlen_tensor,
            max_seqlen_ptr=max_seqlen_tensor if is_max_seqlen_tensor else max_seqlen,
            B=B,
            H=H,
            BLOCK_SIZE_H=BLOCK_SIZE_H,
        )
