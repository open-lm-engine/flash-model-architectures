# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import torch
import triton
import triton.language as tl
from torch.library import custom_op

from ....constants import LIBRARY_NAME
from ....math import ceil_divide, get_next_power_of_2
from ....triton_math import clamp, matmul, tanh_backward
from .backward import _get_autotune_configs


@triton.jit
def _load_input_state(
    h0_ptr,
    h0_stride,
    BLOCK_ID_N,
    BLOCK_B,
    BLOCK_H,
    mask_bh,
    BLOCK_SIZE_B,
    BLOCK_SIZE_H,
    dtype,
):
    if h0_ptr is None:
        y_prev = tl.zeros((BLOCK_SIZE_B, BLOCK_SIZE_H), dtype=dtype)
    else:
        y_ptrs = h0_ptr + BLOCK_B[:, None] * h0_stride[0] + BLOCK_ID_N * h0_stride[1] + BLOCK_H[None, :] * h0_stride[2]
        y_prev = tl.load(y_ptrs, mask=mask_bh)

    return y_prev


@triton.autotune(configs=_get_autotune_configs(), key=["BLOCK_SIZE_H"], reset_to_zero=["dW_ptr"])
@triton.jit
def rnn_varlen_backward_triton_kernel(
    W_ptr,
    W_stride,
    y_ptr,
    y_stride,
    h0_ptr,
    h0_stride,
    dy_ptr,
    cu_seqlens_ptr,
    IS_MAX_SEQLEN_TENSOR: tl.constexpr,
    max_seqlen_ptr,
    dx_ptr,
    dW_ptr,
    gradient_clipping,
    B,
    H,
    BLOCK_SIZE_B: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
):
    BLOCK_ID_B = tl.program_id(axis=0)
    BLOCK_ID_N = tl.program_id(axis=1)

    BLOCK_B = BLOCK_ID_B * BLOCK_SIZE_B + tl.arange(0, BLOCK_SIZE_B)
    BLOCK_H = tl.arange(0, BLOCK_SIZE_H)
    BLOCK_W = BLOCK_ID_N * W_stride[0] + BLOCK_H[:, None] * W_stride[1] + BLOCK_H[None, :] * W_stride[2]

    mask_b = BLOCK_B < B
    mask_h = BLOCK_H < H

    mask_bh = mask_b[:, None] & mask_h[None, :]
    mask_hh = mask_h[:, None] & mask_h[None, :]

    dh = tl.zeros((BLOCK_SIZE_B, BLOCK_SIZE_H), dtype=W_ptr.dtype.element_ty)
    dW = tl.zeros((BLOCK_SIZE_H, BLOCK_SIZE_H), dtype=tl.float32)

    W = tl.load(W_ptr + BLOCK_W, mask=mask_hh)

    cu_seqlens_ptrs = cu_seqlens_ptr + BLOCK_B[:, None]
    start = tl.load(cu_seqlens_ptrs, mask=mask_b[:, None])
    end = tl.load(cu_seqlens_ptrs + 1, mask=mask_b[:, None])

    if IS_MAX_SEQLEN_TENSOR:
        max_seqlen = tl.load(max_seqlen_ptr)
    else:
        max_seqlen = max_seqlen_ptr

    end -= 1

    BLOCK = end * y_stride[0] + BLOCK_ID_N * y_stride[1] + BLOCK_H[None, :] * y_stride[2]
    y = tl.load(y_ptr + BLOCK, mask=mask_bh)

    # backward counting reduces 1 instruction since we need to compare s == 0, otherwise we have to compare s == S - 1
    for _ in range(max_seqlen - 1, -1, -1):
        if gradient_clipping is not None:
            dh = clamp(dh, min_value=-gradient_clipping, max_value=gradient_clipping)

        unfinished = end >= start
        mask = unfinished & mask_h[None, :]

        dy = tl.load(dy_ptr + BLOCK, mask=mask) + dh

        dx_ptrs = dx_ptr + BLOCK
        BLOCK -= y_stride[0]

        y_prev = tl.where(
            start == end,
            _load_input_state(
                h0_ptr=h0_ptr,
                h0_stride=h0_stride,
                BLOCK_ID_N=BLOCK_ID_N,
                BLOCK_B=BLOCK_B,
                BLOCK_H=BLOCK_H,
                mask_bh=mask_bh,
                BLOCK_SIZE_B=BLOCK_SIZE_B,
                BLOCK_SIZE_H=BLOCK_SIZE_H,
                dtype=W.dtype,
            ),
            tl.load(y_ptr + BLOCK, mask=mask & (BLOCK >= 0)),
        )

        dx = dy * tanh_backward(y)
        dh = matmul(A=dx, B=W.T, C=None, output_dtype=dx.dtype)
        dW = matmul(A=y_prev.T, B=dx, C=dW, output_dtype=dW.dtype)

        tl.store(dx_ptrs, dx, mask=mask)
        y = y_prev

        end -= 1

    tl.atomic_add(dW_ptr + BLOCK_W, dW, mask=mask_hh, sem="relaxed")


@custom_op(f"{LIBRARY_NAME}::rnn_varlen_backward_triton", mutates_args={"input_grad", "weight_grad"})
def rnn_varlen_backward_triton(
    weight: torch.Tensor,
    output: torch.Tensor,
    input_state: torch.Tensor | None,
    output_grad: torch.Tensor,
    input_grad: torch.Tensor,
    weight_grad: torch.Tensor,
    cu_seqlens: torch.Tensor,
    max_seqlen_tensor: torch.Tensor | None,
    max_seqlen: int | None,
    gradient_clipping: float | None,
) -> None:
    _, N, H = output.size()
    B = cu_seqlens.size(0) - 1

    BLOCK_SIZE_H = get_next_power_of_2(H)
    BLOCK_SIZE_H = max(16, BLOCK_SIZE_H)
    GRID = lambda meta: (ceil_divide(B, meta["BLOCK_SIZE_B"]), N)

    is_max_seqlen_tensor = max_seqlen_tensor is not None

    with torch.device(output.device):
        rnn_varlen_backward_triton_kernel[GRID](
            W_ptr=weight,
            W_stride=weight.stride(),
            y_ptr=output,
            y_stride=output.stride(),
            h0_ptr=input_state,
            h0_stride=None if input_state is None else input_state.stride(),
            dy_ptr=output_grad,
            cu_seqlens_ptr=cu_seqlens,
            IS_MAX_SEQLEN_TENSOR=is_max_seqlen_tensor,
            max_seqlen_ptr=max_seqlen_tensor if is_max_seqlen_tensor else max_seqlen,
            dx_ptr=input_grad,
            dW_ptr=weight_grad,
            gradient_clipping=gradient_clipping,
            B=B,
            H=H,
            BLOCK_SIZE_H=BLOCK_SIZE_H,
        )
