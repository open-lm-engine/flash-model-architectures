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
from .forward import _get_autotune_configs


@triton.jit
def _load_input_state(
    h0_ptr,
    h0_stride,
    BLOCK_ID_N,
    BLOCK_B,
    BLOCK_H,
    MASK_BH,
    BLOCK_SIZE_B,
    BLOCK_SIZE_H,
    dtype,
):
    if h0_ptr is None:
        y_prev = tl.zeros((BLOCK_SIZE_B, BLOCK_SIZE_H), dtype=dtype)
    else:
        y_ptrs = h0_ptr + BLOCK_B[:, None] * h0_stride[0] + BLOCK_ID_N * h0_stride[1] + BLOCK_H[None, :] * h0_stride[2]
        y_prev = tl.load(y_ptrs, mask=MASK_BH)

    return y_prev


@triton.autotune(configs=_get_autotune_configs(), key=["BLOCK_SIZE_H"], reset_to_zero=["dW_ptr"])
@triton.jit
def rnn_backward_triton_kernel(
    W_ptr,
    W_stride,
    h0_ptr,
    h0_stride,
    y_ptr,
    y_stride,
    dx_ptr,
    dx_stride,
    dW_ptr,
    dW_stride,
    dy_ptr,
    dy_stride,
    cu_seqlens_ptr,
    cu_seqlens_stride,
    IS_MAX_SEQLEN_TENSOR: tl.constexpr,
    max_seqlen_ptr,
    B,
    S,
    H,
    gradient_clipping,
    BLOCK_SIZE_B: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
):
    BLOCK_ID_B = tl.program_id(axis=0)
    BLOCK_ID_N = tl.program_id(axis=1)

    BLOCK_B = BLOCK_ID_B * BLOCK_SIZE_B + tl.arange(0, BLOCK_SIZE_B)
    BLOCK_H = tl.arange(0, BLOCK_SIZE_H)

    MASK_B = BLOCK_B < B
    MASK_H = BLOCK_H < H

    MASK_BH = MASK_B[:, None] & MASK_H[None, :]
    MASK_HH = MASK_H[:, None] & MASK_H[None, :]

    dh = tl.zeros((BLOCK_SIZE_B, BLOCK_SIZE_H), dtype=W_ptr.dtype.element_ty)
    dW = tl.zeros((BLOCK_SIZE_H, BLOCK_SIZE_H), dtype=tl.float32)

    W = tl.load(
        W_ptr + BLOCK_ID_N * W_stride[0] + BLOCK_H[:, None] * W_stride[1] + BLOCK_H[None, :] * W_stride[2],
        mask=MASK_HH,
    )

    IS_VARLEN: tl.constexpr = cu_seqlens_ptr is not None

    if IS_VARLEN:
        cu_seqlens_ptrs = cu_seqlens_ptr + BLOCK_B[:, None] * cu_seqlens_stride[0]
        start = tl.load(cu_seqlens_ptrs, mask=MASK_B[:, None])
        end = tl.load(cu_seqlens_ptrs + cu_seqlens_stride[0], mask=MASK_B[:, None])

        if IS_MAX_SEQLEN_TENSOR:
            S = tl.load(max_seqlen_ptr)
        else:
            S = max_seqlen_ptr

        end -= 1

        y_ptrs = y_ptr + end * y_stride[0] + BLOCK_ID_N * y_stride[1] + BLOCK_H[None, :] * y_stride[2]

        dx_ptrs = dx_ptr + end * dx_stride[0] + BLOCK_ID_N * dx_stride[1] + BLOCK_H[None, :] * dx_stride[2]
        dy_ptrs = dy_ptr + end * dy_stride[0] + BLOCK_ID_N * dy_stride[1] + BLOCK_H[None, :] * dy_stride[2]
    else:
        y_ptrs = (
            y_ptr
            + BLOCK_B[:, None] * y_stride[0]
            + (S - 1) * y_stride[1]
            + BLOCK_ID_N * y_stride[2]
            + BLOCK_H[None, :] * y_stride[3]
        )

        dx_ptrs = (
            dx_ptr
            + BLOCK_B[:, None] * dx_stride[0]
            + (S - 1) * dx_stride[1]
            + BLOCK_ID_N * dx_stride[2]
            + BLOCK_H[None, :] * dx_stride[3]
        )

        dy_ptrs = (
            dy_ptr
            + BLOCK_B[:, None] * dy_stride[0]
            + (S - 1) * dy_stride[1]
            + BLOCK_ID_N * dy_stride[2]
            + BLOCK_H[None, :] * dy_stride[3]
        )

    if IS_VARLEN:
        MASK = (end >= start) & MASK_H[None, :]
    else:
        MASK = MASK_BH

    y = tl.load(y_ptrs, mask=MASK)

    # backward counting reduces 1 instruction since we need to compare s == 0, otherwise we have to compare s == S - 1
    for s in range(S - 1, -1, -1):
        if gradient_clipping is not None:
            dh = clamp(dh, min_value=-gradient_clipping, max_value=gradient_clipping)

        if IS_VARLEN:
            MASK = (end >= start) & MASK_H[None, :]
        else:
            MASK = MASK_BH

        dy = tl.load(dy_ptrs, mask=MASK) + dh

        y_ptrs -= y_stride[1 - IS_VARLEN]

        if IS_VARLEN:
            y_prev = tl.where(
                start == end,
                _load_input_state(
                    h0_ptr=h0_ptr,
                    h0_stride=h0_stride,
                    BLOCK_ID_N=BLOCK_ID_N,
                    BLOCK_B=BLOCK_B,
                    BLOCK_H=BLOCK_H,
                    MASK_BH=MASK_BH,
                    BLOCK_SIZE_B=BLOCK_SIZE_B,
                    BLOCK_SIZE_H=BLOCK_SIZE_H,
                    dtype=W.dtype,
                ),
                tl.load(y_ptrs, mask=MASK),
            )
        elif s == 0:
            if h0_ptr is None:
                y_prev = tl.zeros((BLOCK_SIZE_B, BLOCK_SIZE_H), dtype=W.dtype)
            else:
                y_prev = tl.load(
                    h0_ptr
                    + BLOCK_B[:, None] * h0_stride[0]
                    + BLOCK_ID_N * h0_stride[1]
                    + BLOCK_H[None, :] * h0_stride[2],
                    mask=MASK,
                )
        else:
            y_prev = tl.load(y_ptrs, mask=MASK)

        dx = dy * tanh_backward(y)
        dh = matmul(A=dx, B=W.T, C=None, output_dtype=dx.dtype)
        dW = matmul(A=y_prev.T, B=dx, C=dW, output_dtype=dW.dtype)

        y = y_prev

        tl.store(dx_ptrs, dx, mask=MASK)

        dx_ptrs -= dx_stride[1 - IS_VARLEN]
        dy_ptrs -= dy_stride[1 - IS_VARLEN]

        if IS_VARLEN:
            end -= 1

    tl.atomic_add(
        dW_ptr + BLOCK_ID_N * dW_stride[0] + BLOCK_H[:, None] * dW_stride[1] + BLOCK_H[None, :] * dW_stride[2],
        dW,
        mask=MASK_HH,
        sem="relaxed",
    )


@custom_op(f"{LIBRARY_NAME}::rnn_backward_triton", mutates_args={"input_grad", "weight_grad"})
def rnn_backward_triton(
    weight: torch.Tensor,
    output: torch.Tensor,
    input_state: torch.Tensor | None,
    output_grad: torch.Tensor,
    input_grad: torch.Tensor,
    weight_grad: torch.Tensor,
    cu_seqlens: torch.Tensor | None,
    max_seqlen_tensor: torch.Tensor | None,
    max_seqlen: int | None,
    gradient_clipping: float | None,
) -> None:
    if cu_seqlens is None:
        assert max_seqlen is None
        assert max_seqlen_tensor is None

        B, S, N, H = output.size()
    else:
        B = cu_seqlens.size(0) - 1
        S = None
        _, N, H = output.size()

    is_max_seqlen_tensor = max_seqlen_tensor is not None

    BLOCK_SIZE_H = get_next_power_of_2(H)
    BLOCK_SIZE_H = max(16, BLOCK_SIZE_H)
    GRID = lambda meta: (ceil_divide(B, meta["BLOCK_SIZE_B"]), N)

    with torch.device(output.device):
        rnn_backward_triton_kernel[GRID](
            W_ptr=weight,
            W_stride=weight.stride(),
            h0_ptr=input_state,
            h0_stride=None if input_state is None else input_state.stride(),
            y_ptr=output,
            y_stride=output.stride(),
            dx_ptr=input_grad,
            dx_stride=input_grad.stride(),
            dW_ptr=weight_grad,
            dW_stride=weight_grad.stride(),
            dy_ptr=output_grad,
            dy_stride=output_grad.stride(),
            cu_seqlens_ptr=cu_seqlens,
            cu_seqlens_stride=None if cu_seqlens is None else cu_seqlens.stride(),
            IS_MAX_SEQLEN_TENSOR=is_max_seqlen_tensor,
            max_seqlen_ptr=max_seqlen_tensor if is_max_seqlen_tensor else max_seqlen,
            B=B,
            S=S,
            H=H,
            gradient_clipping=gradient_clipping,
            BLOCK_SIZE_H=BLOCK_SIZE_H,
        )
