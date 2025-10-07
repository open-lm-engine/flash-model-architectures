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


@triton.autotune(configs=_get_autotune_configs(), key=["BLOCK_SIZE_H"], reset_to_zero=["dW_ptr"])
@triton.jit
def rnn_backward_triton_kernel(
    W_ptr,
    W_stride,
    y_ptr,
    y_stride,
    h0_ptr,
    h0_stride,
    dy_ptr,
    dx_ptr,
    dW_ptr,
    gradient_clipping,
    B,
    S,
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

    BLOCK = (
        BLOCK_B[:, None] * y_stride[0]
        + (S - 1) * y_stride[1]
        + BLOCK_ID_N * y_stride[2]
        + BLOCK_H[None, :] * y_stride[3]
    )
    y = tl.load(y_ptr + BLOCK, mask=mask_bh)

    # backward counting reduces 1 instruction since we need to compare s == 0, otherwise we have to compare s == S - 1
    for s in range(S - 1, -1, -1):
        if gradient_clipping is not None:
            dh = clamp(dh, min_value=-gradient_clipping, max_value=gradient_clipping)

        dy = tl.load(dy_ptr + BLOCK, mask=mask_bh) + dh

        dx_ptrs = dx_ptr + BLOCK
        BLOCK -= y_stride[1]

        if s == 0:
            if h0_ptr is None:
                y_prev = tl.zeros((BLOCK_SIZE_B, BLOCK_SIZE_H), dtype=W.dtype)
            else:
                y_prev = tl.load(
                    h0_ptr
                    + BLOCK_B[:, None] * h0_stride[0]
                    + BLOCK_ID_N * h0_stride[1]
                    + BLOCK_H[None, :] * h0_stride[2],
                    mask=mask_bh,
                )
        else:
            y_prev = tl.load(y_ptr + BLOCK, mask=mask_bh)

        dx = dy * tanh_backward(y)
        dh = matmul(A=dx, B=W.T, C=None, output_dtype=dx.dtype)
        dW = matmul(A=y_prev.T, B=dx, C=dW, output_dtype=dW.dtype)

        tl.store(dx_ptrs, dx, mask=mask_bh)
        y = y_prev

    tl.atomic_add(dW_ptr + BLOCK_W, dW, mask=mask_hh, sem="relaxed")


@custom_op(f"{LIBRARY_NAME}::rnn_backward_triton", mutates_args={"input_grad", "weight_grad"})
def rnn_backward_triton(
    weight: torch.Tensor,
    output: torch.Tensor,
    input_state: torch.Tensor | None,
    output_grad: torch.Tensor,
    input_grad: torch.Tensor,
    weight_grad: torch.Tensor,
    gradient_clipping: float | None,
) -> None:
    B, S, N, H = output.size()

    BLOCK_SIZE_H = get_next_power_of_2(H)
    BLOCK_SIZE_H = max(16, BLOCK_SIZE_H)
    GRID = lambda meta: (ceil_divide(B, meta["BLOCK_SIZE_B"]), N)

    with torch.device(output.device):
        rnn_backward_triton_kernel[GRID](
            W_ptr=weight,
            W_stride=weight.stride(),
            y_ptr=output,
            y_stride=output.stride(),
            h0_ptr=input_state,
            h0_stride=None if input_state is None else input_state.stride(),
            dy_ptr=output_grad,
            dx_ptr=input_grad,
            dW_ptr=weight_grad,
            gradient_clipping=gradient_clipping,
            B=B,
            S=S,
            H=H,
            BLOCK_SIZE_H=BLOCK_SIZE_H,
        )
