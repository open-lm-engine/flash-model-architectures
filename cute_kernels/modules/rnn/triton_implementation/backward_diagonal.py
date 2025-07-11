# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import torch
import triton
import triton.language as tl
from torch.library import custom_op

from ....constants import LIBRARY_NAME
from ....math import ceil_divide, get_next_power_of_2
from ....triton_math import clamp, tanh_backward
from .forward_diagonal import _get_autotune_configs


@triton.autotune(configs=_get_autotune_configs(), key=["BLOCK_SIZE_N"], reset_to_zero=["dW_ptr"])
@triton.jit
def diagonal_rnn_backward_triton_kernel(
    W_ptr,
    y_ptr,
    y_stride_b,
    h0_ptr,
    dy_ptr,
    dx_ptr,
    dW_ptr,
    gradient_clipping,
    B,
    S,
    N,
    BLOCK_SIZE_B: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    pid_b = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    indices_b = pid_b * BLOCK_SIZE_B + tl.arange(0, BLOCK_SIZE_B)
    indices_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    mask_b = indices_b < B
    mask_n = indices_n < N

    mask_bn = mask_b[:, None] & mask_n[None, :]

    dh = tl.zeros((BLOCK_SIZE_B, BLOCK_SIZE_N), dtype=W_ptr.dtype.element_ty)
    dW = tl.zeros((BLOCK_SIZE_N,), dtype=tl.float32)

    W = tl.load(W_ptr + indices_n, mask=mask_n)[None, :]

    indices = indices_b[:, None] * y_stride_b + (S - 1) * N + indices_n[None, :]
    y = tl.load(y_ptr + indices, mask=mask_bn)

    # backward counting reduces 1 instruction since we need to compare s == 0, otherwise we have to compare s == S - 1
    for s in range(S - 1, -1, -1):
        if gradient_clipping is not None:
            dh = clamp(dh, min_value=-gradient_clipping, max_value=gradient_clipping)

        dy = tl.load(dy_ptr + indices, mask=mask_bn) + dh

        dx_ptrs = dx_ptr + indices
        indices -= N

        if s == 0:
            if h0_ptr is None:
                y_prev = tl.zeros((BLOCK_SIZE_B, BLOCK_SIZE_N), dtype=W.dtype)
            else:
                y_prev = tl.load(h0_ptr + indices_b[:, None] * N + indices_n[None, :], mask=mask_bn)
        else:
            y_prev = tl.load(y_ptr + indices, mask=mask_bn)

        dx = dy * tanh_backward(y)
        dh = dx * W
        dW += tl.sum(y_prev * dx, axis=0)

        tl.store(dx_ptrs, dx, mask=mask_bn)
        y = y_prev

    tl.atomic_add(dW_ptr + indices_n, dW, mask=mask_n)


@custom_op(f"{LIBRARY_NAME}::diagonal_rnn_backward_triton", mutates_args={"input_grad", "weight_grad"})
def diagonal_rnn_backward_triton(
    weight: torch.Tensor,
    output: torch.Tensor,
    input_state: torch.Tensor | None,
    output_grad: torch.Tensor,
    input_grad: torch.Tensor,
    weight_grad: torch.Tensor,
    gradient_clipping: float | None,
) -> None:
    B, S, N, _ = output.size()

    BLOCK_SIZE_N = min(1024, get_next_power_of_2(N))
    GRID = lambda meta: (ceil_divide(B, meta["BLOCK_SIZE_B"]), ceil_divide(N, meta["BLOCK_SIZE_N"]))

    with torch.device(output.device):
        diagonal_rnn_backward_triton_kernel[GRID](
            W_ptr=weight,
            y_ptr=output,
            y_stride_b=output.stride(0),
            h0_ptr=input_state,
            dy_ptr=output_grad,
            dx_ptr=input_grad,
            dW_ptr=weight_grad,
            gradient_clipping=gradient_clipping,
            B=B,
            S=S,
            N=N,
            BLOCK_SIZE_N=BLOCK_SIZE_N,
        )
