# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import torch
import triton
import triton.language as tl

from ....constants import LIBRARY_NAME
from ....math import ceil_divide, get_next_power_of_2
from ....triton_math import clamp
from ....utils import cute_op
from ...rnn.triton_implementation.backward import _activation_backward
from ...rnn.triton_implementation.backward_diagonal import _load_previous_output
from .forward_diagonal import _get_autotune_configs


@triton.jit
def _hippo_rnn_backward_update(
    y, W, V, hippo_A, hippo_B, dy, dc, dW, y_prev, c_prev, s, ACTIVATION_FUNCTION: tl.constexpr, relu_negative_slope
):
    # r = W * h + V * c + x
    # h = tanh(r)
    # c = (1 - A / (s + 1)) * c + B / (s + 1) * h
    # y = h

    s1 = s + 1
    A = 1 - hippo_A / s1
    B = hippo_B / s1

    dx = _activation_backward(
        y=y, dy=(dc * B + dy), ACTIVATION_FUNCTION=ACTIVATION_FUNCTION, relu_negative_slope=relu_negative_slope
    )

    dh = dx * W
    dc = dx * V + dc * A

    dW += tl.sum(y_prev * dx, axis=0)
    dV += tl.sum(c_prev * dx, axis=0)

    return dx, dW, dV, dh, dc


@triton.autotune(configs=_get_autotune_configs(), key=["BLOCK_SIZE_N"], reset_to_zero=["dW_ptr"])
@triton.jit
def diagonal_hippo_rnn_backward_triton_kernel(
    W_ptr,
    V_ptr,
    hippo_A_ptr,
    hippo_B_ptr,
    y_ptr,
    y_stride_b,
    h0_ptr,
    c0_ptr,
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
    dc = tl.zeros((BLOCK_SIZE_B, BLOCK_SIZE_N), dtype=W_ptr.dtype.element_ty)

    dW = tl.zeros((BLOCK_SIZE_N,), dtype=tl.float32)
    dV = tl.zeros((BLOCK_SIZE_N,), dtype=tl.float32)

    W = tl.load(W_ptr + indices_n, mask=mask_n)[None, :]
    V = tl.load(V_ptr + indices_n, mask=mask_n)[None, :]

    hippo_A = tl.load(hippo_A_ptr)
    hippo_B = tl.load(hippo_B_ptr)

    indices = indices_b[:, None] * y_stride_b + (S - 1) * N + indices_n[None, :]
    y = tl.load(y_ptr + indices, mask=mask_bn)

    # backward counting reduces 1 instruction since we need to compare s == 0, otherwise we have to compare s == S - 1
    for s in range(S - 1, -1, -1):
        if gradient_clipping is not None:
            dh = clamp(dh, min_value=-gradient_clipping, max_value=gradient_clipping)

        dy = tl.load(dy_ptr + indices, mask=mask_bn) + dh

        dx_ptrs = dx_ptr + indices
        indices -= N

        y_prev = _load_previous_output(
            h0_ptr=h0_ptr,
            y_ptrs=y_ptr + indices,
            N=N,
            indices_b=indices_b,
            indices_n=indices_n,
            mask_bn=mask_bn,
            BLOCK_SIZE_B=BLOCK_SIZE_B,
            BLOCK_SIZE_N=BLOCK_SIZE_N,
            s=s,
            dtype=W.dtype,
        )

        c_prev = _load_previous_output(
            h0_ptr=c0_ptr,
            y_ptrs=c_ptr + indices,
            N=N,
            indices_b=indices_b,
            indices_n=indices_n,
            mask_bn=mask_bn,
            BLOCK_SIZE_B=BLOCK_SIZE_B,
            BLOCK_SIZE_N=BLOCK_SIZE_N,
            s=s,
            dtype=W.dtype,
        )

        dx, dW, dV, dh, dc = _hippo_rnn_backward_update(
            y=y,
            W=W,
            dy=dy,
            dW=dW,
            y_prev=y_prev,
            ACTIVATION_FUNCTION="tanh",
            relu_negative_slope=None,
        )

        tl.store(dx_ptrs, dx, mask=mask_bn)
        y = y_prev

    tl.atomic_add(dW_ptr + indices_n, dW, mask=mask_n)


@cute_op(f"{LIBRARY_NAME}::diagonal_hippo_rnn_backward_triton", mutates_args={"input_grad", "weight_grad"})
def diagonal_hippo_rnn_backward_triton(
    weight: torch.Tensor,
    hippo_weight: torch.Tensor,
    hippo_A: torch.Tensor,
    hippo_B: torch.Tensor,
    output: torch.Tensor,
    hippo_output: torch.Tensor,
    input_state: torch.Tensor | None,
    hippo_state: torch.Tensor | None,
    output_grad: torch.Tensor,
    input_grad: torch.Tensor,
    weight_grad: torch.Tensor,
    gradient_clipping: float | None,
) -> None:
    B, S, N, _ = output.size()

    BLOCK_SIZE_N = min(1024, get_next_power_of_2(N))
    GRID = lambda meta: (ceil_divide(B, meta["BLOCK_SIZE_B"]), ceil_divide(N, meta["BLOCK_SIZE_N"]))

    with torch.device(output.device):
        diagonal_hippo_rnn_backward_triton_kernel[GRID](
            W_ptr=weight,
            V_ptr=hippo_weight,
            hippo_A_ptr=hippo_A,
            hippo_B_ptr=hippo_B,
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
