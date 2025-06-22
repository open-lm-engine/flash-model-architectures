# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import torch
import triton
import triton.language as tl

from ....constants import LIBRARY_NAME
from ....math import ceil_divide, get_next_power_of_2
from ....utils import cute_op
from ...rnn.triton_implementation.forward_diagonal import _activation, _get_autotune_configs


@triton.jit
def _rnn_forward_update(h, W, c, V, x, hippo_A, hippo_B, s, ACTIVATION_FUNCTION, relu_negative_slope):
    h = W * h + x
    h += V * c
    h = _activation(x=h, ACTIVATION_FUNCTION=ACTIVATION_FUNCTION, relu_negative_slope=relu_negative_slope)

    A = 1 - hippo_A / s
    B = hippo_B / s
    c = c * A + h * B

    return h, c


@triton.autotune(configs=_get_autotune_configs(), key=["BLOCK_SIZE_N"])
@triton.jit
def diagonal_hippo_rnn_forward_triton_kernel(
    x_ptr,
    x_stride_b,
    W_ptr,
    V_ptr,
    hippo_A_ptr,
    hippo_B_ptr,
    h_ptr,
    c0_ptr,
    y_ptr,
    c_ptr,
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

    W = tl.load(W_ptr + indices_n, mask=mask_n)[None, :]
    V = tl.load(V_ptr + indices_n, mask=mask_n)[None, :]

    if h_ptr is None:
        h = tl.zeros((BLOCK_SIZE_B, BLOCK_SIZE_N), dtype=x_ptr.dtype.element_ty)
    else:
        h = tl.load(h_ptr + indices_b[:, None] * N + indices_n[None, :], mask=mask_bn)

    if c0_ptr is None:
        c = tl.zeros((BLOCK_SIZE_B, BLOCK_SIZE_N), dtype=x_ptr.dtype.element_ty)
    else:
        c = tl.load(c0_ptr + indices_b[:, None] * N + indices_n[None, :], mask=mask_bn)

    hippo_A = tl.load(hippo_A_ptr)
    hippo_B = tl.load(hippo_B_ptr)

    indices = indices_b[:, None] * x_stride_b + indices_n[None, :]

    for s in range(1, S + 1):
        h, c = _rnn_forward_update(
            h=h,
            W=W,
            c=c,
            V=V,
            x=tl.load(x_ptr + indices, mask=mask_bn),
            hippo_A=hippo_A,
            hippo_B=hippo_B,
            s=s,
            ACTIVATION_FUNCTION="tanh",
            relu_negative_slope=None,
        )

        tl.store(y_ptr + indices, h, mask=mask_bn)
        tl.store(c_ptr + indices, c, mask=mask_bn)

        indices += N


@cute_op(f"{LIBRARY_NAME}::diagonal_hippo_rnn_forward_triton", mutates_args={"output", "hippo_output"})
def diagonal_hippo_rnn_forward_triton(
    input: torch.Tensor,
    weight: torch.Tensor,
    hippo_weight: torch.Tensor,
    hippo_A: torch.Tensor,
    hippo_B: torch.Tensor,
    input_state: torch.Tensor | None,
    hippo_state: torch.Tensor | None,
    output: torch.Tensor,
    hippo_output: torch.Tensor,
) -> None:
    B, S, N, _ = input.size()

    BLOCK_SIZE_N = min(1024, get_next_power_of_2(N))
    GRID = lambda meta: (ceil_divide(B, meta["BLOCK_SIZE_B"]), ceil_divide(N, meta["BLOCK_SIZE_N"]))

    with torch.device(input.device):
        diagonal_hippo_rnn_forward_triton_kernel[GRID](
            x_ptr=input,
            x_stride_b=input.stride(0),
            W_ptr=weight,
            V_ptr=hippo_weight,
            hippo_A_ptr=hippo_A,
            hippo_B_ptr=hippo_B,
            h_ptr=input_state,
            c0_ptr=hippo_state,
            y_ptr=output,
            c_ptr=hippo_output,
            B=B,
            S=S,
            N=N,
            BLOCK_SIZE_N=BLOCK_SIZE_N,
        )
