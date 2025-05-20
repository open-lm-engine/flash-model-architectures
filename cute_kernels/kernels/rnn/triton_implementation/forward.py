# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import torch
import triton
import triton.language as tl

from ....constants import LIBRARY_NAME
from ....math import ceil_divide, get_next_power_of_2
from ....triton_math import leaky_relu, sigmoid, tanh
from ....utils import cute_op


@triton.jit
def _activation(x, ACTIVATION_FUNCTION, relu_negative_slope):
    if ACTIVATION_FUNCTION == "leaky_relu":
        x = leaky_relu(x, relu_negative_slope)
    elif ACTIVATION_FUNCTION == "sigmoid":
        x = sigmoid(x)
    elif ACTIVATION_FUNCTION == "tanh":
        x = tanh(x)

    return x


@triton.jit
def _rnn_forward_update(h, W, x, out_dtype, cast_dtype, ACTIVATION_FUNCTION, relu_negative_slope):
    h = tl.dot(h, W, x, allow_tf32=True, out_dtype=out_dtype).to(cast_dtype)
    h = _activation(x=h, ACTIVATION_FUNCTION=ACTIVATION_FUNCTION, relu_negative_slope=relu_negative_slope)
    return h


@triton.jit
def rnn_forward_triton_kernel(
    x_ptr,
    x_stride_b,
    x_stride_s,
    W_ptr,
    W_stride_n,
    HAS_INPUT_STATE: tl.constexpr,
    h_ptr,
    h_stride_b,
    y_ptr,
    ACTIVATION_FUNCTION: tl.constexpr,
    relu_negative_slope,
    B,
    S,
    H,
    BLOCK_SIZE_B: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
):
    pid_b = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    indices_b = pid_b * BLOCK_SIZE_B + tl.arange(0, BLOCK_SIZE_B)
    indices_h = tl.arange(0, BLOCK_SIZE_H)

    mask_b = indices_b < B
    mask_h = indices_h < H
    mask_bh = mask_b[:, None] & mask_h[None, :]

    W = tl.load(
        W_ptr + pid_n * W_stride_n + indices_h[:, None] * H + indices_h[None, :],
        mask=mask_h[:, None] & mask_h[None, :],
    )

    if HAS_INPUT_STATE:
        h = tl.load(h_ptr + indices_b[:, None] * h_stride_b + pid_n * H + indices_h[None, :], mask=mask_bh)
    else:
        h = tl.zeros((BLOCK_SIZE_B, BLOCK_SIZE_H), dtype=x_ptr.dtype.element_ty)

    indices = indices_b[:, None] * x_stride_b + pid_n * H + indices_h[None, :]

    input_dtype = x_ptr.dtype.element_ty
    cast_dtype = input_dtype
    if input_dtype == tl.bfloat16:
        input_dtype = tl.float32
        cast_dtype = tl.bfloat16

    out_dtype = input_dtype

    for _ in range(S):
        h = _rnn_forward_update(
            h=h,
            W=W,
            x=tl.load(x_ptr + indices, mask=mask_bh).to(input_dtype),
            out_dtype=out_dtype,
            cast_dtype=cast_dtype,
            ACTIVATION_FUNCTION=ACTIVATION_FUNCTION,
            relu_negative_slope=relu_negative_slope,
        )

        tl.store(y_ptr + indices, h, mask=mask_bh)

        indices += x_stride_s


@cute_op(f"{LIBRARY_NAME}::rnn_forward_triton", mutates_args={"output"})
def rnn_forward_triton(
    input: torch.Tensor,
    weight: torch.Tensor,
    input_state: torch.Tensor | None,
    output: torch.Tensor,
    activation_function: str,
    relu_negative_slope: float | None,
    BLOCK_SIZE_B: int,
) -> None:
    B, S, N, H = input.size()

    BLOCK_SIZE_H = get_next_power_of_2(H)
    BLOCK_SIZE_H = max(16, BLOCK_SIZE_H)

    has_input_state = input_state is not None

    with torch.device(input.device):
        rnn_forward_triton_kernel[ceil_divide(B, BLOCK_SIZE_B), N](
            x_ptr=input,
            x_stride_b=input.stride(0),
            x_stride_s=input.stride(1),
            W_ptr=weight,
            W_stride_n=weight.stride(0),
            HAS_INPUT_STATE=has_input_state,
            h_ptr=input_state,
            h_stride_b=input_state.stride(0) if has_input_state else None,
            y_ptr=output,
            ACTIVATION_FUNCTION=activation_function,
            relu_negative_slope=relu_negative_slope,
            B=B,
            S=S,
            H=H,
            BLOCK_SIZE_B=BLOCK_SIZE_B,
            BLOCK_SIZE_H=BLOCK_SIZE_H,
        )
