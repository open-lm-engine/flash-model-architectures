# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import triton
import triton.language as tl


@triton.jit
def clamp(x, min_value, max_value):
    dtype = x.dtype

    x = max(min_value, x)
    x = min(max_value, x)
    x = x.to(dtype)

    return x


@triton.jit
def sigmoid(
    x,
    MIN_EXP_FP32: tl.constexpr = -88.3762626647949,
    MAX_EXP_FP32: tl.constexpr = 88.3762626647949,
    output_dtype: tl.constexpr = None,
):
    if output_dtype is None:
        output_dtype = x.dtype

    x = x.to(tl.float32)
    x = clamp(x, min_value=MIN_EXP_FP32, max_value=MAX_EXP_FP32)
    x = 1 / (1 + tl.exp(-x))

    x = x.to(output_dtype)

    return x


@triton.jit
def tanh(x, output_dtype: tl.constexpr = None):
    if output_dtype is None:
        output_dtype = x.dtype

    x = x.to(tl.float32)
    x = 2 * sigmoid(2 * x) - 1
    x = x.to(output_dtype)

    return x


@triton.jit
def relu(x):
    return max(0, x)


@triton.jit
def leaky_relu(x, negative_slope):
    return relu(x) + negative_slope * min(0, x)


@triton.jit
def sigmoid_backward(y):
    dtype = y.dtype

    y = y.to(tl.float32)
    y = y * (1 - y)
    y = y.to(dtype)

    return y


@triton.jit
def tanh_backward(y):
    dtype = y.dtype

    y = y.to(tl.float32)
    y = 1 - y * y
    y = y.to(dtype)

    return y


@triton.jit
def leaky_relu_backward(y, relu_negative_slope):
    return tl.where(y >= 0, 1, relu_negative_slope)


@triton.jit
def matmul(A, B, C, output_dtype: tl.constexpr):
    if A.shape[0] == 1:
        x = tl.sum(A.T * B, axis=0, keep_dims=True)
        if C is not None:
            x += C
        x = x.to(output_dtype)
    elif A.shape[1] == 1:
        x = A * B
        if C is not None:
            x += C
        x = x.to(output_dtype)
    elif B.shape[1] == 1:
        x = tl.sum(A * B.T, axis=1, keep_dims=True)
        if C is not None:
            x += C
        x = x.to(output_dtype)
    else:
        if C is None:
            if output_dtype == tl.bfloat16:
                x = tl.dot(A, B, out_dtype=tl.float32).to(output_dtype)
            else:
                x = tl.dot(A, B, out_dtype=output_dtype)
        else:
            x = tl.dot(A, B, C.to(tl.float32), out_dtype=tl.float32).to(output_dtype)

    return x
