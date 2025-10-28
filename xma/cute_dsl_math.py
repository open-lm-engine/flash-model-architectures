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
def tanh(x, output_dtype: tl.constexpr = None):
    if output_dtype is None:
        output_dtype = x.dtype

    x = x.to(tl.float32)
    x = tl.inline_asm_elementwise("tanh.approx.f32 $0, $1;", "=f,f", [x], dtype=tl.float32, is_pure=True, pack=1)
    x = x.to(output_dtype)

    return x


@triton.jit
def tanh_backward(y):
    dtype = y.dtype

    y = y.to(tl.float32)
    y = 1 - y * y
    y = y.to(dtype)

    return y
