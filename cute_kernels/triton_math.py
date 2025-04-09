import triton
import triton.language as tl


MAX_EXP_FP32: tl.constexpr = 88.3762626647949
MIN_EXP_FP32: tl.constexpr = -88.3762626647949


@triton.jit
def clamp(x):
    x = max(MIN_EXP_FP32, x)
    x = min(MAX_EXP_FP32, x)
    return x


@triton.jit
def sigmoid(x):
    dtype = x.dtype

    x = x.to(tl.float32)
    x = clamp(x)
    x = 1 / (1 + tl.exp(-x))

    x = x.to(dtype)

    return x


@triton.jit
def tanh(x):
    dtype = x.dtype

    x = x.to(tl.float32)
    x = clamp(x)
    x = tl.exp(-x)
    x *= x
    x = (1 - x) / (1 + x)

    x = x.to(dtype)

    return x
