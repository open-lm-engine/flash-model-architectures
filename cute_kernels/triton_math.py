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
def sigmoid(x, MIN_EXP_FP32: tl.constexpr = -88.3762626647949, MAX_EXP_FP32: tl.constexpr = 88.3762626647949):
    dtype = x.dtype

    x = x.to(tl.float32)
    x = clamp(x, min_value=MIN_EXP_FP32, max_value=MAX_EXP_FP32)
    x = 1 / (1 + tl.exp(-x))

    x = x.to(dtype)

    return x


@triton.jit
def tanh(x):
    dtype = x.dtype

    x = x.to(tl.float32)
    x = 2 * sigmoid(2 * x) - 1
    x = x.to(dtype)

    return x
