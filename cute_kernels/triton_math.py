import triton
import triton.language as tl


MAX_EXP_FP32 = 88.3762626647949
MIN_EXP_FP32 = -88.3762626647949


@triton.jit
def sigmoid(x):
    x = x.to(tl.float32)
    x = max(MIN_EXP_FP32, x)
    x = min(MAX_EXP_FP32, x)
    x = 1 / (1 + tl.exp(-x))
    return x
