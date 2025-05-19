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


@triton.jit
def relu(x):
    dtype = x.dtype

    x = max(0, x)
    x = x.to(dtype)

    return x


@triton.jit
def leaky_relu(x, negative_slope):
    dtype = x.dtype

    x = relu(x) + negative_slope * min(0, x)
    x = x.to(dtype)

    return x


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
    dtype = y.dtype

    y = tl.where(y >= 0, 1, relu_negative_slope)
    y = y.to(dtype)

    return y


@triton.jit
def matmul(A, B, C, output_dtype):
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
        is_bf16 = output_dtype == tl.bfloat16
        if is_bf16:
            output_dtype = tl.float32

        if C is None:
            x = tl.dot(A, B, out_dtype=output_dtype)
        else:
            x = tl.dot(A, B, C, out_dtype=output_dtype)

        if is_bf16:
            x = x.to(tl.bfloat16)

    return x
