import torch
import triton
import triton.language as tl

from ....constants import LIBRARY_NAME
from ....math import ceil_divide, get_next_power_of_2
from ....triton_math import clamp
from ....utils import cute_op


@triton.jit
def _sigmoid_backward(y):
    dtype = y.dtype

    y = y.to(tl.float32)
    y = y * (1 - y)
    y = y.to(dtype)

    return y


@triton.jit
def _tanh_backward(y):
    dtype = y.dtype

    y = y.to(tl.float32)
    y = 1 - y * y
    y = y.to(dtype)

    return y


@triton.jit
def _leaky_relu_backward(y, relu_negative_slope):
    dtype = y.dtype

    y = tl.where(y >= 0, 1, relu_negative_slope)
    y = y.to(dtype)

    return y


@triton.jit
def _backward_rnn_update(
    output_grad, output, weight, output_prev, ACTIVATION_FUNCTION: tl.constexpr, relu_negative_slope
):
    if ACTIVATION_FUNCTION == "leaky_relu":
        input_grad = output_grad * _leaky_relu_backward(output, relu_negative_slope)
    elif ACTIVATION_FUNCTION == "sigmoid":
        input_grad = output_grad * _sigmoid_backward(output)
    elif ACTIVATION_FUNCTION == "tanh":
        input_grad = output_grad * _tanh_backward(output)

    input_state_grad = tl.dot(input_grad, weight.T, allow_tf32=True).to(input_grad.dtype)
    weight_grad = tl.dot(output_prev.T, input_grad, weight_grad, allow_tf32=True)

    return input_grad, weight_grad, input_state_grad


@triton.jit
def rnn_backward_triton_kernel(
    weight_ptr,
    weight_stride_n,
    output_ptr,
    output_stride_b,
    output_stride_s,
    HAS_INPUT_STATE: tl.constexpr,
    input_state_ptr,
    input_state_stride_b,
    output_grad_ptr,
    input_grad_ptr,
    weight_grad_ptr,
    HAS_GRADIENT_CLIPPING: tl.constexpr,
    gradient_clipping,
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

    input_state_grad = tl.zeros((BLOCK_SIZE_B, BLOCK_SIZE_H), dtype=weight_ptr.dtype.element_ty)
    weight_grad = tl.zeros((BLOCK_SIZE_H, BLOCK_SIZE_H), dtype=tl.float32)

    weight_ptrs = weight_ptr + pid_n * weight_stride_n + indices_h[:, None] * H + indices_h[None, :]
    weight = tl.load(weight_ptrs, mask=mask_h[:, None] & mask_h[None, :], other=0)

    indices = indices_b[:, None] * output_stride_b + (S - 1) * output_stride_s + pid_n * H + indices_h[None, :]

    output_ptrs = output_ptr + indices
    output = tl.load(output_ptrs, mask=mask_bh, other=0)

    # backward counting reduces 1 instruction since we need to compare s == 0, otherwise we have to compare s == S - 1
    for s in range(S - 1, -1, -1):
        if HAS_GRADIENT_CLIPPING:
            input_state_grad = clamp(input_state_grad, min_value=-gradient_clipping, max_value=gradient_clipping)

        output_grad_ptrs = output_grad_ptr + indices
        output_grad = tl.load(output_grad_ptrs, mask=mask_bh, other=0)
        output_grad += input_state_grad

        if s == 0:
            if HAS_INPUT_STATE:
                input_state_ptrs = (
                    input_state_ptr + indices_b[:, None] * input_state_stride_b + pid_n * H + indices_h[None, :]
                )
                output_prev = tl.load(input_state_ptrs, mask=mask_bh, other=0)
            else:
                output_prev = tl.zeros((BLOCK_SIZE_B, BLOCK_SIZE_H), dtype=weight.dtype)
        else:
            output_ptrs -= output_stride_s
            output_prev = tl.load(output_ptrs, mask=mask_bh, other=0)

        input_grad, weight_grad, input_state_grad = _backward_rnn_update(
            output_grad=output_grad,
            output=output,
            weight=weight,
            output_prev=output_prev,
            ACTIVATION_FUNCTION=ACTIVATION_FUNCTION,
            relu_negative_slope=relu_negative_slope,
        )

        input_grad_ptrs = input_grad_ptr + indices
        tl.store(input_grad_ptrs, input_grad, mask=mask_bh)

        output = output_prev
        indices -= output_stride_s

    weight_grad_ptrs = weight_grad_ptr + pid_n * weight_stride_n + indices_h[:, None] * H + indices_h[None, :]
    tl.store(weight_grad_ptrs, weight_grad, mask=mask_h[:, None] & mask_h[None, :])


@cute_op(f"{LIBRARY_NAME}::rnn_backward_triton", mutates_args={"input_grad", "weight_grad"})
def rnn_backward_triton(
    weight: torch.Tensor,
    output: torch.Tensor,
    input_state: torch.Tensor | None,
    output_grad: torch.Tensor,
    input_grad: torch.Tensor,
    weight_grad: torch.Tensor,
    gradient_clipping: float | None,
    activation_function: str,
    relu_negative_slope: float | None,
    BLOCK_SIZE_B: int,
) -> None:
    B, S, N, H = output.size()

    BLOCK_SIZE_H = get_next_power_of_2(H)
    BLOCK_SIZE_H = max(16, BLOCK_SIZE_H)

    with torch.device(output.device):
        rnn_backward_triton_kernel[ceil_divide(B, BLOCK_SIZE_B), N](
            weight_ptr=weight,
            weight_stride_n=weight.stride(0),
            output_ptr=output,
            output_stride_b=output.stride(0),
            output_stride_s=output.stride(1),
            HAS_INPUT_STATE=input_state is not None,
            input_state_ptr=input_state,
            input_state_stride_b=None if input_state is None else input_state.stride(0),
            output_grad_ptr=output_grad,
            input_grad_ptr=input_grad,
            weight_grad_ptr=weight_grad,
            HAS_GRADIENT_CLIPPING=gradient_clipping is not None,
            gradient_clipping=gradient_clipping,
            ACTIVATION_FUNCTION=activation_function,
            relu_negative_slope=relu_negative_slope,
            B=B,
            S=S,
            H=H,
            BLOCK_SIZE_B=BLOCK_SIZE_B,
            BLOCK_SIZE_H=BLOCK_SIZE_H,
        )
