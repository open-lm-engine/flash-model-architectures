import torch
import triton
import triton.language as tl

from ....constants import LIBRARY_NAME
from ....math import ceil_divide
from ....triton_math import clamp, leaky_relu_backward, sigmoid_backward, tanh_backward
from ....utils import cute_op


@triton.jit
def _rnn_backward_update(
    output, weight, output_grad, weight_grad, output_prev, ACTIVATION_FUNCTION: tl.constexpr, relu_negative_slope
):
    if ACTIVATION_FUNCTION == "leaky_relu":
        input_grad = output_grad * leaky_relu_backward(output, relu_negative_slope)
    elif ACTIVATION_FUNCTION == "sigmoid":
        input_grad = output_grad * sigmoid_backward(output)
    elif ACTIVATION_FUNCTION == "tanh":
        input_grad = output_grad * tanh_backward(output)

    input_state_grad = input_grad * weight
    weight_grad += tl.sum(output_prev * input_grad, axis=0)

    return input_grad, weight_grad, input_state_grad


@triton.jit
def _load_previous_output(
    HAS_INPUT_STATE: tl.constexpr,
    input_state_ptr,
    output_ptrs,
    N,
    indices_b,
    indices_n,
    mask_bn,
    BLOCK_SIZE_B: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    s,
    dtype,
):
    if s == 0:
        if HAS_INPUT_STATE:
            output_prev = tl.load(input_state_ptr + indices_b[:, None] * N + indices_n[None, :], mask=mask_bn)
        else:
            output_prev = tl.zeros((BLOCK_SIZE_B, BLOCK_SIZE_N), dtype=dtype)
    else:
        output_prev = tl.load(output_ptrs, mask=mask_bn)

    return output_prev


@triton.jit
def scalar_rnn_backward_triton_kernel(
    weight_ptr,
    output_ptr,
    output_stride_b,
    HAS_INPUT_STATE: tl.constexpr,
    input_state_ptr,
    output_grad_ptr,
    input_grad_ptr,
    weight_grad_ptr,
    HAS_GRADIENT_CLIPPING: tl.constexpr,
    gradient_clipping,
    ACTIVATION_FUNCTION: tl.constexpr,
    relu_negative_slope,
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

    input_state_grad = tl.zeros((BLOCK_SIZE_B, BLOCK_SIZE_N), dtype=weight_ptr.dtype.element_ty)
    weight_grad = tl.zeros((BLOCK_SIZE_N,), dtype=tl.float32)

    weight = tl.load(weight_ptr + indices_n, mask=mask_n)

    indices = indices_b[:, None] * output_stride_b + (S - 1) * N + indices_n[None, :]
    output = tl.load(output_ptr + indices, mask=mask_bn)

    # backward counting reduces 1 instruction since we need to compare s == 0, otherwise we have to compare s == S - 1
    for s in range(S - 1, -1, -1):
        if HAS_GRADIENT_CLIPPING:
            input_state_grad = clamp(input_state_grad, min_value=-gradient_clipping, max_value=gradient_clipping)

        output_grad = tl.load(output_grad_ptr + indices, mask=mask_bn)
        output_grad += input_state_grad

        input_grad_ptrs = input_grad_ptr + indices
        indices -= N

        output_prev = _load_previous_output(
            HAS_INPUT_STATE=HAS_INPUT_STATE,
            input_state_ptr=input_state_ptr,
            output_ptrs=output_ptr + indices,
            N=N,
            indices_b=indices_b,
            indices_n=indices_n,
            mask_bn=mask_bn,
            BLOCK_SIZE_B=BLOCK_SIZE_B,
            BLOCK_SIZE_N=BLOCK_SIZE_N,
            s=s,
            dtype=weight.dtype,
        )

        input_grad, weight_grad, input_state_grad = _rnn_backward_update(
            output=output,
            weight=weight,
            output_grad=output_grad,
            weight_grad=weight_grad,
            output_prev=output_prev,
            ACTIVATION_FUNCTION=ACTIVATION_FUNCTION,
            relu_negative_slope=relu_negative_slope,
        )

        tl.store(input_grad_ptrs, input_grad, mask=mask_bn)
        output = output_prev

    tl.atomic_add(weight_grad_ptr + indices_n, weight_grad, mask=mask_n)


@cute_op(f"{LIBRARY_NAME}::scalar_rnn_backward_triton", mutates_args={"input_grad", "weight_grad"})
def scalar_rnn_backward_triton(
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
    BLOCK_SIZE_N: int,
) -> None:
    B, S, N, _ = output.size()

    with torch.device(output.device):
        scalar_rnn_backward_triton_kernel[ceil_divide(B, BLOCK_SIZE_B), ceil_divide(N, BLOCK_SIZE_N)](
            weight_ptr=weight,
            output_ptr=output,
            output_stride_b=output.stride(0),
            HAS_INPUT_STATE=input_state is not None,
            input_state_ptr=input_state,
            output_grad_ptr=output_grad,
            input_grad_ptr=input_grad,
            weight_grad_ptr=weight_grad,
            HAS_GRADIENT_CLIPPING=gradient_clipping is not None,
            gradient_clipping=gradient_clipping,
            ACTIVATION_FUNCTION=activation_function,
            relu_negative_slope=relu_negative_slope,
            B=B,
            S=S,
            N=N,
            BLOCK_SIZE_B=BLOCK_SIZE_B,
            BLOCK_SIZE_N=BLOCK_SIZE_N,
        )
