import torch
import triton
import triton.language as tl

from ....constants import LIBRARY_NAME
from ....math import ceil_divide, get_next_power_of_2
from ....triton_math import clamp
from ....utils import cute_op


@triton.jit
def _tanh_backward(y):
    dtype = y.dtype

    y = y.to(tl.float32)
    y = 1 - y * y
    y = y.to(dtype)

    return y


@triton.jit
def rnn_backward_triton_kernel(
    weight_ptr,
    weight_stride_n,
    weight_stride_h,
    output_ptr,
    output_stride_b,
    output_stride_s,
    output_stride_n,
    has_input_state: tl.constexpr,
    input_state_ptr,
    input_state_stride_b,
    input_state_stride_n,
    output_grad_ptr,
    input_grad_ptr,
    weight_grad_ptr,
    has_gradient_clipping: tl.constexpr,
    gradient_clipping,
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

    weight_ptrs = weight_ptr + pid_n * weight_stride_n + indices_h[:, None] * weight_stride_h + indices_h[None, :]
    weight = tl.load(weight_ptrs, mask=mask_h[:, None] & mask_h[None, :], other=0)

    indices = (
        indices_b[:, None] * output_stride_b + (S - 1) * output_stride_s + pid_n * output_stride_n + indices_h[None, :]
    )

    output_ptrs = output_ptr + indices
    output = tl.load(output_ptrs, mask=mask_bh, other=0)

    # backward counting reduces 1 instruction since we need to compare s == 0, otherwise we have to compare s == S - 1
    for s in range(S - 1, -1, -1):
        output_grad_ptrs = output_grad_ptr + indices
        output_grad = tl.load(output_grad_ptrs, mask=mask_bh, other=0)

        if has_gradient_clipping:
            input_state_grad = clamp(input_state_grad, min_value=-gradient_clipping, max_value=gradient_clipping)

        input_grad = (output_grad + input_state_grad) * _tanh_backward(output)

        input_grad_ptrs = input_grad_ptr + indices
        tl.store(input_grad_ptrs, input_grad, mask=mask_bh)

        input_state_grad = tl.dot(input_grad, weight.T, allow_tf32=True).to(input_state_grad.dtype)

        if s == 0:
            if has_input_state:
                output_ptrs = (
                    input_state_ptr
                    + indices_b[:, None] * input_state_stride_b
                    + pid_n * input_state_stride_n
                    + indices_h[None, :]
                )
                output_prev = tl.load(output_ptrs, mask=mask_bh, other=0)
            else:
                output_prev = tl.zeros((BLOCK_SIZE_B, BLOCK_SIZE_H), dtype=weight.dtype)
        else:
            output_ptrs -= output_stride_s
            output_prev = tl.load(output_ptrs, mask=mask_bh, other=0)

        weight_grad = tl.dot(output_prev.T, input_grad, weight_grad, allow_tf32=True)
        output = output_prev

        indices -= output_stride_s

    weight_grad_ptrs = (
        weight_grad_ptr + pid_n * weight_stride_n + indices_h[:, None] * weight_stride_h + indices_h[None, :]
    )
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
    BLOCK_SIZE_B: int,
) -> None:
    B, S, N, H = output.size()

    BLOCK_SIZE_H = get_next_power_of_2(H)
    BLOCK_SIZE_H = max(16, BLOCK_SIZE_H)

    with torch.device(output.device):
        rnn_backward_triton_kernel[ceil_divide(B, BLOCK_SIZE_B), N](
            weight_ptr=weight,
            weight_stride_n=weight.stride(0),
            weight_stride_h=weight.stride(1),
            output_ptr=output,
            output_stride_b=output.stride(0),
            output_stride_s=output.stride(1),
            output_stride_n=output.stride(2),
            has_input_state=input_state is not None,
            input_state_ptr=input_state,
            input_state_stride_b=None if input_state is None else input_state.stride(0),
            input_state_stride_n=None if input_state is None else input_state.stride(1),
            output_grad_ptr=output_grad,
            input_grad_ptr=input_grad,
            weight_grad_ptr=weight_grad,
            has_gradient_clipping=gradient_clipping is not None,
            gradient_clipping=gradient_clipping,
            B=B,
            S=S,
            H=H,
            BLOCK_SIZE_B=BLOCK_SIZE_B,
            BLOCK_SIZE_H=BLOCK_SIZE_H,
        )
