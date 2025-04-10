import torch
import triton
import triton.language as tl

from ....constants import LIBRARY_NAME
from ....math import ceil_divide, get_next_power_of_2
from ....utils import cute_op


_KERNEL_NAME = "rnn_backward_triton"


@triton.jit
def _rnn_backward_triton_kernel(
    input_ptr,
    input_stride_b,
    input_stride_s,
    input_stride_n,
    weight_ptr,
    weight_stride_n,
    weight_stride_h,
    output_ptr,
    output_stride_b,
    output_stride_s,
    output_stride_n,
    output_grad_ptr,
    input_grad_ptr,
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

    input_state_grad = tl.zeros((BLOCK_SIZE_B, BLOCK_SIZE_H), dtype=input_ptr.dtype.element_ty)

    weight_ptrs = weight_ptr + pid_n * weight_stride_n + indices_h[:, None] * weight_stride_h + indices_h[None, :]
    weight = tl.load(weight_ptrs, mask=mask_h[:, None] & mask_h[None, :], other=0)

    for s in range(S - 1, -1, -1):
        output_grad_ptrs = (
            output_grad_ptr
            + indices_b[:, None] * input_stride_b
            + s * input_stride_s
            + pid_n * input_stride_n
            + indices_h[None, :]
        )
        output_grad = tl.load(output_grad_ptrs, mask=mask_bh, other=0)

        output_ptrs = (
            output_ptr
            + indices_b[:, None] * output_stride_b
            + s * output_stride_s
            + pid_n * output_stride_n
            + indices_h[None, :]
        )
        output = tl.load(output_ptrs, mask=mask_bh, other=0).to(tl.float32)

        r = (1 - output * output).to(output_grad.dtype)
        input_grad = (output_grad + input_state_grad) * r

        input_grad_ptrs = (
            input_grad_ptr
            + indices_b[:, None] * input_stride_b
            + s * input_stride_s
            + pid_n * input_stride_n
            + indices_h[None, :]
        )
        tl.store(input_grad_ptrs, input_grad, mask=mask_bh)

        input_state_grad = tl.dot(input_grad, weight.T).to(input_state_grad.dtype)


@cute_op(f"{LIBRARY_NAME}::{_KERNEL_NAME}", mutates_args={"input_grad"})
def rnn_backward_triton(
    input: torch.Tensor,
    weight: torch.Tensor,
    output: torch.Tensor,
    output_grad: torch.Tensor,
    input_grad: torch.Tensor,
    BLOCK_SIZE_B: int,
) -> None:
    B, S, N, H = input.size()

    BLOCK_SIZE_H = get_next_power_of_2(H)
    BLOCK_SIZE_H = max(16, BLOCK_SIZE_H)

    with torch.device(input.device):
        _rnn_backward_triton_kernel[ceil_divide(B, BLOCK_SIZE_B), N](
            input_ptr=input,
            input_stride_b=input.stride(0),
            input_stride_s=input.stride(1),
            input_stride_n=input.stride(2),
            weight_ptr=weight,
            weight_stride_n=weight.stride(0),
            weight_stride_h=weight.stride(1),
            output_ptr=output,
            output_stride_b=output.stride(0),
            output_stride_s=output.stride(1),
            output_stride_n=output.stride(2),
            output_grad_ptr=output_grad,
            input_grad_ptr=input_grad,
            B=B,
            S=S,
            H=H,
            BLOCK_SIZE_B=BLOCK_SIZE_B,
            BLOCK_SIZE_H=BLOCK_SIZE_H,
        )
