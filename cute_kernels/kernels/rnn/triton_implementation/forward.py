import torch
import triton
import triton.language as tl

from ....constants import LIBRARY_NAME
from ....math import ceil_divide, get_next_power_of_2
from ....triton_math import tanh
from ....utils import cute_op


_KERNEL_NAME = "rnn_forward_triton"


@triton.jit
def _rnn_forward_triton_kernel(
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
    B,
    S,
    N,
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

    input_state = tl.zeros((BLOCK_SIZE_B, BLOCK_SIZE_H), dtype=input_ptr.dtype.element_ty)

    for s in range(S):
        input_ptrs = (
            input_ptr
            + indices_b[:, None] * input_stride_b
            + s * input_stride_s
            + pid_n * input_stride_n
            + indices_h[None, :]
        )
        input = tl.load(input_ptrs, mask=mask_bh, other=0).to(tl.float32)

        weight_ptrs = weight_ptr + pid_n * weight_stride_n + indices_h[:, None] * weight_stride_h + indices_h[None, :]
        weight = tl.load(weight_ptrs, mask=mask_h[:, None] & mask_h[None, :], other=0)

        input = tl.dot(input_state, weight, input, out_dtype=tl.float32)

        input_state = input.to(tl.float32)
        input_state = tanh(input_state)
        input_state = input_state.to(input_ptr.dtype.element_ty)

        output_ptrs = (
            output_ptr
            + indices_b[:, None, None, None] * output_stride_b
            + s * output_stride_s
            + pid_n * output_stride_n
            + indices_h[None, None, None, :]
        )
        tl.store(output_ptrs, input_state[:, None, None, :], mask=mask_bh[:, None, None, :])


@cute_op(f"{LIBRARY_NAME}::{_KERNEL_NAME}", mutates_args={"output"})
def rnn_forward_triton(
    input: torch.Tensor,
    weight: torch.Tensor,
    output: torch.Tensor,
    input_state: torch.Tensor | None,
    BLOCK_SIZE_B: int,
) -> None:
    B, S, N, H = input.size()

    BLOCK_SIZE_H = get_next_power_of_2(H)
    BLOCK_SIZE_H = max(16, BLOCK_SIZE_H)

    with torch.device(input.device):
        _rnn_forward_triton_kernel[ceil_divide(B, BLOCK_SIZE_B), N](
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
            B=B,
            S=S,
            N=N,
            H=H,
            BLOCK_SIZE_B=BLOCK_SIZE_B,
            BLOCK_SIZE_H=BLOCK_SIZE_H,
        )
