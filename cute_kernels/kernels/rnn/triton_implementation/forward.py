import torch
import triton
import triton.language as tl

from ....constants import LIBRARY_NAME
from ....math import ceil_divide
from ....triton_math import tanh
from ....utils import cute_op


_KERNEL_NAME = "rnn_forward_triton"


@triton.jit
def _rnn_forward_triton_kernel(
    input_ptr,
    weight_ptr,
    output_ptr,
    has_input_state: tl.constexpr,
    input_state_ptr,
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

    if not has_input_state:
        input_state = tl.zeros((BLOCK_SIZE_B, BLOCK_SIZE_H), dtype=input_ptr.dtype.element_ty)

    for s in range(S):
        input_ptrs = input_ptr + indices_b[:, None] * S * N * H + s * N * H + pid_n * H + indices_h[None, :]
        input = tl.load(input_ptrs, mask=mask_bh).to(tl.float32)

        weight_ptrs = weight_ptr + pid_n * H * H + indices_h[:, None] * H + indices_h[None, :]
        weight = tl.load(weight_ptrs, mask=mask_h[:, None] & mask_h[None, :], other=0)

        input = tl.dot(input_state, weight, input, out_dtype=tl.float32)

        input_state = input.to(tl.float32)
        input_state = tanh(input_state)
        input_state = input_state.to(input_ptr.dtype.element_ty)

        output_ptrs = (
            output_ptr + indices_b[:, None, None] * S * N * H + s * N * H + pid_n * H + indices_h[None, None, :]
        )
        tl.store(output_ptrs, input_state[:, None, :], mask=mask_bh[:, None, :])


@cute_op(f"{LIBRARY_NAME}::{_KERNEL_NAME}", mutates_args={"output"})
def rnn_forward_triton(
    input: torch.Tensor,
    weight: torch.Tensor,
    output: torch.Tensor,
    input_state: torch.Tensor | None,
    BLOCK_SIZE_B: int,
    BLOCK_SIZE_H: int,
) -> None:
    B, S, N, H = input.size()

    with torch.device(input.device):
        _rnn_forward_triton_kernel[ceil_divide(B, BLOCK_SIZE_B), N](
            input_ptr=input,
            weight_ptr=weight,
            output_ptr=output,
            has_input_state=input_state is not None,
            input_state_ptr=input_state,
            B=B,
            S=S,
            N=N,
            H=H,
            BLOCK_SIZE_B=BLOCK_SIZE_B,
            BLOCK_SIZE_H=BLOCK_SIZE_H,
        )
