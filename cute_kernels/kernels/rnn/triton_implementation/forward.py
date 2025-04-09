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
    BLOCK_SIZE_I: tl.constexpr,
    allow_tf32: tl.constexpr = True,
):
    pid_b = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    pid_i = tl.program_id(axis=2)

    indices_b = pid_b * BLOCK_SIZE_B + tl.arange(0, BLOCK_SIZE_B)
    indices_i = pid_i * BLOCK_SIZE_I + tl.arange(0, BLOCK_SIZE_I)

    mask_b = indices_b < B
    mask_i = indices_i < H
    mask_bi = mask_b[:, None] & mask_i[None, :]

    if has_input_state:
        input_state_ptrs = input_state_ptr + indices_b[:, None] * H + indices_i[None, :]
        input_state = tl.load(input_state_ptrs, mask=mask_bi)
    else:
        input_state = tl.zeros((BLOCK_SIZE_B, BLOCK_SIZE_I), dtype=tl.float32)

    for s in range(S):
        for h in range(tl.cdiv(H, BLOCK_SIZE_H)):
            indices_h = h * BLOCK_SIZE_H + tl.arange(0, BLOCK_SIZE_H)
            mask_h = indices_h < H

            indices = pid_n * H * H + indices_i[:, None] * H + indices_h[None, :]
            mask = mask_i[:, None] & mask_h[None, :]

            weight_ptrs = weight_ptr + indices
            # weight -> (BLOCK_SIZE_I, BLOCK_SIZE_H)
            weight = tl.load(weight_ptrs, mask=mask)

            indices = indices_b[:, None] * S * N * H + s * N * H + pid_i * H + indices_h[None, :]
            mask = mask_b[:, None] & mask_h[None, :]

            input_ptrs = input_ptr + indices
            # input -> (BLOCK_SIZE_B, BLOCK_SIZE_H)
            input = tl.load(input_ptrs, mask=mask)

            input_state = tl.dot(input_state, weight.T, input, allow_tf32=allow_tf32, out_dtype=tl.float32)
            input_state = tanh(input_state)

        output_ptrs = output_ptr + indices_b[:, None] * S * H + s * H + indices_i[None, :]
        tl.store(output_ptrs, input_state[:, None, :], mask=mask_bi[:, None, :])


@cute_op(f"{LIBRARY_NAME}::{_KERNEL_NAME}", mutates_args={"output"})
def rnn_forward_triton(
    input: torch.Tensor,
    weight: torch.Tensor,
    output: torch.Tensor,
    input_state: torch.Tensor | None,
    BLOCK_SIZE_B: int,
    BLOCK_SIZE_H: int,
    BLOCK_SIZE_I: int,
) -> None:
    B, S, N, H = input.size()

    with torch.device(input.device):
        _rnn_forward_triton_kernel[ceil_divide(B, BLOCK_SIZE_B), N, ceil_divide(H, BLOCK_SIZE_I)](
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
            BLOCK_SIZE_I=BLOCK_SIZE_I,
        )
