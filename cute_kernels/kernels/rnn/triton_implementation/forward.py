import torch
import triton
import triton.language as tl

from ....constants import LIBRARY_NAME
from ....math import ceil_divide
from ....utils import cute_op


_KERNEL_NAME = "rnn_forward_triton"


@triton.jit
def _rnn_forward_triton_kernel(
    x_ptr,
    w_ptr,
    bias_ptr,
    y_ptr,
    has_input_state: tl.constexpr,
    input_state_ptr,
    B,
    S,
    N,
    H,
    I,
    BLOCK_SIZE_B: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_I: tl.constexpr,
    allow_tf32: tl.constexpr,
):
    pid_b = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    pid_i = tl.program_id(axis=2)

    indices_b = pid_b * BLOCK_SIZE_B + tl.arange(0, BLOCK_SIZE_B)
    indices_i = pid_i * BLOCK_SIZE_I + tl.arange(0, BLOCK_SIZE_I)

    mask_b = indices_b < B
    mask_i = indices_i < I

    if has_input_state:
        input_state_ptrs = input_state_ptr + indices_b[:, None] * I + indices_i[None, :]
        input_state = tl.load(input_state_ptrs, mask=mask_b[:, None] & mask_i[None, :])

    for s in range(S):
        accumulator = tl.zeros((BLOCK_SIZE_B, BLOCK_SIZE_I), dtype=tl.float32)

        for h in range(tl.cdiv(H, BLOCK_SIZE_H)):
            indices_h = h * BLOCK_SIZE_H + tl.arange(0, BLOCK_SIZE_H)
            mask_h = indices_h < H

            indices = pid_n * I * H + indices_i[:, None] * H + indices_h[None, :]
            mask = mask_i[:, None] & mask_h[None, :]

            w_ptrs = w_ptr + indices
            # w -> (BLOCK_SIZE_I, BLOCK_SIZE_H)
            w = tl.load(w_ptrs, mask=mask)

            indices = indices_b[:, None] * S * N * H + s * N * H + pid_i * H + indices_h[None, :]
            mask = mask_b[:, None] & mask_h[None, :]

            x_ptrs = x_ptr + indices
            # x -> (BLOCK_SIZE_B, BLOCK_SIZE_H)
            x = tl.load(x_ptrs, mask=mask)

            accumulator = tl.dot(x, w.T, accumulator, allow_tf32=allow_tf32, out_dtype=tl.float32)


@cute_op(f"{LIBRARY_NAME}::{_KERNEL_NAME}", mutates_args={"y", "output_state"})
def rnn_forward_triton(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    y: torch.Tensor,
    input_state: torch.Tensor | None,
    output_state: torch.Tensor,
    BLOCK_SIZE_B: int,
    BLOCK_SIZE_N: int,
) -> None:
    B, S, N, H = x.size()

    with torch.device(x.device):
        _rnn_forward_triton_kernel[ceil_divide(B, BLOCK_SIZE_B), N, ceil_divide(I, BLOCK_SIZE_I)](
            input_ptr=x, W_ptr=weight, bias_ptr=bias, y_ptr=y, input_state_ptr=input_state
        )
