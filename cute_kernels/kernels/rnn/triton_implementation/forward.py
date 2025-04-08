import torch
import triton
import triton.language as tl

from ....constants import LIBRARY_NAME
from ....math import ceil_divide
from ....utils import cute_op


_KERNEL_NAME = "rnn_forward_triton"


@triton.jit
def _tanh(x):
    return 2 * tl.sigmoid(x) - 1


@triton.jit
def _rnn_forward_triton_kernel(
    input_ptr,
    weight_ptr,
    has_bias: tl.constexpr,
    bias_ptr,
    output_ptr,
    has_input_state: tl.constexpr,
    starting_state_ptr,
    B,
    S,
    N,
    H,
    I,
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
    mask_i = indices_i < I
    mask_bi = mask_b[:, None] & mask_i[None, :]

    if has_input_state:
        starting_state_ptrs = starting_state_ptr + indices_b[:, None] * I + indices_i[None, :]
        starting_state = tl.load(starting_state_ptrs, mask=mask_bi)
    else:
        starting_state = tl.zeros((BLOCK_SIZE_B, BLOCK_SIZE_I), dtype=tl.float32)

    for s in range(S):
        for h in range(tl.cdiv(H, BLOCK_SIZE_H)):
            indices_h = h * BLOCK_SIZE_H + tl.arange(0, BLOCK_SIZE_H)
            mask_h = indices_h < H

            indices = pid_n * I * H + indices_i[:, None] * H + indices_h[None, :]
            mask = mask_i[:, None] & mask_h[None, :]

            weight_ptrs = weight_ptr + indices
            # weight -> (BLOCK_SIZE_I, BLOCK_SIZE_H)
            weight = tl.load(weight_ptrs, mask=mask)

            indices = indices_b[:, None] * S * N * H + s * N * H + pid_i * H + indices_h[None, :]
            mask = mask_b[:, None] & mask_h[None, :]

            input_ptrs = input_ptr + indices
            # input -> (BLOCK_SIZE_B, BLOCK_SIZE_H)
            input = tl.load(input_ptrs, mask=mask)

            starting_state = tl.dot(input, weight.T, starting_state, allow_tf32=allow_tf32, out_dtype=tl.float32)

            if has_bias:
                bias_ptrs = bias_ptr + pid_n * I + indices_i[:, None]
                bias = tl.load(bias_ptrs, mask=mask_i[None, :])
                starting_state += bias

            starting_state = _tanh(starting_state)

        output_ptrs = output_ptr + indices_b[:, None] * S * I + s * I + indices_i[None, :]
        tl.store(output_ptrs, starting_state[:, None, :], mask=mask_bi[:, None, :])


@cute_op(f"{LIBRARY_NAME}::{_KERNEL_NAME}", mutates_args={"y", "output_state"})
def rnn_forward_triton(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    output: torch.Tensor,
    starting_state: torch.Tensor | None,
    state_size: int,
    BLOCK_SIZE_B: int,
    BLOCK_SIZE_H: int,
    BLOCK_SIZE_I: int,
) -> None:
    B, S, N, H = input.size()

    with torch.device(input.device):
        _rnn_forward_triton_kernel[ceil_divide(B, BLOCK_SIZE_B), N, ceil_divide(state_size, BLOCK_SIZE_I)](
            input_ptr=input,
            weight_ptr=weight,
            has_bias=bias is not None,
            bias_ptr=bias,
            output_ptr=output,
            has_input_state=starting_state is not None,
            input_state_ptr=starting_state,
            B=B,
            S=S,
            N=N,
            H=H,
            I=state_size,
            BLOCK_SIZE_B=BLOCK_SIZE_B,
            BLOCK_SIZE_H=BLOCK_SIZE_H,
            BLOCK_SIZE_I=BLOCK_SIZE_I,
        )
