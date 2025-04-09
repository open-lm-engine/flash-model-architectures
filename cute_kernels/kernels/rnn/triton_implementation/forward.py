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
    BLOCK_SIZE_HI: tl.constexpr,
    BLOCK_SIZE_HO: tl.constexpr,
    allow_tf32: tl.constexpr = False,
):
    pid_b = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    pid_ho = tl.program_id(axis=2)

    indices_b = pid_b * BLOCK_SIZE_B + tl.arange(0, BLOCK_SIZE_B)
    indices_ho = pid_ho * BLOCK_SIZE_HO + tl.arange(0, BLOCK_SIZE_HO)

    mask_b = indices_b < B
    mask_ho = indices_ho < H
    mask_bi = mask_b[:, None] & mask_ho[None, :]

    if has_input_state:
        input_state_ptrs = input_state_ptr + indices_b[:, None] * H + indices_ho[None, :]
        input_state = tl.load(input_state_ptrs, mask=mask_bi)
    else:
        input_state = tl.zeros((BLOCK_SIZE_B, BLOCK_SIZE_HO), dtype=input_ptr.dtype.element_ty)

    for s in range(S):
        for h in range(tl.cdiv(H, BLOCK_SIZE_HI)):
            indices_hi = h * BLOCK_SIZE_HI + tl.arange(0, BLOCK_SIZE_HI)
            mask_hi = indices_hi < H

            indices = pid_n * H * H + indices_hi[:, None] * H + indices_ho[None, :]
            mask = mask_hi[:, None] & mask_ho[None, :]

            weight_ptrs = weight_ptr + indices
            weight = tl.load(weight_ptrs, mask=mask)

            indices = indices_b[:, None] * S * N * H + s * N * H + pid_n * H + indices_ho[None, :]
            input_ptrs = input_ptr + indices
            input = tl.load(input_ptrs, mask=mask_bi).to(tl.float32)

            # weight -> (BLOCK_SIZE_HI, BLOCK_SIZE_HO)
            # input -> (BLOCK_SIZE_B, BLOCK_SIZE_HO)

            input_state = tl.dot(input_state, weight.T, input, allow_tf32=allow_tf32, out_dtype=tl.float32)
            input_state = tanh(input_state)
            input_state = input_state.to(input_ptr.dtype.element_ty)

        output_ptrs = (
            output_ptr + indices_b[:, None, None] * S * N * H + s * N * H + pid_n * h + indices_ho[None, None, :]
        )
        tl.store(output_ptrs, input_state[:, None, :], mask=mask_bi[:, None, :])


@cute_op(f"{LIBRARY_NAME}::{_KERNEL_NAME}", mutates_args={"output"})
def rnn_forward_triton(
    input: torch.Tensor,
    weight: torch.Tensor,
    output: torch.Tensor,
    input_state: torch.Tensor | None,
    BLOCK_SIZE_B: int,
    BLOCK_SIZE_HI: int,
    BLOCK_SIZE_HO: int,
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
            BLOCK_SIZE_HI=BLOCK_SIZE_HI,
            BLOCK_SIZE_HO=BLOCK_SIZE_HO,
        )
