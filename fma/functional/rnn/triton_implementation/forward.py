# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import torch
import triton
import triton.language as tl
from torch.library import custom_op

from ....constants import LIBRARY_NAME
from ....math import ceil_divide, get_next_power_of_2, get_powers_of_2
from ....triton_math import matmul, tanh


def _get_autotune_configs() -> list[triton.Config]:
    configs = []
    for num_warps in get_powers_of_2(4, 8):
        for num_stages in range(1, 5):
            for BLOCK_SIZE_B in [1] + get_powers_of_2(16, 32):
                configs.append(
                    triton.Config({"BLOCK_SIZE_B": BLOCK_SIZE_B}, num_stages=num_stages, num_warps=num_warps)
                )

    return configs


@triton.autotune(configs=_get_autotune_configs(), key=["BLOCK_SIZE_H"])
@triton.jit
def rnn_forward_triton_kernel(
    x_ptr,
    x_stride,
    W_ptr,
    W_stride,
    h0_ptr,
    h0_stride,
    y_ptr,
    B,
    S,
    H,
    BLOCK_SIZE_B: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
):
    BLOCK_ID_B = tl.program_id(axis=0)
    BLOCK_ID_N = tl.program_id(axis=1)

    BLOCK_B = BLOCK_ID_B * BLOCK_SIZE_B + tl.arange(0, BLOCK_SIZE_B)
    BLOCK_H = tl.arange(0, BLOCK_SIZE_H)

    mask_b = BLOCK_B < B
    mask_h = BLOCK_H < H

    mask_bh = mask_b[:, None] & mask_h[None, :]

    W = tl.load(
        W_ptr + BLOCK_ID_N * W_stride[0] + BLOCK_H[:, None] * W_stride[1] + BLOCK_H[None, :] * W_stride[2],
        mask=mask_h[:, None] & mask_h[None, :],
    )

    if h0_ptr is None:
        h = tl.zeros((BLOCK_SIZE_B, BLOCK_SIZE_H), dtype=x_ptr.dtype.element_ty)
    else:
        h = tl.load(
            h0_ptr + BLOCK_B[:, None] * h0_stride[0] + BLOCK_ID_N * h0_stride[1] + BLOCK_H[None, :] * h0_stride[2],
            mask=mask_bh,
        )

    BLOCK = BLOCK_B[:, None] * x_stride[0] + BLOCK_ID_N * x_stride[2] + BLOCK_H[None, :] * x_stride[3]

    for _ in range(S):
        x = tl.load(x_ptr + BLOCK, mask=mask_bh)
        h = matmul(A=h, B=W, C=x, output_dtype=tl.float32)
        h = tanh(h, output_dtype=x.dtype)
        tl.store(y_ptr + BLOCK, h, mask=mask_bh)

        BLOCK += x_stride[1]


@custom_op(f"{LIBRARY_NAME}::rnn_forward_triton", mutates_args={"output"})
def rnn_forward_triton(
    input: torch.Tensor, weight: torch.Tensor, input_state: torch.Tensor | None, output: torch.Tensor
) -> None:
    B, S, N, H = input.size()

    BLOCK_SIZE_H = get_next_power_of_2(H)
    BLOCK_SIZE_H = max(16, BLOCK_SIZE_H)
    GRID = lambda meta: (ceil_divide(B, meta["BLOCK_SIZE_B"]), N)

    with torch.device(input.device):
        rnn_forward_triton_kernel[GRID](
            x_ptr=input,
            x_stride=input.stride(),
            W_ptr=weight,
            W_stride=weight.stride(),
            h0_ptr=input_state,
            h0_stride=None if input_state is None else input_state.stride(),
            y_ptr=output,
            B=B,
            S=S,
            H=H,
            BLOCK_SIZE_H=BLOCK_SIZE_H,
        )
