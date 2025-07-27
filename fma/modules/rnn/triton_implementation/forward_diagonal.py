# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import torch
import triton
import triton.language as tl
from torch.library import custom_op

from ....constants import LIBRARY_NAME
from ....math import ceil_divide, get_next_power_of_2, get_powers_of_2
from ....triton_math import tanh


def _get_autotune_configs() -> list[triton.Config]:
    configs = []
    for num_warps in get_powers_of_2(4, 8):
        for num_stages in range(1, 5):
            for BLOCK_SIZE_B in get_powers_of_2(1, 32):
                configs.append(
                    triton.Config({"BLOCK_SIZE_B": BLOCK_SIZE_B}, num_stages=num_stages, num_warps=num_warps)
                )

    return configs


@triton.autotune(configs=_get_autotune_configs(), key=["BLOCK_SIZE_N"])
@triton.jit
def diagonal_rnn_forward_triton_kernel(
    x_ptr,
    x_stride_b,
    W_ptr,
    h0_ptr,
    y_ptr,
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

    W = tl.load(W_ptr + indices_n, mask=mask_n)[None, :]

    if h0_ptr is None:
        h = tl.zeros((BLOCK_SIZE_B, BLOCK_SIZE_N), dtype=x_ptr.dtype.element_ty)
    else:
        h = tl.load(h0_ptr + indices_b[:, None] * N + indices_n[None, :], mask=mask_bn)

    indices = indices_b[:, None] * x_stride_b + indices_n[None, :]

    for _ in range(S):
        x = tl.load(x_ptr + indices, mask=mask_bn)
        h = W * h + x
        h = tanh(h)
        tl.store(y_ptr + indices, h, mask=mask_bn)

        indices += N


@custom_op(f"{LIBRARY_NAME}::diagonal_rnn_forward_triton", mutates_args={"output"})
def diagonal_rnn_forward_triton(
    input: torch.Tensor, weight: torch.Tensor, input_state: torch.Tensor | None, output: torch.Tensor
) -> None:
    B, S, N, _ = input.size()

    BLOCK_SIZE_N = min(1024, get_next_power_of_2(N))
    GRID = lambda meta: (ceil_divide(B, meta["BLOCK_SIZE_B"]), ceil_divide(N, meta["BLOCK_SIZE_N"]))

    with torch.device(input.device):
        diagonal_rnn_forward_triton_kernel[GRID](
            x_ptr=input,
            x_stride_b=input.stride(0),
            W_ptr=weight,
            h0_ptr=input_state,
            y_ptr=output,
            B=B,
            S=S,
            N=N,
            BLOCK_SIZE_N=BLOCK_SIZE_N,
        )
