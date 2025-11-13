# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import torch
import triton
import triton.language as tl
from torch.library import custom_op

from ....constants import LIBRARY_NAME
from ....math import ceil_divide, get_next_power_of_2, get_powers_of_2
from ....triton_utils import matmul, tanh


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
    y_stride,
    cu_seqlens_ptr,
    cu_seqlens_stride,
    IS_MAX_SEQLEN_TENSOR: tl.constexpr,
    max_seqlen_ptr,
    B,
    S,
    H,
    Gx,
    Gw,
    BLOCK_SIZE_B: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
):
    BLOCK_ID_B = tl.program_id(axis=0)
    BLOCK_ID_N = tl.program_id(axis=1)

    BLOCK_ID_Nx = BLOCK_ID_N // Gx
    BLOCK_ID_Nw = BLOCK_ID_N // Gw

    BLOCK_B = BLOCK_ID_B * BLOCK_SIZE_B + tl.arange(0, BLOCK_SIZE_B)
    BLOCK_H = tl.arange(0, BLOCK_SIZE_H)

    MASK_B = BLOCK_B < B
    MASK_H = BLOCK_H < H

    MASK_BH = MASK_B[:, None] & MASK_H[None, :]
    MASK_HH = MASK_H[:, None] & MASK_H[None, :]

    W = tl.load(
        W_ptr + BLOCK_ID_Nw * W_stride[0] + BLOCK_H[:, None] * W_stride[1] + BLOCK_H[None, :] * W_stride[2],
        mask=MASK_HH,
    )

    if h0_ptr is None:
        h = tl.zeros((BLOCK_SIZE_B, BLOCK_SIZE_H), dtype=x_ptr.dtype.element_ty)
    else:
        h = tl.load(
            h0_ptr + BLOCK_B[:, None] * h0_stride[0] + BLOCK_ID_N * h0_stride[1] + BLOCK_H[None, :] * h0_stride[2],
            mask=MASK_BH,
        )

    IS_VARLEN: tl.constexpr = cu_seqlens_ptr is not None

    if IS_VARLEN:
        cu_seqlens_ptrs = cu_seqlens_ptr + BLOCK_B[:, None] * cu_seqlens_stride[0]
        start = tl.load(cu_seqlens_ptrs, mask=MASK_B[:, None])
        end = tl.load(cu_seqlens_ptrs + cu_seqlens_stride[0], mask=MASK_B[:, None])

        S = tl.load(max_seqlen_ptr) if IS_MAX_SEQLEN_TENSOR else max_seqlen_ptr

        x_ptrs = x_ptr + start * x_stride[0] + BLOCK_ID_Nx * x_stride[1] + BLOCK_H[None, :] * x_stride[2]
        y_ptrs = y_ptr + start * y_stride[0] + BLOCK_ID_N * y_stride[1] + BLOCK_H[None, :] * y_stride[2]
    else:
        x_ptrs = x_ptr + BLOCK_B[:, None] * x_stride[0] + BLOCK_ID_Nx * x_stride[2] + BLOCK_H[None, :] * x_stride[3]
        y_ptrs = y_ptr + BLOCK_B[:, None] * y_stride[0] + BLOCK_ID_N * y_stride[2] + BLOCK_H[None, :] * y_stride[3]

    for _ in range(S):
        MASK = ((start < end) & MASK_H[None, :]) if IS_VARLEN else MASK_BH

        x = tl.load(x_ptrs, mask=MASK)
        h = matmul(A=h, B=W, C=x, output_dtype=tl.float32)
        h = tanh(h, output_dtype=x.dtype)
        tl.store(y_ptrs, h, mask=MASK)

        x_ptrs += x_stride[1 - IS_VARLEN]
        y_ptrs += y_stride[1 - IS_VARLEN]

        if IS_VARLEN:
            start += 1


@custom_op(f"{LIBRARY_NAME}::rnn_forward_triton", mutates_args={"output"})
def rnn_forward_triton(
    input: torch.Tensor,
    weight: torch.Tensor,
    input_state: torch.Tensor | None,
    output: torch.Tensor,
    cu_seqlens: torch.Tensor | None,
    max_seqlen_tensor: torch.Tensor | None,
    max_seqlen: int | None,
) -> None:
    if cu_seqlens is None:
        B, S, Nx, H = input.size()
    else:
        B = cu_seqlens.size(0) - 1
        S = None
        _, Nx, H = input.size()

    Nw = weight.size(0)
    N = max(Nx, Nw)

    is_max_seqlen_tensor = max_seqlen_tensor is not None

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
            y_stride=output.stride(),
            cu_seqlens_ptr=cu_seqlens,
            cu_seqlens_stride=None if cu_seqlens is None else cu_seqlens.stride(),
            IS_MAX_SEQLEN_TENSOR=is_max_seqlen_tensor,
            max_seqlen_ptr=max_seqlen_tensor if is_max_seqlen_tensor else max_seqlen,
            B=B,
            S=S,
            H=H,
            Gx=N // Nx,
            Gw=N // Nw,
            BLOCK_SIZE_H=BLOCK_SIZE_H,
        )
