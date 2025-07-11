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
    for num_warps in get_powers_of_2(1, 8):
        for num_stages in range(1, 5):
            for BLOCK_SIZE_B in [1] + get_powers_of_2(16, 32):
                configs.append(
                    triton.Config({"BLOCK_SIZE_B": BLOCK_SIZE_B}, num_stages=num_stages, num_warps=num_warps)
                )

    return configs


@triton.autotune(configs=_get_autotune_configs(), key=["BLOCK_SIZE_H", "BLOCK_SIZE_D"])
@triton.jit
def hippo_rnn_forward_triton_kernel(
    x_ptr,
    x_stride_b,
    x_stride_s,
    W_ptr,
    W_stride_n,
    Wh_ptr,
    Wh_stride_n,
    Wc_ptr,
    hippo_A_ptr,
    hippo_B_ptr,
    h0_ptr,
    h0_stride_b,
    c0_ptr,
    c0_stride_b,
    y_ptr,
    c_ptr,
    c_stride_b,
    c_stride_s,
    B,
    S,
    H,
    D,
    BLOCK_SIZE_B: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
):
    pid_b = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    indices_b = pid_b * BLOCK_SIZE_B + tl.arange(0, BLOCK_SIZE_B)
    indices_h = tl.arange(0, BLOCK_SIZE_H)
    indices_d = tl.arange(0, BLOCK_SIZE_D)

    mask_b = indices_b < B
    mask_h = indices_h < H
    mask_d = indices_d < D

    mask_bh = mask_b[:, None] & mask_h[None, :]
    mask_bd = mask_b[:, None] & mask_d[None, :]

    W = tl.load(
        W_ptr + pid_n * W_stride_n + indices_h[:, None] * H + indices_h[None, :],
        mask=mask_h[:, None] & mask_h[None, :],
    )

    Wh = tl.load(
        Wh_ptr + pid_n * Wh_stride_n + indices_d[:, None] * H + indices_h[None, :],
        mask=mask_d[:, None] & mask_h[None, :],
    )

    Wc = tl.load(Wc_ptr + pid_n * H + indices_h, mask=mask_h)

    if h0_ptr is None:
        h = tl.zeros((BLOCK_SIZE_B, BLOCK_SIZE_H), dtype=x_ptr.dtype.element_ty)
    else:
        h = tl.load(h0_ptr + indices_b[:, None] * h0_stride_b + pid_n * H + indices_h[None, :], mask=mask_bh)

    if c0_ptr is None:
        c = tl.zeros((BLOCK_SIZE_B, BLOCK_SIZE_D), dtype=x_ptr.dtype.element_ty)
    else:
        c = tl.load(c0_ptr + indices_b[:, None] * c0_stride_b + pid_n * D + indices_d[None, :], mask=mask_bd)

    I = tl.where(indices_d[:, None] == indices_d[None, :], 1, 0).to(x_ptr.dtype.element_ty)

    hippo_A = tl.load(
        hippo_A_ptr + indices_d[:, None] * D + indices_d[None, :], mask=mask_d[:, None] & mask_d[None, :]
    )
    hippo_B = tl.load(hippo_B_ptr + indices_d, mask=mask_d)

    indices_x = indices_b[:, None] * x_stride_b + pid_n * H + indices_h[None, :]
    indices_c = indices_b[:, None] * c_stride_b + pid_n * D + indices_d[None, :]

    for s in range(1, S + 1):
        x = tl.load(x_ptr + indices_x, mask=mask_bh)

        h = matmul(A=h, B=W, C=x, output_dtype=tl.float32)
        h = matmul(A=c, B=Wh, C=h, output_dtype=tl.float32)
        h = tanh(h, output_dtype=x.dtype)

        s1 = (1 / s).to(c.dtype)
        f = matmul(A=h, B=Wc[:, None], C=None, output_dtype=x.dtype)
        c = matmul(A=c, B=(I - hippo_A * s1).T, C=None, output_dtype=tl.float32)
        c = matmul(A=f, B=(hippo_B * s1)[None, :], C=c, output_dtype=x.dtype)

        tl.store(y_ptr + indices_x, h, mask=mask_bh)
        tl.store(c_ptr + indices_c, c, mask=mask_bd)

        indices_x += x_stride_s
        indices_c += c_stride_s


@custom_op(f"{LIBRARY_NAME}::hippo_rnn_forward_triton", mutates_args={"output", "hippo_output"})
def hippo_rnn_forward_triton(
    input: torch.Tensor,
    weight: torch.Tensor,
    hippo_weight: torch.Tensor,
    compress_weight: torch.Tensor,
    hippo_A: torch.Tensor,
    hippo_B: torch.Tensor,
    input_state: torch.Tensor | None,
    hippo_state: torch.Tensor | None,
    output: torch.Tensor,
    hippo_output: torch.Tensor,
) -> None:
    B, S, N, H = input.size()
    D = hippo_A.size(0)

    BLOCK_SIZE_H = get_next_power_of_2(H)
    BLOCK_SIZE_H = max(16, BLOCK_SIZE_H)

    if D == 1:
        BLOCK_SIZE_D = 1
    else:
        BLOCK_SIZE_D = get_next_power_of_2(D)
        BLOCK_SIZE_D = max(16, BLOCK_SIZE_D)

    GRID = lambda meta: (ceil_divide(B, meta["BLOCK_SIZE_B"]), N)

    with torch.device(input.device):
        hippo_rnn_forward_triton_kernel[GRID](
            x_ptr=input,
            x_stride_b=input.stride(0),
            x_stride_s=input.stride(1),
            W_ptr=weight,
            W_stride_n=weight.stride(0),
            Wh_ptr=hippo_weight,
            Wh_stride_n=hippo_weight.stride(0),
            Wc_ptr=compress_weight,
            hippo_A_ptr=hippo_A,
            hippo_B_ptr=hippo_B,
            h0_ptr=input_state,
            h0_stride_b=None if input_state is None else input_state.stride(0),
            c0_ptr=hippo_state,
            c0_stride_b=None if hippo_state is None else hippo_state.stride(0),
            y_ptr=output,
            c_ptr=hippo_output,
            c_stride_b=hippo_output.stride(0),
            c_stride_s=hippo_output.stride(1),
            B=B,
            S=S,
            H=H,
            D=D,
            BLOCK_SIZE_H=BLOCK_SIZE_H,
            BLOCK_SIZE_D=BLOCK_SIZE_D,
        )
