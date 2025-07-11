# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import torch
import triton
import triton.language as tl
from torch.library import custom_op

from ....constants import LIBRARY_NAME
from ....math import ceil_divide, get_next_power_of_2, get_powers_of_2
from ....triton_math import clamp, matmul, tanh_backward


def _get_autotune_configs() -> list[triton.Config]:
    configs = []
    for num_warps in get_powers_of_2(1, 8):
        for num_stages in range(1, 5):
            for BLOCK_SIZE_B in [1] + get_powers_of_2(16, 32):
                configs.append(
                    triton.Config({"BLOCK_SIZE_B": BLOCK_SIZE_B}, num_stages=num_stages, num_warps=num_warps)
                )

    return configs


@triton.autotune(
    configs=_get_autotune_configs(),
    key=["BLOCK_SIZE_H", "BLOCK_SIZE_D"],
    reset_to_zero=["dW_ptr", "dWh_ptr", "dWc_ptr"],
)
@triton.jit
def hippo_rnn_backward_triton_kernel(
    W_ptr,
    W_stride_n,
    Wh_ptr,
    Wh_stride_n,
    Wc_ptr,
    hippo_A_ptr,
    hippo_B_ptr,
    y_ptr,
    y_stride_b,
    y_stride_s,
    c_ptr,
    c_stride_b,
    c_stride_s,
    h0_ptr,
    h0_stride_b,
    c0_ptr,
    c0_stride_b,
    dy_ptr,
    dx_ptr,
    dW_ptr,
    dWh_ptr,
    dWc_ptr,
    gradient_clipping,
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
    mask_hh = mask_h[:, None] & mask_h[None, :]
    mask_bd = mask_b[:, None] & mask_d[None, :]
    mask_dh = mask_d[:, None] & mask_h[None, :]

    dh = tl.zeros((BLOCK_SIZE_B, BLOCK_SIZE_H), dtype=W_ptr.dtype.element_ty)
    dc = tl.zeros((BLOCK_SIZE_B, BLOCK_SIZE_D), dtype=W_ptr.dtype.element_ty)
    dW = tl.zeros((BLOCK_SIZE_H, BLOCK_SIZE_H), dtype=tl.float32)
    dWh = tl.zeros((BLOCK_SIZE_D, BLOCK_SIZE_H), dtype=tl.float32)
    dWc = tl.zeros((BLOCK_SIZE_H, 1), dtype=tl.float32)

    indices_W = pid_n * W_stride_n + indices_h[:, None] * H + indices_h[None, :]
    indices_Wh = pid_n * Wh_stride_n + indices_d[:, None] * H + indices_h[None, :]
    indices_Wc = pid_n * H + indices_h

    W = tl.load(W_ptr + indices_W, mask=mask_hh)
    Wh = tl.load(Wh_ptr + indices_Wh, mask=mask_dh)
    Wc = tl.load(Wc_ptr + indices_Wc, mask=mask_h)

    indices_y = indices_b[:, None] * y_stride_b + (S - 1) * y_stride_s + pid_n * H + indices_h[None, :]
    indices_c = indices_b[:, None] * c_stride_b + (S - 1) * c_stride_s + pid_n * D + indices_d[None, :]

    I = tl.where(indices_d[:, None] == indices_d[None, :], 1, 0).to(y_ptr.dtype.element_ty)

    hippo_A = tl.load(
        hippo_A_ptr + indices_d[:, None] * D + indices_d[None, :], mask=mask_d[:, None] & mask_d[None, :]
    )
    hippo_B = tl.load(hippo_B_ptr + indices_d, mask=mask_d)

    y = tl.load(y_ptr + indices_y, mask=mask_bh)

    # backward counting reduces 1 instruction since we need to compare s == 0, otherwise we have to compare s == S - 1
    for s in range(S, 0, -1):
        if gradient_clipping is not None:
            dh = clamp(dh, min_value=-gradient_clipping, max_value=gradient_clipping)

        dy = tl.load(dy_ptr + indices_y, mask=mask_bh) + dh

        s1 = (1 / s).to(y.dtype)
        df = matmul(A=dc, B=(hippo_B * s1)[:, None], C=None, output_dtype=y.dtype)
        dy = matmul(A=df, B=Wc[None, :], C=dy, output_dtype=y.dtype)
        dWc = matmul(A=y.T, B=df, C=None, output_dtype=dWc.dtype)

        dx_ptrs = dx_ptr + indices_y
        indices_y -= y_stride_s
        indices_c -= c_stride_s

        if s == 1:
            if h0_ptr is None:
                y_prev = tl.zeros((BLOCK_SIZE_B, BLOCK_SIZE_H), dtype=W.dtype)
            else:
                y_prev = tl.load(
                    h0_ptr + indices_b[:, None] * h0_stride_b + pid_n * H + indices_h[None, :], mask=mask_bh
                )

            if c0_ptr is None:
                c_prev = tl.zeros((BLOCK_SIZE_B, BLOCK_SIZE_D), dtype=W.dtype)
            else:
                c_prev = tl.load(
                    c0_ptr + indices_b[:, None] * c0_stride_b + pid_n * D + indices_d[None, :], mask=mask_bd
                )
        else:
            y_prev = tl.load(y_ptr + indices_y, mask=mask_bh)
            c_prev = tl.load(c_ptr + indices_c, mask=mask_bd)

        dx = dy * tanh_backward(y)
        dh = matmul(A=dx, B=W.T, C=None, output_dtype=dx.dtype)
        dW = matmul(A=y_prev.T, B=dx, C=dW, output_dtype=dW.dtype)
        dWh = matmul(A=c_prev.T, B=dx, C=dWh, output_dtype=dWh.dtype)
        _dc = matmul(A=dx, B=Wh.T, C=None, output_dtype=dx.dtype)

        dc = matmul(A=dc, B=I - hippo_A * s1, C=_dc, output_dtype=dx.dtype)

        tl.store(dx_ptrs, dx, mask=mask_bh)
        y = y_prev

    tl.atomic_add(dW_ptr + indices_W, dW, mask=mask_hh)
    tl.atomic_add(dWh_ptr + indices_Wh, dWh, mask=mask_dh)
    tl.atomic_add(dWc_ptr + indices_Wc[:, None], dWc, mask=mask_h[:, None])


@custom_op(
    f"{LIBRARY_NAME}::hippo_rnn_backward_triton",
    mutates_args={"input_grad", "weight_grad", "hippo_weight_grad", "compress_weight_grad"},
)
def hippo_rnn_backward_triton(
    weight: torch.Tensor,
    hippo_weight: torch.Tensor,
    compress_weight: torch.Tensor,
    hippo_A: torch.Tensor,
    hippo_B: torch.Tensor,
    output: torch.Tensor,
    hippo_output: torch.Tensor,
    input_state: torch.Tensor | None,
    hippo_state: torch.Tensor | None,
    output_grad: torch.Tensor,
    input_grad: torch.Tensor,
    weight_grad: torch.Tensor,
    hippo_weight_grad: torch.Tensor,
    compress_weight_grad: torch.Tensor,
    gradient_clipping: float | None,
) -> None:
    B, S, N, H = output.size()
    D = hippo_A.size(0)

    BLOCK_SIZE_H = get_next_power_of_2(H)
    BLOCK_SIZE_H = max(16, BLOCK_SIZE_H)

    if D == 1:
        BLOCK_SIZE_D = 1
    else:
        BLOCK_SIZE_D = get_next_power_of_2(D)
        BLOCK_SIZE_D = max(16, BLOCK_SIZE_D)

    GRID = lambda meta: (ceil_divide(B, meta["BLOCK_SIZE_B"]), N)

    with torch.device(output.device):
        hippo_rnn_backward_triton_kernel[GRID](
            W_ptr=weight,
            W_stride_n=weight.stride(0),
            Wh_ptr=hippo_weight,
            Wh_stride_n=hippo_weight.stride(0),
            Wc_ptr=compress_weight,
            hippo_A_ptr=hippo_A,
            hippo_B_ptr=hippo_B,
            y_ptr=output,
            y_stride_b=output.stride(0),
            y_stride_s=output.stride(1),
            c_ptr=hippo_output,
            c_stride_b=hippo_output.stride(0),
            c_stride_s=hippo_output.stride(1),
            h0_ptr=input_state,
            h0_stride_b=None if input_state is None else input_state.stride(0),
            c0_ptr=hippo_state,
            c0_stride_b=None if hippo_state is None else hippo_state.stride(0),
            dy_ptr=output_grad,
            dx_ptr=input_grad,
            dW_ptr=weight_grad,
            dWh_ptr=hippo_weight_grad,
            dWc_ptr=compress_weight_grad,
            gradient_clipping=gradient_clipping,
            B=B,
            S=S,
            H=H,
            D=D,
            BLOCK_SIZE_H=BLOCK_SIZE_H,
            BLOCK_SIZE_D=BLOCK_SIZE_D,
        )
