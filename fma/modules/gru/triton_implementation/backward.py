# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import torch
import triton
import triton.language as tl
from torch.library import custom_op

from ....constants import LIBRARY_NAME
from ....math import ceil_divide, get_next_power_of_2
from ....triton_math import clamp, matmul, sigmoid_backward, tanh_backward
from ...rnn.triton_implementation.backward import _get_autotune_configs
from .forward import _get_autotune_configs


@triton.autotune(configs=_get_autotune_configs(), key=["BLOCK_SIZE_H"], reset_to_zero=["dW_ptr", "dWf_ptr", "dWr_ptr"])
@triton.jit
def gru_backward_triton_kernel(
    W_ptr,
    W_stride,
    y_ptr,
    y_stride,
    Wf_ptr,
    f_ptr,
    dxf_ptr,
    dWf_ptr,
    Wr_ptr,
    r_ptr,
    dxr_ptr,
    dWr_ptr,
    z_ptr,
    h0_ptr,
    h0_stride,
    dy_ptr,
    dx_ptr,
    dW_ptr,
    gradient_clipping,
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
    BLOCK_W = BLOCK_ID_N * W_stride[0] + BLOCK_H[:, None] * W_stride[1] + BLOCK_H[None, :] * W_stride[2]

    mask_b = BLOCK_B < B
    mask_h = BLOCK_H < H

    mask_bh = mask_b[:, None] & mask_h[None, :]
    mask_hh = mask_h[:, None] & mask_h[None, :]

    dh = tl.zeros((BLOCK_SIZE_B, BLOCK_SIZE_H), dtype=W_ptr.dtype.element_ty)
    dW = tl.zeros((BLOCK_SIZE_H, BLOCK_SIZE_H), dtype=tl.float32)
    dWf = tl.zeros((BLOCK_SIZE_H, BLOCK_SIZE_H), dtype=tl.float32)
    dWr = tl.zeros((BLOCK_SIZE_H, BLOCK_SIZE_H), dtype=tl.float32)

    W = tl.load(W_ptr + BLOCK_W, mask=mask_hh)
    Wf = tl.load(Wf_ptr + BLOCK_W, mask=mask_hh)
    Wr = tl.load(Wr_ptr + BLOCK_W, mask=mask_hh)

    indices = (
        BLOCK_B[:, None] * y_stride[0]
        + (S - 1) * y_stride[1]
        + BLOCK_ID_N * y_stride[2]
        + BLOCK_H[None, :] * y_stride[3]
    )

    # backward counting reduces 1 instruction since we need to compare s == 0, otherwise we have to compare s == S - 1
    for s in range(S - 1, -1, -1):
        if gradient_clipping is not None:
            dh = clamp(dh, min_value=-gradient_clipping, max_value=gradient_clipping)

        dy = tl.load(dy_ptr + indices, mask=mask_bh) + dh
        f = tl.load(f_ptr + indices, mask=mask_bh)
        r = tl.load(r_ptr + indices, mask=mask_bh)
        z = tl.load(z_ptr + indices, mask=mask_bh)

        dx_ptrs = dx_ptr + indices
        dxf_ptrs = dxf_ptr + indices
        dxr_ptrs = dxr_ptr + indices

        indices -= y_stride[1]

        if s == 0:
            if h0_ptr is None:
                y_prev = tl.zeros((BLOCK_SIZE_B, BLOCK_SIZE_H), dtype=W.dtype)
            else:
                y_prev = tl.load(
                    h0_ptr
                    + BLOCK_B[:, None] * h0_stride[0]
                    + BLOCK_ID_N * h0_stride[1]
                    + BLOCK_H[None, :] * h0_stride[2],
                    mask=mask_bh,
                )
        else:
            y_prev = tl.load(y_ptr + indices, mask=mask_bh)

        dh = f * dy
        dz = dy * (1 - f)
        df = dy * (y_prev - z)

        dx = dz * tanh_backward(z)
        drh = matmul(A=dx, B=W.T, C=None, output_dtype=dx.dtype)
        dW = matmul(A=(r * y_prev).T, B=dx, C=dW, output_dtype=dW.dtype)
        tl.store(dx_ptrs, dx, mask=mask_bh)

        dh += drh * r

        dxf = df * sigmoid_backward(f)
        dh = matmul(A=dxf, B=Wf.T, C=dh, output_dtype=dx.dtype)
        dWf = matmul(A=y_prev.T, B=dxf, C=dWf, output_dtype=dW.dtype)
        tl.store(dxf_ptrs, dxf, mask=mask_bh)

        dxr = drh * y_prev * sigmoid_backward(r)
        dh = matmul(A=dxr, B=Wr.T, C=dh, output_dtype=dx.dtype)
        dWr = matmul(A=y_prev.T, B=dxr, C=dWr, output_dtype=dW.dtype)
        tl.store(dxr_ptrs, dxr, mask=mask_bh)

    tl.atomic_add(dW_ptr + BLOCK_W, dW, mask=mask_hh, sem="relaxed")
    tl.atomic_add(dWf_ptr + BLOCK_W, dWf, mask=mask_hh, sem="relaxed")
    tl.atomic_add(dWr_ptr + BLOCK_W, dWr, mask=mask_hh, sem="relaxed")


@custom_op(
    f"{LIBRARY_NAME}::gru_backward_triton",
    mutates_args={
        "forget_input_grad",
        "forget_weight_grad",
        "reset_input_grad",
        "reset_weight_grad",
        "input_grad",
        "weight_grad",
    },
)
def gru_backward_triton(
    weight: torch.Tensor,
    output: torch.Tensor,
    forget_weight: torch.Tensor,
    forget_gate: torch.Tensor,
    forget_input_grad: torch.Tensor,
    forget_weight_grad: torch.Tensor,
    reset_weight: torch.Tensor,
    reset_gate: torch.Tensor,
    reset_input_grad: torch.Tensor,
    reset_weight_grad: torch.Tensor,
    output_update: torch.Tensor,
    input_state: torch.Tensor | None,
    output_grad: torch.Tensor,
    input_grad: torch.Tensor,
    weight_grad: torch.Tensor,
    gradient_clipping: float | None,
) -> None:
    B, S, N, H = output.size()

    BLOCK_SIZE_H = get_next_power_of_2(H)
    BLOCK_SIZE_H = max(16, BLOCK_SIZE_H)
    GRID = lambda meta: (ceil_divide(B, meta["BLOCK_SIZE_B"]), N)

    with torch.device(output.device):
        gru_backward_triton_kernel[GRID](
            W_ptr=weight,
            W_stride=weight.stride(),
            y_ptr=output,
            y_stride=output.stride(),
            Wf_ptr=forget_weight,
            f_ptr=forget_gate,
            dxf_ptr=forget_input_grad,
            dWf_ptr=forget_weight_grad,
            Wr_ptr=reset_weight,
            r_ptr=reset_gate,
            dxr_ptr=reset_input_grad,
            dWr_ptr=reset_weight_grad,
            z_ptr=output_update,
            h0_ptr=input_state,
            h0_stride_b=None if input_state is None else input_state.stride(0),
            dy_ptr=output_grad,
            dx_ptr=input_grad,
            dW_ptr=weight_grad,
            gradient_clipping=gradient_clipping,
            B=B,
            S=S,
            H=H,
            BLOCK_SIZE_H=BLOCK_SIZE_H,
        )
