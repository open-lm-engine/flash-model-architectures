# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import torch
import triton
import triton.language as tl

from ....constants import LIBRARY_NAME
from ....math import get_next_power_of_2
from ....triton_math import clamp, matmul
from ....utils import cute_op
from ...rnn.triton_implementation.backward import _activation_backward, _get_autotune_configs


@triton.jit
def _rnn_backward_update(y, W, dy, y_prev, ACTIVATION_FUNCTION: tl.constexpr, relu_negative_slope):
    dx = _activation_backward(
        y=y, dy=dy, ACTIVATION_FUNCTION=ACTIVATION_FUNCTION, relu_negative_slope=relu_negative_slope
    )

    dh = matmul(A=dx, B=W.T, C=None, output_dtype=dx.dtype)
    dW = matmul(A=y_prev.T, B=dx, output_dtype=dx.dtype)

    return dx, dW, dh


@triton.jit
def _load_previous_output(
    HAS_INPUT_STATE: tl.constexpr,
    h_ptr,
    h_stride_b,
    y_ptrs,
    pid_n,
    H,
    pid_b,
    indices_h,
    mask_h,
    BLOCK_SIZE_H: tl.constexpr,
    s,
    dtype,
):
    if s == 0:
        if HAS_INPUT_STATE:
            y_prev = tl.load(h_ptr + pid_b * h_stride_b + pid_n * H + indices_h[None, :], mask=pid_b)
        else:
            y_prev = tl.zeros((1, BLOCK_SIZE_H), dtype=dtype)
    else:
        y_prev = tl.load(y_ptrs, mask=mask_h[None, :])

    return y_prev


@triton.autotune(configs=_get_autotune_configs(), key=["BLOCK_SIZE_H"], reset_to_zero=["dW_ptr", "dWf_ptr", "dWr_ptr"])
@triton.jit
def gru_backward_input_dependent_weights_triton_kernel(
    W_ptr,
    W_stride_b,
    W_stride_s,
    W_stride_n,
    y_ptr,
    y_stride_b,
    y_stride_s,
    Wf_ptr,
    f_ptr,
    dxf_ptr,
    dWf_ptr,
    Wr_ptr,
    r_ptr,
    dxr_ptr,
    dWr_ptr,
    z_ptr,
    HAS_INPUT_STATE: tl.constexpr,
    h_ptr,
    h_stride_b,
    dy_ptr,
    dx_ptr,
    dW_ptr,
    HAS_GRADIENT_CLIPPING: tl.constexpr,
    gradient_clipping,
    S,
    H,
    BLOCK_SIZE_H: tl.constexpr,
):
    pid_b = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    indices_h = tl.arange(0, BLOCK_SIZE_H)

    mask_h = indices_h < H
    mask_hh = mask_h[:, None] & mask_h[None, :]

    dh = tl.zeros((1, BLOCK_SIZE_H), dtype=W_ptr.dtype.element_ty)

    indices = pid_b * y_stride_b + (S - 1) * y_stride_s + pid_n * H + indices_h[None, :]
    indices_W = (
        pid_b * W_stride_b + (S - 1) * W_stride_s + pid_n * W_stride_n + indices_h[None, :] * H + indices_h[None, :]
    )

    # backward counting reduces 1 instruction since we need to compare s == 0, otherwise we have to compare s == S - 1
    for s in range(S - 1, -1, -1):
        if HAS_GRADIENT_CLIPPING:
            dh = clamp(dh, min_value=-gradient_clipping, max_value=gradient_clipping)

        dy = tl.load(dy_ptr + indices, mask=mask_h[None, :]) + dh
        f = tl.load(f_ptr + indices, mask=mask_h[None, :])
        r = tl.load(r_ptr + indices, mask=mask_h[None, :])
        z = tl.load(z_ptr + indices, mask=mask_h[None, :])

        dx_ptrs = dx_ptr + indices
        dxf_ptrs = dxf_ptr + indices
        dxr_ptrs = dxr_ptr + indices

        indices -= y_stride_s

        y_prev = _load_previous_output(
            HAS_INPUT_STATE=HAS_INPUT_STATE,
            h_ptr=h_ptr,
            h_stride_b=h_stride_b,
            y_ptrs=y_ptr + indices,
            pid_n=pid_n,
            H=H,
            pid_b=pid_b,
            indices_h=indices_h,
            mask_h=mask_h,
            BLOCK_SIZE_H=BLOCK_SIZE_H,
            s=s,
            dtype=W_ptr.dtype.element_ty,
        )

        dh = f * dy
        dz = dy * (1 - f)
        df = dy * (y_prev - z)

        dx, dW, drh = _rnn_backward_update(
            y=z,
            W=tl.load(W_ptr + indices_W, mask=mask_hh),
            dy=dz,
            y_prev=r * y_prev,
            ACTIVATION_FUNCTION="tanh",
            relu_negative_slope=None,
        )

        dh += drh * r
        tl.store(dx_ptrs, dx, mask=mask_h[None, :])
        tl.store(dW_ptr + indices_W, dW, mask=mask_hh)

        dxf, dWf, _dh = _rnn_backward_update(
            y=f,
            W=tl.load(Wf_ptr + indices_W, mask=mask_hh),
            dy=df,
            y_prev=y_prev,
            ACTIVATION_FUNCTION="sigmoid",
            relu_negative_slope=None,
        )

        dh += _dh
        tl.store(dxf_ptrs, dxf, mask=mask_h[None, :])
        tl.store(dWf_ptr + indices_W, dWf, mask=mask_hh)

        dr = drh * y_prev

        dxr, dWr, _dh = _rnn_backward_update(
            y=r,
            W=tl.load(Wr_ptr + indices_W, mask=mask_hh),
            dy=dr,
            y_prev=y_prev,
            ACTIVATION_FUNCTION="sigmoid",
            relu_negative_slope=None,
        )

        dh += _dh
        tl.store(dxr_ptrs, dxr, mask=mask_h[None, :])
        tl.store(dWr_ptr + indices_W, dWr, mask=mask_hh)

        indices_W -= W_stride_s


@cute_op(
    f"{LIBRARY_NAME}::gru_backward_input_dependent_weights_triton",
    mutates_args={
        "forget_input_grad",
        "forget_weight_grad",
        "reset_input_grad",
        "reset_weight_grad",
        "input_grad",
        "weight_grad",
    },
)
def gru_backward_input_dependent_weights_triton(
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

    with torch.device(output.device):
        gru_backward_input_dependent_weights_triton_kernel[B, N](
            W_ptr=weight,
            W_stride_n=weight.stride(0),
            y_ptr=output,
            y_stride_b=output.stride(0),
            y_stride_s=output.stride(1),
            Wf_ptr=forget_weight,
            f_ptr=forget_gate,
            dxf_ptr=forget_input_grad,
            dWf_ptr=forget_weight_grad,
            Wr_ptr=reset_weight,
            r_ptr=reset_gate,
            dxr_ptr=reset_input_grad,
            dWr_ptr=reset_weight_grad,
            z_ptr=output_update,
            HAS_INPUT_STATE=input_state is not None,
            h_ptr=input_state,
            h_stride_b=None if input_state is None else input_state.stride(0),
            dy_ptr=output_grad,
            dx_ptr=input_grad,
            dW_ptr=weight_grad,
            HAS_GRADIENT_CLIPPING=gradient_clipping is not None,
            gradient_clipping=gradient_clipping,
            S=S,
            H=H,
            BLOCK_SIZE_H=BLOCK_SIZE_H,
        )
