# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import torch
import triton
import triton.language as tl
from torch.library import custom_op

from ....constants import LIBRARY_NAME
from ....math import ceil_divide, get_next_power_of_2
from ....triton_math import matmul, sigmoid, tanh
from ...rnn.triton_implementation.forward import _get_autotune_configs


@triton.autotune(configs=_get_autotune_configs(), key=["BLOCK_SIZE_H"])
@triton.jit
def gru_forward_triton_kernel(
    x_ptr,
    x_stride,
    W_ptr,
    W_stride,
    xf_ptr,
    Wf_ptr,
    f_ptr,
    xr_ptr,
    Wr_ptr,
    r_ptr,
    z_ptr,
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
    mask_hh = mask_h[:, None] & mask_h[None, :]

    BLOCK = BLOCK_ID_N * W_stride + BLOCK_H[:, None] * H + BLOCK_H[None, :]

    W = tl.load(W_ptr + BLOCK, mask=mask_hh)
    Wf = tl.load(Wf_ptr + BLOCK, mask=mask_hh)
    Wr = tl.load(Wr_ptr + BLOCK, mask=mask_hh)

    if h0_ptr is None:
        h = tl.zeros((BLOCK_SIZE_B, BLOCK_SIZE_H), dtype=x_ptr.dtype.element_ty)
    else:
        h = tl.load(
            h0_ptr + BLOCK_B[:, None] * h0_stride[0] + BLOCK_ID_N * h0_stride[1] + BLOCK_H[None, :] * h0_stride[2],
            mask=mask_bh,
        )

    BLOCK = BLOCK_B[:, None] * x_stride[0] + BLOCK_ID_N * x_stride[2] + BLOCK_H[None, :] * x_stride[3]

    for _ in range(S):
        x = tl.load(xr_ptr + BLOCK, mask=mask_bh)
        r = matmul(A=h, B=Wr, C=x, output_dtype=tl.float32)
        r = sigmoid(r, output_dtype=x.dtype)
        tl.store(r_ptr + BLOCK, r, mask=mask_bh)

        x = tl.load(x_ptr + BLOCK, mask=mask_bh)
        z = matmul(A=h * r, B=W, C=x, output_dtype=tl.float32)
        z = tanh(z, output_dtype=x.dtype)
        tl.store(z_ptr + BLOCK, z, mask=mask_bh)

        x = tl.load(xf_ptr + BLOCK, mask=mask_bh)
        f = matmul(A=h, B=Wf, C=x, output_dtype=tl.float32)
        f = sigmoid(f, output_dtype=x.dtype)
        tl.store(f_ptr + BLOCK, f, mask=mask_bh)

        h = f * h + (1 - f) * z
        tl.store(y_ptr + BLOCK, h, mask=mask_bh)

        BLOCK += x_stride[1]


@custom_op(
    f"{LIBRARY_NAME}::gru_forward_triton", mutates_args={"forget_gate", "reset_gate", "output_update", "output"}
)
def gru_forward_triton(
    input: torch.Tensor,
    weight: torch.Tensor,
    forget_input: torch.Tensor,
    forget_weight: torch.Tensor,
    forget_gate: torch.Tensor,
    reset_input: torch.Tensor,
    reset_weight: torch.Tensor,
    reset_gate: torch.Tensor,
    output_update: torch.Tensor,
    input_state: torch.Tensor | None,
    output: torch.Tensor,
) -> None:
    B, S, N, H = input.size()

    BLOCK_SIZE_H = get_next_power_of_2(H)
    BLOCK_SIZE_H = max(16, BLOCK_SIZE_H)
    GRID = lambda meta: (ceil_divide(B, meta["BLOCK_SIZE_B"]), N)

    with torch.device(input.device):
        gru_forward_triton_kernel[GRID](
            x_ptr=input,
            x_stride=input.stride(),
            W_ptr=weight,
            W_stride=weight.stride(),
            xf_ptr=forget_input,
            Wf_ptr=forget_weight,
            f_ptr=forget_gate,
            xr_ptr=reset_input,
            Wr_ptr=reset_weight,
            r_ptr=reset_gate,
            z_ptr=output_update,
            h0_ptr=input_state,
            h0_stride=None if input_state is None else input_state.stride(),
            y_ptr=output,
            B=B,
            S=S,
            H=H,
            BLOCK_SIZE_H=BLOCK_SIZE_H,
        )
