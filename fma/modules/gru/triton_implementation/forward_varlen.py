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
@triton.heuristics({"NEEDS_MASK_B": lambda args: args["B"] % args["BLOCK_SIZE_B"] != 0})
@triton.jit
def gru_varlen_forward_triton_kernel(
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
    cu_seqlens_ptr,
    IS_MAX_SEQLEN_TENSOR: tl.constexpr,
    max_seqlen_ptr,
    B,
    H,
    BLOCK_SIZE_B: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
    NEEDS_MASK_B: tl.constexpr,
    NEEDS_MASK_H: tl.constexpr,
):
    BLOCK_ID_B = tl.program_id(axis=0)
    BLOCK_ID_N = tl.program_id(axis=1)

    BLOCK_B = BLOCK_ID_B * BLOCK_SIZE_B + tl.arange(0, BLOCK_SIZE_B)
    BLOCK_H = tl.arange(0, BLOCK_SIZE_H)

    if NEEDS_MASK_B:
        MASK_B = BLOCK_B < B
    else:
        MASK_B = None

    if NEEDS_MASK_H:
        MASK_H = BLOCK_H < H
        MASK_HH = MASK_H[:, None] & MASK_H[None, :]
    else:
        MASK_H = None
        MASK_HH = None

    if NEEDS_MASK_B and NEEDS_MASK_H:
        MASK_BH = MASK_B[:, None] & MASK_H[None, :]
    elif NEEDS_MASK_B:
        MASK_BH = MASK_B[:, None]
    elif NEEDS_MASK_H:
        MASK_BH = MASK_H[None, :]
    else:
        MASK_BH = None

    BLOCK = BLOCK_ID_N * W_stride[0] + BLOCK_H[:, None] * W_stride[1] + BLOCK_H[None, :] * W_stride[2]

    W = tl.load(W_ptr + BLOCK, mask=MASK_HH)
    Wf = tl.load(Wf_ptr + BLOCK, mask=MASK_HH)
    Wr = tl.load(Wr_ptr + BLOCK, mask=MASK_HH)

    if h0_ptr is None:
        h = tl.zeros((BLOCK_SIZE_B, BLOCK_SIZE_H), dtype=x_ptr.dtype.element_ty)
    else:
        h = tl.load(
            h0_ptr + BLOCK_B[:, None] * h0_stride[0] + BLOCK_ID_N * h0_stride[1] + BLOCK_H[None, :] * h0_stride[2],
            mask=MASK_BH,
        )

    cu_seqlens_ptrs = cu_seqlens_ptr + BLOCK_B[:, None]
    start = tl.load(cu_seqlens_ptrs, mask=MASK_B[:, None])
    end = tl.load(cu_seqlens_ptrs + 1, mask=MASK_B[:, None])

    if IS_MAX_SEQLEN_TENSOR:
        max_seqlen = tl.load(max_seqlen_ptr)
    else:
        max_seqlen = max_seqlen_ptr

    BLOCK = start * x_stride[0] + BLOCK_ID_N * x_stride[1] + BLOCK_H[None, :] * x_stride[2]

    for _ in range(max_seqlen):
        MASK = (start < end) & MASK_H[None, :]

        x = tl.load(xr_ptr + BLOCK, mask=MASK)
        r = matmul(A=h, B=Wr, C=x, output_dtype=tl.float32)
        r = sigmoid(r, output_dtype=x.dtype)
        tl.store(r_ptr + BLOCK, r, mask=MASK)

        x = tl.load(x_ptr + BLOCK, mask=MASK)
        z = matmul(A=h * r, B=W, C=x, output_dtype=tl.float32)
        z = tanh(z, output_dtype=x.dtype)
        tl.store(z_ptr + BLOCK, z, mask=MASK)

        x = tl.load(xf_ptr + BLOCK, mask=MASK)
        f = matmul(A=h, B=Wf, C=x, output_dtype=tl.float32)
        f = sigmoid(f, output_dtype=x.dtype)
        tl.store(f_ptr + BLOCK, f, mask=MASK)

        h = f * h + (1 - f) * z
        tl.store(y_ptr + BLOCK, h, mask=MASK)

        BLOCK += x_stride[0]
        start += 1


@custom_op(
    f"{LIBRARY_NAME}::gru_varlen_forward_triton", mutates_args={"forget_gate", "reset_gate", "output_update", "output"}
)
def gru_varlen_forward_triton(
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
    cu_seqlens: torch.Tensor,
    max_seqlen_tensor: torch.Tensor | None,
    max_seqlen: int | None,
) -> None:
    B = cu_seqlens.size(0) - 1
    _, N, H = input.size()

    BLOCK_SIZE_H = get_next_power_of_2(H)
    BLOCK_SIZE_H = max(16, BLOCK_SIZE_H)
    GRID = lambda meta: (ceil_divide(B, meta["BLOCK_SIZE_B"]), N)

    is_max_seqlen_tensor = max_seqlen_tensor is not None

    with torch.device(input.device):
        gru_varlen_forward_triton_kernel[GRID](
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
            cu_seqlens_ptr=cu_seqlens,
            IS_MAX_SEQLEN_TENSOR=is_max_seqlen_tensor,
            max_seqlen_ptr=max_seqlen_tensor if is_max_seqlen_tensor else max_seqlen,
            B=B,
            H=H,
            BLOCK_SIZE_H=BLOCK_SIZE_H,
            NEEDS_MASK_H=H % BLOCK_SIZE_H != 0,
        )
