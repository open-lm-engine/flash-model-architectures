# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import torch
import triton
import triton.language as tl

from ....triton_math import matmul


@triton.jit
def softmax_attention_forward_triton_kernel(
    q_ptr,
    q_stride,
    k_ptr,
    k_stride,
    v_ptr,
    v_stride,
    y_ptr,
    y_stride,
    s_ptr,
    s_stride,
    multiplier,
    Q,
    K,
    H,
    BLOCK_SIZE_Q: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
):
    BLOCK_ID_B = tl.program_id(0)
    BLOCK_ID_N = tl.program_id(1)
    BLOCK_ID_Q = tl.program_id(2)

    BLOCK_Q = BLOCK_ID_Q * BLOCK_SIZE_Q + tl.arange(0, BLOCK_SIZE_Q)
    MASK_Q = BLOCK_Q < Q

    BLOCK_K = tl.arange(0, BLOCK_SIZE_K)

    BLOCK_H = tl.arange(0, BLOCK_SIZE_H)
    MASK_H = BLOCK_H < H

    q_ptrs = (
        q_ptr
        + BLOCK_ID_B * q_stride[0]
        + BLOCK_Q[:, None] * q_stride[1]
        + BLOCK_ID_N * q_stride[2]
        + BLOCK_SIZE_H[None, :] * q_stride[3]
    )

    k_ptrs = (
        k_ptr
        + BLOCK_ID_B * k_stride[0]
        + BLOCK_K[:, None] * k_stride[1]
        + BLOCK_ID_N * k_stride[2]
        + BLOCK_SIZE_H[None, :] * k_stride[3]
    )

    q = tl.load(q_ptrs, mask=MASK_Q[:, None] & MASK_H[None, :])

    for i in range(tl.cdiv(K, BLOCK_SIZE_K)):
        MASK_K = BLOCK_K < K

        k = tl.load(k_ptrs, mask=MASK_K[:, None] & MASK_H[None, :])

        s = matmul(A=q, B=k.T, C=None, output_dtype=tl.float32)
        if multiplier is not None:
            s *= multiplier

        BLOCK_K += BLOCK_SIZE_K
        k_ptrs += BLOCK_SIZE_K * k_stride[1]


def softmax_attention_forward_triton(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    output: torch.Tensor,
    attention_scores: torch.Tensor | None,
    attention_multiplier: float | None = None,
) -> None:
    B, Q, N, H = query.size()
    K = key.size(1)

    with torch.device(query.device):
        softmax_attention_forward_triton_kernel[B, N, Q](
            q_ptr=query,
            q_stride=query.stride(),
            k_ptr=key,
            k_stride=key.stride(),
            v_ptr=value,
            v_stride=value.stride(),
            y_ptr=output,
            y_stride=output.stride(),
            s_ptr=attention_scores,
            s_stride=None if attention_scores is None else attention_scores.stride(),
            Q=Q,
            K=K,
            H=H,
            multiplier=attention_multiplier,
        )
