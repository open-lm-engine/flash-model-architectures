# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import triton
import triton.language as tl


@triton.jit
def softmax_attention_forward_triton_kernel(
    q_ptr,
    q_stride,
    k_ptr,
    k_stride,
    v_ptr,
    v_stride,
    attention_multiplier,
    m_ptr,
    m_stride,
    cu_seqlens_ptr,
    cu_seqlens_stride,
    IS_MAX_SEQLEN_TENSOR: tl.constexpr,
    max_seqlen_ptr,
    BLOCK_SIZE_Sq: tl.constexpr,
    BLOCK_SIZE_Sk: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_V: tl.constexpr,
):
    BLOCK_ID_Sq = tl.program_id(0)
    BLOCK_ID_N = tl.program_id(1)
    BLOCK_ID_B = tl.program_id(2)

    BLOCK_Sq = BLOCK_ID_Sq * BLOCK_SIZE_Sq + tl.arange(0, BLOCK_SIZE_Sq)
    BLOCK_K = tl.arange(0, BLOCK_SIZE_K)
    BLOCK_V = tl.arange(0, BLOCK_SIZE_V)

    q_ptrs = (
        q_ptr
        + BLOCK_ID_B * q_stride[0]
        + BLOCK_Sq[:, None] * q_stride[1]
        + BLOCK_ID_N * q_stride[2]
        + BLOCK_K[None, :] * q_stride[3]
    )

    k_ptrs = k_ptr + BLOCK_ID_B * k_stride[0] + BLOCK_ID_N * k_stride[2] + BLOCK_K[None, :] * k_stride[3]
    v_ptrs = v_ptr + BLOCK_ID_B * v_stride[0] + BLOCK_ID_N * v_stride[2] + BLOCK_V[None, :] * v_stride[3]
