# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import torch
import triton
import triton.language as tl

from ....custom_op import xma_op
from ....math import ceil_divide
from ....triton_utils import matmul


@triton.jit
def linear_attention_forward_chunked_triton_kernel(
    k_ptr,
    k_stride,
    v_ptr,
    v_stride,
    h0_ptr,
    h0_stride,
    h_ptr,
    h_stride,
    cu_seqlens_ptr,
    cu_seqlens_stride,
    IS_MAX_SEQLEN_TENSOR: tl.constexpr,
    max_seqlen_ptr,
    B,
    S,
    N: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    Gk: tl.constexpr,
    Gv: tl.constexpr,
    BLOCK_SIZE_S: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_V: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
):
    tl.static_assert(CHUNK_SIZE % BLOCK_SIZE_S == 0)

    BLOCK_ID_BN = tl.program_id(0)
    BLOCK_ID_K = tl.program_id(1)
    BLOCK_ID_V = tl.program_id(2)

    BLOCK_ID_B = BLOCK_ID_BN // N
    BLOCK_ID_N = BLOCK_ID_BN % N

    BLOCK_ID_Nk = BLOCK_ID_N // Gk
    BLOCK_ID_Nv = BLOCK_ID_N // Gv

    if CHUNK_SIZE is None:
        CHUNK_SIZE = BLOCK_SIZE_S

    BLOCK_K = BLOCK_ID_K * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
    BLOCK_V = BLOCK_ID_V * BLOCK_SIZE_V + tl.arange(0, BLOCK_SIZE_V)
    BLOCK_S = tl.arange(0, BLOCK_SIZE_S)

    MASK_K = BLOCK_K < K
    MASK_V = BLOCK_V < V
    MASK_KV = MASK_K[:, None] & MASK_V[None, :]

    k_ptrs = (
        k_ptr
        + BLOCK_ID_B * k_stride[0]
        + BLOCK_S[:, None] * k_stride[1]
        + BLOCK_ID_Nk * k_stride[2]
        + BLOCK_K[None, :] * k_stride[3]
    )

    v_ptrs = (
        v_ptr
        + BLOCK_ID_B * v_stride[0]
        + BLOCK_S[:, None] * v_stride[1]
        + BLOCK_ID_Nv * v_stride[2]
        + BLOCK_V[None, :] * v_stride[3]
    )

    h_ptrs = (
        h_ptr
        + BLOCK_ID_B * h_stride[0]
        + BLOCK_ID_N * h_stride[2]
        + BLOCK_K[:, None] * h_stride[3]
        + BLOCK_V[None, :] * h_stride[4]
    )

    if h0_ptr is None:
        h = tl.zeros((BLOCK_SIZE_K, BLOCK_SIZE_V), dtype=tl.float32)
    else:
        h = tl.load(
            h0_ptr
            + BLOCK_ID_B * h0_stride[0]
            + BLOCK_ID_N * h0_stride[1]
            + BLOCK_K[:, None] * h0_stride[2]
            + BLOCK_V[None, :] * h0_stride[3],
            mask=MASK_KV,
        ).to(tl.float32)

    NUM_BLOCKS_S = tl.cdiv(S, BLOCK_SIZE_S)

    for s in range(NUM_BLOCKS_S):
        if (s > 0 and (s * BLOCK_SIZE_S) % CHUNK_SIZE == 0) or s == NUM_BLOCKS_S - 1:
            tl.store(h_ptrs, h, mask=MASK_KV)
            h_ptrs += h_stride[1]

        MASK_S = BLOCK_S < S

        k = tl.load(k_ptrs, mask=MASK_S[:, None] & MASK_K[None, :])
        v = tl.load(v_ptrs, mask=MASK_S[:, None] & MASK_V[None, :])

        h = matmul(A=k.T, B=v, C=h)

        BLOCK_S += BLOCK_SIZE_S
        k_ptrs += BLOCK_SIZE_S * k_stride[1]
        v_ptrs += BLOCK_SIZE_S * v_stride[1]


@xma_op(mutates_args={"h"})
def linear_attention_forward_chunked_triton(
    k: torch.Tensor,
    v: torch.Tensor,
    N: int,
    h0: torch.Tensor | None,
    cu_seqlens: torch.Tensor | None,
    max_seqlen_tensor: torch.Tensor | None,
    max_seqlen: int | None,
    CHUNK_SIZE: int | None,
) -> None:
    if cu_seqlens is None:
        assert max_seqlen is None
        assert max_seqlen_tensor is None

        B, S, Nk, K = k.size()
    else:
        B = cu_seqlens.size(0) - 1
        S = None
        _, Nk, K = k.size()

    Nv, V = v.size()[-2:]

    is_max_seqlen_tensor = max_seqlen_tensor is not None

    GRID = lambda meta: (N, ceil_divide(B, meta["BLOCK_SIZE_B"]))

    with torch.device(k.device):
        linear_attention_forward_chunked_triton_kernel[GRID](
            k_ptr=k,
            k_stride=k.stride(),
            v_ptr=v,
            v_stride=v.stride(),
            h0_ptr=h0,
            h0_stride=None if h0 is None else h0.stride(),
            cu_seqlens_ptr=cu_seqlens,
            cu_seqlens_stride=None if cu_seqlens is None else cu_seqlens.stride(),
            IS_MAX_SEQLEN_TENSOR=is_max_seqlen_tensor,
            max_seqlen_ptr=max_seqlen_tensor if is_max_seqlen_tensor else max_seqlen,
            B=B,
            S=S,
            K=K,
            V=V,
            Gk=N // Nk,
            Gv=N // Nv,
            CHUNK_SIZE=CHUNK_SIZE,
        )
