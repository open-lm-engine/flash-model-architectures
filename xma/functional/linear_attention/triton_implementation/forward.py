# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import torch
import triton
import triton.language as tl

from ....custom_op import xma_op
from ....math import ceil_divide, get_next_power_of_2
from ..utils import _get_num_heads


@triton.jit
def linear_attention_forward_triton_kernel(
    q_ptr,
    q_stride,
    k_ptr,
    k_stride,
    v_ptr,
    v_stride,
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
    K: tl.constexpr,
    V: tl.constexpr,
    Gq: tl.constexpr,
    Gk: tl.constexpr,
    Gv: tl.constexpr,
    BLOCK_SIZE_S: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_V: tl.constexpr,
):
    BLOCK_ID_N = tl.program_id(0)
    BLOCK_ID_B = tl.program_id(1)

    NUM_BLOCKS_S = tl.cdiv(S, BLOCK_SIZE_S)


@xma_op(mutates_args={"y"})
def linear_attention_forward_triton(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    h0: torch.Tensor | None,
    y: torch.Tensor,
    cu_seqlens: torch.Tensor | None,
    max_seqlen_tensor: torch.Tensor | None,
    max_seqlen: int | None,
) -> None:
    if cu_seqlens is None:
        assert max_seqlen is None
        assert max_seqlen_tensor is None

        B, S, _, K = q.size()
    else:
        B = cu_seqlens.size(0) - 1
        S = None
        K = q.size(-1)

    V = v.size(-1)

    Nq, Nk, Nv, N = _get_num_heads(q=q, k=k, v=v, run_check=False)
    is_max_seqlen_tensor = max_seqlen_tensor is not None

    GRID = lambda meta: (N, ceil_divide(B, meta["BLOCK_SIZE_B"]))

    with torch.device(q.device):
        linear_attention_forward_triton_kernel[GRID](
            q_ptr=q,
            q_stride=q.stride(),
            k_ptr=k,
            k_stride=k.stride(),
            v_ptr=v,
            v_stride=v.stride(),
            h0_ptr=h0,
            h0_stride=None if h0 is None else h0.stride(),
            y_ptr=y,
            y_stride=y.stride(),
            cu_seqlens_ptr=cu_seqlens,
            cu_seqlens_stride=None if cu_seqlens is None else cu_seqlens.stride(),
            IS_MAX_SEQLEN_TENSOR=is_max_seqlen_tensor,
            max_seqlen_ptr=max_seqlen_tensor if is_max_seqlen_tensor else max_seqlen,
            B=B,
            S=S,
            K=K,
            V=V,
            Gq=N // Nq,
            Gk=N // Nk,
            Gv=N // Nv,
        )
