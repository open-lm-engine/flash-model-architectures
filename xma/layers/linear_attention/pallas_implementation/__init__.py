# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

import torch

from .backward import _linear_attention_backward_pallas
from .forward import _linear_attention_forward_pallas
from .state_passing import _state_passing_pallas


_BLOCK_SIZE_S = 512
_BLOCK_SIZE_V = 256


class _LinearAttentionPallas(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        h0: torch.Tensor | None,
        attention_multiplier: float,
        cu_seqlens: torch.Tensor | None,
        max_seqlen: int | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert cu_seqlens is None

        y, ht = _linear_attention_forward_pallas(q=q, k=k, v=v, h0=h0, attention_multiplier=attention_multiplier)

        ctx.h0_is_none = h0 is None
        ctx.attention_multiplier = attention_multiplier

        ctx.save_for_backward(*((q, k, v) if h0 is None else (q, k, v, h0)))

        return y, ht

    @staticmethod
    def backward(
        ctx, dy: torch.Tensor, dht: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None, None, None, None]:
        if ctx.h0_is_none:
            q, k, v = ctx.saved_tensors
            h0 = None
        else:
            q, k, v, h0 = ctx.saved_tensors

        B, S, Nq, K = q.size()
        Nk = k.size(-2)
        Nv, V = v.shape[-2:]
        N = max(Nq, Nk, Nv)

        Gq = N // Nq
        Gk = N // Nk
        Gv = N // Nv

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        dy = dy.transpose(1, 2)

        h = _state_passing_pallas(k=k, v=v, h0=h0, N=N, BLOCK_SIZE_S=_BLOCK_SIZE_S, BLOCK_SIZE_V=_BLOCK_SIZE_V)

        dq, dk, dv, dh0 = _linear_attention_backward_pallas(
            q=q,
            k=k,
            v=v,
            h=h,
            dy=dy,
            dh=dht,
            attention_multiplier=ctx.attention_multiplier,
            BLOCK_SIZE_S=_BLOCK_SIZE_S,
            BLOCK_SIZE_V=_BLOCK_SIZE_V,
        )

        dq = dq.transpose(1, 2).reshape(B, S, Nq, Gq, K).sum(dim=3)
        dk = dk.transpose(1, 2).reshape(B, S, Nk, Gk, K).sum(dim=3)
        dv = dv.transpose(1, 2).reshape(B, S, Nv, Gv, V).sum(dim=3)

        if h0 is None:
            dh0 = None

        return dq, dk, dv, dh0, None, None, None
