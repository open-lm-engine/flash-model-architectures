# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

import torch

from .backward import _linear_attention_backward_pallas
from .forward import _linear_attention_forward_pallas


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

        dq, dk, dv, dh0 = _linear_attention_backward_pallas(
            q=q,
            k=k,
            v=v,
            h0=h0,
            dy=dy,
            dh=dht,
            attention_multiplier=ctx.attention_multiplier,
        )

        if h0 is None:
            dh0 = None

        return dq, dk, dv, dh0, None, None, None
