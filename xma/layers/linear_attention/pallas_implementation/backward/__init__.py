# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

import torch

from .backward import _backward_core
from .state_passing import _state_passing_core


def _linear_attention_backward_pallas(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    h0: torch.Tensor | None,
    dy: torch.Tensor,
    dh: torch.Tensor | None,
    attention_multiplier: float,
    BLOCK_SIZE_S: int = 128,
    BLOCK_SIZE_V: int = 128,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    B, S, Nq, K = q.shape
    Nk = k.size(-2)
    Nv, V = v.size()[-2:]

    N = max(Nq, Nk, Nv)

    Gq = N // Nq
    Gk = N // Nk
    Gv = N // Nv

    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)
    dy = dy.transpose(1, 2)

    h = _state_passing_core(k=k, v=v, h0=h0, N=N, BLOCK_SIZE_S=BLOCK_SIZE_S, BLOCK_SIZE_V=BLOCK_SIZE_V)

    dq, dk, dv, dh0 = _backward_core(
        q=q,
        k=k,
        v=v,
        h=h,
        dy=dy,
        dh=dh,
        attention_multiplier=attention_multiplier,
        BLOCK_SIZE_S=BLOCK_SIZE_S,
        BLOCK_SIZE_V=BLOCK_SIZE_V,
    )

    dq = dq.transpose(1, 2)
    dk = dk.transpose(1, 2)
    dv = dv.transpose(1, 2)

    dq = dq.reshape(B, S, Nq, Gq, K).sum(dim=3)
    dk = dk.reshape(B, S, Nk, Gk, K).sum(dim=3)
    dv = dv.reshape(B, S, Nv, Gv, V).sum(dim=3)

    return dq, dk, dv, dh0
