# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import torch

from ...accelerator import KernelBackend
from ...custom_op import CustomOp


def _get_num_heads(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, run_check: bool) -> tuple[int, int, int, int]:
    Nq = q.size(-2)
    Nk = k.size(-2)
    Nv = v.size(-2)

    N = max(Nq, Nk, Nv)

    if run_check:
        assert N % Nq == 0
        assert N % Nk == 0
        assert N % Nv == 0

    return Nq, Nk, Nv, N


class _LinearAttention(CustomOp):
    @staticmethod
    def forward_backward_torch(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        h0: torch.Tensor | None,
        cu_seqlens: torch.Tensor | None = None,
        max_seqlen: torch.Tensor | int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        Nq, Nk, Nv, N = _get_num_heads(q=q, k=k, v=v, run_check=False)

        y_shape = list(q.size())
        y_shape[-2] = N
        y = torch.empty(y_shape, device=q.device, dtype=q.dtype)

        if cu_seqlens is None:
            B, S, _, K = q.size()
        else:
            B = cu_seqlens.size(0) - 1
            S = max_seqlen.item() if isinstance(max_seqlen, torch.Tensor) else max_seqlen
            K = q.size(-1)

        V = v.size(-1)

        Gq = N // Nq
        Gk = N // Nk
        Gv = N // Nv

        q = q.repeat_interleave(Gq, dim=-2)
        k = k.repeat_interleave(Gk, dim=-2)
        v = v.repeat_interleave(Gv, dim=-2)

        if h0 is not None:
            h0 = torch.zeros(B, S, N, K, V, dtype=q.dtype, device=q.device)

        if cu_seqlens is not None:
            h0 = h0.clone()
            start = cu_seqlens[:-1]
            end = cu_seqlens[1:]

        for s in range(S):
            if cu_seqlens is None:
                h = h0 + k[..., None] * v[..., None, :]
                y[:, s] = (q[..., None, :] @ h).squeeze(-2)

                h0[unfinished] = h
            else:
                offset = start + s
                unfinished = offset < end
                offset_unfinished = offset[unfinished]

                h = h0[unfinished, :, None, :] + k[offset_unfinished, ..., None] * v[offset_unfinished, :, None, :]
                y[offset_unfinished]

                h0[unfinished] = h

        return y, h

    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        h0: torch.Tensor | None,
        cu_seqlens: torch.Tensor | None = None,
        max_seqlen: torch.Tensor | int | None = None,
        *,
        kernel_backend: KernelBackend,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        Nq, Nk, Nv, N = _get_num_heads(q=q, k=k, v=v, run_check=False)
