# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import torch
import torch.nn.functional as F

from ...accelerator import KernelBackend
from ...custom_op import CustomOp
from ..sequence_packing import unpack_sequence


class _SoftmaxAttention(CustomOp):
    @staticmethod
    def forward_backward_torch(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_multiplier: float | None,
        attention_mask: torch.Tensor | None,
        causal: bool,
        cu_seqlens: torch.Tensor | None,
        max_seqlen: torch.Tensor | int | None,
    ) -> torch.Tensor:
        if cu_seqlens is not None:
            attention_mask = unpack_sequence(
                inputs=torch.ones_like(cu_seqlens, dtype=torch.int32),
                cu_seqlens=cu_seqlens,
                batch_size=cu_seqlens.size(0) - 1,
                sequence_length=max_seqlen.item() if isinstance(max_seqlen, torch.Tensor) else max_seqlen,
                kernel_backend=KernelBackend.torch,
            )

        x = F.scaled_dot_product_attention(
            query=q,
            key=k,
            value=v,
            attn_mask=attention_mask,
            dropout_p=0,
            is_causal=causal,
            scale=attention_multiplier,
            enable_gqa=True,
        )

        return x


def softmax_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_multiplier: float | None = None,
    attention_mask: torch.Tensor | None = None,
    causal: bool = False,
    cu_seqlens: torch.Tensor | None = None,
    max_seqlen: torch.Tensor | int | None = None,
    *,
    kernel_backend: KernelBackend | None = None,
) -> torch.Tensor:
    assert query.dim() == 3 + (cu_seqlens is None)
    assert key.dim() == 3 + (cu_seqlens is None)
    assert value.dim() == 3 + (cu_seqlens is None)

    if cu_seqlens is None:
        assert max_seqlen is None
    else:
        assert attention_mask is None
        assert max_seqlen is not None

    if attention_mask is not None:
        assert cu_seqlens is None
        assert max_seqlen is None

    return _SoftmaxAttention.run(
        q=query,
        k=key,
        v=value,
        attention_multiplier=attention_multiplier,
        attention_mask=attention_mask,
        causal=causal,
        cu_seqlens=cu_seqlens,
        max_seqlen=max_seqlen,
        kernel_backend=kernel_backend,
    )
