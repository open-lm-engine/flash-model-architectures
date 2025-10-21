# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import torch

from ...utils import empty_like_contiguous
from .triton_implementation import softmax_attention_forward_triton


class _SoftmaxAttention(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_multiplier: float | None,
        output_attention_scores: bool,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        B, Q, N = query.size()[:-1]
        K = query.size(1)

        output = empty_like_contiguous(query)

        if output_attention_scores:
            attention_scores = torch.empty(B, N, Q, K, dtype=query.dtype, device=query.device)

        softmax_attention_forward_triton(
            query=query,
            key=key,
            value=value,
            output=output,
            attention_scores=attention_scores,
            attention_multiplier=attention_multiplier,
        )

        ctx.save_for_backward(query, key, value)

        return output, attention_scores


def softmax_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_multiplier: float | None = None,
    output_attention_scores: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    return _SoftmaxAttention.apply(query, key, value, attention_multiplier, output_attention_scores)
