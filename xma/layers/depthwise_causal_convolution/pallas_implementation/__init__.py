# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

import torch

from .backward import _depthwise_causal_convolution_backward_pallas
from .forward import _depthwise_causal_convolution_forward_pallas


def _get_block_size_s(H: int) -> int:
    # the kernel stack holds several (BLOCK_SIZE_S, H) fp32 tiles and vmem overflows once
    # BLOCK_SIZE_S * H exceeds 2**19 (measured), so shrink the block for wide H. Narrow H
    # wants the block as big as possible instead: more rows per program means fewer HBM
    # round-trips, which is what sets the kernel's bandwidth at small hidden sizes
    block_size = 1 << ((1 << 19) // H).bit_length() - 1
    return min(1024, max(8, block_size))


class _DepthwiseCausalConvolutionPallas(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        W: torch.Tensor,
        b: torch.Tensor | None,
        h0: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        y, ht = _depthwise_causal_convolution_forward_pallas(
            x=x, W=W, b=b, h0=h0, BLOCK_SIZE_S=_get_block_size_s(x.shape[-1])
        )

        ctx.save_for_backward(x, W, b, h0)

        return y, ht

    @staticmethod
    def backward(
        ctx, dy: torch.Tensor, dht: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        x, W, b, h0 = ctx.saved_tensors

        dx, dW, db, dh0 = _depthwise_causal_convolution_backward_pallas(
            x=x, W=W, b=b, h0=h0, dy=dy, dht=dht, BLOCK_SIZE_S=_get_block_size_s(x.shape[-1])
        )

        return dx, dW, db, dh0
