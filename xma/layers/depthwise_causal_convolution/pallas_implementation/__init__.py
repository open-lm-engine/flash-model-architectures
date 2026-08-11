# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

import torch

from .backward import _depthwise_causal_convolution_backward_pallas
from .forward import _depthwise_causal_convolution_forward_pallas


_BLOCK_SIZE_S = 256


class _DepthwiseCausalConvolutionPallas(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
        input_state: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        y, ht = _depthwise_causal_convolution_forward_pallas(
            x=x, W=weight, b=bias, h0=input_state, BLOCK_SIZE_S=_BLOCK_SIZE_S
        )
        ctx.save_for_backward(x, weight, bias, input_state)
        return y, ht

    @staticmethod
    def backward(
        ctx, dy: torch.Tensor, dht: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        x, weight, bias, input_state = ctx.saved_tensors

        dx, dW, db, dh0 = _depthwise_causal_convolution_backward_pallas(
            x=x, W=weight, b=bias, h0=input_state, dy=dy, dht=dht, BLOCK_SIZE_S=_BLOCK_SIZE_S
        )

        return dx, dW, db, dh0
