# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

import torch

from .backward import _depthwise_causal_convolution_backward_pallas
from .forward import _depthwise_causal_convolution_forward_pallas


class _DepthwiseCausalConvolutionPallas(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
        input_state: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        y, ht = _depthwise_causal_convolution_forward_pallas(x=x, W=weight, b=bias, h0=input_state)

        ctx.bias_is_none = bias is None
        ctx.input_state_is_none = input_state is None

        tensors_to_save = [x, weight]
        if bias is not None:
            tensors_to_save.append(bias)
        if input_state is not None:
            tensors_to_save.append(input_state)

        ctx.save_for_backward(*tensors_to_save)

        return y, ht

    @staticmethod
    def backward(
        ctx, dy: torch.Tensor, dht: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        saved_tensors = list(ctx.saved_tensors)

        x = saved_tensors.pop(0)
        weight = saved_tensors.pop(0)
        bias = None if ctx.bias_is_none else saved_tensors.pop(0)
        input_state = None if ctx.input_state_is_none else saved_tensors.pop(0)

        dx, dW, db, dh0 = _depthwise_causal_convolution_backward_pallas(
            x=x, W=weight, b=bias, h0=input_state, dy=dy, dht=dht
        )

        return dx, dW, db, dh0
