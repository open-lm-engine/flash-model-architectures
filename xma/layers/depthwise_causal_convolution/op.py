# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

import torch

from ...accelerator import KernelBackend
from ...custom_op import CustomOp
from ...utils import is_torch_xla_available


class _DepthwiseCausalConvolution(CustomOp): ...


if is_torch_xla_available():
    from .pallas_implementation import _DepthwiseCausalConvolutionPallas

    _DepthwiseCausalConvolution[KernelBackend.pallas] = _DepthwiseCausalConvolutionPallas


def depthwise_causal_convolution(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    input_state: torch.Tensor | None = None,
    output_state: bool = False,
    *,
    kernel_backend: KernelBackend | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """computes depthwise causal 1D convolution using a hand-written Pallas TPU kernel, bound into torch via
    torch_xla: `output[b, t, h] = bias[h] + sum_k weight[h, k] * z[b, t + k, h]` where `z` is `input` preceded
    by `kernel_size - 1` raw history positions taken from `input_state` (or 0 if `input_state` is None).

    :param input: input tensor of shape (B, S, H)
    :type input: torch.Tensor
    :param weight: depthwise convolution weight of shape (H, K), K being the kernel size
    :type weight: torch.Tensor
    :param bias: bias tensor of shape (H,). None means no bias is added. Defaults to None.
    :type bias: torch.Tensor | None
    :param input_state: the `K - 1` raw (pre-convolution) input positions preceding `input` that can actually
        affect the output (the very first of the `K` history positions never does - `output[t]` only ever
        reads the last `K - 1` history positions plus the current input), of shape (B, H, K - 1). None is
        equivalent to a 0 tensor. Defaults to None.
    :type input_state: torch.Tensor | None
    :param output_state: whether to also return the trailing `K - 1` raw input positions for use as
        `input_state` in a subsequent call. Defaults to False.
    :type output_state: bool
    :param kernel_backend: only KernelBackend.pallas is currently supported. None defaults to
        KernelBackend.pallas. Defaults to None.
    :type kernel_backend: KernelBackend | None
    :return: output tensor of shape (B, S, H), and the output state of shape (B, H, K - 1) if `output_state`
        is True else None.
    :rtype: tuple[torch.Tensor, torch.Tensor | None]
    """

    B, _, H = input.shape
    K = weight.shape[-1]

    assert weight.ndim == 2
    assert weight.shape[0] == H
    assert K > 1

    if bias is not None:
        assert bias.shape == (H,)

    if input_state is not None:
        assert input_state.shape == (B, H, K - 1)

    input, input_state = _DepthwiseCausalConvolution.run(
        x=input, weight=weight, bias=bias, input_state=input_state, kernel_backend=kernel_backend
    )

    input_state = input_state.transpose(1, 2) if output_state else None

    return input, input_state
