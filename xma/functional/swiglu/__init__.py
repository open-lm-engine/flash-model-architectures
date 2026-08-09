# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

import torch

from ...accelerator import KernelBackend
from ...math import divide_if_divisible
from ...utils import is_cute_dsl_available, is_triton_available
from . import mps_implementation, torch_implementation
from .op import _Swiglu, _SwigluPacked


if is_cute_dsl_available():
    from . import cuda_implementation

if is_triton_available():
    from . import triton_implementation


def swiglu(gate: torch.Tensor, up: torch.Tensor, *, kernel_backend: KernelBackend | None = None) -> torch.Tensor:
    """
    computes swiglu activation as `up * gate * sigmoid(gate)`

    :param gate: `gate` activation tensor
    :type gate: torch.Tensor
    :param up: `up` activation tensor
    :type up: torch.Tensor
    :param kernel_backend: KernelBackend
    :type kernel_backend: KernelBackend | None
    :return: output tensor
    :rtype: Tensor
    """

    assert gate.size() == up.size(), "tensors gate and up should have same shape"
    assert gate.type() == up.type(), "tensors gate and up should have same dtype"

    original_shape = gate.size()
    gate = gate.flatten(0, -2)
    up = up.flatten(0, -2)

    y = _Swiglu.run(g=gate, u=up, kernel_backend=kernel_backend)
    y = y.view(original_shape)

    return y


def swiglu_packed(x: torch.Tensor, *, kernel_backend: KernelBackend | None = None) -> torch.Tensor:
    """
    computes swiglu activation by splitting the tensor `x` into 2 parts: gate and up activations. The tensor has
    interleaved values of gate, up, gate, up, ...

    :param x: input activation
    :type x: torch.Tensor
    :param kernel_backend: KernelBackend
    :type kernel_backend: KernelBackend | None
    :return: output tensor
    :rtype: Tensor
    """

    original_shape = x.size()
    x = x.flatten(0, -2)

    H = divide_if_divisible(original_shape[-1], 2)

    y = _SwigluPacked.run(x=x, kernel_backend=kernel_backend)
    y = y.view(*original_shape[:-1], H)

    return y
