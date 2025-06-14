# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import torch

from ...kernel_backend import KernelBackend
from ...utils import ensure_contiguous
from .cuda_implementation import add_tensor_cuda
from .triton_implementation import add_tensor_triton


class _AddTensor_Cute(torch.autograd.Function):
    @staticmethod
    @ensure_contiguous
    def forward(ctx, x: torch.Tensor, y: torch.Tensor, kernel_backend: KernelBackend) -> torch.Tensor:
        assert x.size() == y.size(), "tensors x and y should have same shape"
        assert x.type() == y.type(), "tensors x and y should have same dtype"

        output = torch.empty_like(x)

        if kernel_backend == KernelBackend.cuda:
            add_tensor_cuda(x=x, y=y, output=output, BLOCK_SIZE=1024)
        elif kernel_backend == KernelBackend.triton:
            add_tensor_triton(x=x, y=y, output=output)
        else:
            raise ValueError("unexpected kernel_backend")

        return output

    @staticmethod
    def backward(ctx, output_grad: torch.Tensor) -> tuple[torch.Tensor | None]:
        return output_grad, output_grad, None


def add_tensor_cute(
    x: torch.Tensor, y: torch.Tensor, *, kernel_backend: KernelBackend = KernelBackend.cuda
) -> torch.Tensor:
    """add 2 tensors

    Args:
        x (torch.Tensor): first tensor
        y (torch.Tensor): second tensor
        kernel_backend (KernelBackend, optional): kernel backend to prioritize.
            Defaults to KernelBackend.cuda.

    Returns:
        torch.Tensor: output tensor
    """

    if kernel_backend == KernelBackend.torch:
        output = x + y
    else:
        output = _AddTensor_Cute.apply(x, y, kernel_backend)

    return output
