# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import torch

from ...kernel_backend import KernelBackend
from .cuda_implementation import add_scalar_cuda
from .triton_implementation import add_scalar_triton


class _AddScalar_Cute(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, y: float, kernel_backend: KernelBackend) -> torch.Tensor:
        output = torch.empty_like(x)

        if kernel_backend == KernelBackend.cuda:
            add_scalar_cuda(x=x, y=y, output=output, BLOCK_SIZE=1024)
        elif kernel_backend == KernelBackend.triton:
            add_scalar_triton(x=x, y=y, output=output)
        else:
            raise ValueError("unexpected kernel_backend")

        return output

    @staticmethod
    def backward(ctx, output_grad: torch.Tensor) -> tuple[torch.Tensor | None]:
        return output_grad, None, None


def add_scalar_cute(
    x: torch.Tensor, y: int | float, *, kernel_backend: KernelBackend = KernelBackend.cuda
) -> torch.Tensor:
    """
    Args:
        x (torch.Tensor): input tensor
        y (int | float): float value to add to `x`
        kernel_backend (KernelBackend, optional): kernel backend to prioritize.
            Defaults to KernelBackend.cuda.

    Returns:
        torch.Tensor: output tensor
    """

    assert isinstance(x, torch.Tensor)
    assert isinstance(y, (int, float)), "y needs to be a numeric type"

    if y == 0:
        return x

    if kernel_backend == KernelBackend.torch:
        output = x + y
    else:
        output = _AddScalar_Cute.apply(x, y, kernel_backend)

    return output
