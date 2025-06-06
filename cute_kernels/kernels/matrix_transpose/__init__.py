# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import torch

from ...cutotune import CutoTuneParameter
from ...kernel_backend import KernelBackend
from ...utils import ensure_contiguous
from .triton_implementation import matrix_transpose_triton


class _MatrixTranspose_Cute(torch.autograd.Function):
    @staticmethod
    @ensure_contiguous
    def forward(ctx, x: torch.Tensor, kernel_backend: KernelBackend | CutoTuneParameter) -> torch.Tensor:
        assert x.dim() == 2
        assert kernel_backend == KernelBackend.triton or isinstance(kernel_backend, CutoTuneParameter)

        M, N = x.size()

        output = torch.empty(N, M, device=x.device, dtype=x.dtype)
        matrix_transpose_triton(x=x, output=output)

        return output

    @staticmethod
    @ensure_contiguous
    def backward(ctx, output_grad: torch.Tensor) -> torch.Tensor:
        M, N = output_grad.size()

        x_grad = torch.empty(N, M, device=output_grad.device, dtype=output_grad.dtype)
        matrix_transpose_triton(x=output_grad, output=x_grad)

        return x_grad, None


def matrix_transpose_cute(
    x: torch.Tensor, *, kernel_backend: KernelBackend | CutoTuneParameter = KernelBackend.triton
) -> torch.Tensor:
    """transposes a matrix

    Args:
        x (torch.Tensor): input tensor

    Returns:
        torch.Tensor: output tensor
    """

    if kernel_backend == KernelBackend.torch:
        x = x.T
    else:
        x = _MatrixTranspose_Cute.apply(x, kernel_backend)

    return x
