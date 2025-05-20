# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import torch

from ...utils import ensure_contiguous
from .torch_implementation import matrix_transpose_torch
from .triton_implementation import matrix_transpose_triton


class _MatrixTranspose_Cute(torch.autograd.Function):
    @staticmethod
    @ensure_contiguous
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        assert x.dim() == 2
        M, N = x.size()

        output = torch.empty(M, N, device=x.device, dtype=x.dtype)
        matrix_transpose_triton(x=x, output=output)

        return output

    @staticmethod
    @ensure_contiguous
    def backward(ctx, output_grad: torch.Tensor) -> torch.Tensor:
        M, N = output_grad.size()

        x_grad = torch.empty(M, N, device=output_grad.device, dtype=output_grad.dtype)
        matrix_transpose_triton(x=output_grad, output=x_grad)

        return x_grad


def matrix_transpose_cute(x: torch.Tensor) -> torch.Tensor:
    """transposes a matrix

    Args:
        x (torch.Tensor): input tensor

    Returns:
        torch.Tensor: output tensor
    """

    return _MatrixTranspose_Cute.apply(x)
