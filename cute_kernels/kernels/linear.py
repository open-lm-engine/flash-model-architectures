# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import torch
import torch.nn.functional as F

from ..cutotune import CutoTuneParameter
from ..kernel_backend import KernelBackend
from ..utils import ensure_contiguous
from .gemm import gemm_cute


class _Linear_Cute(torch.autograd.Function):
    @staticmethod
    @ensure_contiguous
    def forward(
        ctx,
        input: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
        kernel_backend: KernelBackend | CutoTuneParameter,
    ) -> torch.Tensor:
        assert kernel_backend == KernelBackend.triton or isinstance(kernel_backend, CutoTuneParameter)

        ctx.save_for_backward(input, weight)
        ctx.has_bias = bias is not None

        # NOTE this can be a single kernel but I am lazy
        output = gemm_cute(A=input, B=weight, C=None, is_A_transposed=False, is_B_transposed=True, alpha=1, beta=0)

        if bias is not None:
            output += bias

        return output

    @staticmethod
    @ensure_contiguous
    def backward(ctx, output_grad: torch.Tensor) -> torch.Tensor:
        input, weight = ctx.saved_tensors

        input_grad = gemm_cute(
            A=output_grad, B=weight, C=None, is_A_transposed=False, is_B_transposed=False, alpha=1, beta=0
        )

        weight_grad = gemm_cute(
            A=output_grad, B=input, C=None, is_A_transposed=True, is_B_transposed=False, alpha=1, beta=0
        )

        bias_grad = output_grad.sum(dim=0) if ctx.has_bias else None

        return input_grad, weight_grad, bias_grad, None


def linear_cute(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    *,
    kernel_backend: KernelBackend | CutoTuneParameter = KernelBackend.triton,
) -> torch.Tensor:
    """linear layer computation `input` @ `weight` + `bias`

    Args:
        input (torch.Tensor): input tensor
        weight (torch.Tensor): weight tensor
        bias (torch.Tensor | None, optional): bias tensor. Defaults to None.

    Returns:
        torch.Tensor: output tensor
    """

    if kernel_backend == KernelBackend.torch:
        input = F.linear(input, weight, bias)
    else:
        input = _Linear_Cute.apply(input, weight, bias, kernel_backend)

    return input
