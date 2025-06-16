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
        kernel_backend_forward: KernelBackend | CutoTuneParameter,
        kernel_backend_backward: KernelBackend | CutoTuneParameter,
    ) -> torch.Tensor:
        assert kernel_backend_forward == KernelBackend.triton or isinstance(kernel_backend_forward, CutoTuneParameter)
        assert kernel_backend_backward == KernelBackend.triton or isinstance(
            kernel_backend_backward, CutoTuneParameter
        )

        ctx.save_for_backward(input, weight)
        ctx.has_bias = bias is not None
        ctx.kernel_backend_backward = kernel_backend_backward

        # NOTE this can be a single kernel but I am lazy
        output = gemm_cute(
            A=input,
            B=weight,
            C=None,
            is_A_transposed=False,
            is_B_transposed=True,
            alpha=1,
            beta=0,
            kernel_backend=kernel_backend_forward,
        )

        if bias is not None:
            output += bias

        return output

    @staticmethod
    @ensure_contiguous
    def backward(ctx, output_grad: torch.Tensor) -> torch.Tensor:
        input, weight = ctx.saved_tensors
        kernel_backend_backward = ctx.kernel_backend_backward

        input_grad = gemm_cute(
            A=output_grad,
            B=weight,
            C=None,
            is_A_transposed=False,
            is_B_transposed=False,
            alpha=1,
            beta=0,
            kernel_backend=kernel_backend_backward,
        )

        weight_grad = gemm_cute(
            A=output_grad,
            B=input,
            C=None,
            is_A_transposed=True,
            is_B_transposed=False,
            alpha=1,
            beta=0,
            kernel_backend=kernel_backend_backward,
        )

        bias_grad = output_grad.sum(dim=0) if ctx.has_bias else None

        return input_grad, weight_grad, bias_grad, None, None


def linear_cute(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    *,
    kernel_backend_forward: KernelBackend | CutoTuneParameter = KernelBackend.triton,
    kernel_backend_backward: KernelBackend | CutoTuneParameter = KernelBackend.triton,
) -> torch.Tensor:
    """linear layer computation `input` @ `weight` + `bias`

    Args:
        input (torch.Tensor): input tensor
        weight (torch.Tensor): weight tensor
        bias (torch.Tensor | None, optional): bias tensor. Defaults to None.
        kernel_backend_forward (KernelBackend, optional): kernel backend to prioritize. Defaults
            to KernelBackend.triton.
        kernel_backend_backward (KernelBackend, optional): kernel backend to prioritize. Defaults
            to KernelBackend.triton.

    Returns:
        torch.Tensor: output tensor
    """

    if kernel_backend_forward == KernelBackend.torch:
        assert kernel_backend_backward == KernelBackend.torch
        input = F.linear(input, weight, bias)
    else:
        input = _Linear_Cute.apply(input, weight, bias, kernel_backend_forward, kernel_backend_backward)

    return input
