import torch

from ...utils import ensure_contiguous
from ..gemm import gemm_cute
from .torch_implementation import linear_torch


class _Linear_Cute(torch.autograd.Function):
    @staticmethod
    @ensure_contiguous
    def forward(ctx, input: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None) -> torch.Tensor:
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

        return input_grad, weight_grad, bias_grad, *[None] * 2


def linear_cute(input: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
    """linear layer computation `input` @ `weight` + `bias`

    Args:
        input (torch.Tensor): input tensor
        weight (torch.Tensor): weight tensor
        bias (torch.Tensor | None, optional): bias tensor. Defaults to None.

    Returns:
        torch.Tensor: output tensor
    """

    return _Linear_Cute.apply(input, weight, bias)
