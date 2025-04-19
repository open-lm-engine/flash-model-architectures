import torch

from ...utils import ensure_contiguous
from ..gemm import gemm_cute
from .torch_implementation import linear_torch


class _Linear_Cute(torch.autograd.Function):
    @staticmethod
    @ensure_contiguous
    def forward(
        ctx,
        input: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
        kernel_backend_forward: str,
        kernel_backend_backward: str,
    ) -> torch.Tensor:
        ctx.save_for_backward(input, weight)
        ctx.has_bias = bias is not None
        ctx.kernel_backend_backward = kernel_backend_backward

        if kernel_backend_forward == "triton":
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
        else:
            raise ValueError(f"unexpected kernel_backend_forward ({kernel_backend_forward})")

        return output

    @staticmethod
    @ensure_contiguous
    def backward(ctx, output_grad: torch.Tensor) -> torch.Tensor:
        input, weight = ctx.saved_tensors
        kernel_backend_backward = ctx.kernel_backend_backward

        if kernel_backend_backward == "triton":
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
        else:
            raise ValueError(f"unexpected kernel_backend_backward ({kernel_backend_backward})")

        return input_grad, weight_grad, bias_grad, *[None] * 9


def linear_cute(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    kernel_backend_forward: str = "triton",
    kernel_backend_backward: str = "triton",
) -> torch.Tensor:
    return _Linear_Cute.apply(input, weight, bias, kernel_backend_forward, kernel_backend_backward)
