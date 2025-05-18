import torch

from ...utils import ensure_contiguous, get_num_elements_and_hidden_size
from .torch_implementation import rmsnorm_torch
from .triton_implementation import rmsnorm_backward_triton, rmsnorm_forward_triton


class _RMSNorm_Cute(torch.autograd.Function):
    @staticmethod
    @ensure_contiguous
    def forward(
        ctx, x: torch.Tensor, weight: torch.Tensor | None, eps: float | None, memory_efficient: bool
    ) -> torch.Tensor:
        if weight is not None:
            assert weight.dim() == 1, "weight should be 1D"
            assert weight.size(-1) == x.size(-1), "hidden size for x and weight tensor is different"
            assert weight.type() == x.type(), "tensors weight and y should have same dtype"

        if eps is None:
            eps = torch.finfo(x.dtype).eps

        B, _ = get_num_elements_and_hidden_size(x)

        output = torch.empty_like(x)
        rmsnorm_denominator = None if memory_efficient else torch.empty(B, device=x.device, dtype=torch.float32)

        rmsnorm_forward_triton(x=x, weight=weight, output=output, eps=eps, rmsnorm_denominator=rmsnorm_denominator)

        ctx.save_for_backward(x, weight, rmsnorm_denominator)
        ctx.eps = eps

        return output

    @staticmethod
    @ensure_contiguous
    def backward(ctx, output_grad: torch.Tensor) -> tuple[torch.Tensor | None]:
        x, weight, rmsnorm_denominator = ctx.saved_tensors
        x_grad = torch.empty_like(x)
        weight_grad = None if weight is None else torch.zeros_like(weight, dtype=torch.float32)

        rmsnorm_backward_triton(
            x=x,
            weight=weight,
            output_grad=output_grad,
            rmsnorm_denominator=rmsnorm_denominator,
            x_grad=x_grad,
            weight_grad=weight_grad,
            eps=ctx.eps,
        )

        if weight_grad is not None:
            weight_grad = weight_grad.type_as(weight)

        return x_grad, weight_grad, None, None


def rmsnorm_cute(
    x: torch.Tensor, weight: torch.Tensor | None, eps: float | None, memory_efficient: bool = False
) -> torch.Tensor:
    """RMSNorm computation

    Args:
        x (torch.Tensor): input activation
        weight (torch.Tensor | None): RMSNorm weight
        eps (float | None): epsilon
        memory_efficient (bool, optional): memory efficient = False caches RMSNorm's denominator in the forward.
            Defaults to False.

    Returns:
        torch.Tensor: output tensor
    """

    return _RMSNorm_Cute.apply(x, weight, eps, memory_efficient)
