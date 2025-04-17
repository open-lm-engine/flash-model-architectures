import torch

from ...constants import MAX_TRITON_BLOCK_SIZE
from ...math import get_next_power_of_2
from ...utils import ensure_contiguous
from .forward import _forward
from .torch_implementation import fused_residual_add_rmsnorm_torch
from .triton_implementation import fused_residual_add_rmsnorm_backward_triton


class _FusedResidualAddRMSNorm_Cute(torch.autograd.Function):
    @staticmethod
    @ensure_contiguous
    def forward(
        ctx,
        x: torch.Tensor,
        residual: torch.Tensor,
        weight: torch.Tensor | None,
        eps: float | None,
        multiplier: float | None,
        memory_efficient: bool,
    ) -> tuple[torch.Tensor]:
        if weight is not None:
            assert weight.dim() == 1, "weight should be 1D"
            assert weight.size(-1) == x.size(-1), "hidden size for x and weight tensor is different"
            assert weight.type() == x.type(), "tensors weight and y should have same dtype"

        is_x_1d = x.dim() == 1
        if is_x_1d:
            x = x.unsqueeze(0)

        if eps is None:
            eps = torch.finfo(x.dtype).eps

        output, added_x_residual, rmsnorm_denominator = _forward(
            x=x,
            residual=residual,
            weight=weight,
            eps=eps,
            multiplier=multiplier,
            memory_efficient=memory_efficient,
        )

        ctx.save_for_backward(added_x_residual, weight, rmsnorm_denominator)

        if is_x_1d:
            output = output.squeeze(0)
            added_x_residual = added_x_residual.squeeze(0)

        ctx.is_x_1d = is_x_1d
        ctx.eps = eps
        ctx.multiplier = multiplier

        return output, added_x_residual

    @staticmethod
    @ensure_contiguous
    def backward(ctx, output_grad: torch.Tensor, added_x_residual_grad: torch.Tensor) -> tuple[torch.Tensor | None]:
        added_x_residual, weight, rmsnorm_denominator = ctx.saved_tensors

        hidden_size = added_x_residual.size(-1)

        x_grad = torch.empty_like(added_x_residual)
        residual_grad = torch.empty_like(added_x_residual)
        weight_grad = None if weight is None else torch.zeros_like(weight, dtype=torch.float32)

        BLOCK_SIZE_B = 1
        BLOCK_SIZE_H = get_next_power_of_2(hidden_size)
        assert BLOCK_SIZE_H <= MAX_TRITON_BLOCK_SIZE

        fused_residual_add_rmsnorm_backward_triton(
            added_x_residual=added_x_residual,
            weight=weight,
            output_grad=output_grad,
            added_x_residual_grad=added_x_residual_grad,
            rmsnorm_denominator=rmsnorm_denominator,
            x_grad=x_grad,
            residual_grad=residual_grad,
            weight_grad=weight_grad,
            eps=ctx.eps,
            multiplier=ctx.multiplier,
            BLOCK_SIZE_B=BLOCK_SIZE_B,
            BLOCK_SIZE_H=BLOCK_SIZE_H,
        )

        if weight_grad is not None:
            weight_grad = weight_grad.type_as(weight)

        return x_grad, residual_grad, weight_grad, *[None] * 3


def fused_residual_add_rmsnorm_cute(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor | None,
    eps: float | None,
    multiplier: float | None = None,
    memory_efficient: bool = False,
) -> tuple[torch.Tensor]:
    return _FusedResidualAddRMSNorm_Cute.apply(x, residual, weight, eps, multiplier, memory_efficient)
