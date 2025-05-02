import torch

from ...utils import ensure_contiguous, get_num_elements_and_hidden_size
from .torch_implementation import fused_residual_add_rmsnorm_torch
from .triton_implementation import (
    fused_residual_add_rmsnorm_backward_triton,
    fused_residual_add_rmsnorm_forward_triton,
)


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
        BLOCK_SIZE_B_forward: int,
        BLOCK_SIZE_B_backward: int,
    ) -> tuple[torch.Tensor]:
        if weight is not None:
            assert weight.dim() == 1, "weight should be 1D"
            assert weight.size(-1) == x.size(-1), "hidden size for x and weight tensor is different"
            assert weight.type() == x.type(), "tensors weight and y should have same dtype"

        if eps is None:
            eps = torch.finfo(x.dtype).eps

        B, _ = get_num_elements_and_hidden_size(x)

        output = torch.empty_like(x)
        added_x_residual = torch.empty_like(x)
        rmsnorm_denominator = None if memory_efficient else torch.empty(B, device=x.device, dtype=torch.float32)

        fused_residual_add_rmsnorm_forward_triton(
            x=x,
            residual=residual,
            weight=weight,
            output=output,
            eps=eps,
            multiplier=multiplier,
            added_x_residual=added_x_residual,
            rmsnorm_denominator=rmsnorm_denominator,
            BLOCK_SIZE_B=BLOCK_SIZE_B_forward,
        )

        ctx.save_for_backward(added_x_residual, weight, rmsnorm_denominator)
        ctx.eps = eps
        ctx.multiplier = multiplier
        ctx.BLOCK_SIZE_B_backward = BLOCK_SIZE_B_backward

        return output, added_x_residual

    @staticmethod
    @ensure_contiguous
    def backward(ctx, output_grad: torch.Tensor, added_x_residual_grad: torch.Tensor) -> tuple[torch.Tensor | None]:
        added_x_residual, weight, rmsnorm_denominator = ctx.saved_tensors
        x_grad = torch.empty_like(added_x_residual)
        residual_grad = torch.empty_like(added_x_residual)
        weight_grad = None if weight is None else torch.zeros_like(weight, dtype=torch.float32)

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
            BLOCK_SIZE_B=ctx.BLOCK_SIZE_B_backward,
        )

        if weight_grad is not None:
            weight_grad = weight_grad.type_as(weight)

        return x_grad, residual_grad, weight_grad, *[None] * 5


def fused_residual_add_rmsnorm_cute(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor | None,
    eps: float | None,
    multiplier: float | None = None,
    memory_efficient: bool = False,
    *,
    BLOCK_SIZE_B_forward: int = 1,
    BLOCK_SIZE_B_backward: int = 1,
) -> tuple[torch.Tensor]:
    """fused residual add RMSNorm computation

    Args:
        x (torch.Tensor): input activation
        residual (torch.Tensor): residual activation
        weight (torch.Tensor | None): RMSNorm weight
        eps (float | None): epsilon
        multiplier (float | None, optional): if not None, pre-multiplies `x` with `multiplier`. Defaults to None.
        memory_efficient (bool, optional): memory efficient = False caches RMSNorm's denominator in the forward.
            Defaults to False.
        BLOCK_SIZE_B_forward (int, optional): block size along the batch dimension for forward. Defaults to 1.
        BLOCK_SIZE_B_backward (int, optional): block size along the batch dimension for backward. Defaults to 1.

    Returns:
        tuple[torch.Tensor]: output activations, updated residual stream
    """

    return _FusedResidualAddRMSNorm_Cute.apply(
        x, residual, weight, eps, multiplier, memory_efficient, BLOCK_SIZE_B_forward, BLOCK_SIZE_B_backward
    )
