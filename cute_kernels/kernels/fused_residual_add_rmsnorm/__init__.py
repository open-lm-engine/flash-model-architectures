import torch

from ...constants import MAX_TRITON_BLOCK_SIZE
from ...math import ceil_divide, get_next_power_of_2
from ...utils import ensure_contiguous, get_num_elements_and_hidden_size, get_sm_count
from .torch_implementation import fused_residual_add_rmsnorm_torch
from .triton_implementation import (
    _fused_residual_add_rmsnorm_backward_triton_kernel,
    _fused_residual_add_rmsnorm_forward_triton_kernel,
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

        B, H = get_num_elements_and_hidden_size(x)
        BLOCK_SIZE_B = 1
        BLOCK_SIZE_H = get_next_power_of_2(H)
        assert BLOCK_SIZE_H <= MAX_TRITON_BLOCK_SIZE

        output = torch.empty_like(x)
        added_x_residual = torch.empty_like(x)
        rmsnorm_denominator = None if memory_efficient else torch.empty(B, device=x.device, dtype=torch.float32)

        with torch.cuda.device(x.device):
            _fused_residual_add_rmsnorm_forward_triton_kernel[ceil_divide(B, H),](
                x_ptr=x,
                residual_ptr=residual,
                has_weight=weight is not None,
                weight_ptr=weight,
                output_ptr=output,
                eps=eps,
                has_multiplier=multiplier is not None and multiplier != 1,
                multiplier=multiplier,
                added_x_residual_ptr=added_x_residual,
                has_rmsnorm_denominator=rmsnorm_denominator is not None,
                rmsnorm_denominator_ptr=rmsnorm_denominator,
                B=B,
                H=H,
                BLOCK_SIZE_B=BLOCK_SIZE_B,
                BLOCK_SIZE_H=BLOCK_SIZE_H,
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

        x_grad = torch.empty_like(added_x_residual)
        residual_grad = torch.empty_like(added_x_residual)
        weight_grad = None if weight is None else torch.zeros_like(weight, dtype=torch.float32)

        B, H = get_num_elements_and_hidden_size(added_x_residual)
        BLOCK_SIZE_B = 1
        BLOCK_SIZE_H = get_next_power_of_2(H)
        assert BLOCK_SIZE_H <= MAX_TRITON_BLOCK_SIZE

        sm_count = get_sm_count(added_x_residual.device)
        num_programs = min(sm_count, ceil_divide(B, BLOCK_SIZE_B))
        multiplier = ctx.multiplier

        with torch.cuda.device(added_x_residual.device):
            _fused_residual_add_rmsnorm_backward_triton_kernel[num_programs,](
                added_x_residual_ptr=added_x_residual,
                has_weight=weight is not None,
                weight_ptr=weight,
                output_grad_ptr=output_grad,
                added_x_residual_grad_ptr=added_x_residual_grad,
                x_grad_ptr=x_grad,
                residual_grad_ptr=residual_grad,
                weight_grad_ptr=weight_grad,
                eps=ctx.eps,
                has_multiplier=multiplier is not None and multiplier != 1,
                multiplier=multiplier,
                has_rmsnorm_denominator=rmsnorm_denominator is not None,
                rmsnorm_denominator_ptr=rmsnorm_denominator,
                B=B,
                H=H,
                BLOCK_SIZE_B=BLOCK_SIZE_B,
                BLOCK_SIZE_H=BLOCK_SIZE_H,
            )

        if weight_grad is not None:
            weight_grad = weight_grad.type_as(weight)

        if ctx.is_x_1d:
            x_grad = x_grad.squeeze(0)
            residual_grad = residual_grad.squeeze(0)

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
