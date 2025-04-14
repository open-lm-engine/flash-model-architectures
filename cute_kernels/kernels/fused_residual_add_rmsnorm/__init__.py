import torch

from ...constants import MAX_TRITON_BLOCK_SIZE
from ...cutotune import CutoTuneParameter
from ...math import ceil_divide, get_next_power_of_2
from ...utils import ensure_contiguous
from .backward import _backward
from .torch_implementation import fused_residual_add_rmsnorm_torch
from .triton_implementation import _fused_residual_add_rmsnorm_forward_triton_kernel


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

        num_elements, hidden_size = get_num_elements_and_hidden_size(x)

        output = torch.empty_like(x)
        added_x_residual = torch.empty_like(x)
        rmsnorm_denominator = (
            None if memory_efficient else torch.empty(num_elements, device=x.device, dtype=torch.float32)
        )

        BLOCK_SIZE_H = get_next_power_of_2(hidden_size)
        assert BLOCK_SIZE_H <= MAX_TRITON_BLOCK_SIZE

        if BLOCK_SIZE_H < hidden_size:
            raise ValueError(f"hidden_size should be more than the BLOCK_SIZE_H")

        with torch.cuda.device(x.device):
            _fused_residual_add_rmsnorm_forward_triton_kernel[(ceil_divide(num_elements, BLOCK_SIZE_B),)](
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
                B=num_elements,
                H=hidden_size,
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

        x_grad, residual_grad, weight_grad = _backward(
            added_x_residual=added_x_residual,
            weight=weight,
            eps=ctx.eps,
            multiplier=ctx.multiplier,
            rmsnorm_denominator=rmsnorm_denominator,
            output_grad=output_grad,
            added_x_residual_grad=added_x_residual_grad,
        )

        if ctx.is_x_1d:
            x_grad = x_grad.squeeze(0)
            residual_grad = residual_grad.squeeze(0)

        return x_grad, residual_grad, weight_grad, *[None] * 9


def fused_residual_add_rmsnorm_cute(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor | None,
    eps: float | None,
    multiplier: float | None = None,
    memory_efficient: bool = False,
) -> tuple[torch.Tensor]:
    return _FusedResidualAddRMSNorm_Cute.apply(x, residual, weight, eps, multiplier, memory_efficient)
