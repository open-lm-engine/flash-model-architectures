# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import torch
import torch.nn.functional as F

from ...constants import MAX_TRITON_BLOCK_SIZE
from ...counters import increment_counter
from ...cutotune import CutoTuneParameter
from ...enums import KernelBackend
from ...math import ceil_divide, get_next_power_of_2
from ...utils import ensure_contiguous, get_num_elements_and_hidden_size, get_sm_count
from .triton_implementation import (
    fused_residual_add_rmsnorm_backward_triton_kernel,
    fused_residual_add_rmsnorm_forward_triton_kernel,
)


class _FusedResidualAddRMSNorm(torch.autograd.Function):
    @staticmethod
    @ensure_contiguous
    def forward(
        ctx,
        x: torch.Tensor,
        residual: torch.Tensor | None,
        weight: torch.Tensor | None,
        eps: float | None,
        multiplier: float | None,
        memory_efficient: bool,
        kernel_backend: KernelBackend,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        assert kernel_backend == KernelBackend.triton or isinstance(kernel_backend, CutoTuneParameter)

        if weight is not None:
            assert weight.dim() == 1, "weight should be 1D"
            assert weight.size(-1) == x.size(-1), "hidden size for x and weight tensor is different"
            assert weight.type() == x.type(), "tensors weight and y should have same dtype"

        if eps is None:
            eps = torch.finfo(x.dtype).eps

        B, H = get_num_elements_and_hidden_size(x)
        has_residual = residual is not None

        output = torch.empty_like(x)
        added_x_residual = torch.empty_like(x) if has_residual else None
        rmsnorm_denominator = None if memory_efficient else torch.empty(B, device=x.device, dtype=torch.float32)

        BLOCK_SIZE_B = 1
        BLOCK_SIZE_H = get_next_power_of_2(H)
        assert BLOCK_SIZE_H <= MAX_TRITON_BLOCK_SIZE
        NUM_WARPS = 8

        fused_residual_add_rmsnorm_forward_triton_kernel[ceil_divide(B, BLOCK_SIZE_B),](
            x_ptr=x,
            residual_ptr=residual,
            weight_ptr=weight,
            output_ptr=output,
            eps=eps,
            multiplier=multiplier,
            added_x_residual_ptr=added_x_residual,
            rmsnorm_denominator_ptr=rmsnorm_denominator,
            B=B,
            H=H,
            BLOCK_SIZE_B=BLOCK_SIZE_B,
            BLOCK_SIZE_H=BLOCK_SIZE_H,
            num_warps=NUM_WARPS,
        )

        if residual is None:
            ctx.save_for_backward(x, weight, rmsnorm_denominator)
        else:
            ctx.save_for_backward(added_x_residual, weight, rmsnorm_denominator)

        ctx.eps = eps
        ctx.has_residual = has_residual
        ctx.multiplier = multiplier

        return output, added_x_residual

    @staticmethod
    @ensure_contiguous
    def backward(ctx, output_grad: torch.Tensor, added_x_residual_grad: torch.Tensor) -> tuple[torch.Tensor | None]:
        has_residual = ctx.has_residual

        added_x_residual, weight, rmsnorm_denominator = ctx.saved_tensors
        x_grad = torch.empty_like(added_x_residual)
        residual_grad = torch.empty_like(added_x_residual) if has_residual else None
        weight_grad = None if weight is None else torch.zeros_like(weight, dtype=torch.float32)

        if not has_residual:
            assert added_x_residual_grad is None

        B, H = get_num_elements_and_hidden_size(added_x_residual)

        BLOCK_SIZE_B = 1
        BLOCK_SIZE_H = get_next_power_of_2(H)
        assert BLOCK_SIZE_H <= MAX_TRITON_BLOCK_SIZE
        NUM_WARPS = 8

        sm_count = get_sm_count(added_x_residual.device)
        num_programs = min(sm_count, ceil_divide(B, BLOCK_SIZE_B))

        fused_residual_add_rmsnorm_backward_triton_kernel[num_programs,](
            added_x_residual_ptr=added_x_residual,
            weight_ptr=weight,
            output_grad_ptr=output_grad,
            added_x_residual_grad_ptr=added_x_residual_grad,
            x_grad_ptr=x_grad,
            residual_grad_ptr=residual_grad,
            weight_grad_ptr=weight_grad,
            eps=ctx.eps,
            multiplier=ctx.multiplier,
            rmsnorm_denominator_ptr=rmsnorm_denominator,
            B=B,
            H=H,
            BLOCK_SIZE_B=BLOCK_SIZE_B,
            BLOCK_SIZE_H=BLOCK_SIZE_H,
            num_warps=NUM_WARPS,
        )

        if weight_grad is not None:
            weight_grad = weight_grad.type_as(weight)

        return x_grad, residual_grad, weight_grad, *[None] * 4


def fused_residual_add_rmsnorm(
    x: torch.Tensor,
    residual: torch.Tensor | None,
    weight: torch.Tensor | None,
    eps: float | None,
    multiplier: float | None = None,
    memory_efficient: bool = False,
    kernel_backend: KernelBackend | CutoTuneParameter = KernelBackend.triton,
) -> tuple[torch.Tensor, torch.Tensor]:
    """fused residual add RMSNorm computation

    Args:
        x (torch.Tensor): input activation
        residual (torch.Tensor): residual activation
        weight (torch.Tensor | None): RMSNorm weight
        eps (float | None): epsilon
        multiplier (float | None, optional): if not None, pre-multiplies `x` with `multiplier`. Defaults to None.
        memory_efficient (bool, optional): memory efficient = False caches RMSNorm's denominator in the forward.
            Defaults to False.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: output activations, updated residual stream
    """

    if kernel_backend == KernelBackend.torch:
        if multiplier not in [None, 1]:
            x = x * multiplier

        if residual is not None:
            x = x + residual
            residual = x

        x = F.rms_norm(x, normalized_shape=(x.size(-1),), weight=weight, eps=eps)
    else:
        increment_counter(fused_residual_add_rmsnorm)
        x, residual = _FusedResidualAddRMSNorm.apply(
            x, residual, weight, eps, multiplier, memory_efficient, kernel_backend
        )

    return x, residual
