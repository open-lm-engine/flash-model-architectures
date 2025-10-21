# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import torch
import torch.nn.functional as F

from ...counters import increment_counter
from ...enums import KernelBackend
from ...utils import empty_like_contiguous, get_num_elements_and_hidden_size, get_sm_count, zeros_like_contiguous
from .triton_implementation import (
    fused_residual_add_rmsnorm_backward_triton,
    fused_residual_add_rmsnorm_forward_triton,
)


class _FusedResidualAddRMSNorm(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        residual: torch.Tensor | None,
        weight: torch.Tensor | None,
        eps: float | None,
        multiplier: float | None,
        memory_efficient: bool,
        deterministic: bool,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if eps is None:
            eps = torch.finfo(x.dtype).eps

        B, _ = get_num_elements_and_hidden_size(x)
        has_residual = residual is not None

        output = empty_like_contiguous(x)
        added_x_residual = empty_like_contiguous(x) if has_residual else None
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
        )

        ctx.save_for_backward(added_x_residual if has_residual else x, weight, rmsnorm_denominator)
        ctx.eps = eps
        ctx.has_residual = has_residual
        ctx.multiplier = multiplier
        ctx.deterministic = deterministic

        return output, added_x_residual

    @staticmethod
    def backward(ctx, output_grad: torch.Tensor, added_x_residual_grad: torch.Tensor) -> tuple[torch.Tensor | None]:
        has_residual = ctx.has_residual
        deterministic = ctx.deterministic

        added_x_residual, weight, rmsnorm_denominator = ctx.saved_tensors
        x_grad = empty_like_contiguous(added_x_residual)
        residual_grad = empty_like_contiguous(added_x_residual) if has_residual else None

        if weight is not None:
            if deterministic:
                weight_grad = torch.empty(
                    get_sm_count(x_grad.device), *weight.size(), dtype=weight.dtype, device=weight.device
                )
            else:
                weight_grad = zeros_like_contiguous(weight, dtype=torch.float32)
        else:
            weight_grad = None

        if not has_residual:
            assert added_x_residual_grad is None

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
            deterministic=deterministic,
        )

        if weight_grad is not None:
            if deterministic:
                weight_grad = weight_grad.sum(0)
            else:
                weight_grad = weight_grad.type_as(weight)

        return x_grad, residual_grad, weight_grad, *[None] * 4


def fused_residual_add_rmsnorm(
    x: torch.Tensor,
    residual: torch.Tensor | None,
    weight: torch.Tensor | None,
    eps: float | None,
    multiplier: float | None = None,
    memory_efficient: bool = False,
    deterministic: bool = False,
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
        deterministic (bool, optional): whether to use deterministic backward. Defaults to False.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: output activations, updated residual stream
    """

    if weight is not None:
        assert weight.dim() == 1, "weight should be 1D"
        assert weight.size(-1) == x.size(-1), "hidden size for x and weight tensor is different"
        assert weight.type() == x.type(), "tensors weight and y should have same dtype"

    kernel_backend = KernelBackend.get_kernel_backend_from_device(x)

    if kernel_backend == KernelBackend.torch:
        if multiplier not in [None, 1]:
            x = x * multiplier

        if residual is not None:
            x = x + residual
            residual = x

        x = F.rms_norm(x, normalized_shape=(x.size(-1),), weight=weight, eps=eps)
    else:
        assert kernel_backend in [KernelBackend.cuda, KernelBackend.triton]
        increment_counter(fused_residual_add_rmsnorm)

        is_flat = x.dim() == 1
        if is_flat:
            x = x[None, :]

            if residual is not None:
                residual = residual[None, :]

        x, residual = _FusedResidualAddRMSNorm.apply(
            x, residual, weight, eps, multiplier, memory_efficient, deterministic
        )

        if is_flat:
            x = x.squeeze(0)

            if residual is not None:
                residual = residual.squeeze(0)

    return x, residual
