# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import torch
import torch.nn.functional as F

from ...custom_op import CustomOp, ctx_needs_gradients, ctx_save_for_backward
from ...enums import KernelBackend
from ...utils import (
    empty_like_contiguous,
    get_num_elements_and_hidden_size,
    get_sm_count,
    is_triton_available,
    zeros_like_contiguous,
)


if is_triton_available():
    from .triton_implementation import (
        fused_residual_add_rmsnorm_backward_triton,
        fused_residual_add_rmsnorm_forward_triton,
    )


class _FusedResidualAddRMSNorm(CustomOp):
    @staticmethod
    def forward_backward_torch(
        x: torch.Tensor,
        residual: torch.Tensor | None,
        weight: torch.Tensor | None,
        eps: float | None,
        multiplier: float | None,
        memory_efficient: bool,
        deterministic: bool,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if multiplier not in [None, 1]:
            x = x * multiplier

        if residual is not None:
            x = x + residual
            residual = x

        x = F.rms_norm(x, normalized_shape=(x.size(-1),), weight=weight, eps=eps)

        return x, residual

    @staticmethod
    def forward_triton(
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

        rmsnorm_denominator = None
        if ctx_needs_gradients(ctx) and not memory_efficient:
            rmsnorm_denominator = torch.empty(B, device=x.device, dtype=torch.float32)

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

        ctx_save_for_backward(ctx, added_x_residual if has_residual else x, weight, rmsnorm_denominator)
        ctx.eps = eps
        ctx.has_residual = has_residual
        ctx.multiplier = multiplier
        ctx.deterministic = deterministic

        return output, added_x_residual

    @staticmethod
    def backward_triton(
        ctx, output_grad: torch.Tensor, added_x_residual_grad: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None, None, None, None, None]:
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
    *,
    kernel_backend: KernelBackend | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
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
        tuple[torch.Tensor, torch.Tensor | None]: output activations, updated residual stream
    """

    if weight is not None:
        assert weight.dim() == 1, "weight should be 1D"
        assert weight.size(-1) == x.size(-1), "hidden size for x and weight tensor is different"
        assert weight.type() == x.type(), "tensors weight and y should have same dtype"

    # if 1D -> make 2D
    is_flat = x.dim() == 1
    if is_flat:
        x = x[None, ...]

        if residual is not None:
            residual = residual[None, :]

    x, residual = _FusedResidualAddRMSNorm.run(
        x=x,
        residual=residual,
        weight=weight,
        eps=eps,
        multiplier=multiplier,
        memory_efficient=memory_efficient,
        deterministic=deterministic,
        kernel_backend=kernel_backend,
    )

    # convert back to 1D
    if is_flat:
        x = x.squeeze(0)

        if residual is not None:
            residual = residual.squeeze(0)

    return x, residual
