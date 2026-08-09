# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

import torch

from ....accelerator import Accelerator
from ....custom_op import ctx_needs_gradients, ctx_save_for_backward
from .backward import _fused_residual_add_rmsnorm_backward_triton
from .forward import _fused_residual_add_rmsnorm_forward_triton


def _forward_impl(
    x: torch.Tensor,
    r: torch.Tensor | None,
    W: torch.Tensor | None,
    eps: float | None,
    multiplier: float | None,
    output_std: bool,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    if eps is None:
        eps = torch.finfo(x.dtype).eps

    B = x.size(0)

    y = torch.empty_like(x, memory_format=torch.contiguous_format)
    xr = None if r is None else torch.empty_like(x, memory_format=torch.contiguous_format)
    s = torch.empty(B, device=x.device, dtype=torch.float32) if output_std else None

    _fused_residual_add_rmsnorm_forward_triton(x=x, r=r, W=W, y=y, eps=eps, multiplier=multiplier, xr=xr, s=s)

    return y, xr, s


def _backward_impl(
    xr: torch.Tensor,
    W: torch.Tensor | None,
    s: torch.Tensor | None,
    dy: torch.Tensor,
    dxr: torch.Tensor | None,
    has_residual: bool,
    multiplier: float | None,
    eps: float | None,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    dx = torch.empty_like(xr, memory_format=torch.contiguous_format)
    dr = torch.empty_like(xr, memory_format=torch.contiguous_format) if has_residual else None

    dW = (
        None
        if W is None
        else torch.empty(Accelerator.get_core_count(), *W.size(), dtype=torch.float32, device=dx.device)
    )

    if not has_residual:
        assert dxr is None

    if eps is None:
        eps = torch.finfo(dy.dtype).eps

    _fused_residual_add_rmsnorm_backward_triton(
        xr=xr,
        W=W,
        dy=dy,
        dxr=dxr,
        s=s,
        dx=dx,
        dr=dr,
        dW=dW,
        eps=eps,
        multiplier=multiplier,
    )

    if dW is not None:
        dW = dW.sum(0)
        dW = dW.type_as(W)

    return dx, dr, dW


class _FusedResidualAddRMSNormTriton(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        r: torch.Tensor | None,
        W: torch.Tensor | None,
        eps: float | None,
        multiplier: float | None,
        memory_efficient: bool,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        y, xr, s = _forward_impl(
            x=x,
            r=r,
            W=W,
            eps=eps,
            multiplier=multiplier,
            output_std=ctx_needs_gradients(ctx) and not memory_efficient,
        )

        has_residual = r is not None

        ctx_save_for_backward(ctx, xr if has_residual else x, W, s)
        ctx.eps = eps
        ctx.has_residual = has_residual
        ctx.multiplier = multiplier

        return y, xr

    @staticmethod
    def backward(
        ctx, dy: torch.Tensor, dxr: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None, None, None, None]:
        xr, W, s = ctx.saved_tensors

        dx, dr, dW = _backward_impl(
            xr=xr, W=W, s=s, dy=dy, dxr=dxr, has_residual=ctx.has_residual, multiplier=ctx.multiplier, eps=ctx.eps
        )

        return dx, dr, dW, None, None, None
