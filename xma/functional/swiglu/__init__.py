# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import torch
import torch.nn.functional as F

from ...custom_op import CustomOp, ctx_save_for_backward
from ...kernel_backend import KernelBackend
from ...math import divide_if_divisible
from ...utils import (
    empty_like_contiguous,
    ensure_contiguous,
    is_cute_dsl_available,
    is_torch_xla_available,
    is_triton_available,
)


if is_cute_dsl_available():
    from .cuda_implementation import swiglu_backward_cuda, swiglu_forward_cuda


if is_triton_available():
    from .triton_implementation import swiglu_backward_triton, swiglu_forward_triton
if is_torch_xla_available():
    from .pallas_implementation import swiglu_forward_pallas


class _Swiglu(CustomOp):
    @staticmethod
    def forward_backward_torch(g: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        dtype = g.dtype

        g = g.float()
        u = u.float()

        y = u * F.silu(g)
        y = y.to(dtype)

        return y

    @staticmethod
    @ensure_contiguous
    def forward_cuda(ctx, g: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        y = empty_like_contiguous(g)
        swiglu_forward_cuda(g=g.flatten(0, -2), u=u.flatten(0, -2), y=y.flatten(0, -2))

        ctx_save_for_backward(ctx, g, u)

        return y

    @staticmethod
    @ensure_contiguous
    def backward_cuda(ctx, dy: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        g, u = ctx.saved_tensors
        dg = empty_like_contiguous(g)
        du = empty_like_contiguous(u)

        swiglu_backward_cuda(
            g=g.flatten(0, -2),
            u=u.flatten(0, -2),
            dy=dy.flatten(0, -2),
            dg=dg.flatten(0, -2),
            du=du.flatten(0, -2),
        )

        return dg, du

    @staticmethod
    @ensure_contiguous
    def forward_pallas(ctx, g: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        ctx_save_for_backward(ctx, g, u)
        return swiglu_forward_pallas(g=g, u=u)

    @staticmethod
    def forward_triton(ctx, g: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        y = empty_like_contiguous(g)
        swiglu_forward_triton(g=g, u=u, y=y)

        ctx_save_for_backward(ctx, g, u)

        return y

    @staticmethod
    def backward_triton(ctx, dy: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        g, u = ctx.saved_tensors
        dg = empty_like_contiguous(g)
        du = empty_like_contiguous(u)

        swiglu_backward_triton(g=g, u=u, dy=dy, dg=dg, du=du)

        return dg, du


class _SwigluPacked(CustomOp):
    @staticmethod
    def forward_backward_torch(x: torch.Tensor) -> torch.Tensor:
        up, gate = x.chunk(2, dim=-1)
        return swiglu(gate=gate, up=up, kernel_backend=KernelBackend.torch)

    @staticmethod
    def forward_cuda(ctx, x: torch.Tensor) -> torch.Tensor:
        ctx_save_for_backward(ctx, x)

        y = torch.empty(*x.size()[:-1], divide_if_divisible(x.size(-1), 2), device=x.device, dtype=x.dtype)
        u, g = x.chunk(2, dim=-1)

        swiglu_forward_cuda(g=g, u=u, y=y)

        return y

    @staticmethod
    def backward_cuda(ctx, dy: torch.Tensor) -> torch.Tensor:
        x = ctx.saved_tensors[0]
        dx = empty_like_contiguous(x)

        u, g = x.chunk(2, dim=-1)
        du, dg = dx.chunk(2, dim=-1)

        swiglu_backward_cuda(g=g, u=u, dy=dy, dg=dg, du=du)

        return dx

    @staticmethod
    def forward_triton(ctx, x: torch.Tensor) -> torch.Tensor:
        ctx_save_for_backward(ctx, x)

        y = torch.empty(*x.size()[:-1], divide_if_divisible(x.size(-1), 2), device=x.device, dtype=x.dtype)
        u, g = x.chunk(2, dim=-1)

        swiglu_forward_triton(g=g, u=u, y=y)

        return y

    @staticmethod
    def backward_triton(ctx, dy: torch.Tensor) -> torch.Tensor:
        x = ctx.saved_tensors[0]
        dx = empty_like_contiguous(x)

        u, g = x.chunk(2, dim=-1)
        du, dg = dx.chunk(2, dim=-1)

        swiglu_backward_triton(g=g, u=u, dy=dy, dg=dg, du=du)

        return dx


def swiglu(gate: torch.Tensor, up: torch.Tensor, *, kernel_backend: KernelBackend | None = None) -> torch.Tensor:
    """computes swiglu activation as `up` * `gate` * sigmoid(`gate`)

    Args:
        gate (torch.Tensor): `gate` activation tensor
        up (torch.Tensor): `up` activation tensor

    Returns:
        torch.Tensor: output tensor
    """

    assert gate.size() == up.size(), "tensors gate and up should have same shape"
    assert gate.type() == up.type(), "tensors gate and up should have same dtype"

    return _Swiglu.run(g=gate, u=up, kernel_backend=kernel_backend)


def swiglu_packed(x: torch.Tensor, *, kernel_backend: KernelBackend | None = None) -> torch.Tensor:
    """computes swiglu activation by splitting the tensor `x` into 2 parts: gate and up activations

    Args:
        x (torch.Tensor): input activation

    Returns:
        torch.Tensor: output tensor
    """

    return _SwigluPacked.run(x=x, kernel_backend=kernel_backend)
