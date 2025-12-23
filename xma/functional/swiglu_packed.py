# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import torch

from ..accelerator import KernelBackend
from ..custom_op import CustomOp, ctx_save_for_backward
from ..math import divide_if_divisible
from ..utils import (
    empty_like_contiguous,
    ensure_contiguous,
    is_cute_dsl_available,
    is_torch_neuronx_available,
    is_torch_xla_available,
    is_triton_available,
    make_contiguous,
)
from .swiglu import swiglu


if is_cute_dsl_available():
    from .swiglu import swiglu_backward_cuda, swiglu_forward_cuda

if is_torch_neuronx_available():
    from .swiglu import swiglu_backward_nki, swiglu_forward_nki

if is_torch_xla_available():
    from .swiglu import swiglu_backward_pallas, swiglu_forward_pallas

if is_triton_available():
    from .swiglu import swiglu_backward_triton, swiglu_forward_triton


class _SwigluPacked(CustomOp):
    @staticmethod
    def forward_backward_torch(x: torch.Tensor) -> torch.Tensor:
        up, gate = x.chunk(2, dim=-1)
        return swiglu(gate=gate, up=up, kernel_backend=KernelBackend.torch)

    @staticmethod
    def forward(ctx, x: torch.Tensor, kernel_backend: KernelBackend) -> torch.Tensor:
        ctx.kernel_backend = kernel_backend

        if kernel_backend in [KernelBackend.cuda, KernelBackend.pallas]:
            x = make_contiguous(x)

        ctx_save_for_backward(ctx, x)

        u, g = x.chunk(2, dim=-1)

        if kernel_backend == KernelBackend.pallas:
            return swiglu_forward_pallas(g=g, u=u)

        y = torch.empty(*x.size()[:-1], divide_if_divisible(x.size(-1), 2), device=x.device, dtype=x.dtype)

        if kernel_backend == KernelBackend.cuda:
            swiglu_forward_cuda(g=g, u=u, y=y)
        elif kernel_backend == KernelBackend.nki:
            swiglu_forward_nki(g=g, u=u, y=y)
        elif kernel_backend == KernelBackend.triton:
            swiglu_forward_triton(g=g, u=u, y=y)
        else:
            raise ValueError(f"unexpected kernel_backend ({kernel_backend})")

        return y

    @staticmethod
    @ensure_contiguous
    def backward_cuda(ctx, dy: torch.Tensor) -> torch.Tensor:
        x = ctx.saved_tensors[0]
        dx = empty_like_contiguous(x)

        u, g = x.chunk(2, dim=-1)
        du, dg = dx.chunk(2, dim=-1)

        swiglu_backward_cuda(g=g, u=u, dy=dy, dg=dg, du=du)

        return dx

    @staticmethod
    def backward_nki(ctx, dy: torch.Tensor) -> torch.Tensor:
        x = ctx.saved_tensors[0]
        dx = empty_like_contiguous(x)

        # Trainium doesn't support non-contiguous output tensors
        u, g = x.chunk(2, dim=-1)
        du = empty_like_contiguous(u)
        dg = empty_like_contiguous(g)

        swiglu_backward_nki(g=g, u=u, dy=dy, dg=dg, du=du)
        dx = torch.cat([du, dg], dim=-1)

        return dx

    @staticmethod
    @ensure_contiguous
    def backward_pallas(ctx, dy: torch.Tensor) -> torch.Tensor:
        x = ctx.saved_tensors[0]
        dx = empty_like_contiguous(x)

        u, g = x.chunk(2, dim=-1)
        du, dg = dx.chunk(2, dim=-1)

        dg, du = swiglu_backward_pallas(g=g, u=u, dy=dy)
        dx = torch.cat([du, dg], dim=-1)

        return dx

    @staticmethod
    def backward_triton(ctx, dy: torch.Tensor) -> torch.Tensor:
        x = ctx.saved_tensors[0]
        dx = empty_like_contiguous(x)

        u, g = x.chunk(2, dim=-1)
        du, dg = dx.chunk(2, dim=-1)

        swiglu_backward_triton(g=g, u=u, dy=dy, dg=dg, du=du)

        return dx


def swiglu_packed(x: torch.Tensor, *, kernel_backend: KernelBackend | None = None) -> torch.Tensor:
    """computes swiglu activation by splitting the tensor `x` into 2 parts: gate and up activations

    Args:
        x (torch.Tensor): input activation

    Returns:
        torch.Tensor: output tensor
    """

    original_shape = x.size()
    x = x.flatten(0, -2)

    H = divide_if_divisible(original_shape[-1], 2)

    y = _SwigluPacked.run(x=x, kernel_backend=kernel_backend)
    y = y.view(*original_shape[:-1], H)

    return y
