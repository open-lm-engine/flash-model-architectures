# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

import torch

from ....accelerator import KernelBackend
from ....custom_op import ctx_save_for_backward
from ....math import divide_if_divisible
from ....utils import empty_like_contiguous
from ..op import _Swiglu, _SwigluPacked
from .backward import _swiglu_backward_cuda, _swiglu_packed_backward_cuda
from .forward import _forward_cuda, _packed_forward_cuda


class _SwigluCUDA(torch.autograd.Function):
    @staticmethod
    def forward(ctx, g: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        g = g.contiguous()
        u = u.contiguous()
        ctx_save_for_backward(ctx, g, u)

        y = empty_like_contiguous(g)
        _forward_cuda(g=g, u=u, y=y)

        return y

    @staticmethod
    def backward(ctx, dy: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        g, u = ctx.saved_tensors
        dy = dy.contiguous()

        dg = empty_like_contiguous(g)
        du = empty_like_contiguous(u)
        _swiglu_backward_cuda(g=g, u=u, dy=dy, dg=dg, du=du)

        return dg, du


class _SwigluPackedCUDA(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        ctx_save_for_backward(ctx, x)

        y = torch.empty(*x.size()[:-1], divide_if_divisible(x.size(-1), 2), device=x.device, dtype=x.dtype)
        _packed_forward_cuda(x=x, y=y)

        return y

    @staticmethod
    def backward(ctx, dy: torch.Tensor) -> torch.Tensor:
        x = ctx.saved_tensors[0]

        dx = empty_like_contiguous(x)
        _swiglu_packed_backward_cuda(x=x, dy=dy, dx=dx)

        return dx


_Swiglu[KernelBackend.cuda] = _SwigluCUDA
_SwigluPacked[KernelBackend.cuda] = _SwigluPackedCUDA
