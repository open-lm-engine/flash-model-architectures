# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

import torch

from ....accelerator import KernelBackend
from ....custom_op import ctx_save_for_backward
from ....utils import empty_like_contiguous
from ..op import _Swiglu
from .backward import _backward_triton
from .forward import _forward_triton


def _forward(ctx, g: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    ctx_save_for_backward(ctx, g, u)

    y = empty_like_contiguous(g)
    _forward_triton(g=g, u=u, y=y)

    return y


def _backward(ctx, dy: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    g, u = ctx.saved_tensors

    dg = empty_like_contiguous(g)
    du = empty_like_contiguous(u)
    _backward_triton(g=g, u=u, dy=dy, dg=dg, du=du)

    return dg, du


_Swiglu[KernelBackend.triton] = (_forward, _backward)
