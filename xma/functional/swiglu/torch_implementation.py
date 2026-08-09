# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

import torch
import torch.nn.functional as F

from ...accelerator import KernelBackend
from .op import _Swiglu, _SwigluPacked


def _torch(g: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    dtype = g.dtype

    g = g.float()
    u = u.float()

    y = u * F.silu(g)
    y = y.to(dtype)

    return y


def _torch_packed(x: torch.Tensor) -> torch.Tensor:
    dtype = x.dtype
    x = x.float()

    u = x[..., 1::2]
    g = x[..., ::2]

    x = u * F.silu(g)

    return x.to(dtype)


_Swiglu[KernelBackend.torch] = _torch
_SwigluPacked[KernelBackend.torch] = _torch_packed
