# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import torch

from ...kernel_backend import KernelBackend
from .cuda_implementation import zeros_cuda
from .triton_implementation import fill_triton


@torch.no_grad()
def zeros_cute(
    shape: torch.Size,
    dtype: torch.dtype | None = None,
    device: torch.device | None = None,
    *,
    kernel_backend: KernelBackend = KernelBackend.cuda,
) -> torch.Tensor:
    if kernel_backend == KernelBackend.cuda:
        x = torch.empty(*shape, dtype=dtype, device=device)
        zeros_cuda(x)
    elif kernel_backend == KernelBackend.triton:
        x = torch.empty(*shape, dtype=dtype, device=device)
        fill_triton(x=x, fill_value=0)
    elif kernel_backend == KernelBackend.torch:
        x = torch.zeros(*shape, dtype=dtype, device=device)
    else:
        raise ValueError(f"unexpected kernel_backend ({kernel_backend})")

    return x
