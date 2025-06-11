# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import torch

from ...kernel_backend import KernelBackend


@torch.no_grad()
def zeros_cute(
    shape: torch.Size,
    dtype: torch.dtype | None = None,
    device: torch.device | None = None,
    *,
    kernel_backend: KernelBackend = KernelBackend.cuda,
) -> torch.Tensor:
    if kernel_backend == KernelBackend.torch:
        x = torch.zeros(*shape, dtype=dtype, device=device)
    elif kernel_backend == KernelBackend.triton:
        x = torch.empty(*shape, dtype=dtype, device=device)

    return x
