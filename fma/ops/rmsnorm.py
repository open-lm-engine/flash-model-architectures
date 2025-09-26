# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import torch

from ..cutotune import CutoTuneParameter
from ..enums import KernelBackend
from .fused_residual_add_rmsnorm import fused_residual_add_rmsnorm


def rmsnorm(
    x: torch.Tensor,
    weight: torch.Tensor | None,
    eps: float | None,
    memory_efficient: bool = False,
    deterministic: bool = False,
    *,
    kernel_backend: KernelBackend | CutoTuneParameter = KernelBackend.triton,
) -> torch.Tensor:
    """RMSNorm computation

    Args:
        x (torch.Tensor): input activation
        weight (torch.Tensor | None): RMSNorm weight
        eps (float | None): epsilon
        memory_efficient (bool, optional): memory efficient = False caches RMSNorm's denominator in the forward.
            Defaults to False.
        deterministic (bool, optional): whether to use deterministic backward. Defaults to False.
        kernel_backend (KernelBackend | CutoTuneParameter, optional): kernel backend to prioritize.
            Defaults to KernelBackend.triton.

    Returns:
        torch.Tensor: output tensor
    """

    x, _ = fused_residual_add_rmsnorm(
        x=x,
        residual=None,
        weight=weight,
        eps=eps,
        multiplier=None,
        memory_efficient=memory_efficient,
        deterministic=deterministic,
        kernel_backend=kernel_backend,
    )

    return x
