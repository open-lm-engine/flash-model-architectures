# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import torch

import cutlass.cute as cute


@cute.jit
def fused_residual_add_rmsnorm_cute_dsl(
    x: cute.Tensor,
    residual: cute.Tensor | None,
    weight: cute.Tensor,
    output: cute.Tensor,
    eps: float,
    multiplier: float | None,
    added_x_residual: cute.Tensor | None,
    rmsnorm_denominator: cute.Tensor | None,
) -> None:
    print(x)


@cute.jit
def fused_residual_add_rmsnorm_forward_cute_dsl(
    x: torch.Tensor,
    residual: torch.Tensor | None,
    weight: torch.Tensor,
    output: torch.Tensor,
    eps: float,
    multiplier: float | None,
    added_x_residual: torch.Tensor | None,
    rmsnorm_denominator: torch.Tensor | None,
) -> None: ...
