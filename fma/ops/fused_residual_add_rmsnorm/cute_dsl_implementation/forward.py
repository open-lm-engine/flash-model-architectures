# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

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
