# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import torch
from torch.library import custom_op

import cutlass.cute as cute

from ....constants import LIBRARY_NAME
from ....utils import torch_tensor_to_cute_tensor


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


@custom_op(
    f"{LIBRARY_NAME}::fused_residual_add_rmsnorm_forward_cute_dsl",
    mutates_args={"output", "added_x_residual", "rmsnorm_denominator"},
)
def fused_residual_add_rmsnorm_forward_cute_dsl(
    x: torch.Tensor,
    residual: torch.Tensor | None,
    weight: torch.Tensor,
    output: torch.Tensor,
    eps: float,
    multiplier: float | None,
    added_x_residual: torch.Tensor | None,
    rmsnorm_denominator: torch.Tensor | None,
) -> None:
    x = torch_tensor_to_cute_tensor(x, leading_dim=-1)
    residual = torch_tensor_to_cute_tensor(residual, leading_dim=-1)
    weight = torch_tensor_to_cute_tensor(weight, leading_dim=-1)
    output = torch_tensor_to_cute_tensor(output, leading_dim=-1)
    added_x_residual = torch_tensor_to_cute_tensor(added_x_residual, leading_dim=-1)
    rmsnorm_denominator = torch_tensor_to_cute_tensor(rmsnorm_denominator, leading_dim=-1)
