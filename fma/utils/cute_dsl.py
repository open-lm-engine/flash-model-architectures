# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import torch

import cutlass.cute as cute
from cutlass.torch import from_dlpack


def torch_tensor_to_cute_tensor(
    x: torch.Tensor | None, leading_dim: int, alignment: int = 16, divisibility: int = 1
) -> cute.Tensor:
    if x is None:
        return None

    if leading_dim < 0:
        leading_dim += x.dim()

    x = x.detach()

    x = (
        from_dlpack(x, assumed_align=alignment)
        .mark_layout_dynamic(leading_dim=leading_dim)
        .mark_compact_shape_dynamic(mode=leading_dim, stride_order=x.dim_order(), divisibility=divisibility)
    )

    return x
