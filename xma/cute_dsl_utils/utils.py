# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import torch

import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack


def torch_tensor_to_cute_tensor(x: torch.Tensor, leading_dim: int, alignment: int = 16) -> cute.Tensor:
    x = x.detach()
    x = from_dlpack(x, assumed_align=alignment)
    x = x.mark_layout_dynamic(leading_dim=leading_dim)
    return x
