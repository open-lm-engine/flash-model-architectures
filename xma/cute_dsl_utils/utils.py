# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import torch

import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

from ..math import get_powers_of_2


def torch_tensor_to_cute_tensor(x: torch.Tensor, leading_dim: int) -> cute.Tensor:
    x = x.detach()

    alignment = 4
    for i in get_powers_of_2(4, 16):
        if x.data_ptr() % i != 0:
            break
        else:
            alignment = i

    x = from_dlpack(x, assumed_align=alignment)
    x = x.mark_layout_dynamic(leading_dim=leading_dim)

    return x
