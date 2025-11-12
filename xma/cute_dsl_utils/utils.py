# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import torch

import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

from ..utils import get_alignment


def torch_tensor_to_cute_tensor(x: torch.Tensor) -> cute.Tensor:
    x = x.detach()
    x: cute.Tensor = from_dlpack(x, assumed_align=get_alignment(x))
    x = x.mark_layout_dynamic()
    return x
