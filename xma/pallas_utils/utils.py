# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import torch
import torch.nn.functional as F

from ..math import ceil_divide


def pad_tensor_for_pallas(x: torch.Tensor, padding: tuple[int, int] = (8, 128)) -> torch.Tensor:
    padding = tuple(ceil_divide(x.size(-i), padding[-i]) * padding[-i] - x.size(-i) for i in range(len(padding)))

    if padding != (0, 0):
        x = F.pad(x, pad=padding)

    return x
