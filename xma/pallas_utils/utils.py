# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import torch
import torch.nn.functional as F

from ..math import ceil_divide


def pad_tensor_for_pallas(x: torch.Tensor, padding: tuple[int, int] = (8, 128)) -> torch.Tensor:
    full_padding = []
    for i in range(1, len(padding) + 1):
        pad = ceil_divide(x.size(-i), padding[-i]) * padding[-i] - x.size(-i)
        full_padding.append(0)
        full_padding.append(pad)

    if padding != (0, 0):
        x = F.pad(x, pad=full_padding)

    return x
