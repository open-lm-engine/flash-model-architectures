# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import torch
import torch.nn.functional as F

from ..math import ceil_divide


def pad_tensor_for_pallas(x: torch.Tensor, padding: tuple[int, int, int, int] = (8, 128)) -> torch.Tensor:
    padding = (
        ceil_divide(x.size(-2), padding[-2]) * padding[-2] - x.size(-2),
        ceil_divide(x.size(-1), padding[-1]) * padding[-1] - x.size(-1),
    )
    return F.pad(x, pad=padding)
