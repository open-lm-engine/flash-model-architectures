# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import torch
import torch.nn.functional as F


def rmsnorm_torch(x: torch.Tensor, weight: torch.Tensor | None, eps: float) -> torch.Tensor:
    return F.rms_norm(x, (x.size(-1),), weight=weight, eps=eps)
