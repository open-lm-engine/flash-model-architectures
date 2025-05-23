# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import torch


@torch.no_grad()
def continuous_count_torch(x: torch.Tensor, size: int) -> torch.Tensor:
    return x.bincount(minlength=size).to(torch.uint32)
