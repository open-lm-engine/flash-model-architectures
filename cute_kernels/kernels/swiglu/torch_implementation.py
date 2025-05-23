# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import torch
import torch.nn.functional as F


def swiglu_torch(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    dtype = gate.dtype

    gate = gate.float()
    up = up.float()

    output = up * F.silu(gate)
    output = output.to(dtype)

    return output


def swiglu_packed_torch(x: torch.Tensor) -> torch.Tensor:
    x = x.chunk(2, dim=-1)
    return swiglu_torch(x[1], x[0])
