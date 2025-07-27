# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import torch
import torch.nn.functional as F


def sigmoid(x: torch.Tensor) -> torch.Tensor:
    return F.sigmoid(x.float()).type_as(x)


def tanh(x: torch.Tensor) -> torch.Tensor:
    return F.tanh(x.float()).type_as(x)
