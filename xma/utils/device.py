# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import torch


def is_hip() -> bool:
    return torch.version.hip is not None


def is_nvidia_gpu() -> bool:
    return torch.cuda.is_available() and torch.version.hip is None
