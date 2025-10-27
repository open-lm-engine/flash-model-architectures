# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

from __future__ import annotations

from enum import Enum

import torch


_IS_ROCM_AVAILABLE = torch.version.hip is not None


class KernelBackend(Enum):
    cuda = "cuda"
    nki = "nki"
    pallas = "pallas"
    rocm = "rocm"
    torch = "torch"
    triton = "triton"

    @staticmethod
    def get_kernel_backend_from_device(x: torch.Tensor) -> KernelBackend:
        device_type = x.device.type

        if device_type == "cuda":
            return KernelBackend.rocm if _IS_ROCM_AVAILABLE else KernelBackend.cuda
        elif device_type == "xla":
            return KernelBackend.pallas
        else:
            return KernelBackend.triton
