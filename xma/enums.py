# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

from __future__ import annotations

from contextlib import contextmanager
from enum import Enum

import torch


_IS_ROCM_AVAILABLE = torch.version.hip is not None
_FORCED_KERNEL_BACKEND = None


@contextmanager
def force_kernel_backend(kernel_backend: KernelBackend):
    global _FORCED_KERNEL_BACKEND

    original_value = _FORCED_KERNEL_BACKEND
    _FORCED_KERNEL_BACKEND = kernel_backend

    yield

    _FORCED_KERNEL_BACKEND = original_value


class KernelBackend(Enum):
    cuda = "cuda"
    rocm = "rocm"
    pallas = "pallas"
    # for triton compatible accelerators
    triton = "triton"
    torch = "torch"

    @staticmethod
    def get_kernel_backend_from_device(x: torch.Tensor) -> KernelBackend:
        global _FORCED_KERNEL_BACKEND

        if _FORCED_KERNEL_BACKEND is not None:
            return _FORCED_KERNEL_BACKEND

        device_type = x.device.type

        if device_type == "cuda":
            return KernelBackend.rocm if _IS_ROCM_AVAILABLE else KernelBackend.cuda
        elif device_type == "xla":
            return KernelBackend.pallas
        else:
            return KernelBackend.triton
