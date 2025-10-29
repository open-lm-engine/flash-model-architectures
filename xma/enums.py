# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

from __future__ import annotations

from enum import Enum

import torch

from .utils import is_triton_available


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
            kernel_backend = KernelBackend.rocm if _IS_ROCM_AVAILABLE else KernelBackend.cuda
        elif device_type == "xla":
            kernel_backend = KernelBackend.pallas
        else:
            kernel_backend = KernelBackend.triton

        KernelBackend.verify_kernel_backend(kernel_backend)

        return kernel_backend

    @staticmethod
    def verify_kernel_backend(kernel_backend: KernelBackend) -> None:
        if kernel_backend == KernelBackend.triton:
            assert is_triton_available()
