# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

from __future__ import annotations

from enum import Enum

import torch

from .utils import is_cute_dsl_available, is_triton_available


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

        kernel_backend.verify_kernel_backend()

        return kernel_backend

    def verify_kernel_backend(self) -> None:
        if self == KernelBackend.cuda:
            assert is_cute_dsl_available()
        elif self == KernelBackend.triton:
            assert is_triton_available()
