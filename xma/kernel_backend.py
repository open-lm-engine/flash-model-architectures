# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

from __future__ import annotations

from enum import Enum

import torch

from .utils import is_cute_dsl_available, is_torch_xla_available, is_triton_available


if is_torch_xla_available():
    from torch_xla.core.xla_model import get_rng_state as xla_get_rng_state
    from torch_xla.core.xla_model import set_rng_state as xla_set_rng_state
    from torch_xla.core.xla_model import xla_device


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

    def get_current_device(self) -> None:
        if self in [KernelBackend.cuda, KernelBackend.rocm, KernelBackend.triton]:
            return torch.cuda.current_device()
        elif self == KernelBackend.pallas:
            return xla_device()

    def verify_kernel_backend(self) -> None:
        assert self.is_kernel_backend_compatible_with_current_device()

    def is_kernel_backend_compatible_with_current_device(self) -> bool:
        if self == KernelBackend.cuda:
            return not _IS_ROCM_AVAILABLE and torch.cuda.is_available() and is_cute_dsl_available()
        elif self == KernelBackend.pallas:
            return is_torch_xla_available()
        elif self == KernelBackend.rocm:
            return _IS_ROCM_AVAILABLE and torch.cuda.is_available()
        elif self == KernelBackend.triton:
            return torch.cuda.is_available() and is_triton_available()
