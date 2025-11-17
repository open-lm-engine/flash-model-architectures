# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

from __future__ import annotations

from enum import Enum

import torch

from .utils import is_cute_dsl_available, is_torch_xla_available, is_triton_available


if is_torch_xla_available():
    from torch_xla.core.xla_model import xla_device


_IS_ROCM_AVAILABLE = torch.version.hip is not None


class KernelBackend(Enum):
    cuda = "cuda"
    nki = "nki"
    pallas = "pallas"
    rocm = "rocm"
    torch = "torch"
    triton = "triton"

    def get_accelerator(self) -> Accelerator:
        if self in [KernelBackend.torch, KernelBackend.triton]:
            return Accelerator.get_accelerator()

        mapping = {
            KernelBackend.cuda: Accelerator.cuda,
            KernelBackend.nki: Accelerator.trainium,
            KernelBackend.pallas: Accelerator.tpu,
            KernelBackend.rocm: Accelerator.rocm,
        }

        return mapping[self]

    def is_kernel_backend_compatible_with_current_device(self) -> bool:
        if self == KernelBackend.cuda:
            return not _IS_ROCM_AVAILABLE and torch.cuda.is_available() and is_cute_dsl_available()
        elif self == KernelBackend.pallas:
            return is_torch_xla_available()
        elif self == KernelBackend.rocm:
            return _IS_ROCM_AVAILABLE and torch.cuda.is_available()
        elif self == KernelBackend.triton:
            return torch.cuda.is_available() and is_triton_available()
        elif self == KernelBackend.torch:
            return True
        else:
            raise ValueError(f"unexpected kernel_backend ({self})")


class Accelerator(Enum):
    cpu = "cpu"
    cuda = "cuda"
    rocm = "rocm"
    tpu = "tpu"
    trainium = "trainium"

    @staticmethod
    def get_accelerator() -> Accelerator:
        if torch.cuda.is_available():
            return Accelerator.rocm if _IS_ROCM_AVAILABLE else Accelerator.cuda
        elif is_torch_xla_available():
            return Accelerator.tpu

        return Accelerator.cpu

    @staticmethod
    def get_current_device() -> int | str:
        accelerator = Accelerator.get_accelerator()

        if accelerator in [Accelerator.cuda, Accelerator.rocm]:
            device = torch.cuda.current_device()
        elif accelerator == Accelerator.tpu:
            device = xla_device()
        elif accelerator == Accelerator.cpu:
            device = "cpu"

        return device

    def get_kernel_backend(self) -> KernelBackend:
        accelerator = Accelerator.get_accelerator()

        if accelerator == Accelerator.cuda:
            kernel_backend = KernelBackend.rocm if _IS_ROCM_AVAILABLE else KernelBackend.cuda
        elif accelerator == Accelerator.tpu:
            kernel_backend = KernelBackend.pallas
        else:
            kernel_backend = KernelBackend.triton

        return kernel_backend
