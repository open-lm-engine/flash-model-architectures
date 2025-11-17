# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

from __future__ import annotations

from enum import Enum

import torch

from .utils import is_torch_xla_available


if is_torch_xla_available():
    from torch_xla.core.xla_model import wait_device_ops as xla_wait_device_ops
    from torch_xla.core.xla_model import xla_device


_IS_ROCM_AVAILABLE = torch.version.hip is not None


class KernelBackend(Enum):
    cuda = "cuda"
    nki = "nki"
    pallas = "pallas"
    rocm = "rocm"
    torch = "torch"
    triton = "triton"

    def get_compatible_accelerator(self) -> Accelerator:
        found_accelerator = Accelerator.get_accelerator()

        if self == KernelBackend.torch or (
            self == KernelBackend.triton and found_accelerator in [Accelerator.cuda, Accelerator.rocm]
        ):
            return found_accelerator

        mapping = {
            KernelBackend.cuda: Accelerator.cuda,
            KernelBackend.nki: Accelerator.trainium,
            KernelBackend.pallas: Accelerator.tpu,
            KernelBackend.rocm: Accelerator.rocm,
        }

        return mapping.get(self, None)

    def verify_accelerator(self) -> bool:
        expected_accelerator = self.get_compatible_accelerator()
        found_accelerator = Accelerator.get_accelerator()
        return expected_accelerator == found_accelerator


class Accelerator(Enum):
    cpu = "cpu"
    cuda = "cuda"
    rocm = "rocm"
    tpu = "tpu"
    trainium = "trainium"

    @staticmethod
    def get_accelerator() -> Accelerator:
        if torch.cuda.is_available():
            accelerator = Accelerator.rocm if _IS_ROCM_AVAILABLE else Accelerator.cuda
        elif is_torch_xla_available():
            accelerator = Accelerator.tpu
        else:
            accelerator = Accelerator.cpu

        return accelerator

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

    @staticmethod
    def synchronize() -> None:
        accelerator = Accelerator.get_accelerator()

        if accelerator == Accelerator.cuda:
            torch.cuda.synchronize()
        elif accelerator == Accelerator.tpu:
            xla_wait_device_ops()
