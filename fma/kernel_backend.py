# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

from __future__ import annotations

from contextlib import contextmanager
from enum import Enum

import torch


_IS_ROCM_AVAILABLE = torch.version.hip is not None
_FORCE_TORCH_BACKEND = False
_FORCE_TRITON_BACKEND = False


@contextmanager
def force_torch_backend():
    global _FORCE_TORCH_BACKEND

    original_value = _FORCE_TORCH_BACKEND
    _FORCE_TORCH_BACKEND = True

    yield

    _FORCE_TORCH_BACKEND = original_value


@contextmanager
def force_triton_backend():
    global _FORCE_TRITON_BACKEND

    original_value = _FORCE_TRITON_BACKEND
    _FORCE_TRITON_BACKEND = True

    yield

    _FORCE_TRITON_BACKEND = original_value


class KernelBackend(Enum):
    cpu = "cpu"
    cuda = "cuda"
    rocm = "rocm"
    tpu = "tpu"
    xpu = "xpu"
    # for triton compatible accelerators
    triton = "triton"
    torch = "torch"

    @staticmethod
    def get_kernel_backend_from_device(x: torch.Tensor) -> KernelBackend:
        global _FORCE_TORCH_BACKEND, _FORCE_TRITON_BACKEND

        if _FORCE_TORCH_BACKEND:
            return KernelBackend.torch

        if _FORCE_TRITON_BACKEND:
            return KernelBackend.triton

        device_type = x.device.type

        if device_type == "cuda":
            return KernelBackend.rocm if _IS_ROCM_AVAILABLE else KernelBackend.cuda
        elif device_type == "cpu":
            return KernelBackend.cpu
        elif device_type == "xla":
            return KernelBackend.tpu
        elif device_type == "xpu":
            return KernelBackend.xpu

        raise ValueError(f"Unsupported device type: {device_type}")
