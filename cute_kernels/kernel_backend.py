# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

from contextlib import contextmanager
from enum import Enum

from .cutotune import CutoTuneParameter


class KernelBackend(Enum):
    cuda = "cuda"
    torch = "torch"
    triton = "triton"


def is_cuda_kernel_backend_allowed(kernel_backend: KernelBackend) -> bool:
    return isinstance(kernel_backend, CutoTuneParameter) or kernel_backend in [None, KernelBackend.cuda]


def is_triton_kernel_backend_allowed(kernel_backend: KernelBackend) -> bool:
    return isinstance(kernel_backend, CutoTuneParameter) or kernel_backend in [None, KernelBackend.triton]
