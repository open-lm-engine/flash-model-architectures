from contextlib import contextmanager
from enum import Enum


class KernelBackend(Enum):
    cuda = "cuda"
    triton = "triton"


def is_cuda_kernel_backend_allowed(kernel_backend: KernelBackend) -> bool:
    return kernel_backend in [None, KernelBackend.cuda]


def is_triton_kernel_backend_allowed(kernel_backend: KernelBackend) -> bool:
    return kernel_backend in [None, KernelBackend.triton]
