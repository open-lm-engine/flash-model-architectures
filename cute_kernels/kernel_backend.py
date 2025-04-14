from contextlib import contextmanager
from enum import Enum


class KernelBackend(Enum):
    cuda = "cuda"
    triton = "triton"


_KERNEL_BACKEND: KernelBackend | None = None


@contextmanager
def kernel_backend(backend: KernelBackend | None = None):
    global _KERNEL_BACKEND

    current_backend = _KERNEL_BACKEND
    _KERNEL_BACKEND = backend

    yield

    _KERNEL_BACKEND = current_backend


def is_cuda_kernel_backend_allowed() -> bool:
    global _KERNEL_BACKEND
    return _KERNEL_BACKEND in [None, KernelBackend.cuda]


def is_triton_kernel_backend_allowed() -> bool:
    global _KERNEL_BACKEND
    return _KERNEL_BACKEND in [None, KernelBackend.triton]
