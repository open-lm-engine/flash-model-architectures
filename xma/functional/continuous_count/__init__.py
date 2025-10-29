# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import torch

from ...enums import KernelBackend
from .cuda_implementation import continuous_count_cuda


@torch.no_grad()
def continuous_count(x: torch.Tensor, size: int, *, kernel_backend: KernelBackend | None = None) -> torch.Tensor:
    """counts the number of occurances of the values [0, 1, ..., `size`) in the input tensor (`size` is excluded).
        NOTE: the user is responsible for ensuring that the values lie in the valid range, any values outside this
        range are ignored and not counted.

    Args:
        x (torch.Tensor): input tensor
        size (int): values [0, 1, ..., `size`) are counted (`size` is excluded)

    Returns:
        torch.Tensor: output tensor
    """

    if size == 1:
        return torch.tensor([x.numel()], dtype=torch.uint32, device=x.device)

    assert x.dim() == 1, "x should be 1-dimensional"
    assert x.dtype in [torch.int32, torch.long]

    if kernel_backend is None:
        kernel_backend = KernelBackend.get_kernel_backend_from_device(x)
    else:
        KernelBackend.verify_kernel_backend(kernel_backend)

    if kernel_backend == KernelBackend.torch:
        output = x.bincount(minlength=size).to(torch.uint32)
    elif kernel_backend in [KernelBackend.cuda, KernelBackend.triton]:
        output = torch.empty(size, dtype=torch.uint32, device=x.device)
        continuous_count_cuda(x=x, output=output, E=size, THREAD_BLOCK_CLUSTER_SIZE=1, BLOCK_SIZE=1024)
    else:
        raise ValueError(f"unexpected kernel_backend ({kernel_backend})")

    return output
