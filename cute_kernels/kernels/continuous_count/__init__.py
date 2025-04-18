import torch

from .cuda_implementation import continuous_count_cuda
from .torch_implementation import continuous_count_torch


@torch.no_grad()
def continuous_count_cute(x: torch.Tensor, size: int) -> torch.Tensor:
    if size == 1:
        return torch.tensor([x.numel()], dtype=torch.uint32, device=x.device)

    assert x.dim() == 1, "x should be 1-dimensional"
    assert x.dtype in [torch.int32, torch.long]

    BLOCK_SIZE = 1024
    THREAD_BLOCK_CLUSTER_SIZE = 8

    output = torch.empty(size, dtype=torch.uint32, device=x.device)

    continuous_count_cuda(
        x=x, output=output, C=size, THREAD_BLOCK_CLUSTER_SIZE=THREAD_BLOCK_CLUSTER_SIZE, BLOCK_SIZE=BLOCK_SIZE
    )

    return output
