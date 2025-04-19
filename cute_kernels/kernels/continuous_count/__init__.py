import torch

from .cuda_implementation import continuous_count_cuda
from .torch_implementation import continuous_count_torch


@torch.no_grad()
def continuous_count_cute(
    x: torch.Tensor, size: int, *, BLOCK_SIZE: int = 1024, THREAD_BLOCK_CLUSTER_SIZE: int = 8
) -> torch.Tensor:
    """counts the number of occurances of the values [0, 1, ..., `size`) in the input tensor (`size` is excluded).
        NOTE: the user is responsible for ensuring that the values lie in the valid range, any values outside this
        range are ignored and not counted.

    Args:
        x (torch.Tensor): input tensor
        size (int): values [0, 1, ..., `size`) are counted (`size` is excluded)
        BLOCK_SIZE (int, optional): block size for CUDA backend. Defaults to 1024.
        THREAD_BLOCK_CLUSTER_SIZE (int, optional): thread block cluster size refers to the size of the cluster for
            hierarchical accumulation, 1 means no thread block clusters. Defaults to 8.

    Returns:
        torch.Tensor: output tensor
    """

    if size == 1:
        return torch.tensor([x.numel()], dtype=torch.uint32, device=x.device)

    assert x.dim() == 1, "x should be 1-dimensional"
    assert x.dtype in [torch.int32, torch.long]

    output = torch.empty(size, dtype=torch.uint32, device=x.device)

    continuous_count_cuda(
        x=x, output=output, C=size, THREAD_BLOCK_CLUSTER_SIZE=THREAD_BLOCK_CLUSTER_SIZE, BLOCK_SIZE=BLOCK_SIZE
    )

    return output
