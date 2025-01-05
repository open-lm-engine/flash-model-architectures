import torch

from ...cutotune import CutoTuneParameter
from ...enums import KernelBackend
from ...utils import get_sm_count
from .cuda_implementation import continuous_count_and_sort_cuda
from .torch_implementation import continuous_count_and_sort_torch


@torch.no_grad()
def continuous_count_and_sort_cute(
    x: torch.Tensor,
    size: int,
    kernel_backend: KernelBackend = CutoTuneParameter(),
    BLOCK_SIZE: int = CutoTuneParameter(),
) -> tuple[torch.Tensor]:
    assert x.dim() == 1, "x should be 1-dimensional"
    assert x.dtype in [torch.int32, torch.long]

    output = torch.empty(size, dtype=torch.uint32, device=x.device)

    if kernel_backend == KernelBackend.cuda:
        continuous_count_and_sort_cuda(
            x=x,
            output=output,
            sm_count=get_sm_count(x.device),
            size=size,
            BLOCK_SIZE=BLOCK_SIZE,
        )
    else:
        raise ValueError(f"unexpected kernel_backend ({kernel_backend})")

    return output
