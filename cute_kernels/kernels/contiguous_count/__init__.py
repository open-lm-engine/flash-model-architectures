import torch

from ...constants import COMMON_CUDA_BLOCK_SIZES_POWERS_OF_2
from ...cutotune import CutoTuneParameter, cutotune, get_cartesian_product_cutotune_configs
from ...enums import KernelBackend
from ...utils import get_sm_count
from .cuda_implementation import contiguous_count_cuda
from .torch_implementation import contiguous_count_torch
from .triton_implementation import contiguous_count_triton


@torch.no_grad()
@cutotune(
    get_cartesian_product_cutotune_configs(
        kernel_backend=[KernelBackend.triton],
        BLOCK_SIZE=COMMON_CUDA_BLOCK_SIZES_POWERS_OF_2,
        condition=lambda **kwargs: kwargs["size"] <= 16384,
    )
    + get_cartesian_product_cutotune_configs(
        kernel_backend=[KernelBackend.cuda], BLOCK_SIZE=COMMON_CUDA_BLOCK_SIZES_POWERS_OF_2
    )
)
def _contiguous_count_cute(
    x: torch.Tensor,
    size: int,
    kernel_backend: KernelBackend,
    BLOCK_SIZE: int,
) -> torch.Tensor:
    assert x.dim() == 1, "x should be 1-dimensional"
    assert x.dtype in [torch.int32, torch.long]

    output = torch.zeros(size, dtype=torch.uint32, device=x.device)

    if kernel_backend == KernelBackend.cuda:
        contiguous_count_cuda(
            x=x,
            output=output,
            sm_count=get_sm_count(x.device),
            thread_block_cluster_size=4,
            size=size,
            BLOCK_SIZE=BLOCK_SIZE,
        )
    elif kernel_backend == KernelBackend.triton:
        contiguous_count_triton(x=x, output=output, size=size, BLOCK_SIZE=BLOCK_SIZE)
    else:
        raise ValueError(f"unexpected kernel_backend ({kernel_backend})")

    return output


def contiguous_count_cute(
    x: torch.Tensor,
    size: int,
    kernel_backend: KernelBackend = CutoTuneParameter(),
    BLOCK_SIZE: int = CutoTuneParameter(),
) -> torch.Tensor:
    return _contiguous_count_cute(x=x, size=size, kernel_backend=kernel_backend, BLOCK_SIZE=BLOCK_SIZE)
