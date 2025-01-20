import torch

from ....cutotune import cutotune
from ....enums import KernelBackend
from ..parameters import get_cutotune_parameters
from .cuda_implementation import add_scalar_cuda
from .triton_implementation import add_scalar_triton


_KERNEL_MAP = {KernelBackend.cuda: add_scalar_cuda, KernelBackend.triton: add_scalar_triton}


@cutotune(**get_cutotune_parameters())
def _forward(
    x: torch.Tensor,
    y: float,
    kernel_backend: KernelBackend,
    BLOCK_SIZE: int,
) -> torch.Tensor:
    output = torch.empty_like(x)
    _KERNEL_MAP[kernel_backend](x=x, y=y, output=output, BLOCK_SIZE=BLOCK_SIZE)

    return output
