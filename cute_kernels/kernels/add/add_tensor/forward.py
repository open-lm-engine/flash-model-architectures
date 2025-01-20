import torch

from ....cutotune import cutotune
from ....enums import KernelBackend
from ..parameters import get_cutotune_parameters
from .cuda_implementation import add_tensor_cuda
from .triton_implementation import add_tensor_triton


_KERNEL_MAP = {KernelBackend.cuda: add_tensor_cuda, KernelBackend.triton: add_tensor_triton}


@cutotune(**get_cutotune_parameters())
def _forward(
    x: torch.Tensor,
    y: torch.Tensor,
    kernel_backend: KernelBackend,
    BLOCK_SIZE: int,
) -> torch.Tensor:
    output = torch.empty_like(x)
    _KERNEL_MAP[kernel_backend](x=x, y=y, output=output, BLOCK_SIZE=BLOCK_SIZE)

    return output
