import torch

from ....cutotune import cutotune
from ..parameters import get_cutotune_parameters
from .cuda_implementation import add_scalar_cuda
from .triton_implementation import add_scalar_triton


@cutotune(**get_cutotune_parameters())
def _forward(
    x: torch.Tensor,
    y: float,
    kernel_backend: str,
    BLOCK_SIZE: int,
) -> torch.Tensor:
    if kernel_backend == "cuda":
        function = add_scalar_cuda
    elif kernel_backend == "triton":
        function = add_scalar_triton
    else:
        raise ValueError(f"unexpected kernel_backend ({kernel_backend})")

    output = torch.empty_like(x)
    function(x=x, y=y, output=output, BLOCK_SIZE=BLOCK_SIZE)

    return output
