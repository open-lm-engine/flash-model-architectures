import torch

from ...cutotune import cutotune
from ..add_scalar.parameters import get_cutotune_parameters
from .cuda_implementation import add_tensor_cuda
from .triton_implementation import add_tensor_triton


@cutotune(**get_cutotune_parameters())
def _forward(
    x: torch.Tensor,
    y: torch.Tensor,
    kernel_backend: str,
    BLOCK_SIZE: int,
) -> torch.Tensor:
    if kernel_backend == "cuda":
        function = add_tensor_cuda
    elif kernel_backend == "triton":
        function = add_tensor_triton
    else:
        raise ValueError(f"unexpected kernel_backend ({kernel_backend})")

    output = torch.empty_like(x)
    function(x=x, y=y, output=output, BLOCK_SIZE=BLOCK_SIZE)

    return output
