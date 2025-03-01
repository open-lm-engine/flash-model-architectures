import torch

from ....cutotune import cutotune
from ....math import ceil_divide
from ..parameters import get_cutotune_parameters
from .cuda_implementation import add_tensor_cuda
from .triton_implementation import _add_tensor_triton_kernel


@cutotune(**get_cutotune_parameters())
def _forward(
    x: torch.Tensor,
    y: torch.Tensor,
    kernel_backend: str,
    BLOCK_SIZE: int,
) -> torch.Tensor:
    output = torch.empty_like(x)

    if kernel_backend == "cuda":
        add_tensor_cuda(x=x, y=y, output=output, BLOCK_SIZE=BLOCK_SIZE)
    elif kernel_backend == "triton":
        num_elements = x.numel()
        num_programs = ceil_divide(num_elements, BLOCK_SIZE)

        _add_tensor_triton_kernel[(num_programs,)](
            x_ptr=x, y_ptr=y, output_ptr=output, num_elements=num_elements, BLOCK_SIZE=BLOCK_SIZE
        )
    else:
        raise ValueError(f"unexpected kernel_backend ({kernel_backend})")

    return output
