import torch

from ....cutotune import cutotune
from ....enums import KernelBackend
from ..parameters import get_cutotune_parameters
from .cuda_implementation import add_scalar_forward_cuda
from .triton_implementation import add_scalar_forward_triton


@cutotune(**get_cutotune_parameters())
def _forward(
    x: torch.Tensor,
    y: float,
    kernel_backend: KernelBackend,
    vector_instruction_width: int,
    BLOCK_SIZE: int,
) -> torch.Tensor:
    output = torch.empty_like(x)

    if kernel_backend == KernelBackend.cuda:
        assert x.is_cuda, "tensor x is not on GPU"
        add_scalar_forward_cuda(
            x=x, y=y, output=output, vector_instruction_width=vector_instruction_width, BLOCK_SIZE=BLOCK_SIZE
        )
    elif kernel_backend == KernelBackend.triton:
        assert vector_instruction_width is None
        add_scalar_forward_triton(x=x, y=y, output=output, BLOCK_SIZE=BLOCK_SIZE)
    else:
        raise ValueError(f"unexpected kernel_backend ({kernel_backend})")

    return output
