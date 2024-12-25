import torch

from .....constants import LIBRARY_NAME
from .....math import ceil_divide
from .....utils import cute_op
from .kernels_forward import _add_scalar_forward_triton_kernel


_KERNEL_NAME = "add_scalar_forward_triton"


@cute_op(f"{LIBRARY_NAME}::{_KERNEL_NAME}", mutates_args={"output"})
def add_scalar_forward_triton(x: torch.Tensor, y: float, output: torch.Tensor, BLOCK_SIZE: int) -> None:
    num_elements = x.numel()
    num_programs = ceil_divide(num_elements, BLOCK_SIZE)

    with torch.device(x.device):
        _add_scalar_forward_triton_kernel[(num_programs,)](
            x_ptr=x, y=y, output_ptr=output, num_elements=num_elements, BLOCK_SIZE=BLOCK_SIZE
        )
