import torch
import triton
import triton.language as tl

from .....constants import LIBRARY_NAME
from .....math import ceil_divide
from .....utils import cute_op
from .forward import _add_scalar_forward_triton_kernel


_KERNEL_NAME = "add_scalar_forward_triton"


@triton.jit
def _add_scalar_forward_triton_kernel(x_ptr, y, output_ptr, num_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)

    indices = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = indices < num_elements

    x = tl.load(x_ptr + indices, mask=mask)
    output = x + y

    tl.store(output_ptr + indices, output, mask=mask)


@cute_op(f"{LIBRARY_NAME}::{_KERNEL_NAME}", mutates_args={"output"})
def add_scalar_forward_triton(x: torch.Tensor, y: float, output: torch.Tensor, BLOCK_SIZE: int) -> None:
    num_elements = x.numel()
    num_programs = ceil_divide(num_elements, BLOCK_SIZE)

    with torch.device(x.device):
        _add_scalar_forward_triton_kernel[(num_programs,)](
            x_ptr=x, y=y, output_ptr=output, num_elements=num_elements, BLOCK_SIZE=BLOCK_SIZE
        )
