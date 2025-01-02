import torch
import triton
import triton.language as tl

from .....constants import LIBRARY_NAME
from .....math import ceil_divide
from .....utils import cute_op


_KERNEL_NAME = "contiguous_count_high_atomic_add_triton"


@triton.jit
def _contiguous_count_high_atomic_add_triton_kernel(x_ptr, output_ptr, B, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)

    indices = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = indices < B

    x = tl.load(x_ptr + indices, mask=mask)
    tl.atomic_add(output_ptr + x, 1, mask=mask)


@cute_op(f"{LIBRARY_NAME}::{_KERNEL_NAME}", mutates_args={"output"})
def contiguous_count_high_atomic_add_triton(x: torch.Tensor, output: torch.Tensor, size: int, BLOCK_SIZE: int) -> None:
    B = x.numel()

    with torch.device(x.device):
        _contiguous_count_high_atomic_add_triton_kernel[(ceil_divide(B, BLOCK_SIZE),)](
            x_ptr=x,
            output_ptr=output,
            B=B,
            BLOCK_SIZE=BLOCK_SIZE,
        )
