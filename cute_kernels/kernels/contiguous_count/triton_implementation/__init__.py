import torch

from ....constants import LIBRARY_NAME
from ....math import ceil_divide, get_next_power_of_2
from ....utils import cute_op, get_sm_count
from .kernels_forward import _contiguous_count_triton_kernel


_KERNEL_NAME = "contiguous_count_triton"


@cute_op(f"{LIBRARY_NAME}::{_KERNEL_NAME}", mutates_args={"output"})
def contiguous_count_triton(x: torch.Tensor, output: torch.Tensor, size: int, BLOCK_SIZE: int) -> None:
    B = x.numel()
    BLOCK_SIZE_C = get_next_power_of_2(size)

    sm_count = get_sm_count(x.device)
    num_programs = min(sm_count, ceil_divide(B, BLOCK_SIZE))

    with torch.device(x.device):
        _contiguous_count_triton_kernel[(num_programs,)](
            x_ptr=x,
            output_ptr=output,
            B=B,
            C=size,
            BLOCK_SIZE_B=BLOCK_SIZE,
            BLOCK_SIZE_C=BLOCK_SIZE_C,
        )
