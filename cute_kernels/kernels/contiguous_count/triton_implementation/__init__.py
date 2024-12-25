import torch

from ....constants import LIBRARY_NAME
from ....math import ceil_divide, get_next_power_of_2
from ....utils import cute_op, get_sm_count
from .kernels_forward import _contiguous_count_triton_kernel


_KERNEL_NAME = "contiguous_count_triton"


def _fake(x: torch.Tensor, size: int, BLOCK_SIZE_B: int) -> torch.Tensor:
    return torch.empty(size, dtype=torch.int32, device=x.device)


@cute_op(f"{LIBRARY_NAME}::{_KERNEL_NAME}", mutates_args={}, fake_func=_fake)
def contiguous_count_triton(x: torch.Tensor, size: int, BLOCK_SIZE_B: int) -> torch.Tensor:
    B = x.numel()
    BLOCK_SIZE_C = get_next_power_of_2(size)

    sm_count = get_sm_count(x.device)
    num_programs = min(sm_count, ceil_divide(B, BLOCK_SIZE_B))

    output = torch.zeros(size, dtype=torch.int32, device=x.device)

    with torch.device(x.device):
        _contiguous_count_triton_kernel[(num_programs,)](
            x_ptr=x,
            output_ptr=output,
            B=B,
            C=size,
            BLOCK_SIZE_B=BLOCK_SIZE_B,
            BLOCK_SIZE_C=BLOCK_SIZE_C,
        )

    return output
