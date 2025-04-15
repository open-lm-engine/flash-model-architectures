import torch

from ...constants import MAX_TRITON_BLOCK_SIZE
from ...cutotune import cutotune
from ...math import ceil_divide, get_next_power_of_2
from ...utils import get_num_elements_and_hidden_size
from .parameters import get_cutotune_parameters
from .triton_implementation import _rmsnorm_forward_triton_kernel


def _forward(
    x: torch.Tensor,
    weight: torch.Tensor | None,
    eps: float,
    memory_efficient: bool,
    BLOCK_SIZE_B: int,
    BLOCK_SIZE_H: int,
) -> tuple[torch.Tensor | None]:
    num_elements, hidden_size = get_num_elements_and_hidden_size(x)

    output = torch.empty_like(x)
    rmsnorm_denominator = None if memory_efficient else torch.empty(num_elements, device=x.device, dtype=torch.float32)

    BLOCK_SIZE_H = get_next_power_of_2(hidden_size)
    assert BLOCK_SIZE_H <= MAX_TRITON_BLOCK_SIZE

    num_elements, hidden_size = get_num_elements_and_hidden_size(x)

    if BLOCK_SIZE_H < hidden_size:
        raise ValueError(f"hidden_size should be more than the BLOCK_SIZE_H")

    with torch.cuda.device(x.device):
        _rmsnorm_forward_triton_kernel[(ceil_divide(num_elements, BLOCK_SIZE_B),)](
            x_ptr=x,
            has_weight=weight is not None,
            weight_ptr=weight,
            output_ptr=output,
            eps=eps,
            has_rmsnorm_denominator=rmsnorm_denominator is not None,
            rmsnorm_denominator_ptr=rmsnorm_denominator,
            B=num_elements,
            H=hidden_size,
            BLOCK_SIZE_B=BLOCK_SIZE_B,
            BLOCK_SIZE_H=BLOCK_SIZE_H,
        )

    return output, rmsnorm_denominator
