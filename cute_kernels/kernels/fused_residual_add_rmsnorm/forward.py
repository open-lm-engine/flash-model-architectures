import torch

from ...constants import MAX_TRITON_BLOCK_SIZE
from ...math import get_next_power_of_2
from ...utils import get_num_elements_and_hidden_size
from .triton_implementation import fused_residual_add_rmsnorm_forward_triton


def _forward(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor | None,
    eps: float,
    multiplier: float | None,
    memory_efficient: bool,
) -> tuple[torch.Tensor | None]:
    num_elements, hidden_size = get_num_elements_and_hidden_size(x)

    output = torch.empty_like(x)
    added_x_residual = torch.empty_like(x)
    rmsnorm_denominator = None if memory_efficient else torch.empty(num_elements, device=x.device, dtype=torch.float32)

    BLOCK_SIZE_B = 1
    BLOCK_SIZE_H = get_next_power_of_2(hidden_size)
    assert BLOCK_SIZE_H <= MAX_TRITON_BLOCK_SIZE

    fused_residual_add_rmsnorm_forward_triton(
        x=x,
        residual=residual,
        weight=weight,
        output=output,
        eps=eps,
        multiplier=multiplier,
        added_x_residual=added_x_residual,
        rmsnorm_denominator=rmsnorm_denominator,
        BLOCK_SIZE_B=BLOCK_SIZE_B,
        BLOCK_SIZE_H=BLOCK_SIZE_H,
    )

    return output, added_x_residual, rmsnorm_denominator
