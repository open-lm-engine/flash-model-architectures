import torch

from ...constants import MAX_TRITON_BLOCK_SIZE
from ...cutotune import cutotune
from ...enums import KernelBackend
from ...math import get_next_power_of_2
from .parameters import get_cutotune_parameters
from .triton_implementation import rmsnorm_forward_triton


@cutotune(**get_cutotune_parameters())
def _forward(
    x: torch.Tensor,
    weight: torch.Tensor | None,
    eps: float,
    memory_efficient: bool,
    kernel_backend: KernelBackend,
    BLOCK_SIZE_B: int,
    BLOCK_SIZE_H: int,
) -> tuple[torch.Tensor | None]:
    hidden_size = x.size(-1)
    num_elements = x.numel() // hidden_size

    output = torch.empty_like(x)
    rmsnorm_denominator = None if memory_efficient else torch.empty(num_elements, device=x.device, dtype=torch.float32)

    if kernel_backend == KernelBackend.triton:
        BLOCK_SIZE_H = get_next_power_of_2(hidden_size)
        assert BLOCK_SIZE_H <= MAX_TRITON_BLOCK_SIZE

        rmsnorm_forward_triton(
            x=x,
            weight=weight,
            output=output,
            eps=eps,
            rmsnorm_denominator=rmsnorm_denominator,
            BLOCK_SIZE_B=BLOCK_SIZE_B,
            BLOCK_SIZE_H=BLOCK_SIZE_H,
        )
    else:
        raise ValueError(f"unexpected kernel_backend ({kernel_backend})")

    return output, rmsnorm_denominator
