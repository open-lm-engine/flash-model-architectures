import torch

from ...constants import MAX_TRITON_BLOCK_SIZE
from ...cutotune import cutotune
from ...enums import KernelBackend
from ...math import get_next_power_of_2
from .parameters import get_cutotune_parameters
from .triton_implementation import rmsnorm_backward_triton


@cutotune(**get_cutotune_parameters())
def _backward(
    x: torch.Tensor,
    weight: torch.Tensor | None,
    eps: float,
    rmsnorm_denominator: torch.Tensor,
    output_grad: torch.Tensor,
    kernel_backend: KernelBackend,
    BLOCK_SIZE_B: int,
    BLOCK_SIZE_H: int,
) -> tuple[torch.Tensor | None]:
    hidden_size = x.size(-1)
    x_grad = torch.empty_like(x)

    if kernel_backend == KernelBackend.triton:
        BLOCK_SIZE_H = get_next_power_of_2(hidden_size)
        assert BLOCK_SIZE_H <= MAX_TRITON_BLOCK_SIZE

        weight_grad = rmsnorm_backward_triton(
            x=x,
            weight=weight,
            output_grad=output_grad,
            rmsnorm_denominator=rmsnorm_denominator,
            x_grad=x_grad,
            eps=eps,
            BLOCK_SIZE_B=BLOCK_SIZE_B,
            BLOCK_SIZE_H=BLOCK_SIZE_H,
        )
    else:
        raise ValueError(f"unexpected kernel_backend ({kernel_backend})")

    return x_grad, weight_grad
