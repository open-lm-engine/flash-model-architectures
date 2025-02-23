import torch

from ...cutotune import cutotune
from .parameters import get_cutotune_parameters
from .triton_implementation import swiglu_unchunked_backward_triton


@cutotune(**get_cutotune_parameters())
def _backward(
    x: torch.Tensor,
    output_grad: torch.Tensor,
    kernel_backend: str,
    BLOCK_SIZE_B: int,
    BLOCK_SIZE_H: int,
) -> tuple[torch.Tensor]:
    x_grad = torch.empty_like(x)

    if kernel_backend == "triton":
        swiglu_unchunked_backward_triton(
            x=x, output_grad=output_grad, x_grad=x_grad, BLOCK_SIZE_B=BLOCK_SIZE_B, BLOCK_SIZE_H=BLOCK_SIZE_H
        )
    else:
        raise ValueError(f"unexpected kernel_backend ({kernel_backend})")

    return x_grad
