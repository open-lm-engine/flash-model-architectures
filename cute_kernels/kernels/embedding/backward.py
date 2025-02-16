import torch

from ...cutotune import cutotune
from .parameters import get_cutotune_parameters
from .triton_implementation import embedding_backward_triton


@cutotune(**get_cutotune_parameters())
def _backward(
    input_ids: torch.Tensor,
    weight: torch.Tensor,
    output_grad: torch.Tensor,
    kernel_backend: str,
    BLOCK_SIZE_B: int,
    BLOCK_SIZE_H: int,
) -> torch.Tensor:
    weight_grad = torch.zeros_like(weight)

    if kernel_backend == "triton":
        embedding_backward_triton(
            input_ids=input_ids,
            output_grad=output_grad,
            weight_grad=weight_grad,
            BLOCK_SIZE_B=BLOCK_SIZE_B,
            BLOCK_SIZE_H=BLOCK_SIZE_H,
        )
    else:
        raise ValueError(f"unexpected kernel_backend ({kernel_backend})")

    return weight_grad
