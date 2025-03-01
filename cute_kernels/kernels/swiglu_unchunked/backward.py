import torch

from ...cutotune import cutotune
from ...math import ceil_divide
from ...utils import get_num_elements_and_hidden_size
from .parameters import get_cutotune_parameters
from .triton_implementation import _swiglu_unchunked_backward_triton_kernel


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
        num_elements, hidden_size = get_num_elements_and_hidden_size(x)

        _swiglu_unchunked_backward_triton_kernel[
            (ceil_divide(num_elements, BLOCK_SIZE_B), ceil_divide(hidden_size, BLOCK_SIZE_H))
        ](
            x_ptr=x,
            output_grad_ptr=output_grad,
            x_grad_ptr=x_grad,
            B=num_elements,
            H=hidden_size,
            BLOCK_SIZE_B=BLOCK_SIZE_B,
            BLOCK_SIZE_H=BLOCK_SIZE_H,
        )
    else:
        raise ValueError(f"unexpected kernel_backend ({kernel_backend})")

    return x_grad
