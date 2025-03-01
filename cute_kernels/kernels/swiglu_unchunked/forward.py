import torch

from ...cutotune import cutotune
from ...math import ceil_divide, divide_if_divisible
from ...utils import get_num_elements_and_hidden_size
from .parameters import get_cutotune_parameters
from .triton_implementation import _swiglu_unchunked_forward_triton_kernel


@cutotune(**get_cutotune_parameters())
def _forward(x: torch.Tensor, kernel_backend: str, BLOCK_SIZE_B: int, BLOCK_SIZE_H: int) -> torch.Tensor:
    num_elements, hidden_size = get_num_elements_and_hidden_size(x)
    output = torch.empty(*x.size()[:-1], divide_if_divisible(hidden_size, 2), device=x.device, dtype=x.dtype)

    if kernel_backend == "triton":
        _swiglu_unchunked_forward_triton_kernel[
            (ceil_divide(num_elements, BLOCK_SIZE_B), ceil_divide(hidden_size, BLOCK_SIZE_H))
        ](
            x_ptr=x,
            output_ptr=output,
            B=num_elements,
            H=hidden_size,
            BLOCK_SIZE_B=BLOCK_SIZE_B,
            BLOCK_SIZE_H=BLOCK_SIZE_H,
        )
    else:
        raise ValueError(f"unexpected kernel_backend ({kernel_backend})")

    return output
