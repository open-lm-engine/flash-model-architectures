import torch

from ...cutotune import cutotune
from ...math import ceil_divide
from .parameters import get_cutotune_parameters
from .triton_implementation import _embedding_forward_triton_kernel


@cutotune(**get_cutotune_parameters())
def _forward(
    input_ids: torch.Tensor,
    weight: torch.Tensor,
    kernel_backend: str,
    BLOCK_SIZE_B: int,
    BLOCK_SIZE_H: int,
) -> torch.Tensor:
    num_elements = input_ids.numel()
    hidden_size = weight.size(-1)

    output = torch.empty(num_elements, hidden_size, dtype=weight.dtype, device=input_ids.device)

    if kernel_backend == "triton":
        num_elements = input_ids.numel()
        hidden_size = weight.size(-1)

        with torch.cuda.device(input_ids.device):
            _embedding_forward_triton_kernel[
                (ceil_divide(num_elements, BLOCK_SIZE_B), ceil_divide(hidden_size, BLOCK_SIZE_H))
            ](
                x_ptr=input_ids,
                weight_ptr=weight,
                output_ptr=output,
                B=num_elements,
                H=hidden_size,
                BLOCK_SIZE_B=BLOCK_SIZE_B,
                BLOCK_SIZE_H=BLOCK_SIZE_H,
            )
    else:
        raise ValueError(f"unexpected kernel_backend ({kernel_backend})")

    return output.view(*input_ids.size(), hidden_size)
