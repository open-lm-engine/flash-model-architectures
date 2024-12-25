import torch

from ...cutotune import cutotune
from ...enums import KernelBackend
from .parameters import get_cutotune_parameters
from .triton_implementation import embedding_forward_triton


@cutotune(**get_cutotune_parameters())
def _forward(
    input_ids: torch.Tensor,
    weight: torch.Tensor,
    kernel_backend: KernelBackend,
    BLOCK_SIZE_B: int,
    BLOCK_SIZE_H: int,
) -> torch.Tensor:
    num_elements = input_ids.numel()
    hidden_size = weight.size(-1)

    output = torch.empty(num_elements, hidden_size, dtype=weight.dtype, device=input_ids.device)

    if kernel_backend == KernelBackend.triton:
        embedding_forward_triton(
            input_ids=input_ids, weight=weight, output=output, BLOCK_SIZE_B=BLOCK_SIZE_B, BLOCK_SIZE_H=BLOCK_SIZE_H
        )
    else:
        raise ValueError(f"unexpected kernel_backend ({kernel_backend})")

    return output.view(*input_ids.size(), hidden_size)
