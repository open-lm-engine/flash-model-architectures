import torch

from ...cutotune import cutotune
from ...math import divide_if_divisible
from .parameters import get_cutotune_parameters
from .triton_implementation import swiglu_unchunked_forward_triton


@cutotune(**get_cutotune_parameters())
def _forward(x: torch.Tensor, BLOCK_SIZE_B: int, BLOCK_SIZE_H: int) -> torch.Tensor:
    H = x.size(-1)

    output = torch.empty(*x.size()[:-1], divide_if_divisible(H, 2), device=x.device, dtype=x.dtype)
    swiglu_unchunked_forward_triton(x=x, output=output, BLOCK_SIZE_B=BLOCK_SIZE_B, BLOCK_SIZE_H=BLOCK_SIZE_H)

    return output
