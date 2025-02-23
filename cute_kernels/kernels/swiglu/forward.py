import torch

from ...cutotune import cutotune
from .cuda_implementation import swiglu_forward_cuda
from .parameters import get_cutotune_parameters
from .triton_implementation import swiglu_forward_triton


_KERNEL_MAP = {"cuda": swiglu_forward_cuda, "triton": swiglu_forward_triton}


@cutotune(**get_cutotune_parameters())
def _forward(gate: torch.Tensor, up: torch.Tensor, kernel_backend: str, BLOCK_SIZE: int) -> torch.Tensor:
    output = torch.empty_like(gate)
    _KERNEL_MAP[kernel_backend](gate=gate, up=up, output=output, BLOCK_SIZE=BLOCK_SIZE)

    return output
