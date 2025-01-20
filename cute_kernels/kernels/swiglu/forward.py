import torch

from ...cutotune import cutotune
from ...enums import KernelBackend
from .cuda_implementation import swiglu_forward_cuda
from .parameters import get_cutotune_parameters
from .triton_implementation import swiglu_forward_triton


_KERNEL_MAP = {KernelBackend.cuda: swiglu_forward_cuda, KernelBackend.triton: swiglu_forward_triton}


@cutotune(**get_cutotune_parameters())
def _forward(gate: torch.Tensor, up: torch.Tensor, kernel_backend: KernelBackend, BLOCK_SIZE: int) -> torch.Tensor:
    output = torch.empty_like(gate)
    _KERNEL_MAP[kernel_backend](gate=gate, up=up, output=output, BLOCK_SIZE=BLOCK_SIZE)

    return output
