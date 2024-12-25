import torch

from ...cutotune import cutotune
from ...enums import KernelBackend
from .cuda_implementation import swiglu_forward_cuda
from .parameters import get_cutotune_parameters
from .triton_implementation import swiglu_forward_triton


@cutotune(**get_cutotune_parameters())
def _forward(gate: torch.Tensor, up: torch.Tensor, kernel_backend: KernelBackend, BLOCK_SIZE: int) -> torch.Tensor:
    output = torch.empty_like(gate)

    if kernel_backend == KernelBackend.cuda:
        assert gate.is_cuda, "tensor gate is not on GPU"
        assert up.is_cuda, "tensor up is not on GPU"

        swiglu_forward_cuda(gate=gate, up=up, output=output, BLOCK_SIZE=BLOCK_SIZE)
    elif kernel_backend == KernelBackend.triton:
        swiglu_forward_triton(gate=gate, up=up, output=output, BLOCK_SIZE=BLOCK_SIZE)
    else:
        raise ValueError(f"unexpected kernel_backend ({kernel_backend})")

    return output
