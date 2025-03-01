import torch

from ...cutotune import cutotune
from ...math import ceil_divide
from .cuda_implementation import swiglu_forward_cuda
from .parameters import get_cutotune_parameters
from .triton_implementation import _swiglu_forward_triton_kernel


@cutotune(**get_cutotune_parameters())
def _forward(gate: torch.Tensor, up: torch.Tensor, kernel_backend: str, BLOCK_SIZE: int) -> torch.Tensor:
    output = torch.empty_like(gate)

    if kernel_backend == "cuda":
        swiglu_forward_cuda(gate=gate, up=up, output=output, BLOCK_SIZE=BLOCK_SIZE)
    elif kernel_backend == "triton":
        num_elements = gate.numel()

        _swiglu_forward_triton_kernel[(ceil_divide(num_elements, BLOCK_SIZE),)](
            gate_ptr=gate,
            up_ptr=up,
            output_ptr=output,
            num_elements=num_elements,
            BLOCK_SIZE=BLOCK_SIZE,
        )
    else:
        raise ValueError(f"unexpected kernel_backend ({kernel_backend})")

    return output
