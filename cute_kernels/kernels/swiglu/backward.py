import torch

from ...cutotune import cutotune
from ...enums import KernelBackend
from .cuda_implementation import swiglu_backward_cuda
from .parameters import get_cutotune_parameters
from .triton_implementation import swiglu_backward_triton


@cutotune(**get_cutotune_parameters())
def _backward(
    gate: torch.Tensor,
    up: torch.Tensor,
    output_grad: torch.Tensor,
    kernel_backend: KernelBackend,
    BLOCK_SIZE: int,
) -> tuple[torch.Tensor]:
    gate_grad = torch.empty_like(gate)
    up_grad = torch.empty_like(up)

    if kernel_backend == KernelBackend.cuda:
        swiglu_backward_cuda(
            gate=gate, up=up, output_grad=output_grad, gate_grad=gate_grad, up_grad=up_grad, BLOCK_SIZE=BLOCK_SIZE
        )
    elif kernel_backend == KernelBackend.triton:
        swiglu_backward_triton(
            gate=gate, up=up, output_grad=output_grad, gate_grad=gate_grad, up_grad=up_grad, BLOCK_SIZE=BLOCK_SIZE
        )
    else:
        raise ValueError(f"unexpected kernel_backend ({kernel_backend})")

    return gate_grad, up_grad
