import torch

from ....constants import LIBRARY_NAME
from ....math import ceil_divide
from ....utils import cute_op
from .kernels_backward import _swiglu_backward_triton_kernel
from .kernels_forward import _swiglu_forward_triton_kernel


_FORWARD_KERNEL_NAME = "swiglu_forward_triton"
_BACKWARD_KERNEL_NAME = "swiglu_backward_triton"


@cute_op(f"{LIBRARY_NAME}::{_FORWARD_KERNEL_NAME}", mutates_args={"output"})
def swiglu_forward_triton(gate: torch.Tensor, up: torch.Tensor, output: torch.Tensor, BLOCK_SIZE: int) -> None:
    num_elements = gate.numel()

    with torch.device(gate.device):
        _swiglu_forward_triton_kernel[(ceil_divide(num_elements, BLOCK_SIZE),)](
            gate_ptr=gate,
            up_ptr=up,
            output_ptr=output,
            num_elements=num_elements,
            BLOCK_SIZE=BLOCK_SIZE,
        )


@cute_op(f"{LIBRARY_NAME}::{_BACKWARD_KERNEL_NAME}", mutates_args={"gate_grad", "up_grad"})
def swiglu_backward_triton(
    gate: torch.Tensor,
    up: torch.Tensor,
    output_grad: torch.Tensor,
    gate_grad: torch.Tensor,
    up_grad: torch.Tensor,
    BLOCK_SIZE: int,
) -> None:
    num_elements = gate.numel()

    with torch.device(gate.device):
        _swiglu_backward_triton_kernel[(ceil_divide(num_elements, BLOCK_SIZE),)](
            gate_ptr=gate,
            up_ptr=up,
            output_grad_ptr=output_grad,
            gate_grad_ptr=gate_grad,
            up_grad_ptr=up_grad,
            num_elements=num_elements,
            BLOCK_SIZE=BLOCK_SIZE,
        )
