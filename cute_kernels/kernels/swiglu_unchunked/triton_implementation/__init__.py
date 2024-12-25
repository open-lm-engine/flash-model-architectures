import torch

from ....constants import LIBRARY_NAME
from ....math import ceil_divide
from ....utils import cute_op
from .kernels_backward import _swiglu_unchunked_backward_triton_kernel
from .kernels_forward import _swiglu_unchunked_forward_triton_kernel


_FORWARD_KERNEL_NAME = "swiglu_unchunked_forward_triton"
_BACKWARD_KERNEL_NAME = "swiglu_unchunked_backward_triton"


@cute_op(f"{LIBRARY_NAME}::{_FORWARD_KERNEL_NAME}", mutates_args={"output"})
def swiglu_unchunked_forward_triton(
    x: torch.Tensor, output: torch.Tensor, BLOCK_SIZE_B: int, BLOCK_SIZE_H: int
) -> None:
    H = x.size(-1)
    B = x.numel() // H

    with torch.device(x.device):
        _swiglu_unchunked_forward_triton_kernel[(ceil_divide(B, BLOCK_SIZE_B), ceil_divide(H, BLOCK_SIZE_H))](
            x_ptr=x,
            output_ptr=output,
            B=B,
            H=H,
            BLOCK_SIZE_B=BLOCK_SIZE_B,
            BLOCK_SIZE_H=BLOCK_SIZE_H,
        )


@cute_op(f"{LIBRARY_NAME}::{_BACKWARD_KERNEL_NAME}", mutates_args={"x_grad"})
def swiglu_unchunked_backward_triton(
    x: torch.Tensor,
    output_grad: torch.Tensor,
    x_grad: torch.Tensor,
    BLOCK_SIZE_B: int,
    BLOCK_SIZE_H: int,
) -> None:
    H = x.size(-1)
    B = x.numel() // H

    with torch.device(x.device):
        _swiglu_unchunked_backward_triton_kernel[(ceil_divide(B, BLOCK_SIZE_B), ceil_divide(H, BLOCK_SIZE_H))](
            x_ptr=x,
            output_grad_ptr=output_grad,
            x_grad_ptr=x_grad,
            B=B,
            H=H,
            BLOCK_SIZE_B=BLOCK_SIZE_B,
            BLOCK_SIZE_H=BLOCK_SIZE_H,
        )
