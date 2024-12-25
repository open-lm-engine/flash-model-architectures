import torch

from ....constants import LIBRARY_NAME
from ....math import ceil_divide
from ....utils import cute_op
from .kernels_backward import _embedding_backward_triton_kernel
from .kernels_forward import _embedding_forward_triton_kernel


_FORWARD_KERNEL_NAME = "embedding_forward_triton"
_BACKWARD_KERNEL_NAME = "embedding_backward_triton"


@cute_op(f"{LIBRARY_NAME}::{_FORWARD_KERNEL_NAME}", mutates_args={"output"})
def embedding_forward_triton(
    input_ids: torch.Tensor,
    weight: torch.Tensor,
    output: torch.Tensor,
    BLOCK_SIZE_B: int,
    BLOCK_SIZE_H: int,
) -> None:
    num_elements = input_ids.numel()
    hidden_size = weight.size(-1)

    with torch.device(input_ids.device):
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


@cute_op(f"{LIBRARY_NAME}::{_BACKWARD_KERNEL_NAME}", mutates_args={"weight_grad"})
def embedding_backward_triton(
    input_ids: torch.Tensor,
    output_grad: torch.Tensor,
    weight_grad: torch.Tensor,
    BLOCK_SIZE_B: int,
    BLOCK_SIZE_H: int,
) -> None:
    num_elements = input_ids.numel()
    hidden_size = weight_grad.size(-1)

    accumulate_in_fp32 = weight_grad.dtype == torch.bfloat16
    if accumulate_in_fp32:
        weight_grad = weight_grad.float()

    with torch.device(input_ids.device):
        _embedding_backward_triton_kernel[
            (ceil_divide(num_elements, BLOCK_SIZE_B), ceil_divide(hidden_size, BLOCK_SIZE_H))
        ](
            x_ptr=input_ids,
            output_grad_ptr=output_grad,
            weight_grad_ptr=weight_grad,
            B=num_elements,
            H=hidden_size,
            accumulate_in_fp32=accumulate_in_fp32,
            BLOCK_SIZE_B=BLOCK_SIZE_B,
            BLOCK_SIZE_H=BLOCK_SIZE_H,
        )
