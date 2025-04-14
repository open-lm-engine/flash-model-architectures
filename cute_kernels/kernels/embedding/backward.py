import torch

from ...math import ceil_divide
from .triton_implementation import _embedding_backward_triton_kernel


def _backward(
    input_ids: torch.Tensor,
    weight: torch.Tensor,
    output_grad: torch.Tensor,
    kernel_backend: str,
    BLOCK_SIZE_B: int,
    BLOCK_SIZE_H: int,
) -> torch.Tensor:
    weight_grad = torch.zeros_like(weight)

    if kernel_backend == "triton":
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
    else:
        raise ValueError(f"unexpected kernel_backend ({kernel_backend})")

    return weight_grad
