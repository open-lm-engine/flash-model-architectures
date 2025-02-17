import torch

from ...cutotune import CutoTuneConfig, cutotune
from ...utils import get_num_elements_and_hidden_size
from .triton_implementation import softmax_backward_full_row_triton


@cutotune(
    configs=[CutoTuneConfig(dict(kernel_backend="triton"))],
    default_config=CutoTuneConfig(dict(kernel_backend="triton")),
    triggers={"output.dtype"},
)
def _backward(
    output: torch.Tensor,
    output_grad: torch.Tensor,
    kernel_backend: str,
    BLOCK_SIZE_B: int,
    BLOCK_SIZE_H: int,
) -> torch.Tensor:
    x_grad = torch.empty_like(output)
    _, hidden_size = get_num_elements_and_hidden_size(x_grad)

    if kernel_backend == "triton":
        softmax_backward_full_row_triton(
            output=output,
            output_grad=output_grad,
            x_grad=x_grad,
            BLOCK_SIZE_B=BLOCK_SIZE_B,
            BLOCK_SIZE_H=BLOCK_SIZE_H,
        )
    else:
        raise ValueError(f"unexpected kernel_backend ({kernel_backend})")

    return x_grad
