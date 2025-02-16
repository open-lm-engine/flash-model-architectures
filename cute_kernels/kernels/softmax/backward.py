import torch

from ...constants import MAX_TRITON_BLOCK_SIZE
from ...cutotune import CutoTuneConfig, cutotune, get_cartesian_product_cutotune_configs
from ...math import get_next_power_of_2
from ...utils import get_num_elements_and_hidden_size
from .triton_implementation import softmax_backward_full_row_triton


@cutotune(
    configs=get_cartesian_product_cutotune_configs(
        kernel_backend=["triton"], triton_kernel_algorithm=["full_row_softmax"]
    ),
    default_config=CutoTuneConfig(dict(kernel_backend="triton", triton_kernel_algorithm="full_row_softmax")),
    triggers={"output.dtype"},
    functional_triggers={
        "next_power_of_2(hidden_size)": lambda **kwargs: get_next_power_of_2(kwargs["output"].size(-1))
    },
)
def _backward(
    output: torch.Tensor,
    output_grad: torch.Tensor,
    kernel_backend: str,
    triton_kernel_algorithm: str,
    BLOCK_SIZE_B: int,
    BLOCK_SIZE_H: int,
) -> torch.Tensor:
    x_grad = torch.empty_like(output)
    _, hidden_size = get_num_elements_and_hidden_size(x_grad)

    if kernel_backend == "triton":
        if triton_kernel_algorithm == "full_row_softmax":
            BLOCK_SIZE_H = get_next_power_of_2(hidden_size)
            assert BLOCK_SIZE_H <= MAX_TRITON_BLOCK_SIZE

            softmax_backward_full_row_triton(
                output=output,
                output_grad=output_grad,
                x_grad=x_grad,
                BLOCK_SIZE_B=BLOCK_SIZE_B,
                BLOCK_SIZE_H=BLOCK_SIZE_H,
            )
        elif triton_kernel_algorithm == "online_softmax":
            pass
        else:
            raise ValueError(f"unexpected triton_kernel_algorithm ({triton_kernel_algorithm})")
    else:
        raise ValueError(f"unexpected kernel_backend ({kernel_backend})")

    return x_grad
