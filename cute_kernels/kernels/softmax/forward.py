import torch

from ...constants import MAX_TRITON_BLOCK_SIZE
from ...cutotune import CutoTuneConfig, cutotune, get_cartesian_product_cutotune_configs
from ...math import get_next_power_of_2
from ...utils import get_num_elements_and_hidden_size
from .triton_implementation import full_row_softmax_forward_triton, online_softmax_forward_triton


@cutotune(
    configs=get_cartesian_product_cutotune_configs(
        kernel_backend=["triton"], triton_kernel_algorithm=["full_row_softmax"]
    ),
    default_config=CutoTuneConfig(dict(kernel_backend="triton", triton_kernel_algorithm="full_row_softmax")),
    triggers={"x.dtype"},
    functional_triggers={"next_power_of_2(hidden_size)": lambda **kwargs: get_next_power_of_2(kwargs["x"].size(-1))},
)
def _forward(
    x: torch.Tensor,
    kernel_backend: str,
    triton_kernel_algorithm: str,
    BLOCK_SIZE_B: int,
    BLOCK_SIZE_H: int,
) -> torch.Tensor:
    output = torch.empty_like(x)
    _, hidden_size = get_num_elements_and_hidden_size(x)

    if kernel_backend == "triton":
        if triton_kernel_algorithm == "full_row_softmax":
            BLOCK_SIZE_H = get_next_power_of_2(hidden_size)
            assert BLOCK_SIZE_H <= MAX_TRITON_BLOCK_SIZE

            full_row_softmax_forward_triton(x=x, output=output, BLOCK_SIZE_B=BLOCK_SIZE_B, BLOCK_SIZE_H=BLOCK_SIZE_H)
        elif triton_kernel_algorithm == "online_softmax":
            online_softmax_forward_triton(x=x, output=output, BLOCK_SIZE_B=BLOCK_SIZE_B, BLOCK_SIZE_H=BLOCK_SIZE_H)
        else:
            raise ValueError(f"unexpected triton_kernel_algorithm ({triton_kernel_algorithm})")
    else:
        raise ValueError(f"unexpected kernel_backend ({kernel_backend})")

    return output
