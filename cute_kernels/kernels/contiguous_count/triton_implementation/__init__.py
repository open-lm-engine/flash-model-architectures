import torch

from ....cutotune import CutoTuneConfig, cutotune, get_cartesian_product_cutotune_configs
from ....math import get_next_power_of_2
from .forward import contiguous_count_high_atomic_add_triton, contiguous_count_low_atomic_add_triton


@cutotune(
    configs=get_cartesian_product_cutotune_configs(
        use_low_atomic_add=[False, True], condition=lambda **kwargs: kwargs["size"] <= 16384
    )
    + get_cartesian_product_cutotune_configs(use_low_atomic_add=[False]),
    default_config=CutoTuneConfig(dict(use_low_atomic_add=False)),
)
def contiguous_count_triton(
    x: torch.Tensor, output: torch.Tensor, size: int, BLOCK_SIZE: int, use_low_atomic_add: bool
) -> None:
    if use_low_atomic_add:
        contiguous_count_low_atomic_add_triton(
            x=x, output=output, size=size, BLOCK_SIZE=BLOCK_SIZE, BLOCK_SIZE_C=get_next_power_of_2(size)
        )
    else:
        contiguous_count_high_atomic_add_triton(x=x, output=output, BLOCK_SIZE=BLOCK_SIZE)
