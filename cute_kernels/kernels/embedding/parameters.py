from ...constants import MAX_TRITON_BLOCK_SIZE
from ...cutotune import CutoTuneConfig, get_cartesian_product_cutotune_configs
from ...enums import KernelBackend
from ...math import get_powers_of_2


def get_cutotune_parameters() -> dict:
    return dict(
        configs=get_cartesian_product_cutotune_configs(
            kernel_backend=[KernelBackend.triton],
            BLOCK_SIZE_B=get_powers_of_2(128, MAX_TRITON_BLOCK_SIZE),
            BLOCK_SIZE_H=get_powers_of_2(128, MAX_TRITON_BLOCK_SIZE),
            condition=lambda **kwargs: 1024
            <= kwargs["BLOCK_SIZE_B"] * kwargs["BLOCK_SIZE_H"]
            <= MAX_TRITON_BLOCK_SIZE,
        ),
        default_config=CutoTuneConfig(
            {"kernel_backend": KernelBackend.triton, "BLOCK_SIZE_B": 128, "BLOCK_SIZE_H": 128}
        ),
        triggers={"weight.dtype"},
    )
