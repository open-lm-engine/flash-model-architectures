from ...cutotune import CutoTuneConfig, get_cartesian_product_cutotune_configs
from ...math import get_powers_of_2


def get_cutotune_parameters() -> dict:
    return dict(
        configs=get_cartesian_product_cutotune_configs(
            kernel_backend=["triton"],
            BLOCK_SIZE_B=get_powers_of_2(64, 1024),
            BLOCK_SIZE_H=[64],
        ),
        default_config=CutoTuneConfig(dict(kernel_backend="triton", BLOCK_SIZE_B=64, BLOCK_SIZE_H=64)),
        triggers={"x.dtype"},
    )
