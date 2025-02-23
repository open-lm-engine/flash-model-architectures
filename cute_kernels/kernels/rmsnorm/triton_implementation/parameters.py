from ....constants import MAX_TRITON_BLOCK_SIZE
from ....cutotune import CutoTuneConfig, get_cartesian_product_cutotune_configs
from ....math import get_powers_of_2


def get_cutotune_parameters() -> dict:
    return dict(
        configs=get_cartesian_product_cutotune_configs(
            BLOCK_SIZE_B=get_powers_of_2(1, MAX_TRITON_BLOCK_SIZE),
            condition=lambda **kwargs: 1024
            <= kwargs["BLOCK_SIZE_B"] * kwargs["BLOCK_SIZE_H"]
            <= MAX_TRITON_BLOCK_SIZE,
        ),
        default_config=CutoTuneConfig({"BLOCK_SIZE_B": 1}),
        triggers={"x.dtype", "BLOCK_SIZE_H"},
        functional_triggers={"has_weight": lambda **kwargs: kwargs["weight"] is not None},
    )
