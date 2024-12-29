from ....constants import COMMON_TRITON_BLOCK_SIZES_POWERS_OF_2, MAX_TRITON_BLOCK_SIZE
from ....cutotune import CutoTuneConfig
from ....math import get_powers_of_2


def get_cutotune_parameters() -> dict:
    return dict(
        configs=[
            CutoTuneConfig(
                {"BLOCK_SIZE_B": BLOCK_SIZE_B},
                condition=lambda **kwargs: 1024
                <= kwargs["BLOCK_SIZE_B"] * kwargs["BLOCK_SIZE_H"]
                <= MAX_TRITON_BLOCK_SIZE,
            )
            for BLOCK_SIZE_B in get_powers_of_2(1, 32) + COMMON_TRITON_BLOCK_SIZES_POWERS_OF_2
        ],
        default_config=CutoTuneConfig({"BLOCK_SIZE_B": 1}),
        triggers={"x.dtype", "BLOCK_SIZE_H"},
        functional_triggers={"has_weight": lambda **kwargs: kwargs["weight"] is not None},
    )
