from ...constants import COMMON_CUDA_BLOCK_SIZES_POWERS_OF_2, COMMON_TRITON_BLOCK_SIZES_POWERS_OF_2
from ...cutotune import CutoTuneConfig, get_cartesian_product_cutotune_configs
from ...enums import KernelBackend


def get_cutotune_parameters() -> dict:
    return dict(
        configs=get_cartesian_product_cutotune_configs(
            kernel_backend=[KernelBackend.cuda], BLOCK_SIZE=COMMON_CUDA_BLOCK_SIZES_POWERS_OF_2
        )
        + get_cartesian_product_cutotune_configs(
            kernel_backend=[KernelBackend.triton], BLOCK_SIZE=COMMON_TRITON_BLOCK_SIZES_POWERS_OF_2
        ),
        default_config=CutoTuneConfig(dict(kernel_backend=KernelBackend.triton, BLOCK_SIZE=1024)),
        triggers={"gate.dtype"},
    )
