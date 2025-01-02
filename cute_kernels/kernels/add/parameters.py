import torch

from ...constants import (
    COMMON_CUDA_BLOCK_SIZES_POWERS_OF_2,
    COMMON_TRITON_BLOCK_SIZES_POWERS_OF_2,
    COMMON_VECTOR_INSTRUCTION_WIDTHS,
    MAX_CUDA_BLOCK_SIZE,
    MAX_FP16_BF16_INSTRUCTION_WIDTH,
)
from ...cutotune import CutoTuneConfig, get_cartesian_product_cutotune_configs
from ...enums import KernelBackend


def get_cutotune_parameters() -> dict:
    return dict(
        configs=get_cartesian_product_cutotune_configs(
            kernel_backend=[KernelBackend.cuda],
            vector_instruction_width=COMMON_VECTOR_INSTRUCTION_WIDTHS,
            BLOCK_SIZE=COMMON_CUDA_BLOCK_SIZES_POWERS_OF_2,
        )
        + get_cartesian_product_cutotune_configs(
            kernel_backend=[KernelBackend.cuda],
            vector_instruction_width=[MAX_FP16_BF16_INSTRUCTION_WIDTH],
            BLOCK_SIZE=COMMON_CUDA_BLOCK_SIZES_POWERS_OF_2,
            condition=lambda **kwargs: kwargs["x"].dtype in [torch.float16, torch.bfloat16],
        )
        + get_cartesian_product_cutotune_configs(
            kernel_backend=[KernelBackend.triton],
            vector_instruction_width=[None],
            BLOCK_SIZE=COMMON_TRITON_BLOCK_SIZES_POWERS_OF_2,
        ),
        default_config=CutoTuneConfig(
            dict(kernel_backend=KernelBackend.triton, vector_instruction_width=None, BLOCK_SIZE=MAX_CUDA_BLOCK_SIZE)
        ),
        triggers={"x.dtype"},
    )
