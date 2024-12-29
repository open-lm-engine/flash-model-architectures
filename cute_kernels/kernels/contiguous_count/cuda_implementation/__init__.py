import torch

from ....constants import COMMON_CUDA_BLOCK_SIZES_POWERS_OF_2, LIBRARY_NAME, MAX_CUDA_BLOCK_SIZE
from ....cutotune import CutoTuneConfig, cutotune, get_cartesian_product_cutotune_configs
from ....jit import cpp_jit
from ....math import get_next_power_of_2
from ....utils import cute_op


_KERNEL_NAME = "contiguous_count_cuda"


@cutotune(
    configs=get_cartesian_product_cutotune_configs(BLOCK_SIZE=COMMON_CUDA_BLOCK_SIZES_POWERS_OF_2),
    default_config=CutoTuneConfig({"BLOCK_SIZE": MAX_CUDA_BLOCK_SIZE}),
    functional_triggers={"next_power_of_2(size)": lambda **kwargs: get_next_power_of_2(kwargs["size"])},
)
@cute_op(f"{LIBRARY_NAME}::{_KERNEL_NAME}", mutates_args={"output"})
@cpp_jit(_KERNEL_NAME)
def contiguous_count_cuda(
    x: torch.Tensor, output: torch.Tensor, sm_count: int, thread_block_cluster_size: int, size: int, BLOCK_SIZE: int
) -> None: ...
