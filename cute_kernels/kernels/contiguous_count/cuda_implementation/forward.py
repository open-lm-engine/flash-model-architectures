import torch

from ....constants import COMMON_CUDA_BLOCK_SIZES_POWERS_OF_2, LIBRARY_NAME
from ....cutotune import CutoTuneConfig, cutotune, get_cartesian_product_cutotune_configs
from ....jit import cpp_jit
from ....utils import cute_op


_KERNEL_NAME = "contiguous_count_cuda"


@cutotune(
    get_cartesian_product_cutotune_configs(
        BLOCK_SIZE=COMMON_CUDA_BLOCK_SIZES_POWERS_OF_2, thread_block_cluster_size=[1, 2, 4, 8]
    ),
    default_config=CutoTuneConfig(dict(BLOCK_SIZE=1024)),
)
@cute_op(f"{LIBRARY_NAME}::{_KERNEL_NAME}", mutates_args={"output"})
@cpp_jit(_KERNEL_NAME)
def contiguous_count_cuda(
    x: torch.Tensor, output: torch.Tensor, sm_count: int, thread_block_cluster_size: int, size: int, BLOCK_SIZE: int
) -> None: ...
