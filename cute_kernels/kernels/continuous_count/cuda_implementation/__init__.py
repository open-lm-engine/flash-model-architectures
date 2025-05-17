import torch

from ....constants import LIBRARY_NAME
from ....cutotune import CutoTuneConfig, cutotune
from ....jit import cpp_jit
from ....utils import cute_op


@cutotune(
    configs=[
        CutoTuneConfig({"BLOCK_SIZE": 1024, "THREAD_BLOCK_CLUSTER_SIZE": thread_block_cluster_size})
        for thread_block_cluster_size in [1, 2, 4, 8]
    ],
)
@cute_op(f"{LIBRARY_NAME}::continuous_count_cuda", mutates_args={"output"})
@cpp_jit()
def continuous_count_cuda(
    x: torch.Tensor, output: torch.Tensor, C: int, THREAD_BLOCK_CLUSTER_SIZE: int, BLOCK_SIZE: int
) -> None: ...
