import torch

from ....constants import LIBRARY_NAME
from ....jit import cpp_jit
from ....utils import cute_op


_KERNEL_NAME = "contiguous_count_cuda"


@cute_op(f"{LIBRARY_NAME}::{_KERNEL_NAME}", mutates_args={"output"})
@cpp_jit(_KERNEL_NAME)
def contiguous_count_cuda(
    x: torch.Tensor, output: torch.Tensor, sm_count: int, thread_block_cluster_size: int, size: int, BLOCK_SIZE_B: int
) -> None: ...
