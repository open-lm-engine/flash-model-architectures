import torch

from ....constants import LIBRARY_NAME
from ....jit import cpp_jit
from ....utils import cute_op


_KERNEL_NAME = "continuous_count_and_sort_cuda"


@cute_op(f"{LIBRARY_NAME}::{_KERNEL_NAME}", mutates_args={"output"})
@cpp_jit(_KERNEL_NAME)
def continuous_count_and_sort_cuda(
    x: torch.Tensor, count: torch.Tensor, sm_count: int, size: int, BLOCK_SIZE: int
) -> None: ...
