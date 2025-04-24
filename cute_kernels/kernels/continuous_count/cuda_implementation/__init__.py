import torch

from ....constants import LIBRARY_NAME
from ....jit import cpp_jit
from ....utils import cute_op


@cute_op(f"{LIBRARY_NAME}::continuous_count_cuda", mutates_args={"output"})
@cpp_jit()
def continuous_count_cuda(
    x: torch.Tensor, output: torch.Tensor, C: int, THREAD_BLOCK_CLUSTER_SIZE: int, BLOCK_SIZE: int
) -> None: ...
