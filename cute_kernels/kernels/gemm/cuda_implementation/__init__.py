import torch

from ....constants import LIBRARY_NAME
from ....jit import cpp_jit
from ....utils import cute_op


_KERNEL_NAME = "naive_gemm_cuda"


@cute_op(f"{LIBRARY_NAME}::{_KERNEL_NAME}", mutates_args={"c"})
@cpp_jit(_KERNEL_NAME)
def naive_gemm_cuda(
    a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, BLOCK_SIZE_M: int, BLOCK_SIZE_N: int
) -> None: ...
