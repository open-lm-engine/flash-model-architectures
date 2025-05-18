import torch

from ....constants import LIBRARY_NAME
from ....jit import cpp_jit
from ....utils import cute_op


@cute_op(f"{LIBRARY_NAME}::add_scalar_cuda", mutates_args={"output"})
@cpp_jit()
def add_scalar_cuda(x: torch.Tensor, y: float, output: torch.Tensor, BLOCK_SIZE: int) -> None: ...
