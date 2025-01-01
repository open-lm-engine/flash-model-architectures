import torch

from .....constants import LIBRARY_NAME
from .....jit import cpp_jit
from .....utils import cute_op


_KERNEL_NAME = "add_tensor_cuda"


@cute_op(f"{LIBRARY_NAME}::{_KERNEL_NAME}", mutates_args={"output"})
@cpp_jit(_KERNEL_NAME)
def add_tensor_cuda(
    x: torch.Tensor, y: torch.Tensor, output: torch.Tensor, vector_instruction_width: int, BLOCK_SIZE: int
) -> None: ...
