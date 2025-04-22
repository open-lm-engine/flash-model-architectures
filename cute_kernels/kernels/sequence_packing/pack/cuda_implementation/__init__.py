import torch

from .....constants import LIBRARY_NAME
from .....jit import cpp_jit
from .....utils import cute_op


_KERNEL_NAME = "pack_sequence_cuda"


@cute_op(f"{LIBRARY_NAME}::{_KERNEL_NAME}", mutates_args={"output"})
@cpp_jit(_KERNEL_NAME)
def pack_sequence_cuda(
    x: torch.Tensor,
    output: torch.Tensor,
    cu_seqlens: torch.Tensor,
    max_seqlen_tensor: torch.Tensor,
    max_seqlen: int,
    BLOCK_SIZE: int,
) -> None: ...
