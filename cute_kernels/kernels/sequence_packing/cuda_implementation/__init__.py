import torch

from ....constants import LIBRARY_NAME
from ....jit import cpp_jit
from ....utils import cute_op


@cute_op(f"{LIBRARY_NAME}::pack_sequence_cuda", mutates_args={"output"})
@cpp_jit()
def pack_sequence_cuda(
    x: torch.Tensor,
    output: torch.Tensor,
    seqlens: torch.Tensor,
    max_seqlen_tensor: torch.Tensor | None,
    max_seqlen: int | None,
    padding_side: str,
    BLOCK_SIZE: int,
) -> None: ...
