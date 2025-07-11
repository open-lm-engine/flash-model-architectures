# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import torch
from torch.library import custom_op

from ....constants import LIBRARY_NAME
from ....jit import cpp_jit


@custom_op(f"{LIBRARY_NAME}::pack_unpack_sequence_cuda", mutates_args={"output"})
@cpp_jit()
def pack_unpack_sequence_cuda(
    x: torch.Tensor,
    output: torch.Tensor,
    cu_seqlens: torch.Tensor,
    padding_side: str,
    pack: bool,
    BLOCK_SIZE: int,
) -> None: ...
