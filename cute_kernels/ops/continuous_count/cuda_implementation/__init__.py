# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import torch
from torch.library import custom_op

from ....constants import LIBRARY_NAME
from ....jit import cpp_jit


@custom_op(f"{LIBRARY_NAME}::continuous_count_cuda", mutates_args={"output"})
@cpp_jit()
def continuous_count_cuda(
    x: torch.Tensor, output: torch.Tensor, E: int, THREAD_BLOCK_CLUSTER_SIZE: int, BLOCK_SIZE: int
) -> None: ...
