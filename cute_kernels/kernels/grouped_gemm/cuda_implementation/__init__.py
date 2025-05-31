# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import torch

from ....constants import LIBRARY_NAME
from ....jit import cpp_jit
from ....utils import cute_op


_KERNEL_NAME = "grouped_gemm_cuda"


@cute_op(f"{LIBRARY_NAME}::{_KERNEL_NAME}", mutates_args={})
@cpp_jit()
def grouped_gemm_cuda(A: torch.Tensor, B: torch.Tensor, output: torch.Tensor, alpha: float, beta: float) -> None: ...
