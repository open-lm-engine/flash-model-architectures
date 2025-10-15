# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import torch
from torch.library import custom_op

from ....constants import LIBRARY_NAME
from ....jit import cpp_jit


_KERNEL_NAME = "grouped_gemm_cuda"


@custom_op(f"{LIBRARY_NAME}::{_KERNEL_NAME}", mutates_args={"output"})
@cpp_jit()
def grouped_gemm_cuda(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor | None,
    output: torch.Tensor,
    M_array: torch.Tensor,
    N_array: torch.Tensor,
    K_array: torch.Tensor,
    is_A_transposed: bool,
    is_B_transposed: bool,
    alpha: float,
    beta: float,
    benchmark: bool,
) -> None: ...
