# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import torch

from ....custom_op import xma_op
from ....jit import cpp_jit
from ....utils import ensure_contiguous


@ensure_contiguous
@xma_op(mutates_args={"output"})
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
