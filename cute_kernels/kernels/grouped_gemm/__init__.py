# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import torch

from ...utils import ensure_contiguous
from .cuda_implementation import grouped_gemm_cuda


@ensure_contiguous
def grouped_gemm_cute(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor | None,
    M_array: torch.Tensor,
    N_array: torch.Tensor,
    K_array: torch.Tensor,
    alpha: float = 1,
    beta: float = 0,
    is_A_transposed: bool = False,
    is_B_transposed: bool = False,
    benchmark: bool = False,
) -> torch.Tensor:
    assert beta == 0
    assert C is None

    E = A.size(0)
    M = A.size(2 if is_A_transposed else 1)
    N = B.size(1 if is_B_transposed else 2)

    output = torch.empty(E, M, N, device=A.device, dtype=torch.bfloat16)

    grouped_gemm_cuda(
        A=A,
        B=B,
        C=C,
        output=output,
        M_array=M_array,
        N_array=N_array,
        K_array=K_array,
        is_A_transposed=is_A_transposed,
        is_B_transposed=is_B_transposed,
        alpha=alpha,
        beta=beta,
        benchmark=benchmark,
    )

    return output
