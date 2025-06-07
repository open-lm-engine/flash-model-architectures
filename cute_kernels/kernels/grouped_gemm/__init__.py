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
    assert A.dim() == 2
    assert B.dim() == 2
    assert beta == 0
    assert C is None

    if C is not None:
        assert C.dim() == 2

    E = M_array.size(0)
    assert N_array.size(0) == E
    assert K_array.size(0) == E

    if is_A_transposed:
        if is_B_transposed:
            # A -> K x sum(M)
            # B -> (E x N) x K
            # C -> sum(M) x N
            output = torch.empty(A.size(1), B.size(0) // E, device=A.device, dtype=A.dtype)
        else:
            # A -> K x sum(M)
            # B -> (E x K) x N
            # C -> sum(M) x N
            output = torch.empty(A.size(1), B.size(1), device=A.device, dtype=A.dtype)
    else:
        if is_B_transposed:
            # A -> sum(M) x K
            # B -> (E x N) x K
            # C -> sum(M) x N
            output = torch.empty(A.size(0), B.size(0) // E, device=A.device, dtype=A.dtype)
        else:
            # A -> sum(M) x K
            # B -> (E x K) x N
            # C -> sum(M) x N
            output = torch.empty(A.size(0), B.size(1), device=A.device, dtype=A.dtype)

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
