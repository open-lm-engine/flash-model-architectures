# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import torch


def gemm_torch(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    alpha: float = 1,
    beta: float = 1,
    is_A_transposed: bool = False,
    is_B_transposed: bool = False,
) -> torch.Tensor:
    assert B.size(1 if is_B_transposed else 0) == A.size(0 if is_A_transposed else 1)

    if is_A_transposed:
        A = A.T

    if is_B_transposed:
        B = B.T

    if beta == 0:
        assert C is None

        output = A @ B
        if alpha != 1:
            output *= alpha
    else:
        assert C is not None
        output = torch.addmm(C, A, B, alpha=alpha, beta=beta)

    return output
