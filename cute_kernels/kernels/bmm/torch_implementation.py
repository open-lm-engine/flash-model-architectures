# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import torch


def bmm_torch(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    alpha: float = 1,
    beta: float = 1,
    is_A_transposed: bool = False,
    is_B_transposed: bool = False,
) -> torch.Tensor:
    if is_A_transposed:
        A = A.T

    if is_B_transposed:
        B = B.T

    if beta == 0:
        assert C is None

        output = torch.bmm(A, B)
        if alpha != 1:
            output *= alpha
    else:
        assert C is not None
        output = torch.baddbmm(C, A, B, alpha=alpha, beta=beta)

    return output
