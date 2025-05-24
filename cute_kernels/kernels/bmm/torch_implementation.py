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
    assert B.size(2 if is_B_transposed else 1) == A.size(1 if is_A_transposed else 2)

    if is_A_transposed:
        A = A.transpose(1, 2)

    if is_B_transposed:
        B = B.transpose(1, 2)

    if beta == 0:
        assert C is None

        output = torch.bmm(A, B)
        if alpha != 1:
            output *= alpha
    else:
        assert C is not None
        output = torch.baddbmm(C, A, B, alpha=alpha, beta=beta)

    return output
