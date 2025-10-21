# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import torch

from .bmm import bmm


def gemm(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor | None,
    is_A_transposed: bool = False,
    is_B_transposed: bool = False,
    alpha: float = 1,
    beta: float = 1,
) -> torch.Tensor:
    """computes `alpha` * (`A` @ `B`) + `beta` * `C`

    Args:
        A (torch.Tensor): `A` matrix
        B (torch.Tensor): `B` matrix
        C (torch.Tensor | None): `C` matrix, function returns `A` @ `B` if C is None
        is_A_transposed (bool, optional): whether A has shape K x M. Defaults to False.
        is_B_transposed (bool, optional): whether B has shape N x K. Defaults to False.
        alpha (float, optional): alpha. Defaults to 1.
        beta (float, optional): beta. Defaults to 1.

    Raises:
        ValueError: if unexpected `kernel_backend` is passed

    Returns:
        torch.Tensor: output tensor
    """

    assert A.dim() == 2
    assert B.dim() == 2

    D = bmm(
        A=A[None, ...],
        B=B[None, ...],
        C=None if C is None else C[None, ...],
        is_A_transposed=is_A_transposed,
        is_B_transposed=is_B_transposed,
        alpha=alpha,
        beta=beta,
    )

    D = D.squeeze(0)

    return D
