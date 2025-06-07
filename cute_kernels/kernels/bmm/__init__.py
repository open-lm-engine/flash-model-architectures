# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import torch

from ...cutotune import CutoTuneParameter
from ...kernel_backend import KernelBackend
from ...utils import ensure_contiguous
from .triton_implementation import bmm_triton


@ensure_contiguous
def bmm_cute(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor | None,
    is_A_transposed: bool = False,
    is_B_transposed: bool = False,
    alpha: float = 1,
    beta: float = 1,
    *,
    kernel_backend: KernelBackend | CutoTuneParameter = KernelBackend.triton,
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
        kernel_backend (KernelBackend | CutoTuneParameter, optional): kernel backend to prioritize.
            Defaults to KernelBackend.triton.

    Raises:
        ValueError: if unexpected `kernel_backend` is passed

    Returns:
        torch.Tensor: output tensor
    """

    assert A.dim() == 3
    assert B.dim() == 3

    L, M, K = A.size()
    if is_A_transposed:
        M, K = K, M

    assert B.size(2 if is_B_transposed else 1) == K
    N = B.size(1 if is_B_transposed else 2)

    if beta == 0:
        assert C is None
    else:
        assert C is not None
        assert C.size() == (L, M, N)

    if kernel_backend == KernelBackend.torch:
        if is_A_transposed:
            A = A.transpose(1, 2)

        if is_B_transposed:
            B = B.transpose(1, 2)

        if beta == 0:
            output = torch.bmm(A, B)
            if alpha != 1:
                output = alpha * output
        else:
            output = torch.baddbmm(C, A, B, alpha=alpha, beta=beta)
    elif kernel_backend == KernelBackend.triton:
        output = torch.empty(L, M, N, dtype=A.dtype, device=A.device)

        bmm_triton(
            A=A,
            B=B,
            C=C,
            output=output,
            is_A_transposed=is_A_transposed,
            is_B_transposed=is_B_transposed,
            alpha=alpha,
            beta=beta,
        )
    else:
        raise ValueError(f"unexpected kernel_backend ({kernel_backend})")

    return output
