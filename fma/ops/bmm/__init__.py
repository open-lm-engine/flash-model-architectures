# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import torch

from ...cutotune import CutoTuneParameter
from ...enums import KernelBackend
from ...math import ceil_divide
from ...utils import ensure_contiguous
from .triton_implementation import bmm_triton_kernel


@ensure_contiguous
def bmm(
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
        torch.Tensor: D tensor
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
            D = torch.bmm(A, B)
            if alpha != 1:
                D = alpha * D
        else:
            D = torch.baddbmm(C, A, B, alpha=alpha, beta=beta)
    elif kernel_backend == KernelBackend.triton:
        D = torch.empty(L, M, N, dtype=A.dtype, device=A.device)

        L, M, K = A.size()
        if is_A_transposed:
            M, K = K, M

        N = B.size(1 if is_B_transposed else 2)

        GRID = lambda meta: (L, ceil_divide(M, meta["BLOCK_SIZE_M"]) * ceil_divide(N, meta["BLOCK_SIZE_N"]))

        bmm_triton_kernel[GRID](
            A_ptr=A,
            B_ptr=B,
            C_ptr=C,
            D_ptr=D,
            alpha=alpha,
            beta=beta,
            IS_A_TRANSPOSED=is_A_transposed,
            IS_B_TRANSPOSED=is_B_transposed,
            M=M,
            K=K,
            N=N,
        )
    else:
        raise ValueError(f"unexpected kernel_backend ({kernel_backend})")

    return D
