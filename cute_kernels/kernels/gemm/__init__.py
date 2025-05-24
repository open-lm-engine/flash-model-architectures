# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import torch

from ...utils import ensure_contiguous
from ..bmm.triton_implementation import bmm_triton
from .cuda_implementation import (
    cutlass_gemm_cuda,
    cutlass_tensorcore_mma_gemm_cuda,
    naive_gemm_cuda,
    shared_memory_gemm_cuda,
)
from .torch_implementation import gemm_torch


@ensure_contiguous
def gemm_cute(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor | None,
    is_A_transposed: bool = False,
    is_B_transposed: bool = False,
    alpha: float = 1,
    beta: float = 1,
    *,
    kernel_backend: str = "triton",
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
        kernel_backend (str, optional): kernel backend to use. Defaults to triton.

    Raises:
        ValueError: if unexpected `kernel_backend` is passed

    Returns:
        torch.Tensor: output tensor
    """

    assert A.dim() == 2
    assert B.dim() == 2

    M, K = A.size()
    if is_A_transposed:
        M, K = K, M

    assert B.size(1 if is_B_transposed else 0) == K
    N = B.size(0 if is_B_transposed else 1)

    output = torch.empty(M, N, dtype=A.dtype, device=A.device)

    if beta == 0:
        assert C is None
    else:
        assert C is not None
        assert C.size() == (M, N)

    if kernel_backend == "cutlass_tensorcore_mma_gemm_cuda":
        cutlass_tensorcore_mma_gemm_cuda(
            A=A,
            B=B,
            C=C,
            output=output,
            is_A_transposed=is_A_transposed,
            is_B_transposed=is_B_transposed,
            alpha=alpha,
            beta=beta,
        )
    elif kernel_backend == "cutlass":
        cutlass_gemm_cuda(
            A=A,
            B=B,
            C=C,
            output=output,
            is_A_transposed=is_A_transposed,
            is_B_transposed=is_B_transposed,
            alpha=alpha,
            beta=beta,
        )
    elif kernel_backend == "shared_memory_cuda":
        BLOCK_SIZE = 32

        shared_memory_gemm_cuda(
            A=A,
            B=B,
            C=C,
            output=output,
            is_A_transposed=is_A_transposed,
            is_B_transposed=is_B_transposed,
            alpha=alpha,
            beta=beta,
            BLOCK_SIZE=BLOCK_SIZE,
        )
    elif kernel_backend == "naive_cuda":
        BLOCK_SIZE_M = 16
        BLOCK_SIZE_N = 16

        naive_gemm_cuda(
            A=A,
            B=B,
            C=C,
            output=output,
            is_A_transposed=is_A_transposed,
            is_B_transposed=is_B_transposed,
            alpha=alpha,
            beta=beta,
            BLOCK_SIZE_M=BLOCK_SIZE_M,
            BLOCK_SIZE_N=BLOCK_SIZE_N,
        )
    elif kernel_backend == "triton":
        bmm_triton(
            A=A.unsqueeze(0),
            B=B.unsqueeze(0),
            C=None if C is None else C.unsqueeze(0),
            output=output.unsqueeze(0),
            is_A_transposed=is_A_transposed,
            is_B_transposed=is_B_transposed,
            alpha=alpha,
            beta=beta,
        )
    else:
        raise ValueError(f"unexpected kernel_backend ({kernel_backend})")

    return output
