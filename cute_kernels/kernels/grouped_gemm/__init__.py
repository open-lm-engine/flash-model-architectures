# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import torch

from .cuda_implementation import grouped_gemm_cuda


def grouped_gemm_cute(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor | None,
    alpha: float = 1,
    beta: float = 0,
    is_A_transposed: bool = False,
    is_B_transposed: bool = False,
) -> torch.Tensor:
    assert beta == 0
    assert C is None

    E = A.size(0)
    M = A.size(2 if is_A_transposed else 1)
    K = A.size(1 if is_A_transposed else 2)
    N = B.size(1 if is_B_transposed else 2)

    M_array = torch.tensor([M] * E, device=torch.cuda.current_device(), dtype=torch.uint32)
    N_array = torch.full_like(M_array, fill_value=N)
    K_array = torch.full_like(M_array, fill_value=K)

    output = torch.empty(E, M, N, device=torch.cuda.current_device(), dtype=torch.bfloat16)

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
    )

    # print(A)
    # print(B)

    D = []
    for i in range(E):
        a = A[i]
        if is_A_transposed:
            a = a.T
        b = B[i]
        if is_B_transposed:
            b = b.T

        D.append(alpha * a @ b + beta)

    D = torch.stack(D)

    print((output - D).abs().max())

    return output
