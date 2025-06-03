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

    E = 7
    K = 16
    M = 8
    N = 24

    M_array = torch.tensor([M] * E, device=torch.cuda.current_device(), dtype=torch.uint32)
    N_array = torch.full_like(M_array, fill_value=N)
    K_array = torch.full_like(M_array, fill_value=K)

    is_A_transposed = False
    is_B_transposed = False

    A = torch.randint(
        -8, 9, (E, K, M) if is_A_transposed else (E, M, K), device=torch.cuda.current_device(), dtype=torch.bfloat16
    )
    B = torch.randint(
        -8, 9, (E, N, K) if is_B_transposed else (E, K, N), device=torch.cuda.current_device(), dtype=torch.bfloat16
    )

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
