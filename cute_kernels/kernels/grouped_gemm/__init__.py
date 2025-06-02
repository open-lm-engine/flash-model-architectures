# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import torch

from .cuda_implementation import grouped_gemm_cuda


def grouped_gemm_cute(alpha: float = 1, beta: float = 0) -> torch.Tensor:
    E = 16
    K = 4096
    M = 4096
    N = 512
    TK = 65536
    assert TK % E == 0

    M_offsets = torch.tensor(list(range(0, 48000, 3000)) + [TK], device=torch.cuda.current_device(), dtype=torch.int32)

    M_array = (M_offsets[1:] - M_offsets[:-1]).to(torch.uint32)
    N_array = torch.full_like(M_array, fill_value=N)
    K_array = torch.full_like(M_array, fill_value=K)

    A = torch.randint(-8, 9, (TK, K), device=torch.cuda.current_device(), dtype=torch.bfloat16)
    B = torch.randn(E, N, K, device=torch.cuda.current_device(), dtype=torch.bfloat16)
    output = torch.randn(E, M, N, device=torch.cuda.current_device(), dtype=torch.bfloat16)

    grouped_gemm_cuda(
        A=A,
        B=B,
        output=output,
        M_array=M_array,
        N_array=N_array,
        K_array=K_array,
        is_A_transposed=False,
        is_B_transposed=True,
        alpha=alpha,
        beta=beta,
    )

    return output
