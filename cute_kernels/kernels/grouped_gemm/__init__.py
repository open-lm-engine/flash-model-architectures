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

    expert_offsets = torch.tensor(
        list(range(0, 48000, 3000)) + [TK], device=torch.cuda.current_device(), dtype=torch.uint32
    )

    A = torch.randn(TK, K, device=torch.cuda.current_device(), dtype=torch.bfloat16)
    B = torch.randn(E, N, K, device=torch.cuda.current_device(), dtype=torch.bfloat16)
    output = torch.randn(E, M, N, device=torch.cuda.current_device(), dtype=torch.bfloat16)

    grouped_gemm_cuda(
        A=A,
        B=B,
        output=output,
        M_offsets=expert_offsets,
        M=M,
        K_offsets=None,
        K=K,
        N_offsets=None,
        N=N,
        is_A_transposed=False,
        is_B_transposed=True,
        alpha=alpha,
        beta=beta,
    )

    return output
