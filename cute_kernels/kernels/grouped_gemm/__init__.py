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

    A = torch.empty(TK, K, device=torch.cuda.current_device(), dtype=torch.bfloat16)
    B = torch.empty(E, N, K, device=torch.cuda.current_device(), dtype=torch.bfloat16)
    output = torch.empty(E, M, N, device=torch.cuda.current_device(), dtype=torch.bfloat16)

    return grouped_gemm_cuda(A=A, B=B, output=output, expert_offsets=expert_offsets, alpha=alpha, beta=beta)
