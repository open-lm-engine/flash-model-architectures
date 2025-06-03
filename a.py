# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import torch

from cute_kernels.kernels.grouped_gemm import grouped_gemm_cute


E = 7
K = 16
M = 8
N = 24

is_A_transposed = False
is_B_transposed = False

A = torch.randint(
    -8, 9, (E, K, M) if is_A_transposed else (E, M, K), device=torch.cuda.current_device(), dtype=torch.bfloat16
)
B = torch.randint(
    -8, 9, (E, N, K) if is_B_transposed else (E, K, N), device=torch.cuda.current_device(), dtype=torch.bfloat16
)

output = grouped_gemm_cute(A=A, B=B, C=None, is_A_transposed=is_A_transposed, is_B_transposed=is_B_transposed)

D = []
for i in range(E):
    a = A[i]
    if is_A_transposed:
        a = a.T
    b = B[i]
    if is_B_transposed:
        b = b.T

    D.append(a @ b)

D = torch.stack(D)

print((output - D).abs().max())
