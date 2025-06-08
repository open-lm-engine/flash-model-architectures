# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import torch

from cute_kernels import grouped_gemm_cute


E = 16
K = 4096
M = 4096
N = 512


def get_tensors(is_A_transposed, is_B_transposed, M_array, N_array, K_array):
    As = []
    Bs = []

    for M, N, K in zip(M_array, N_array, K_array):
        A = torch.randint(
            -8, 9, (K, M) if is_A_transposed else (M, K), device=torch.cuda.current_device(), dtype=torch.bfloat16
        )
        As.append(A)

        B = torch.randint(
            -8, 9, (N, K) if is_B_transposed else (K, N), device=torch.cuda.current_device(), dtype=torch.bfloat16
        )
        Bs.append(B)

    return As, Bs


M_array = torch.tensor(
    [512, 3172, 2048, 512, 3172, 2048, 512, 3172, 2048, 512, 3172, 2048, 512, 3172, 2048, 36876],
    device=torch.cuda.current_device(),
    dtype=torch.uint32,
)
N_array = torch.full_like(M_array, fill_value=N)
K_array = torch.full_like(M_array, fill_value=K)


for is_A_transposed in [False, True]:
    for is_B_transposed in [False, True]:
        print(is_A_transposed, is_B_transposed)
        As, Bs = get_tensors(is_A_transposed, is_B_transposed, M_array, N_array, K_array)

        A = torch.cat([i.view(-1) for i in As])
        B = torch.cat([i.view(-1) for i in Bs])

        for i in range(10):
            output = grouped_gemm_cute(
                A=A,
                B=B,
                C=None,
                M_array=M_array,
                N_array=N_array,
                K_array=K_array,
                output_shape=(E * M * N,),
                is_A_transposed=is_A_transposed,
                is_B_transposed=is_B_transposed,
            )

        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)

        s.record()
        for i in range(10):
            output = grouped_gemm_cute(
                A=A,
                B=B,
                C=None,
                M_array=M_array,
                N_array=N_array,
                K_array=K_array,
                output_shape=(E * M * N,),
                is_A_transposed=is_A_transposed,
                is_B_transposed=is_B_transposed,
            )
        e.record()

        torch.cuda.synchronize()

        t = s.elapsed_time(e) / 10 / 1e3
        print(2 * M * N * K * E / t / 1e12)

        D = []
        for A, B in zip(As, Bs):
            if is_A_transposed:
                A = A.T
            if is_B_transposed:
                B = B.T

            D.append(A @ B)

        D = torch.cat([i.view(-1) for i in D])

        print((output - D).abs().max())
        print()
