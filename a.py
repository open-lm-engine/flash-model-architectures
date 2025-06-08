# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import torch

from cute_kernels import grouped_gemm_cute


E = 16
K = 4096
M = 4096
N = 512


def get_tensors(is_A_transposed, is_B_transposed):
    As = []
    Bs = []

    for i in range(E):
        A = torch.randint(
            -8, 9, (K, M) if is_A_transposed else (M, K), device=torch.cuda.current_device(), dtype=torch.bfloat16
        )
        As.append(A)

        B = torch.randint(
            -8, 9, (N, K) if is_B_transposed else (K, N), device=torch.cuda.current_device(), dtype=torch.bfloat16
        )
        Bs.append(B)

    return As, Bs


for is_A_transposed in [False, True]:
    for is_B_transposed in [False, True]:
        As, Bs = get_tensors(is_A_transposed, is_B_transposed)

        A = torch.cat(As)
        B = torch.cat(Bs)

        M_array = torch.tensor([M] * E, device=torch.cuda.current_device(), dtype=torch.uint32)
        N_array = torch.full_like(M_array, fill_value=N)
        K_array = torch.full_like(M_array, fill_value=K)

        torch_profiler = torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            schedule=torch.profiler.schedule(wait=5, warmup=5, active=1, repeat=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler("tmp"),
            record_shapes=True,
        )

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
            # torch_profiler.step()

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
            # torch_profiler.step()
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

        D = torch.cat(D)
        output = output.view_as(D)

        print((output - D).abs().max())
        print()
