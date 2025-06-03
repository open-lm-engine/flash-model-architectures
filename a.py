# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import torch

from cute_kernels.kernels.grouped_gemm import grouped_gemm_cute


E = 16
K = 4096
M = 4096
N = 512

is_A_transposed = False
is_B_transposed = False

A = torch.randint(
    -8, 9, (E, K, M) if is_A_transposed else (E, M, K), device=torch.cuda.current_device(), dtype=torch.bfloat16
)
B = torch.randint(
    -8, 9, (E, N, K) if is_B_transposed else (E, K, N), device=torch.cuda.current_device(), dtype=torch.bfloat16
)

M_array = torch.tensor([M] * E, device=torch.cuda.current_device(), dtype=torch.uint32)
N_array = torch.full_like(M_array, fill_value=N)
K_array = torch.full_like(M_array, fill_value=K)

ptr_A = torch.empty(E, device=A.device, dtype=torch.uint64)
ptr_B = torch.empty(E, device=A.device, dtype=torch.uint64)
output = torch.empty(E, M, N, device=A.device, dtype=torch.bfloat16)

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
        ptr_A=ptr_A,
        ptr_B=ptr_B,
        output=output,
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
        ptr_A=ptr_A,
        ptr_B=ptr_B,
        output=output,
        is_A_transposed=is_A_transposed,
        is_B_transposed=is_B_transposed,
    )
    # torch_profiler.step()
e.record()

torch.cuda.synchronize()

t = s.elapsed_time(e) / 10 / 1e3
print(2 * M * N * K * E / t / 1e12)

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
