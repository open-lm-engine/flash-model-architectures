# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

from functools import partial

import torch
from tabulate import tabulate

from cute_kernels import KernelBackend, device_synchronize, gemm_cute


torch._inductor.config.max_autotune_gemm_backends = "TRITON"
torch.backends.cuda.matmul.allow_tf32 = True

n = 100

headers = ["dtype", "torch TFLOPs", "torch compile TFLOPs", "triton TFLOPs"]
kernels = [
    partial(gemm_cute, kernel_backend=KernelBackend.torch),
    partial(torch.compile(gemm_cute, mode="max-autotune"), kernel_backend=KernelBackend.torch),
    partial(gemm_cute, kernel_backend=KernelBackend.triton),
]

table = []

for dtype in [torch.float16, torch.bfloat16, torch.float32]:
    row = [str(dtype)]
    for kernel in kernels:
        x = torch.randn(4096, 4096, device=torch.cuda.current_device(), dtype=dtype)
        w = torch.randn(4096, 4096, device=torch.cuda.current_device(), dtype=dtype)

        for i in range(n):
            z = kernel(x, w, C=None, beta=0)

        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)

        s.record()
        for i in range(n):
            z = kernel(x, w, C=None, beta=0)
        e.record()

        device_synchronize()

        t = s.elapsed_time(e) / n / 1e3
        row.append(2 * x.size(0) * x.size(1) * w.size(0) / t / 1e12)

    table.append(row)


print(tabulate(table, headers=headers))
