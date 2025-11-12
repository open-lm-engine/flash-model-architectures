# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

from functools import partial

import torch
from tabulate import tabulate

from xma import KernelBackend, bmm, device_synchronize, swiglu


torch._inductor.config.max_autotune_gemm_backends = "TRITON"
torch.backends.cuda.matmul.allow_tf32 = True

n = 100

headers = ["dtype", "torch BW", "torch compile BW", "CUDA BW", "triton BW"]
kernels = [
    partial(swiglu, kernel_backend=KernelBackend.torch),
    partial(torch.compile(swiglu, dynamic=True), kernel_backend=KernelBackend.torch),
    partial(swiglu, kernel_backend=KernelBackend.cuda),
    partial(swiglu, kernel_backend=KernelBackend.triton),
]

table = []
B = 16 * 4096
H = 4096

for dtype in [torch.float16, torch.bfloat16, torch.float32]:
    row = [str(dtype)]
    u = torch.randn(B, H, device=torch.cuda.current_device(), dtype=dtype)
    g = torch.randn(B, H, device=torch.cuda.current_device(), dtype=dtype)

    for kernel in kernels:
        for i in range(n):
            z = kernel(g, u)

        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)

        s.record()
        for i in range(n):
            z = kernel(g, u)
        e.record()

        device_synchronize()

        t = s.elapsed_time(e) / n / 1e3
        row.append(3 * B * H * dtype.itemsize / t / (1024**4))

    table.append(row)


print(tabulate(table, headers=headers))
