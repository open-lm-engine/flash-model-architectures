# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import torch
from tabulate import tabulate

from xma import KernelBackend, device_synchronize, swiglu


torch._inductor.config.max_autotune_gemm_backends = "TRITON"
torch.backends.cuda.matmul.allow_tf32 = True

n = 100

headers = ["kernel", "torch BW", "torch compile BW", "CUDA BW", "triton BW"]
kernels = [
    (swiglu, KernelBackend.cuda),
    (swiglu, KernelBackend.pallas),
    (swiglu, KernelBackend.torch),
    (torch.compile(swiglu, dynamic=True), KernelBackend.torch),
    (swiglu, KernelBackend.triton),
]
dtypes = [torch.float16, torch.bfloat16, torch.float32]

table = [str(dtype) for dtype in dtypes]
B = 16 * 4096
H = 4096

run_forward = False

for kernel, kernel_backend in kernels:
    row = []

    if not kernel_backend.is_kernel_backend_compatible_with_current_device():
        for _ in range(len(dtypes)):
            row.append("NA")
        continue

    u = torch.randn(B, H, device=kernel_backend.get_current_device(), dtype=dtype, requires_grad=not run_forward)
    g = torch.randn(B, H, device=kernel_backend.get_current_device(), dtype=dtype, requires_grad=not run_forward)

    if not run_forward:
        dy = torch.randn(B, H, device=kernel_backend.get_current_device(), dtype=dtype)

    for dtype in dtypes:
        if not run_forward:
            z = kernel(g, u, kernel_backend=kernel_backend)

        for i in range(n):
            if run_forward:
                z = kernel(g, u, kernel_backend=kernel_backend)
            else:
                torch.autograd.grad(z, (g, u), grad_outputs=dy, retain_graph=True)

        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)

        s.record()
        for i in range(n):
            if run_forward:
                z = kernel(g, u, kernel_backend=kernel_backend)
            else:
                torch.autograd.grad(z, (g, u), grad_outputs=dy, retain_graph=True)
        e.record()

        device_synchronize()

        t = s.elapsed_time(e) / n / 1e3
        row.append((3 if run_forward else 5) * B * H * dtype.itemsize / t / 1e12)

    table.append(row)


print(tabulate(table, headers=headers))
