# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

from functools import partial

import torch
from tabulate import tabulate

from xma import RNN, Accelerator, KernelBackend, rmsnorm


n = 100

kernels = [
    (rmsnorm, KernelBackend.torch, "torch"),
    (torch.compile(rmsnorm, dynamic=True), KernelBackend.torch, "torch compile"),
    (rmsnorm, KernelBackend.triton, "triton"),
    (partial(rmsnorm, deterministic=False), KernelBackend.triton, "triton with atomic_add"),
    (partial(rmsnorm, deterministic=True), KernelBackend.triton, "triton without atomic_add"),
]
dtypes = [torch.float32, torch.bfloat16, torch.float16]
headers = ["kernel"] + dtypes
table = []

B = 16 * 4096
H = 4096

run_forward = True

for kernel, kernel_backend, row_header in kernels:
    row = [row_header]

    if not kernel_backend.verify_accelerator():
        for _ in range(len(dtypes)):
            row.append("NA")
        table.append(row)
        continue

    device = kernel_backend.get_compatible_accelerator().get_current_device()

    for dtype in dtypes:
        x = torch.randn(B, H, device=device, dtype=dtype, requires_grad=not run_forward)
        W = torch.randn(H, device=device, dtype=dtype, requires_grad=not run_forward)

        if not run_forward:
            dy = torch.randn(B, H, device=device, dtype=dtype, requires_grad=not run_forward)
            z = kernel(x=x, weight=W, eps=None, kernel_backend=kernel_backend)

        for i in range(n):
            if run_forward:
                z = kernel(x=x, weight=W, eps=None)
            else:
                torch.autograd.grad(z, (x, W), grad_outputs=dy, retain_graph=True)

        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)

        s.record()
        for i in range(n):
            if run_forward:
                z = kernel(x=x, weight=W, eps=None, kernel_backend=kernel_backend)
            else:
                torch.autograd.grad(z, (x, W), grad_outputs=dy, retain_graph=True)
        e.record()

        Accelerator.synchronize()

        t = s.elapsed_time(e) / n / 1e3
        row.append(((B + 1) * H if run_forward else 5) * dtype.itemsize / t / 1e12)

    table.append(row)


print(tabulate(table, headers=headers))
