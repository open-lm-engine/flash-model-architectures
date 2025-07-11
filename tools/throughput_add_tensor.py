# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

from functools import partial

import torch
from tabulate import tabulate

from cute_kernels import KernelBackend, add_tensor_cute, device_synchronize


n = 100

headers = ["dtype", "torch", "CUDA kernel", "triton kernel"]
kernels = [
    partial(add_tensor_cute, kernel_backend=KernelBackend.torch),
    partial(add_tensor_cute, kernel_backend=KernelBackend.cuda),
    partial(add_tensor_cute, kernel_backend=KernelBackend.triton),
]

table = []

for dtype in [torch.float32, torch.float16, torch.bfloat16]:
    row = [str(dtype)]
    for kernel in kernels:
        x = torch.randn(104857600, device=torch.cuda.current_device(), dtype=dtype)
        y = torch.randn(104857600, device=torch.cuda.current_device(), dtype=dtype)

        for i in range(n):
            z = kernel(x, y)

        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)

        s.record()
        for i in range(n):
            z = kernel(x, y)
        e.record()

        device_synchronize()

        row.append(s.elapsed_time(e) / n)
    table.append(row)


print(tabulate(table, headers=headers))
