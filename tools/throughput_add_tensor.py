from functools import partial

import torch
from tabulate import tabulate

from cute_kernels import KernelBackend, add_tensor_cute, add_tensor_torch, device_synchronize


n = 100

headers = ["dtype", "torch", "cuda", "triton"]
kernels = [
    add_tensor_torch,
    partial(add_tensor_cute, kernel_backend=KernelBackend.cuda, BLOCK_SIZE=1024),
    partial(add_tensor_cute, kernel_backend=KernelBackend.triton, BLOCK_SIZE=1024),
]

table = []

for dtype in [torch.float16, torch.bfloat16, torch.float32]:
    row = [str(dtype)]
    for kernel in kernels:
        x = torch.randn(10485760, device=torch.cuda.current_device(), dtype=dtype)
        y = torch.randn(10485760, device=torch.cuda.current_device(), dtype=dtype)

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
