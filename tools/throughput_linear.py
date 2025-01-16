import torch
from tabulate import tabulate

from cute_kernels import device_synchronize, linear_cute, linear_torch


n = 100

headers = ["dtype", "torch TFLOPs", "torch compile TFLOPs", "triton TFLOPs"]
kernels = [linear_torch, torch.compile(linear_torch, mode="max-autotune"), linear_cute]

table = []

for dtype in [torch.float16, torch.bfloat16, torch.float32]:
    row = [str(dtype)]
    for kernel in kernels:
        x = torch.randn(4 * 4096, 4096, device=torch.cuda.current_device(), dtype=dtype)
        w = torch.randn(4096, 4096, device=torch.cuda.current_device(), dtype=dtype)

        for i in range(n):
            z = kernel(x, w)

        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)

        s.record()
        for i in range(n):
            z = kernel(x, w)
        e.record()

        device_synchronize()

        t = s.elapsed_time(e) / n / 1e3
        row.append(2 * x.size(0) * x.size(1) * w.size(0) / t / 1e12)

    table.append(row)


print(tabulate(table, headers=headers))
