import torch
from tabulate import tabulate

from cute_kernels import add_tensor_cute, add_tensor_torch, device_synchronize


n = 100

headers = ["dtype", "torch", "kernel"]
kernels = [add_tensor_torch, add_tensor_cute]

table = []

for dtype in [torch.float32]:
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
