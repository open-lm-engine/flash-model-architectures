# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

from functools import partial

import torch
from tabulate import tabulate

from cute_kernels import RNN, KernelBackend, device_synchronize


n = 100

B = 16
S = 4096
N = 64
H = 768

with torch.device(torch.cuda.current_device()):
    rnn = RNN(input_size=H, state_size=1024, output_size=H, num_heads=N, add_bias=False, gradient_clipping=None)

headers = ["dtype", "torch", "kernel"]

table = []

for dtype in [torch.float32]:
    row = [str(dtype)]

    kernels = [
        partial(rnn.to(dtype), kernel_backend=KernelBackend.torch),
        partial(rnn.to(dtype), kernel_backend=KernelBackend.triton),
    ]

    for kernel in kernels:
        input = torch.randn(B, S, H, device=torch.cuda.current_device(), dtype=dtype)

        for i in range(n):
            z = kernel(input)

        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)

        s.record()
        for i in range(n):
            z = kernel(input)
        e.record()

        device_synchronize()

        row.append(s.elapsed_time(e) / n)
    table.append(row)


print(tabulate(table, headers=headers))
