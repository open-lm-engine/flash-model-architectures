from typing import Callable

import torch
from transformers import set_seed

from cute_kernels import (
    KernelBackend,
    add_scalar_cute,
    add_tensor_cute,
    continuous_count_cute,
    embedding_cute,
    gemm_cute,
    get_all_cutotune_caches,
    get_powers_of_2,
    rmsnorm_cute,
    save_cutotune_cache,
    swiglu_cute,
    swiglu_unchunked_cute,
)
from cute_kernels.kernels.gemm import CUDAKernelAlgorithm


def forward_backward(kernel: Callable, *args, **kwargs) -> None:
    output = kernel(*args, **kwargs)
    output.sum().backward()


set_seed(42)
all_dtypes = [torch.float32, torch.float16, torch.bfloat16]


for dtype in all_dtypes:
    size = 104857600
    x = torch.randn(size, dtype=dtype, device=torch.cuda.current_device(), requires_grad=True)

    forward_backward(add_scalar_cute, x, 3)
    forward_backward(add_tensor_cute, x, x)
    forward_backward(swiglu_cute, x, x)

    size = (81920, 8192)
    forward_backward(
        swiglu_unchunked_cute,
        torch.randn(size, dtype=dtype, device=torch.cuda.current_device(), requires_grad=True),
    )

    for power_of_2 in get_powers_of_2(1, 65536):
        size = (2048, power_of_2)
        x = torch.randn(size, dtype=dtype, device=torch.cuda.current_device(), requires_grad=True)

        forward_backward(rmsnorm_cute, x, weight=None, eps=1e-5)
        forward_backward(
            rmsnorm_cute,
            x,
            weight=torch.randn(x.size(-1), dtype=dtype, device=torch.cuda.current_device(), requires_grad=True),
            eps=1e-5,
        )

    input_ids_size = (32, 4096)
    weight_size = (131072, 4096)
    forward_backward(
        embedding_cute,
        input_ids=torch.randint(
            0, weight_size[0] - 1, input_ids_size, device=torch.cuda.current_device(), dtype=torch.long
        ),
        weight=torch.randn(weight_size, device=torch.cuda.current_device(), dtype=dtype, requires_grad=True),
    )

    input_size = (4096, 4096)
    weight_size = (4096, 4096)

    for is_a_transposed in [False, True]:
        for is_b_transposed in [False, True]:
            gemm_cute(
                a=torch.randn(*input_size, device=torch.cuda.current_device(), dtype=dtype, requires_grad=True),
                b=torch.randn(*weight_size, device=torch.cuda.current_device(), dtype=dtype, requires_grad=True),
                c=None,
                is_a_transposed=is_a_transposed,
                is_b_transposed=is_b_transposed,
                cuda_kernel_algorithm=CUDAKernelAlgorithm.naive,
                beta=0,
            )

    gemm_cute(
        a=torch.randn(*input_size, device=torch.cuda.current_device(), dtype=dtype, requires_grad=True),
        b=torch.randn(*weight_size, device=torch.cuda.current_device(), dtype=dtype, requires_grad=True),
        c=None,
        is_a_transposed=False,
        is_b_transposed=False,
        kernel_backend=KernelBackend.cuda,
        cuda_kernel_algorithm=CUDAKernelAlgorithm.shared_memory,
        beta=0,
    )

size = 104857600
for dtype in [torch.long, torch.int32]:
    for n in get_powers_of_2(1, 16384):
        x = torch.randint(0, n, (size,), dtype=dtype, device=torch.cuda.current_device())
        continuous_count_cute(x, n)


for function_hash in get_all_cutotune_caches():
    save_cutotune_cache(function_hash)
