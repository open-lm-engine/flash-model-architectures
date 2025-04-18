import torch

from ...kernel_backend import KernelBackend, is_cuda_kernel_backend_allowed, is_triton_kernel_backend_allowed
from ...math import ceil_divide
from ...utils import is_nvidia_gpu
from .cuda_implementation import add_scalar_cuda
from .torch_implementation import add_scalar_torch
from .triton_implementation import _add_scalar_triton_kernel


class _AddScalar_Cute(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        y: float,
        kernel_backend: KernelBackend,
        BLOCK_SIZE_CUDA: int,
        BLOCK_SIZE_TRITON: int,
        NUM_WARPS_TRITON: int,
    ) -> torch.Tensor:
        output = torch.empty_like(x)

        if is_cuda_kernel_backend_allowed(kernel_backend) and is_nvidia_gpu() and x.is_cuda:
            add_scalar_cuda(x=x, y=y, output=output, BLOCK_SIZE=BLOCK_SIZE_CUDA)
        elif is_triton_kernel_backend_allowed(kernel_backend):
            N = x.numel()
            num_programs = ceil_divide(N, BLOCK_SIZE_TRITON)

            with torch.cuda.device(x.device):
                _add_scalar_triton_kernel[num_programs,](
                    x_ptr=x, y=y, output_ptr=output, N=N, BLOCK_SIZE=BLOCK_SIZE_TRITON, num_warps=NUM_WARPS_TRITON
                )
        else:
            raise ValueError("unexpected kernel_backend")

        return output

    @staticmethod
    def backward(ctx, output_grad: torch.Tensor) -> tuple[torch.Tensor | None]:
        return output_grad, *[None] * 5


def add_scalar_cute(
    x: torch.Tensor,
    y: float,
    *,
    kernel_backend: KernelBackend = KernelBackend.cuda,
    # cuda
    BLOCK_SIZE_CUDA: int = 1024,
    # triton
    BLOCK_SIZE_TRITON: int = 4096,
    NUM_WARPS_TRITON: int = 32,
) -> torch.Tensor:
    if y == 0:
        return x

    return _AddScalar_Cute.apply(x, y, kernel_backend, BLOCK_SIZE_CUDA, BLOCK_SIZE_TRITON, NUM_WARPS_TRITON)
