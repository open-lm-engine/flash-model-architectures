import torch

from ...kernel_backend import KernelBackend, is_cuda_kernel_backend_allowed, is_triton_kernel_backend_allowed
from ...math import ceil_divide
from ...utils import ensure_same_strides, is_nvidia_gpu
from .cuda_implementation import add_tensor_cuda
from .torch_implementation import add_tensor_torch
from .triton_implementation import add_tensor_triton_kernel


class _AddTensor_Cute(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        y: torch.Tensor,
        kernel_backend: KernelBackend,
        BLOCK_SIZE_CUDA: int,
        BLOCK_SIZE_TRITON: int,
        NUM_WARPS_TRITON: int,
    ) -> torch.Tensor:
        assert x.size() == y.size(), "tensors x and y should have same shape"
        assert x.type() == y.type(), "tensors x and y should have same dtype"

        x, y = ensure_same_strides(x, y)
        output = torch.empty_like(x)

        if is_cuda_kernel_backend_allowed(kernel_backend) and is_nvidia_gpu() and x.is_cuda and y.is_cuda:
            add_tensor_cuda(x=x, y=y, output=output, BLOCK_SIZE=BLOCK_SIZE_CUDA)
        elif is_triton_kernel_backend_allowed(kernel_backend):
            N = x.numel()

            with torch.cuda.device(x.device):
                add_tensor_triton_kernel[ceil_divide(N, BLOCK_SIZE_TRITON),](
                    x_ptr=x,
                    y_ptr=y,
                    output_ptr=output,
                    N=N,
                    BLOCK_SIZE=BLOCK_SIZE_TRITON,
                    num_warps=NUM_WARPS_TRITON,
                )
        else:
            raise ValueError("unexpected kernel_backend")

        return output

    @staticmethod
    def backward(ctx, output_grad: torch.Tensor) -> tuple[torch.Tensor | None]:
        return output_grad, output_grad, *[None] * 4


def add_tensor_cute(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    kernel_backend: KernelBackend = KernelBackend.cuda,
    BLOCK_SIZE_CUDA: int = 1024,
    BLOCK_SIZE_TRITON: int = 4096,
    NUM_WARPS_TRITON: int = 32,
) -> torch.Tensor:
    """add 2 tensors

    Args:
        x (torch.Tensor): first tensor
        y (torch.Tensor): second tensor
        kernel_backend (KernelBackend, optional): kernel backend to prioritize. Defaults to KernelBackend.cuda.
        BLOCK_SIZE_CUDA (int, optional): block size for CUDA backend. Defaults to 1024.
        BLOCK_SIZE_TRITON (int, optional): block size for triton backend. Defaults to 4096.
        NUM_WARPS_TRITON (int, optional): warps for triton backend. Defaults to 32.

    Returns:
        torch.Tensor: output tensor
    """

    return _AddTensor_Cute.apply(x, y, kernel_backend, BLOCK_SIZE_CUDA, BLOCK_SIZE_TRITON, NUM_WARPS_TRITON)
