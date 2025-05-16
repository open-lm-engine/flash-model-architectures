import torch

from ...kernel_backend import KernelBackend, is_cuda_kernel_backend_allowed, is_triton_kernel_backend_allowed
from ...utils import ensure_same_strides, is_nvidia_gpu
from .cuda_implementation import add_tensor_cuda
from .torch_implementation import add_tensor_torch
from .triton_implementation import add_tensor_triton


class _AddTensor_Cute(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, x: torch.Tensor, y: torch.Tensor, kernel_backend: KernelBackend, BLOCK_SIZE_CUDA: int
    ) -> torch.Tensor:
        assert x.size() == y.size(), "tensors x and y should have same shape"
        assert x.type() == y.type(), "tensors x and y should have same dtype"

        x, y = ensure_same_strides(x, y)
        output = torch.empty_like(x)

        if is_cuda_kernel_backend_allowed(kernel_backend) and is_nvidia_gpu() and x.is_cuda and y.is_cuda:
            add_tensor_cuda(x=x, y=y, output=output, BLOCK_SIZE=BLOCK_SIZE_CUDA)
        elif is_triton_kernel_backend_allowed(kernel_backend):
            add_tensor_triton(x=x, y=y, output=output)
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
) -> torch.Tensor:
    """add 2 tensors

    Args:
        x (torch.Tensor): first tensor
        y (torch.Tensor): second tensor
        kernel_backend (KernelBackend, optional): kernel backend to prioritize. Defaults to KernelBackend.cuda.
        BLOCK_SIZE_CUDA (int, optional): block size for CUDA backend. Defaults to 1024.

    Returns:
        torch.Tensor: output tensor
    """

    return _AddTensor_Cute.apply(x, y, kernel_backend, BLOCK_SIZE_CUDA)
