import torch

from ...kernel_backend import is_cuda_kernel_backend_allowed, is_triton_kernel_backend_allowed
from ...math import ceil_divide
from ...utils import ensure_same_strides, is_nvidia_gpu
from .cuda_implementation import add_tensor_cuda
from .torch_implementation import add_tensor_torch
from .triton_implementation import _add_tensor_triton_kernel


class _AddTensor_Cute(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        assert x.size() == y.size(), "tensors x and y should have same shape"
        assert x.type() == y.type(), "tensors x and y should have same dtype"

        BLOCK_SIZE = 1024

        x, y = ensure_same_strides(x, y)
        output = torch.empty_like(x)

        if is_cuda_kernel_backend_allowed() and is_nvidia_gpu() and x.is_cuda and y.is_cuda:
            add_tensor_cuda(x=x, y=y, output=output, BLOCK_SIZE=BLOCK_SIZE)
        elif is_triton_kernel_backend_allowed():
            num_elements = x.numel()
            num_programs = ceil_divide(num_elements, BLOCK_SIZE)

            with torch.cuda.device(x.device):
                _add_tensor_triton_kernel[(num_programs,)](
                    x_ptr=x, y_ptr=y, output_ptr=output, num_elements=num_elements, BLOCK_SIZE=BLOCK_SIZE
                )
        else:
            raise ValueError("unexpected kernel_backend")

        return output

    @staticmethod
    def backward(ctx, output_grad: torch.Tensor) -> tuple[torch.Tensor | None]:
        return output_grad, output_grad


def add_tensor_cute(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return _AddTensor_Cute.apply(x, y)
