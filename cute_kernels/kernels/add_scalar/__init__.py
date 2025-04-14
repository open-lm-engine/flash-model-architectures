import torch

from ...kernel_backend import is_cuda_kernel_backend_allowed, is_triton_kernel_backend_allowed
from ...math import ceil_divide
from ...utils import is_nvidia_gpu
from .cuda_implementation import add_scalar_cuda
from .torch_implementation import add_scalar_torch
from .triton_implementation import _add_scalar_triton_kernel


class _AddScalar_Cute(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, y: float) -> torch.Tensor:
        output = torch.empty_like(x)
        BLOCK_SIZE = 1024

        if is_nvidia_gpu() and is_cuda_kernel_backend_allowed() and x.is_cuda:
            add_scalar_cuda(x=x, y=y, output=output, BLOCK_SIZE=BLOCK_SIZE)
        elif is_triton_kernel_backend_allowed():
            num_elements = x.numel()
            num_programs = ceil_divide(num_elements, BLOCK_SIZE=BLOCK_SIZE)

            with torch.cuda.device(x.device):
                _add_scalar_triton_kernel[(num_programs,)](
                    x_ptr=x, y=y, output_ptr=output, num_elements=num_elements, BLOCK_SIZE=BLOCK_SIZE
                )
        else:
            raise ValueError("unexpected kernel_backend")

        return output

    @staticmethod
    def backward(ctx, output_grad: torch.Tensor) -> tuple[torch.Tensor | None]:
        return output_grad, None


def add_scalar_cute(x: torch.Tensor, y: float) -> torch.Tensor:
    return x if y == 0 else _AddScalar_Cute.apply(x, y)
