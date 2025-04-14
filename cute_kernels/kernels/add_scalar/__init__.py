import torch

from ...math import ceil_divide
from ...utils import is_nvidia_gpu
from .cuda_implementation import add_scalar_cuda
from .torch_implementation import add_scalar_torch
from .triton_implementation import _add_scalar_triton_kernel


class _AddScalar_Cute(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, y: float, kernel_backend: str) -> torch.Tensor:
        output = torch.empty_like(x)
        BLOCK_SIZE = 1024

        if is_nvidia_gpu() and x.is_cuda:
            add_scalar_cuda(x=x, y=y, output=output, BLOCK_SIZE=BLOCK_SIZE)
        else:
            num_elements = x.numel()
            num_programs = ceil_divide(num_elements, BLOCK_SIZE=BLOCK_SIZE)

            with torch.cuda.device(x.device):
                _add_scalar_triton_kernel[(num_programs,)](
                    x_ptr=x, y=y, output_ptr=output, num_elements=num_elements, BLOCK_SIZE=BLOCK_SIZE
                )

        return output

    @staticmethod
    def backward(ctx, output_grad: torch.Tensor) -> tuple[torch.Tensor | None]:
        return output_grad, None, None


def add_scalar_cute(x: torch.Tensor, y: float, kernel_backend: str | None = None) -> torch.Tensor:
    return x if y == 0 else _AddScalar_Cute.apply(x, y, kernel_backend)
