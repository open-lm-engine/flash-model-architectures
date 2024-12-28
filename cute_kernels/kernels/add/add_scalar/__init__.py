import torch

from ....cutotune import CutoTuneParameter
from ....enums import KernelBackend
from .forward import _forward
from .torch_implementation import add_scalar_torch


class _AddScalar_Cute(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        y: float,
        kernel_backend: KernelBackend,
        vector_instruction_width: int,
        BLOCK_SIZE: int,
    ) -> torch.Tensor:
        return _forward(
            x=x,
            y=y,
            kernel_backend=kernel_backend,
            vector_instruction_width=vector_instruction_width,
            BLOCK_SIZE=BLOCK_SIZE,
        )

    @staticmethod
    def backward(ctx, output_grad: torch.Tensor) -> tuple[torch.Tensor | None]:
        return output_grad, *[None] * 4


def add_scalar_cute(
    x: torch.Tensor,
    y: torch.Tensor,
    kernel_backend: KernelBackend = CutoTuneParameter(),
    vector_instruction_width: int = CutoTuneParameter(),
    BLOCK_SIZE: int = CutoTuneParameter(),
) -> torch.Tensor:
    if y == 0:
        return x

    return _AddScalar_Cute.apply(x, y, kernel_backend, vector_instruction_width, BLOCK_SIZE)
