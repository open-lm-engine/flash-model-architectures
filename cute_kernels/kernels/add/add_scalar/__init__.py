import torch

from ....cutotune import CutoTuneParameter
from .forward import _forward
from .torch_implementation import add_scalar_torch


class _AddScalar_Cute(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        y: float,
        kernel_backend: str,
        BLOCK_SIZE: int,
    ) -> torch.Tensor:
        assert x.dtype in [torch.float32, torch.float16, torch.bfloat16]

        return _forward(
            x=x,
            y=y,
            kernel_backend=kernel_backend,
            BLOCK_SIZE=BLOCK_SIZE,
        )

    @staticmethod
    def backward(ctx, output_grad: torch.Tensor) -> tuple[torch.Tensor | None]:
        return output_grad, *[None] * 3


def add_scalar_cute(
    x: torch.Tensor,
    y: torch.Tensor,
    kernel_backend: str = CutoTuneParameter(),
    BLOCK_SIZE: int = CutoTuneParameter(),
) -> torch.Tensor:
    return x if y == 0 else _AddScalar_Cute.apply(x, y, kernel_backend, BLOCK_SIZE)
