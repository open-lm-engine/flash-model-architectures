import torch

from ....cutotune import CutoTuneParameter
from ....utils import ensure_same_strides
from .forward import _forward
from .torch_implementation import add_tensor_torch


class _AddTensor_Cute(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        y: torch.Tensor,
        kernel_backend: str,
        BLOCK_SIZE: int,
    ) -> torch.Tensor:
        assert x.size() == y.size(), "tensors x and y should have same shape"
        assert x.type() == y.type(), "tensors x and y should have same dtype"
        assert x.type() in [torch.float32, torch.float16, torch.bfloat16]

        x, y = ensure_same_strides(x, y)

        return _forward(
            x=x,
            y=y,
            kernel_backend=kernel_backend,
            BLOCK_SIZE=BLOCK_SIZE,
        )

    @staticmethod
    def backward(ctx, output_grad: torch.Tensor) -> tuple[torch.Tensor | None]:
        return output_grad, output_grad, *[None] * 2


def add_tensor_cute(
    x: torch.Tensor,
    y: torch.Tensor,
    kernel_backend: str = CutoTuneParameter(),
    BLOCK_SIZE: int = CutoTuneParameter(),
) -> torch.Tensor:
    return _AddTensor_Cute.apply(x, y, kernel_backend, BLOCK_SIZE)
