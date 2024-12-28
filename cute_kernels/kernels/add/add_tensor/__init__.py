import torch

from ....cutotune import CutoTuneParameter
from ....enums import KernelBackend
from ....utils import ensure_same_strides
from .forward import _forward
from .torch_implementation import add_tensor_torch


class _AddTensor_Cute(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        y: torch.Tensor,
        kernel_backend: KernelBackend,
        vector_instruction_width: int,
        BLOCK_SIZE: int,
    ) -> torch.Tensor:
        assert x.size() == y.size(), "tensors x and y should have same shape"
        assert x.type() == y.type(), "tensors x and y should have same dtype"

        x, y = ensure_same_strides(x, y)

        return _forward(
            x=x,
            y=y,
            kernel_backend=kernel_backend,
            vector_instruction_width=vector_instruction_width,
            BLOCK_SIZE=BLOCK_SIZE,
        )

    @staticmethod
    def backward(ctx, output_grad: torch.Tensor) -> tuple[torch.Tensor | None]:
        return output_grad, output_grad, *[None] * 3


def add_tensor_cute(
    x: torch.Tensor,
    y: torch.Tensor,
    kernel_backend: KernelBackend = CutoTuneParameter(),
    vector_instruction_width: int = CutoTuneParameter(),
    BLOCK_SIZE: int = CutoTuneParameter(),
) -> torch.Tensor:
    return _AddTensor_Cute.apply(x, y, kernel_backend, vector_instruction_width, BLOCK_SIZE)
