import torch

from ...cutotune import CutoTuneParameter
from .torch_implementation import cross_entropy_torch
from .triton_implementation import cross_entropy_forward_triton


class _CrossEntropy_Cute(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        pass

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor | None]:
        pass


def cross_entropy_cute(
    x: torch.Tensor,
    labels: torch.Tensor,
    BLOCK_SIZE_B: int = CutoTuneParameter(),
    BLOCK_SIZE_V: int = CutoTuneParameter(),
) -> torch.Tensor:
    return _CrossEntropy_Cute.apply(x, labels, BLOCK_SIZE_B, BLOCK_SIZE_V)
