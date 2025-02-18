import torch

from ...cutotune import CutoTuneParameter
from .torch_implementation import cross_entropy_torch
from .triton_implementation import cross_entropy_forward_triton


class _CrossEntropy_Cute(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, labels: torch.Tensor, BLOCK_SIZE_B: int, BLOCK_SIZE_V: int) -> torch.Tensor:
        assert x.dim() == 2, "x should be 2 dimensional"
        assert labels.dim() == 1, "labels should be 1 dimensional"
        assert x.size(0) == labels.size(0), "x and labels have different number of elements along dim 0"

        loss = torch.zeros(1, device=x.device, dtype=torch.float32)

        cross_entropy_forward_triton(
            x=x, labels=labels, loss=loss, V=x.size(-1), BLOCK_SIZE_B=BLOCK_SIZE_B, BLOCK_SIZE_V=BLOCK_SIZE_V
        )

        return loss.squeeze()

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
