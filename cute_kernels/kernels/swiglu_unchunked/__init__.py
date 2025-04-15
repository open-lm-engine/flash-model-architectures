import torch

from ...cutotune import CutoTuneParameter
from ...utils import ensure_contiguous
from .backward import _backward
from .forward import _forward
from .torch_implementation import swiglu_unchunked_torch


class _SwigluUnchunked_Cute(torch.autograd.Function):
    @staticmethod
    @ensure_contiguous
    def forward(
        ctx,
        x: torch.Tensor,
        BLOCK_SIZE_B_forward: int,
        BLOCK_SIZE_H_forward: int,
        BLOCK_SIZE_B_backward: int,
        BLOCK_SIZE_H_backward: int,
    ) -> torch.Tensor:
        ctx.save_for_backward(x)
        ctx.BLOCK_SIZE_B_backward = BLOCK_SIZE_B_backward
        ctx.BLOCK_SIZE_H_backward = BLOCK_SIZE_H_backward

        return _forward(x=x, BLOCK_SIZE_B=BLOCK_SIZE_B_forward, BLOCK_SIZE_H=BLOCK_SIZE_H_forward)

    @staticmethod
    @ensure_contiguous
    def backward(ctx, output_grad: torch.Tensor) -> tuple[torch.Tensor | None]:
        x = ctx.saved_tensors[0]

        x_grad = _backward(
            x=x,
            output_grad=output_grad,
            BLOCK_SIZE_B=ctx.BLOCK_SIZE_B_backward,
            BLOCK_SIZE_H=ctx.BLOCK_SIZE_H_backward,
        )

        return x_grad, *[None] * 6


def swiglu_unchunked_cute(
    x: torch.Tensor,
    BLOCK_SIZE_B_forward: int = CutoTuneParameter(),
    BLOCK_SIZE_H_forward: int = CutoTuneParameter(),
    BLOCK_SIZE_B_backward: int = CutoTuneParameter(),
    BLOCK_SIZE_H_backward: int = CutoTuneParameter(),
) -> torch.Tensor:
    return _SwigluUnchunked_Cute.apply(
        x,
        BLOCK_SIZE_B_forward,
        BLOCK_SIZE_H_forward,
        BLOCK_SIZE_B_backward,
        BLOCK_SIZE_H_backward,
    )
