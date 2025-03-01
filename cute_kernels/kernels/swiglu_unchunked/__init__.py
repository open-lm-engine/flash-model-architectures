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
        kernel_backend_forward: str,
        kernel_backend_backward: str,
        BLOCK_SIZE_B_forward: int,
        BLOCK_SIZE_H_forward: int,
        BLOCK_SIZE_B_backward: int,
        BLOCK_SIZE_H_backward: int,
    ) -> torch.Tensor:
        assert x.dtype in [torch.float32, torch.float16, torch.bfloat16]

        ctx.save_for_backward(x)
        ctx.kernel_backend_backward = kernel_backend_backward
        ctx.BLOCK_SIZE_B_backward = BLOCK_SIZE_B_backward
        ctx.BLOCK_SIZE_H_backward = BLOCK_SIZE_H_backward

        return _forward(
            x=x,
            kernel_backend=kernel_backend_forward,
            BLOCK_SIZE_B=BLOCK_SIZE_B_forward,
            BLOCK_SIZE_H=BLOCK_SIZE_H_forward,
        )

    @staticmethod
    @ensure_contiguous
    def backward(ctx, output_grad: torch.Tensor) -> tuple[torch.Tensor | None]:
        x = ctx.saved_tensors[0]

        x_grad = _backward(
            x=x,
            output_grad=output_grad,
            kernel_backend=ctx.kernel_backend_backward,
            BLOCK_SIZE_B=ctx.BLOCK_SIZE_B_backward,
            BLOCK_SIZE_H=ctx.BLOCK_SIZE_H_backward,
        )

        return x_grad, *[None] * 6


def swiglu_unchunked_cute(
    x: torch.Tensor,
    kernel_backend_forward: str = CutoTuneParameter(),
    kernel_backend_backward: str = CutoTuneParameter(),
    BLOCK_SIZE_B_forward: int = CutoTuneParameter(),
    BLOCK_SIZE_H_forward: int = CutoTuneParameter(),
    BLOCK_SIZE_B_backward: int = CutoTuneParameter(),
    BLOCK_SIZE_H_backward: int = CutoTuneParameter(),
) -> torch.Tensor:
    return _SwigluUnchunked_Cute.apply(
        x,
        kernel_backend_forward,
        kernel_backend_backward,
        BLOCK_SIZE_B_forward,
        BLOCK_SIZE_H_forward,
        BLOCK_SIZE_B_backward,
        BLOCK_SIZE_H_backward,
    )
