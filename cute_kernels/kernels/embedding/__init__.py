import torch

from ...cutotune import CutoTuneParameter
from ...utils import ensure_contiguous
from .backward import _backward
from .forward import _forward
from .torch_implementation import embedding_torch


class _Embedding_Cute(torch.autograd.Function):
    @staticmethod
    @ensure_contiguous
    def forward(
        ctx,
        input_ids: torch.Tensor,
        weight: torch.Tensor,
        kernel_backend_forward: str,
        kernel_backend_backward: str,
        BLOCK_SIZE_B_forward: int,
        BLOCK_SIZE_B_backward: int,
        BLOCK_SIZE_H_forward: int,
        BLOCK_SIZE_H_backward: int,
    ) -> torch.Tensor:
        output = _forward(
            input_ids=input_ids,
            weight=weight,
            kernel_backend=kernel_backend_forward,
            BLOCK_SIZE_B=BLOCK_SIZE_B_forward,
            BLOCK_SIZE_H=BLOCK_SIZE_H_forward,
        )

        ctx.save_for_backward(input_ids, weight)
        ctx.kernel_backend_backward = kernel_backend_backward
        ctx.BLOCK_SIZE_B_backward = BLOCK_SIZE_B_backward
        ctx.BLOCK_SIZE_H_backward = BLOCK_SIZE_H_backward

        return output

    @staticmethod
    @ensure_contiguous
    def backward(ctx, output_grad: torch.Tensor) -> torch.Tensor:
        input_ids, weight = ctx.saved_tensors
        kernel_backend_backward = ctx.kernel_backend_backward
        BLOCK_SIZE_B_backward = ctx.BLOCK_SIZE_B_backward
        BLOCK_SIZE_H_backward = ctx.BLOCK_SIZE_H_backward

        weight_grad = _backward(
            input_ids=input_ids,
            weight=weight,
            output_grad=output_grad,
            kernel_backend=kernel_backend_backward,
            BLOCK_SIZE_B=BLOCK_SIZE_B_backward,
            BLOCK_SIZE_H=BLOCK_SIZE_H_backward,
        )

        return None, weight_grad, *[None] * 6


def embedding_cute(
    input_ids: torch.Tensor,
    weight: torch.Tensor,
    kernel_backend_forward: str = CutoTuneParameter(),
    kernel_backend_backward: str = CutoTuneParameter(),
    BLOCK_SIZE_B_forward: int = CutoTuneParameter(),
    BLOCK_SIZE_B_backward: int = CutoTuneParameter(),
    BLOCK_SIZE_H_forward: int = CutoTuneParameter(),
    BLOCK_SIZE_H_backward: int = CutoTuneParameter(),
) -> torch.Tensor:
    return _Embedding_Cute.apply(
        input_ids,
        weight,
        kernel_backend_forward,
        kernel_backend_backward,
        BLOCK_SIZE_B_forward,
        BLOCK_SIZE_B_backward,
        BLOCK_SIZE_H_forward,
        BLOCK_SIZE_H_backward,
    )
