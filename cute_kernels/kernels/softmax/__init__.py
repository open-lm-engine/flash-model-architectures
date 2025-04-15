import torch

from ...cutotune import CutoTuneParameter
from ...utils import ensure_contiguous
from .torch_implementation import softmax_torch
from .triton_implementation import softmax_backward_triton, softmax_forward_triton


class _Softmax_Cute(torch.autograd.Function):
    @staticmethod
    @ensure_contiguous
    def forward(
        ctx,
        x: torch.Tensor,
        logits_multiplier: float,
        BLOCK_SIZE_B_forward: int,
        BLOCK_SIZE_H_forward: int,
        BLOCK_SIZE_B_backward: int,
        BLOCK_SIZE_H_backward: int,
    ) -> torch.Tensor:
        if x.size(-1) == 1:
            return torch.ones_like(x)

        ctx.save_for_backward(x)

        is_x_1d = x.dim() == 1
        if is_x_1d:
            x = x.unsqueeze(0)

        output = torch.empty_like(x)

        softmax_forward_triton(
            x=x,
            output=output,
            logits_multiplier=logits_multiplier,
            BLOCK_SIZE_B=BLOCK_SIZE_B_forward,
            BLOCK_SIZE_H=BLOCK_SIZE_H_forward,
        )

        if is_x_1d:
            output = output.squeeze(0)

        ctx.save_for_backward(output)
        ctx.logits_multiplier = logits_multiplier
        ctx.BLOCK_SIZE_B_backward = BLOCK_SIZE_B_backward
        ctx.BLOCK_SIZE_H_backward = BLOCK_SIZE_H_backward

        return output

    @staticmethod
    @ensure_contiguous
    def backward(ctx, output_grad: torch.Tensor) -> tuple[torch.Tensor | None]:
        if output_grad.size(-1) == 1:
            x_grad = torch.zeros_like(output_grad)
        else:
            output = ctx.saved_tensors[0]
            x_grad = torch.empty_like(output)

            softmax_backward_triton(
                output=output,
                output_grad=output_grad,
                x_grad=x_grad,
                logits_multiplier=ctx.logits_multiplier,
                BLOCK_SIZE_B=ctx.BLOCK_SIZE_B_backward,
                BLOCK_SIZE_H=ctx.BLOCK_SIZE_H_backward,
            )

        return x_grad, *[None] * 8


def softmax_cute(
    x: torch.Tensor,
    logits_multiplier: float = 1,
    BLOCK_SIZE_B_forward: int = CutoTuneParameter(),
    BLOCK_SIZE_H_forward: int = CutoTuneParameter(),
    BLOCK_SIZE_B_backward: int = CutoTuneParameter(),
    BLOCK_SIZE_H_backward: int = CutoTuneParameter(),
) -> torch.Tensor:
    return _Softmax_Cute.apply(
        x,
        logits_multiplier,
        BLOCK_SIZE_B_forward,
        BLOCK_SIZE_H_forward,
        BLOCK_SIZE_B_backward,
        BLOCK_SIZE_H_backward,
    )
