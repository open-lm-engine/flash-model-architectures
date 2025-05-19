import torch

from ...utils import ensure_contiguous
from .torch_implementation import softmax_torch
from .triton_implementation import softmax_backward_triton, softmax_forward_triton


class _Softmax_Cute(torch.autograd.Function):
    @staticmethod
    @ensure_contiguous
    def forward(ctx, x: torch.Tensor, logits_multiplier: float | None) -> torch.Tensor:
        output = torch.empty_like(x)

        softmax_forward_triton(x=x, output=output, logits_multiplier=logits_multiplier)

        ctx.save_for_backward(output)
        ctx.logits_multiplier = logits_multiplier

        return output

    @staticmethod
    @ensure_contiguous
    def backward(ctx, output_grad: torch.Tensor) -> tuple[torch.Tensor | None]:
        output = ctx.saved_tensors[0]
        x_grad = torch.empty_like(output)

        softmax_backward_triton(
            output=output, output_grad=output_grad, x_grad=x_grad, logits_multiplier=ctx.logits_multiplier
        )

        return x_grad, None


def softmax_cute(x: torch.Tensor, logits_multiplier: float | None = None) -> torch.Tensor:
    """computes softmax activation

    Args:
        x (torch.Tensor): input activation tensor
        logits_multiplier (float, optional): pre-multiplies `x` with `logits_multiplier` before computing softmax.
            Defaults to None.

    Returns:
        torch.Tensor: output tensor
    """

    return _Softmax_Cute.apply(x, logits_multiplier)
