# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import torch
import torch.nn.functional as F

from ...custom_op import CustomOp
from ...enums import KernelBackend
from ...utils import empty_like_contiguous
from .triton_implementation import softmax_backward_triton, softmax_forward_triton


class _Softmax(CustomOp):
    @staticmethod
    def forward_backward_torch(x: torch.Tensor, logits_multiplier: float | None) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()

        if logits_multiplier is not None:
            x = x * logits_multiplier

        x = F.softmax(x, dim=-1)
        x = x.to(dtype)

        return x

    @staticmethod
    def forward_triton(ctx, x: torch.Tensor, logits_multiplier: float | None) -> torch.Tensor:
        output = empty_like_contiguous(x)

        softmax_forward_triton(x=x, output=output, logits_multiplier=logits_multiplier)

        ctx.save_for_backward(output)
        ctx.logits_multiplier = logits_multiplier

        return output

    @staticmethod
    def backward_triton(ctx, output_grad: torch.Tensor) -> tuple[torch.Tensor | None]:
        output = ctx.saved_tensors[0]
        x_grad = empty_like_contiguous(output)

        softmax_backward_triton(
            output=output, output_grad=output_grad, x_grad=x_grad, logits_multiplier=ctx.logits_multiplier
        )

        return x_grad, None


def softmax(
    x: torch.Tensor, logits_multiplier: float | None = None, kernel_backend: KernelBackend | None = None
) -> torch.Tensor:
    """computes softmax activation

    Args:
        x (torch.Tensor): input activation tensor
        logits_multiplier (float, optional): pre-multiplies `x` with `logits_multiplier` before computing softmax.
            Defaults to None.

    Returns:
        torch.Tensor: output tensor
    """

    return _Softmax.run(x, logits_multiplier, kernel_backend=kernel_backend)
