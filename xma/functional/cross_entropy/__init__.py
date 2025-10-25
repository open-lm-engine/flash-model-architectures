# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import torch
import torch.nn.functional as F

from ...custom_op import CustomOp, ctx_needs_gradients, ctx_save_for_backward
from ...enums import KernelBackend
from ...utils import empty_like_contiguous, get_num_elements_and_hidden_size
from .triton_implementation import cross_entropy_forward_backward_triton


class _CrossEntropy(CustomOp):
    @staticmethod
    def forward_backward_torch(
        x: torch.Tensor, labels: torch.Tensor, reduction: str = "mean", logits_multiplier: float | None = None
    ) -> torch.Tensor:
        x = x.float()

        if logits_multiplier not in [None, 1]:
            x = x * logits_multiplier

        return F.cross_entropy(x, labels, reduction=reduction)

    @staticmethod
    def forward_triton(
        ctx, x: torch.Tensor, labels: torch.Tensor, reduction: str, logits_multiplier: float | None
    ) -> torch.Tensor:
        loss = torch.zeros((), device=x.device, dtype=torch.float32)
        x_grad = empty_like_contiguous(x) if ctx_needs_gradients(ctx) else None

        cross_entropy_forward_backward_triton(
            x=x, labels=labels, loss=loss, x_grad=x_grad, logits_multiplier=logits_multiplier, reduction=reduction
        )

        ctx_save_for_backward(ctx, x_grad)

        return loss

    @staticmethod
    def backward_triton(ctx, output_grad: torch.Tensor) -> tuple[torch.Tensor, None, None, None]:
        x_grad = ctx.saved_tensors[0]
        x_grad *= output_grad

        return x_grad, *[None] * 3


def cross_entropy(
    x: torch.Tensor,
    labels: torch.Tensor,
    reduction: str = "mean",
    logits_multiplier: float | None = None,
    *,
    kernel_backend: KernelBackend | None = None,
) -> torch.Tensor:
    """compute cross entropy loss

    Args:
        x (torch.Tensor): logits
        labels (torch.Tensor): labels
        reduction (str, optional): reduction should be either sum or mean. Defaults to "mean".
        logits_multiplier (float | None, optional): logits multiplier pre-multiplies logits, None implies 1.
            Defaults to None.

    Returns:
        torch.Tensor: loss
    """

    assert reduction in ["sum", "mean"]
    assert x.dim() == 2, "x should be 2 dimensional"
    assert labels.dim() == 1, "labels should be 1 dimensional"
    assert (
        labels.size(0) == get_num_elements_and_hidden_size(x)[0]
    ), "x and labels have different number of elements along batch dimension"

    return _CrossEntropy.run(
        x=x, labels=labels, reduction=reduction, logits_multiplier=logits_multiplier, kernel_backend=kernel_backend
    )
