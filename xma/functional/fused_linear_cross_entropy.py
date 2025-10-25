# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import torch
import torch.nn.functional as F

from ..custom_op import CustomOp, ctx_needs_gradients, ctx_save_for_backward
from ..enums import KernelBackend
from ..math import ceil_divide, get_next_power_of_2
from ..utils import empty_like_contiguous, zeros_like_contiguous
from .cross_entropy import cross_entropy, cross_entropy_forward_backward_triton


class _FusedLinearCrossEntropy(CustomOp):
    def forward_backward_torch(
        x: torch.Tensor,
        weight: torch.Tensor,
        labels: torch.Tensor,
        reduction: str,
        logits_multiplier: float | None,
    ) -> torch.Tensor:
        x = F.linear(x, weight)
        x = cross_entropy(
            x=x,
            labels=labels,
            reduction=reduction,
            logits_multiplier=logits_multiplier,
            kernel_backend=KernelBackend.torch,
        )

        return x

    @staticmethod
    def forward_triton(
        ctx,
        x: torch.Tensor,
        weight: torch.Tensor,
        labels: torch.Tensor,
        reduction: str,
        logits_multiplier: float | None,
    ) -> torch.Tensor:
        B, H = x.size()
        V = weight.size(0)

        # NOTE chunking is copied from liger kernel
        memory_increase_factor = ceil_divide(V, H)
        # chunk_size needed to reduce memory increase back to 1
        chunk_size = get_next_power_of_2(ceil_divide(B, memory_increase_factor))
        num_chunks = ceil_divide(B, chunk_size)

        loss = torch.zeros((), device=x.device, dtype=torch.float32)

        needs_grad = ctx_needs_gradients(ctx)
        x_grad = empty_like_contiguous(x) if needs_grad else None
        weight_grad = zeros_like_contiguous(weight) if needs_grad else None

        for i in range(num_chunks):
            start = i * chunk_size
            end = (i + 1) * chunk_size
            end = min(end, B)

            _x = x[start:end]
            _logits = _x @ weight.T

            _logits_grad = empty_like_contiguous(_logits)
            _labels = labels[start:end]

            cross_entropy_forward_backward_triton(
                x=_logits,
                labels=_labels,
                loss=loss,
                x_grad=_logits_grad,
                logits_multiplier=logits_multiplier,
                reduction="sum",
            )

            if needs_grad:
                x_grad[start:end] = _logits_grad @ weight
                torch.addmm(weight_grad, _logits_grad.T, _x, alpha=1, beta=1, out=weight_grad)

        if reduction == "mean":
            loss /= B
            x_grad /= B
            weight_grad /= B

        ctx_save_for_backward(ctx, x_grad, weight_grad)

        return loss

    @staticmethod
    def backward_triton(ctx, output_grad: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None, None, None, None]:
        x_grad, weight_grad = ctx.saved_tensors

        x_grad *= output_grad
        weight_grad *= output_grad

        return x_grad, weight_grad, None, None, None


def fused_linear_cross_entropy(
    x: torch.Tensor,
    weight: torch.Tensor,
    labels: torch.Tensor,
    reduction: str = "mean",
    logits_multiplier: float | None = None,
    *,
    kernel_backend: KernelBackend | None = None,
) -> torch.Tensor:
    """compute cross entropy loss without materializing the full output logits matrix

    Args:
        x (torch.Tensor): logits
        weight (torch.Tensor): vocab weight
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
    assert x.size(0) == labels.size(0), "x and labels have different number of elements along dim 0"
    assert x.size(-1) == weight.size(-1)

    x = _FusedLinearCrossEntropy.run(
        x=x,
        weight=weight,
        labels=labels,
        reduction=reduction,
        logits_multiplier=logits_multiplier,
        kernel_backend=kernel_backend,
    )

    return x
