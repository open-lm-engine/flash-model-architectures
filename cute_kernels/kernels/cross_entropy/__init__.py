# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import torch
import torch.nn.functional as F

from ...cutotune import CutoTuneParameter
from ...kernel_backend import KernelBackend
from ...utils import ensure_contiguous, get_num_elements_and_hidden_size
from .triton_implementation import cross_entropy_forward_backward_triton


class _CrossEntropy_Cute(torch.autograd.Function):
    @staticmethod
    @ensure_contiguous
    def forward(
        ctx,
        x: torch.Tensor,
        labels: torch.Tensor,
        reduction: str,
        logits_multiplier: float | None,
        kernel_backend: KernelBackend | CutoTuneParameter,
    ) -> torch.Tensor:
        assert reduction in ["sum", "mean"]
        assert x.dim() == 2, "x should be 2 dimensional"
        assert labels.dim() == 1, "labels should be 1 dimensional"
        assert (
            labels.size(0) == get_num_elements_and_hidden_size(x)[0]
        ), "x and labels have different number of elements along batch dimension"
        assert kernel_backend == KernelBackend.triton or isinstance(kernel_backend, CutoTuneParameter)

        loss = torch.tensor(0, device=x.device, dtype=torch.float32)
        x_grad = torch.empty_like(x)

        cross_entropy_forward_backward_triton(
            x=x, labels=labels, loss=loss, x_grad=x_grad, logits_multiplier=logits_multiplier, reduction=reduction
        )

        ctx.save_for_backward(x_grad)

        return loss

    @staticmethod
    def backward(ctx, output_grad: torch.Tensor) -> tuple[torch.Tensor | None]:
        x_grad = ctx.saved_tensors[0]
        x_grad *= output_grad

        return x_grad, *[None] * 4


def cross_entropy_cute(
    x: torch.Tensor,
    labels: torch.Tensor,
    reduction: str = "mean",
    logits_multiplier: float | None = None,
    *,
    kernel_backend: KernelBackend | CutoTuneParameter = KernelBackend.triton,
) -> torch.Tensor:
    """compute cross entropy loss

    Args:
        x (torch.Tensor): logits
        labels (torch.Tensor): labels
        reduction (str, optional): reduction should be either sum or mean. Defaults to "mean".
        logits_multiplier (float | None, optional): logits multiplier pre-multiplies logits, None implies 1.
            Defaults to None.
        kernel_backend (KernelBackend | CutoTuneParameter, optional): kernel backend to prioritize.
            Defaults to KernelBackend.triton.

    Returns:
        torch.Tensor: loss
    """

    if kernel_backend == KernelBackend.torch:
        x = x.float()

        if logits_multiplier not in [None, 1]:
            x = x * logits_multiplier

        x = F.cross_entropy(x, labels, reduction=reduction)
    else:
        x = _CrossEntropy_Cute.apply(x, labels, reduction, logits_multiplier, kernel_backend)

    return x
