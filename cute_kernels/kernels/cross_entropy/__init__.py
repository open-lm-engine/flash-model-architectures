import torch

from ...cutotune import CutoTuneParameter
from ...utils import ensure_contiguous
from ..softmax import _forward as _softmax_forward
from .torch_implementation import cross_entropy_torch
from .triton_implementation import cross_entropy_forward_triton


class _CrossEntropy_Cute(torch.autograd.Function):
    @staticmethod
    @ensure_contiguous
    def forward(
        ctx,
        x: torch.Tensor,
        labels: torch.Tensor,
        reduction: str,
        logits_multiplier: float,
        BLOCK_SIZE_B_forward: int,
        BLOCK_SIZE_V_forward: int,
        kernel_backend_backward: str,
        BLOCK_SIZE_B_backward: int,
        BLOCK_SIZE_V_backward: int,
    ) -> torch.Tensor:
        assert reduction in ["sum", "mean"]
        assert x.dim() == 2, "x should be 2 dimensional"
        assert labels.dim() == 1, "labels should be 1 dimensional"
        assert x.size(0) == labels.size(0), "x and labels have different number of elements along dim 0"

        ctx.save_for_backward(x, labels)

        ctx.reduction = reduction
        ctx.logits_multiplier = logits_multiplier
        ctx.kernel_backend_backward = kernel_backend_backward
        ctx.BLOCK_SIZE_B_backward = BLOCK_SIZE_B_backward
        ctx.BLOCK_SIZE_V_backward = BLOCK_SIZE_V_backward

        loss = torch.tensor(0, device=x.device, dtype=torch.float32)
        cross_entropy_forward_triton(
            x=x,
            labels=labels,
            loss=loss,
            logits_multiplier=logits_multiplier,
            BLOCK_SIZE_B=BLOCK_SIZE_B_forward,
            BLOCK_SIZE_V=BLOCK_SIZE_V_forward,
            reduction=reduction,
        )

        return loss

    @staticmethod
    @ensure_contiguous
    def backward(ctx, output_grad: torch.Tensor) -> tuple[torch.Tensor | None]:
        x, labels = ctx.saved_tensors
        logits_multiplier = ctx.logits_multiplier

        x_grad = _softmax_forward(
            x=x,
            logits_multiplier=logits_multiplier,
            kernel_backend=ctx.kernel_backend_backward,
            BLOCK_SIZE_B=ctx.BLOCK_SIZE_B_backward,
            BLOCK_SIZE_H=ctx.BLOCK_SIZE_V_backward,
        )

        # I am lazy :)
        # but this can be fused inside the above kernel
        x_grad[torch.arange(labels.size(0), device=labels.device), labels] -= 1
        if logits_multiplier != 1:
            x_grad *= logits_multiplier

        if ctx.reduction == "mean":
            x_grad /= x.size(0)

        return x_grad, *[None] * 8


def cross_entropy_cute(
    x: torch.Tensor,
    labels: torch.Tensor,
    reduction: str = "mean",
    logits_multiplier: float = 1,
    BLOCK_SIZE_B_forward: int = CutoTuneParameter(),
    BLOCK_SIZE_V_forward: int = CutoTuneParameter(),
    kernel_backend_backward: str = CutoTuneParameter(),
    BLOCK_SIZE_B_backward: int = CutoTuneParameter(),
    BLOCK_SIZE_V_backward: int = CutoTuneParameter(),
) -> torch.Tensor:
    return _CrossEntropy_Cute.apply(
        x,
        labels,
        reduction,
        logits_multiplier,
        BLOCK_SIZE_B_forward,
        BLOCK_SIZE_V_forward,
        kernel_backend_backward,
        BLOCK_SIZE_B_backward,
        BLOCK_SIZE_V_backward,
    )
