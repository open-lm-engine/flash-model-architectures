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
        kernel_backend: str,
        BLOCK_SIZE_B: int,
        BLOCK_SIZE_V: int,
    ) -> torch.Tensor:
        assert reduction in ["sum", "mean"]
        assert x.dim() == 2, "x should be 2 dimensional"
        assert labels.dim() == 1, "labels should be 1 dimensional"
        assert x.size(0) == labels.size(0), "x and labels have different number of elements along dim 0"

        ctx.reduction = reduction
        ctx.logits_multiplier = logits_multiplier
        ctx.kernel_backend = kernel_backend
        ctx.BLOCK_SIZE_B = BLOCK_SIZE_B
        ctx.BLOCK_SIZE_V = BLOCK_SIZE_V

        loss = torch.tensor(0, device=x.device, dtype=torch.float32)
        x_grad = torch.empty_like(x)

        cross_entropy_forward_triton(
            x=x,
            labels=labels,
            loss=loss,
            x_grad=x_grad,
            logits_multiplier=logits_multiplier,
            BLOCK_SIZE_B=BLOCK_SIZE_B,
            BLOCK_SIZE_V=BLOCK_SIZE_V,
            reduction=reduction,
        )

        # I am lazy :)
        # but this can be fused inside the above kernel
        if logits_multiplier != 1:
            x_grad *= logits_multiplier

        if ctx.reduction == "mean":
            x_grad /= x.size(0)

        ctx.save_for_backward(x_grad)

        return loss

    @staticmethod
    @ensure_contiguous
    def backward(ctx, output_grad: torch.Tensor) -> tuple[torch.Tensor | None]:
        x_grad = ctx.saved_tensors[0]

        return x_grad, *[None] * 8


def cross_entropy_cute(
    x: torch.Tensor,
    labels: torch.Tensor,
    reduction: str = "mean",
    logits_multiplier: float = 1,
    kernel_backend: str = CutoTuneParameter(),
    BLOCK_SIZE_B: int = CutoTuneParameter(),
    BLOCK_SIZE_V: int = CutoTuneParameter(),
) -> torch.Tensor:
    return _CrossEntropy_Cute.apply(
        x,
        labels,
        reduction,
        logits_multiplier,
        kernel_backend,
        BLOCK_SIZE_B,
        BLOCK_SIZE_V,
    )
