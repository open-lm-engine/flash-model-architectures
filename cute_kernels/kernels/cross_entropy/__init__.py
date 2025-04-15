import torch

from ...math import ceil_divide
from ...utils import ensure_contiguous
from .torch_implementation import cross_entropy_torch
from .triton_implementation import _cross_entropy_forward_backward_triton_kernel


class _CrossEntropy_Cute(torch.autograd.Function):
    @staticmethod
    @ensure_contiguous
    def forward(ctx, x: torch.Tensor, labels: torch.Tensor, reduction: str, logits_multiplier: float) -> torch.Tensor:
        assert reduction in ["sum", "mean"]
        assert x.dim() == 2, "x should be 2 dimensional"
        assert labels.dim() == 1, "labels should be 1 dimensional"

        B, V = x.size()
        assert labels.size(0) == B, "x and labels have different number of elements along dim 0"

        BLOCK_SIZE_B = 4
        BLOCK_SIZE_V = 256

        loss = torch.tensor(0, device=x.device, dtype=torch.float32)
        x_grad = torch.empty_like(x)

        with torch.cuda.device(x.device):
            _cross_entropy_forward_backward_triton_kernel[(ceil_divide(num_elements, BLOCK_SIZE_B),)](
                x_ptr=x,
                labels_ptr=labels,
                loss_ptr=loss,
                x_grad_ptr=x_grad,
                logits_multiplier=logits_multiplier,
                B=B,
                V=V,
                BLOCK_SIZE_B=BLOCK_SIZE_B,
                BLOCK_SIZE_V=BLOCK_SIZE_V,
                reduction=reduction,
            )

        ctx.save_for_backward(x_grad)

        return loss

    @staticmethod
    def backward(ctx, output_grad: torch.Tensor) -> tuple[torch.Tensor | None]:
        x_grad = ctx.saved_tensors[0]
        x_grad *= output_grad

        return x_grad, *[None] * 3


def cross_entropy_cute(
    x: torch.Tensor, labels: torch.Tensor, reduction: str = "mean", logits_multiplier: float = 1
) -> torch.Tensor:
    return _CrossEntropy_Cute.apply(x, labels, reduction, logits_multiplier)
