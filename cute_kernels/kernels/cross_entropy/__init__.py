import torch

from ...cutotune import CutoTuneParameter
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
        assert x.size(0) == labels.size(0), "x and labels have different number of elements along dim 0"

        loss = torch.tensor(0, device=x.device, dtype=torch.float32)
        x_grad = torch.empty_like(x)

        num_elements, vocab_size = x.size()
        BLOCK_SIZE_B = 4
        BLOCK_SIZE_V = 256

        with torch.cuda.device(x.device):
            _cross_entropy_forward_backward_triton_kernel[(ceil_divide(num_elements, BLOCK_SIZE_B),)](
                x_ptr=x,
                labels_ptr=labels,
                loss_ptr=loss,
                x_grad_ptr=x_grad,
                logits_multiplier=logits_multiplier,
                B=num_elements,
                V=vocab_size,
                BLOCK_SIZE_B=BLOCK_SIZE_B,
                BLOCK_SIZE_V=BLOCK_SIZE_V,
                reduction=reduction,
            )

        # Meta is on fucking drugs
        # torch compiler doesn't work without this :/
        if torch.compiler.is_compiling():
            x_grad += 0

        ctx.save_for_backward(x_grad)

        return loss

    @staticmethod
    def backward(ctx, output_grad: torch.Tensor) -> tuple[torch.Tensor | None]:
        x_grad = ctx.saved_tensors[0]
        x_grad *= output_grad

        return x_grad, *[None] * 5


def cross_entropy_cute(
    x: torch.Tensor, labels: torch.Tensor, reduction: str = "mean", logits_multiplier: float = 1
) -> torch.Tensor:
    return _CrossEntropy_Cute.apply(x, labels, reduction, logits_multiplier)
