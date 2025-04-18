import torch

from ...math import ceil_divide, get_next_power_of_2
from ...utils import ensure_contiguous
from ..cross_entropy import _cross_entropy_forward_backward_triton_kernel
from .torch_implementation import fused_linear_cross_entropy_torch


class _FusedLinearCrossEntropy_Cute(torch.autograd.Function):
    @staticmethod
    @ensure_contiguous
    def forward(
        ctx, x: torch.Tensor, weight: torch.Tensor, labels: torch.Tensor, reduction: str, logits_multiplier: float
    ) -> torch.Tensor:
        assert reduction in ["sum", "mean"]
        assert x.dim() == 2, "x should be 2 dimensional"
        assert labels.dim() == 1, "labels should be 1 dimensional"
        assert x.size(0) == labels.size(0), "x and labels have different number of elements along dim 0"
        assert x.size(-1) == weight.size(-1)

        batch_size, hidden_size = x.size()
        vocab_size = weight.size(0)

        # NOTE chunking is copied from liger kernel
        memory_increase_factor = ceil_divide(vocab_size, hidden_size)
        # chunk_size needed to reduce memory increase back to 1
        chunk_size = get_next_power_of_2(ceil_divide(batch_size, memory_increase_factor))
        num_chunks = ceil_divide(batch_size, chunk_size)

        loss = torch.tensor(0, device=x.device, dtype=torch.float32)
        x_grad = torch.empty_like(x)
        weight_grad = torch.zeros_like(weight)

        BLOCK_SIZE_B = 4
        BLOCK_SIZE_V = 256

        for i in range(num_chunks):
            start = i * chunk_size
            end = (i + 1) * chunk_size
            end = min(end, batch_size)

            _x = x[start:end]
            _logits = (_x @ weight.T).contiguous()

            _logits_grad = torch.empty_like(_logits)
            _labels = labels[start:end].contiguous()

            with torch.cuda.device(x.device):
                _cross_entropy_forward_backward_triton_kernel[ceil_divide(_logits.size(0), BLOCK_SIZE_B),](
                    x_ptr=_logits,
                    labels_ptr=_labels,
                    loss_ptr=loss,
                    x_grad_ptr=_logits_grad,
                    logits_multiplier=logits_multiplier,
                    B=_logits.size(0),
                    V=vocab_size,
                    BLOCK_SIZE_B=BLOCK_SIZE_B,
                    BLOCK_SIZE_V=BLOCK_SIZE_V,
                    reduction="sum",
                )

            x_grad[start:end] = _logits_grad @ weight
            torch.addmm(weight_grad, _logits_grad.T, _x, alpha=1, beta=1, out=weight_grad)

        if reduction == "mean":
            loss /= batch_size
            x_grad /= batch_size
            weight_grad /= batch_size

        ctx.save_for_backward(x_grad, weight_grad)

        return loss

    @staticmethod
    def backward(ctx, output_grad: torch.Tensor) -> tuple[torch.Tensor | None]:
        x_grad, weight_grad = ctx.saved_tensors

        x_grad *= output_grad
        weight_grad *= output_grad

        return x_grad, weight_grad, *[None] * 3


def fused_linear_cross_entropy_cute(
    x: torch.Tensor, weight: torch.Tensor, labels: torch.Tensor, reduction: str = "mean", logits_multiplier: float = 1
) -> torch.Tensor:
    return _FusedLinearCrossEntropy_Cute.apply(x, weight, labels, reduction, logits_multiplier)
