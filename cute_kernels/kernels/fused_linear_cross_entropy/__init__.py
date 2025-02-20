import torch
import torch.nn.functional as F

from ...cutotune import CutoTuneParameter
from ...math import ceil_divide, get_next_power_of_2
from ...utils import ensure_contiguous
from ..cross_entropy import cross_entropy_forward_triton
from ..softmax import _forward as _softmax_forward
from .torch_implementation import fused_linear_cross_entropy_torch


class _FusedLinearCrossEntropy_Cute(torch.autograd.Function):
    @staticmethod
    @ensure_contiguous
    def forward(
        ctx,
        x: torch.Tensor,
        weight: torch.Tensor,
        labels: torch.Tensor,
        reduction: str,
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
        assert x.size(-1) == weight.size(-1)

        batch_size, hidden_size = x.size()
        vocab_size = weight.size(0)

        # NOTE chunking is copied from liger kernel
        memory_increase_factor = ceil_divide(hidden_size, vocab_size)
        # chunk_size needed to reduce memory increase back to 1
        chunk_size = get_next_power_of_2(ceil_divide(batch_size, memory_increase_factor))
        num_chunks = ceil_divide(batch_size, chunk_size)

        ctx.reduction = reduction
        ctx.kernel_backend_backward = kernel_backend_backward
        ctx.BLOCK_SIZE_B_backward = BLOCK_SIZE_B_backward
        ctx.BLOCK_SIZE_V_backward = BLOCK_SIZE_V_backward
        ctx.chunk_size = chunk_size
        ctx.num_chunks = num_chunks

        loss = torch.tensor(0, device=x.device, dtype=torch.float32)
        x_grad = torch.empty_like(x)
        weight_grad = torch.zeros_like(weight)
        start = 0

        for _ in range(num_chunks):
            end = start + chunk_size

            _x = x[start:end]
            _logits = F.linear(_x, weight).contiguous()

            _labels = labels[start:end].contiguous()

            cross_entropy_forward_triton(
                x=_logits,
                labels=_labels,
                loss=loss,
                V=vocab_size,
                BLOCK_SIZE_B=BLOCK_SIZE_B_forward,
                BLOCK_SIZE_V=BLOCK_SIZE_V_forward,
                reduction="sum",
            )

            # _logits_grad = _softmax_forward(
            #     x=_logits,
            #     kernel_backend=ctx.kernel_backend_backward,
            #     BLOCK_SIZE_B=ctx.BLOCK_SIZE_B_backward,
            #     BLOCK_SIZE_H=ctx.BLOCK_SIZE_V_backward,
            # )

            # # I am lazy :)
            # # but this can be fused inside the above kernel
            # _logits_grad[torch.arange(_labels.size(0), device=_labels.device), _labels] -= 1

            # if reduction == "mean":
            #     _logits_grad /= batch_size

            # x_grad[start:end] = _logits_grad @ weight
            # torch.addmm(weight_grad, _logits_grad.T, _x, alpha=1, beta=1, out=weight_grad)

            start = end

        if reduction == "mean":
            loss /= batch_size

        ctx.save_for_backward(x_grad, weight_grad)

        return loss

    @staticmethod
    def backward(ctx, output_grad: torch.Tensor) -> tuple[torch.Tensor | None]:
        # x_grad, weight_grad = ctx.saved_tensors
        # print(output_grad)

        # if output_grad != 1:
        #     x_grad *= output_grad
        #     weight_grad *= output_grad

        return [None] * 9
        # return output_grad, output_grad, *[None] * 7


def fused_linear_cross_entropy_cute(
    x: torch.Tensor,
    weight: torch.Tensor,
    labels: torch.Tensor,
    reduction: str = "mean",
    BLOCK_SIZE_B_forward: int = CutoTuneParameter(),
    BLOCK_SIZE_V_forward: int = CutoTuneParameter(),
    kernel_backend_backward: str = CutoTuneParameter(),
    BLOCK_SIZE_B_backward: int = CutoTuneParameter(),
    BLOCK_SIZE_V_backward: int = CutoTuneParameter(),
) -> torch.Tensor:
    return _FusedLinearCrossEntropy_Cute.apply(
        x,
        weight,
        labels,
        reduction,
        BLOCK_SIZE_B_forward,
        BLOCK_SIZE_V_forward,
        kernel_backend_backward,
        BLOCK_SIZE_B_backward,
        BLOCK_SIZE_V_backward,
    )
