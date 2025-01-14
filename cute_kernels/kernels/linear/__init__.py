import torch

from ...cutotune import CutoTuneParameter
from .torch_implementation import linear_torch
from .triton_implementation import linear_forward_triton


class _Linear_Cute(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        input: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
        BLOCK_SIZE_M: int,
        BLOCK_SIZE_K: int,
        BLOCK_SIZE_N: int,
    ) -> torch.Tensor:
        ctx.save_for_backward(input, weight, bias)
        output = torch.empty(*input.size()[:-1], weight.size(0), dtype=input.dtype, device=input.device)

        linear_forward_triton(
            input=input,
            weight=weight,
            bias=bias,
            output=output,
            BLOCK_SIZE_M=BLOCK_SIZE_M,
            BLOCK_SIZE_K=BLOCK_SIZE_K,
            BLOCK_SIZE_N=BLOCK_SIZE_N,
        )

        return output

    @staticmethod
    def backward(ctx, output_grad: torch.Tensor) -> torch.Tensor:
        input, weight, bias = ctx.saved_tensors
        return input, weight, bias, None, None, None


def linear_cute(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    BLOCK_SIZE_M: int = CutoTuneParameter(),
    BLOCK_SIZE_K: int = CutoTuneParameter(),
    BLOCK_SIZE_N: int = CutoTuneParameter(),
) -> torch.Tensor:
    return _Linear_Cute.apply(input, weight, bias, BLOCK_SIZE_M, BLOCK_SIZE_K, BLOCK_SIZE_N)
