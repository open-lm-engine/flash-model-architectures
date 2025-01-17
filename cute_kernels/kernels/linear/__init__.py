import torch

from ...cutotune import CutoTuneParameter
from ...utils import ensure_contiguous
from .torch_implementation import linear_torch
from .triton_implementation import linear_forward_triton


class _Linear_Cute(torch.autograd.Function):
    @staticmethod
    @ensure_contiguous
    def forward(
        ctx,
        input: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
        use_tf32: bool,
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
            use_tf32=use_tf32,
            BLOCK_SIZE_M=BLOCK_SIZE_M,
            BLOCK_SIZE_K=BLOCK_SIZE_K,
            BLOCK_SIZE_N=BLOCK_SIZE_N,
            num_warps=CutoTuneParameter(),
            num_stages=CutoTuneParameter(),
        )

        return output

    @staticmethod
    @ensure_contiguous
    def backward(ctx, output_grad: torch.Tensor) -> torch.Tensor:
        input, weight, bias = ctx.saved_tensors
        return input, weight, bias, *[None] * 4


def linear_cute(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    use_tf32: bool = True,
    BLOCK_SIZE_M: int = CutoTuneParameter(),
    BLOCK_SIZE_K: int = CutoTuneParameter(),
    BLOCK_SIZE_N: int = CutoTuneParameter(),
) -> torch.Tensor:
    return _Linear_Cute.apply(input, weight, bias, use_tf32, BLOCK_SIZE_M, BLOCK_SIZE_K, BLOCK_SIZE_N)
