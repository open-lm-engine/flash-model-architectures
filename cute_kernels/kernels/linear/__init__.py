import torch

from ...cutotune import CutoTuneParameter
from ...utils import ensure_contiguous
from .torch_implementation import linear_torch
from .triton_implementation import linear_backward_triton, linear_forward_triton


class _Linear_Cute(torch.autograd.Function):
    @staticmethod
    @ensure_contiguous
    def forward(
        ctx,
        input: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
        use_tf32: bool,
        BLOCK_SIZE_M_forward: int,
        BLOCK_SIZE_K_forward: int,
        BLOCK_SIZE_N_forward: int,
        BLOCK_SIZE_M_backward: int,
        BLOCK_SIZE_K_backward: int,
        BLOCK_SIZE_N_backward: int,
    ) -> torch.Tensor:
        ctx.save_for_backward(input, weight)
        ctx.has_bias = bias is not None
        ctx.use_tf32 = use_tf32
        ctx.BLOCK_SIZE_M_backward = BLOCK_SIZE_M_backward
        ctx.BLOCK_SIZE_K_backward = BLOCK_SIZE_K_backward
        ctx.BLOCK_SIZE_N_backward = BLOCK_SIZE_N_backward

        output = torch.empty(*input.size()[:-1], weight.size(0), dtype=input.dtype, device=input.device)

        linear_forward_triton(
            input=input,
            weight=weight,
            bias=bias,
            output=output,
            use_tf32=use_tf32,
            BLOCK_SIZE_M=BLOCK_SIZE_M_forward,
            BLOCK_SIZE_K=BLOCK_SIZE_K_forward,
            BLOCK_SIZE_N=BLOCK_SIZE_N_forward,
            num_warps=CutoTuneParameter(),
            num_stages=CutoTuneParameter(),
        )

        return output

    @staticmethod
    @ensure_contiguous
    def backward(ctx, output_grad: torch.Tensor) -> torch.Tensor:
        input, weight = ctx.saved_tensors

        input_grad = torch.empty_like(input)
        weight_grad = torch.empty_like(weight)
        bias_grad = None
        if ctx.has_bias:
            bias_grad = torch.empty(weight.size(0), device=weight.device, dtype=weight.dtype)

        # input_grad = output_grad @ weight
        # weight_grad = output_grad.T @ input
        # if bias_grad is not None:
        #     bias_grad = output_grad.sum(dim=0)

        linear_backward_triton(
            input=input,
            output_grad=output_grad,
            weight_grad=weight_grad,
            bias_grad=bias_grad,
            use_tf32=ctx.use_tf32,
            BLOCK_SIZE_M=ctx.BLOCK_SIZE_M_backward,
            BLOCK_SIZE_K=ctx.BLOCK_SIZE_K_backward,
            BLOCK_SIZE_N=ctx.BLOCK_SIZE_N_backward,
            num_warps=CutoTuneParameter(),
            num_stages=CutoTuneParameter(),
        )

        return input_grad, weight_grad, bias_grad, *[None] * 7


def linear_cute(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    use_tf32: bool = True,
    BLOCK_SIZE_M_forward: int = CutoTuneParameter(),
    BLOCK_SIZE_K_forward: int = CutoTuneParameter(),
    BLOCK_SIZE_N_forward: int = CutoTuneParameter(),
    BLOCK_SIZE_M_backward: int = CutoTuneParameter(),
    BLOCK_SIZE_K_backward: int = CutoTuneParameter(),
    BLOCK_SIZE_N_backward: int = CutoTuneParameter(),
) -> torch.Tensor:
    return _Linear_Cute.apply(
        input,
        weight,
        bias,
        use_tf32,
        BLOCK_SIZE_M_forward,
        BLOCK_SIZE_K_forward,
        BLOCK_SIZE_N_forward,
        BLOCK_SIZE_M_backward,
        BLOCK_SIZE_K_backward,
        BLOCK_SIZE_N_backward,
    )
