import torch

from ...utils import ensure_contiguous
from .torch_implementation import rnn_torch
from .triton_implementation import rnn_backward_triton, rnn_forward_triton


class _RNN_Cute(torch.autograd.Function):
    @staticmethod
    @ensure_contiguous
    def forward(
        ctx,
        input: torch.Tensor,
        weight: torch.Tensor,
        input_state: torch.Tensor | None,
        BLOCK_SIZE_B_forward: int,
        BLOCK_SIZE_B_backward: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        output = torch.empty_like(input)

        rnn_forward_triton(
            input=input,
            weight=weight,
            output=output,
            input_state=input_state,
            allow_tf32=True,
            BLOCK_SIZE_B=BLOCK_SIZE_B_forward,
        )

        ctx.save_for_backward(weight, output, input_state)
        ctx.BLOCK_SIZE_B_backward = BLOCK_SIZE_B_backward

        return output

    @staticmethod
    @ensure_contiguous
    def backward(ctx, output_grad: torch.Tensor) -> tuple[torch.Tensor]:
        weight, output, input_state = ctx.saved_tensors

        input_grad = torch.empty_like(output)
        weight_grad = torch.empty_like(weight)

        rnn_backward_triton(
            weight=weight,
            output=output,
            input_state=input_state,
            output_grad=output_grad,
            input_grad=input_grad,
            weight_grad=weight_grad,
            allow_tf32=True,
            BLOCK_SIZE_B=ctx.BLOCK_SIZE_B_backward,
        )

        return input_grad, weight_grad, *[None] * 3


def rnn_cute(
    input: torch.Tensor,
    weight: torch.Tensor,
    input_state: torch.Tensor | None = None,
    BLOCK_SIZE_B_forward: int = 32,
    BLOCK_SIZE_B_backward: int = 32,
) -> torch.Tensor:
    return _RNN_Cute.apply(input, weight, input_state, BLOCK_SIZE_B_forward, BLOCK_SIZE_B_backward)
