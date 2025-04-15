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
        gradient_clipping: float | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if gradient_clipping is not None:
            assert gradient_clipping > 0, "gradient_clipping should be a positive number"

        BLOCK_SIZE_B = 32
        output = torch.empty_like(input)

        rnn_forward_triton(
            input=input,
            weight=weight,
            output=output,
            input_state=input_state,
            allow_tf32=True,
            BLOCK_SIZE_B=BLOCK_SIZE_B,
        )

        ctx.save_for_backward(weight, output, input_state)
        ctx.gradient_clipping = gradient_clipping

        return output

    @staticmethod
    @ensure_contiguous
    def backward(ctx, output_grad: torch.Tensor) -> tuple[torch.Tensor]:
        weight, output, input_state = ctx.saved_tensors
        input_grad = torch.empty_like(output)
        weight_grad = torch.empty_like(weight)

        BLOCK_SIZE_B = 32

        rnn_backward_triton(
            weight=weight,
            output=output,
            input_state=input_state,
            output_grad=output_grad,
            input_grad=input_grad,
            weight_grad=weight_grad,
            gradient_clipping=ctx.gradient_clipping,
            allow_tf32=True,
            BLOCK_SIZE_B=BLOCK_SIZE_B,
        )

        return input_grad, weight_grad, *[None] * 3


def rnn_cute(
    input: torch.Tensor,
    weight: torch.Tensor,
    input_state: torch.Tensor | None = None,
    gradient_clipping: float | None = None,
) -> torch.Tensor:
    return _RNN_Cute.apply(input, weight, input_state, gradient_clipping)
