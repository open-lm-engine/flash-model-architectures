import torch

from ...math import ceil_divide, get_next_power_of_2
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
        BLOCK_SIZE_B_forward: int,
        BLOCK_SIZE_B_backward: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if gradient_clipping is not None:
            assert gradient_clipping > 0, "gradient_clipping should be a positive number"

        output = torch.empty_like(input)

        rnn_forward_triton(
            input=input,
            weight=weight,
            input_state=input_state,
            output=output,
            BLOCK_SIZE_B=BLOCK_SIZE_B_forward,
        )

        ctx.save_for_backward(weight, output, input_state)
        ctx.gradient_clipping = gradient_clipping
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
            gradient_clipping=ctx.gradient_clipping,
            BLOCK_SIZE_B=ctx.BLOCK_SIZE_B_backward,
        )

        return input_grad, weight_grad, *[None] * 4


def rnn_cute(
    input: torch.Tensor,
    weight: torch.Tensor,
    input_state: torch.Tensor | None = None,
    gradient_clipping: float | None = None,
    *,
    BLOCK_SIZE_B_forward: int = 32,
    BLOCK_SIZE_B_backward: int = 32,
) -> torch.Tensor:
    """computes multihead RNN: tanh(`input_state` @ `weight` + `input`)

    Args:
        input (torch.Tensor): input tensor of shape (B, S, N, H) where N is the number of heads and H is the head
            dimension
        weight (torch.Tensor): weight tensor of shape (N, H, H)
        input_state (torch.Tensor | None, optional): starting state of shape (B, N, H), None means starting state
            is 0 tensor. Defaults to None.
        gradient_clipping (float | None, optional): gradient clipping for the state gradient in backward, None
            implies no clipping. Defaults to None.
        BLOCK_SIZE_B_forward (int, optional): block size for forward along batch dimension for forward. Defaults to
            32.
        BLOCK_SIZE_B_backward (int, optional): block size for backward along batch dimension for backward. Defaults to
            32.

    Returns:
        torch.Tensor: output tensor of shape (B, S, N, H)
    """

    return _RNN_Cute.apply(input, weight, input_state, gradient_clipping, BLOCK_SIZE_B_forward, BLOCK_SIZE_B_backward)
