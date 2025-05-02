import torch

from ...math import ceil_divide, get_next_power_of_2
from ...utils import ensure_contiguous
from .torch_implementation import rnn_torch
from .triton_implementation import rnn_backward_triton_kernel, rnn_forward_triton_kernel


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

        B, S, N, H = input.size()

        BLOCK_SIZE_H = get_next_power_of_2(H)
        BLOCK_SIZE_H = max(16, BLOCK_SIZE_H)

        with torch.cuda.device(input.device):
            rnn_forward_triton_kernel[ceil_divide(B, BLOCK_SIZE_B_forward), N](
                input_ptr=input,
                input_stride_b=input.stride(0),
                input_stride_s=input.stride(1),
                input_stride_n=input.stride(2),
                weight_ptr=weight,
                weight_stride_n=weight.stride(0),
                weight_stride_h=weight.stride(1),
                has_input_state=input_state is not None,
                input_state_ptr=input_state,
                output_ptr=output,
                B=B,
                S=S,
                H=H,
                BLOCK_SIZE_B=BLOCK_SIZE_B_forward,
                BLOCK_SIZE_H=BLOCK_SIZE_H,
            )

        ctx.save_for_backward(weight, output, input_state)
        ctx.gradient_clipping = gradient_clipping
        ctx.BLOCK_SIZE_B_backward = BLOCK_SIZE_B_backward
        ctx.BLOCK_SIZE_H = BLOCK_SIZE_H

        return output

    @staticmethod
    @ensure_contiguous
    def backward(ctx, output_grad: torch.Tensor) -> tuple[torch.Tensor]:
        weight, output, input_state = ctx.saved_tensors
        input_grad = torch.empty_like(output)
        weight_grad = torch.empty_like(weight)

        B, S, N, H = output.size()
        BLOCK_SIZE_B = ctx.BLOCK_SIZE_B_backward
        BLOCK_SIZE_H = ctx.BLOCK_SIZE_H
        gradient_clipping = ctx.gradient_clipping

        with torch.cuda.device(output.device):
            rnn_backward_triton_kernel[ceil_divide(B, BLOCK_SIZE_B), N](
                weight_ptr=weight,
                weight_stride_n=weight.stride(0),
                weight_stride_h=weight.stride(1),
                output_ptr=output,
                output_stride_b=output.stride(0),
                output_stride_s=output.stride(1),
                output_stride_n=output.stride(2),
                has_input_state=input_state is not None,
                input_state_ptr=input_state,
                input_state_stride_b=None if input_state is None else input_state.stride(0),
                input_state_stride_n=None if input_state is None else input_state.stride(1),
                output_grad_ptr=output_grad,
                input_grad_ptr=input_grad,
                weight_grad_ptr=weight_grad,
                has_gradient_clipping=gradient_clipping is not None,
                gradient_clipping=gradient_clipping,
                B=B,
                S=S,
                H=H,
                BLOCK_SIZE_B=BLOCK_SIZE_B,
                BLOCK_SIZE_H=BLOCK_SIZE_H,
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
