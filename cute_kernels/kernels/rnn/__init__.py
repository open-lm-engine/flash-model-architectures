import torch

from ...utils import ensure_contiguous
from .torch_implementation import rnn_torch
from .triton_implementation import (
    rnn_backward_triton,
    rnn_forward_triton,
    rnn_varlen_backward_triton,
    rnn_varlen_forward_triton,
    scalar_rnn_forward_triton,
)


class _RNN_Cute(torch.autograd.Function):
    @staticmethod
    @ensure_contiguous
    def forward(
        ctx,
        input: torch.Tensor,
        weight: torch.Tensor,
        input_state: torch.Tensor | None,
        gradient_clipping: float | None,
        cu_seqlens: torch.Tensor | None,
        max_seqlen: torch.Tensor | int | None,
        activation_function: str,
        relu_negative_slope: float | None,
        BLOCK_SIZE_B_forward: int,
        BLOCK_SIZE_B_backward: int,
    ) -> torch.Tensor:
        assert activation_function in ["leaky_relu", "sigmoid", "tanh"]
        assert input.dim() == 4
        assert weight.dim() == 3

        N, H = input.size()[-2:]
        assert weight.size() == (N, H, H)

        if activation_function == "leaky_relu":
            assert relu_negative_slope is not None
        else:
            assert relu_negative_slope is None

        if gradient_clipping is not None and gradient_clipping < 0:
            gradient_clipping = -gradient_clipping

        if cu_seqlens is None:
            assert max_seqlen is None
        else:
            assert max_seqlen is not None
            is_max_seqlen_tensor = isinstance(max_seqlen, torch.Tensor)

        output = torch.empty_like(input)

        if H == 1:
            if cu_seqlens is None:
                scalar_rnn_forward_triton(
                    input=input,
                    weight=weight,
                    input_state=input_state,
                    output=output,
                    activation_function=activation_function,
                    relu_negative_slope=relu_negative_slope,
                    BLOCK_SIZE_B=BLOCK_SIZE_B_forward,
                    BLOCK_SIZE_N=BLOCK_SIZE_N_forward,
                )
            else:
                pass
        else:
            if cu_seqlens is None:
                rnn_forward_triton(
                    input=input,
                    weight=weight,
                    input_state=input_state,
                    output=output,
                    activation_function=activation_function,
                    relu_negative_slope=relu_negative_slope,
                    BLOCK_SIZE_B=BLOCK_SIZE_B_forward,
                )
            else:
                rnn_varlen_forward_triton(
                    input=input,
                    weight=weight,
                    input_state=input_state,
                    output=output,
                    cu_seqlens=cu_seqlens,
                    max_seqlen_tensor=max_seqlen if is_max_seqlen_tensor else None,
                    max_seqlen=None if is_max_seqlen_tensor else max_seqlen,
                    activation_function=activation_function,
                    relu_negative_slope=relu_negative_slope,
                    BLOCK_SIZE_B=BLOCK_SIZE_B_forward,
                )

        ctx.save_for_backward(weight, output, input_state, cu_seqlens, max_seqlen)
        ctx.gradient_clipping = gradient_clipping
        ctx.activation_function = activation_function
        ctx.relu_negative_slope = relu_negative_slope
        ctx.BLOCK_SIZE_B_backward = BLOCK_SIZE_B_backward

        return output

    @staticmethod
    @ensure_contiguous
    def backward(ctx, output_grad: torch.Tensor) -> tuple[torch.Tensor]:
        weight, output, input_state, cu_seqlens, max_seqlen = ctx.saved_tensors
        input_grad = torch.empty_like(output)
        weight_grad = torch.empty_like(weight)

        BLOCK_SIZE_B = ctx.BLOCK_SIZE_B_backward
        gradient_clipping = ctx.gradient_clipping
        activation_function = ctx.activation_function
        relu_negative_slope = ctx.relu_negative_slope

        if cu_seqlens is None:
            rnn_backward_triton(
                weight=weight,
                output=output,
                input_state=input_state,
                output_grad=output_grad,
                input_grad=input_grad,
                weight_grad=weight_grad,
                gradient_clipping=gradient_clipping,
                activation_function=activation_function,
                relu_negative_slope=relu_negative_slope,
                BLOCK_SIZE_B=BLOCK_SIZE_B,
            )
        else:
            is_max_seqlen_tensor = isinstance(max_seqlen, torch.Tensor)

            rnn_varlen_backward_triton(
                weight=weight,
                output=output,
                input_state=input_state,
                output_grad=output_grad,
                input_grad=input_grad,
                weight_grad=weight_grad,
                cu_seqlens=cu_seqlens,
                max_seqlen_tensor=max_seqlen if is_max_seqlen_tensor else None,
                max_seqlen=None if is_max_seqlen_tensor else max_seqlen,
                gradient_clipping=gradient_clipping,
                activation_function=activation_function,
                relu_negative_slope=relu_negative_slope,
                BLOCK_SIZE_B=BLOCK_SIZE_B,
            )

        return input_grad, weight_grad, *[None] * 8


def rnn_cute(
    input: torch.Tensor,
    weight: torch.Tensor,
    input_state: torch.Tensor | None = None,
    gradient_clipping: float | None = None,
    cu_seqlens: torch.Tensor | None = None,
    max_seqlen: torch.Tensor | int | None = None,
    activation_function: str = "tanh",
    relu_negative_slope: float | None = None,
    *,
    BLOCK_SIZE_B_forward: int = 32,
    BLOCK_SIZE_B_backward: int = 32,
) -> torch.Tensor:
    """computes multihead RNN recurrent update over the sequence length: tanh(`input_state` @ `weight` + `input`)

    Args:
        input (torch.Tensor): input tensor of shape (B, S, N, H) where N is the number of heads and H is the head
            dimension. Should have shape (T, N, H) and `cu_seqlens` should be passed.
        weight (torch.Tensor): weight tensor of shape (N, H, H)
        input_state (torch.Tensor | None, optional): starting state of shape (B, N, H), None means starting state
            is 0 tensor. Defaults to None.
        gradient_clipping (float | None, optional): gradient clipping for the state gradient in backward, None
            implies no clipping. Defaults to None.
        cu_seqlens (torch.Tensor | None, optional): cumulative sequence length (must contain 0 as first element). Defaults to None.
        max_seqlen (torch.Tensor | int | None, optional): max sequence length in the batch. Defaults to None.
        activation_function (str): activation function, can be "tanh" or "leaky_relu". Defaults to "tanh".
        relu_negative_slope (float): negative slope for leaky_relu. Defaults to "tanh".
        BLOCK_SIZE_B_forward (int, optional): block size for forward along batch dimension for forward. Defaults to
            32.
        BLOCK_SIZE_B_backward (int, optional): block size for backward along batch dimension for backward. Defaults to
            32.

    Returns:
        torch.Tensor: output tensor of shape (B, S, N, H)
    """

    return _RNN_Cute.apply(
        input,
        weight,
        input_state,
        gradient_clipping,
        cu_seqlens,
        max_seqlen,
        activation_function,
        relu_negative_slope,
        BLOCK_SIZE_B_forward,
        BLOCK_SIZE_B_backward,
    )
