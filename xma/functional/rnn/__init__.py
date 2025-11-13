# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import torch

from ...custom_op import CustomOp, ctx_save_for_backward
from ...enums import KernelBackend
from ...torch_utils import clip_gradients, tanh
from ...utils import (
    empty_like_contiguous,
    get_max_seqlen_and_max_seqlen_tensor,
    is_triton_available,
    zeros_like_contiguous,
)


if is_triton_available():
    from .triton_implementation import rnn_backward_triton, rnn_forward_triton


class _RNN(CustomOp):
    @staticmethod
    def forward_backward_torch(
        input: torch.Tensor,
        weight: torch.Tensor,
        input_state: torch.Tensor | None,
        gradient_clipping: float | None,
        cu_seqlens: torch.Tensor | None,
        max_seqlen: torch.Tensor | int | None,
    ) -> torch.Tensor:
        Nw = weight.size(0)

        if cu_seqlens is None:
            B, S, Nx, H = input.size()
        else:
            T, Nx, H = input.size()
            B = cu_seqlens.size(0) - 1

        N = max(Nx, Nw)
        W = weight[None, ...]

        if input_state is None:
            input_state = torch.zeros(B, N, H, device=input.device, dtype=input.dtype)

        if cu_seqlens is None:
            output = torch.empty(B, S, N, H, device=input.device, dtype=input.dtype)

            for s in range(S):
                # (B, N, 1, H) = (B, N, 1, H) @ (1, Nw, H, H) + (B, Nx, 1, H)
                input_state = input_state[..., None, :] @ W + input[:, s, :, None, :]
                input_state = tanh(input_state)
                input_state = input_state.squeeze(-2)

                if gradient_clipping is not None:
                    input_state = clip_gradients(input_state, gradient_clipping)

                output[:, s] = input_state
        else:
            input_state = input_state.clone()
            output = torch.empty(T, N, H, device=input.device, dtype=input.dtype)

            start = cu_seqlens[:-1]
            end = cu_seqlens[1:]

            for s in range(max_seqlen):
                offset = start + s
                unfinished = offset < end
                offset_unfinished = offset[unfinished]

                # don't update the finished sequences
                # (B, N, 1, H) @ (1, Nw, H, H) + (B, Nx, 1, H)
                new_state = input_state[unfinished][..., None, :] @ W + input[offset_unfinished][:, s, None, :]
                new_state = tanh(new_state)
                new_state = new_state.squeeze(-2)

                if gradient_clipping is not None:
                    new_state = clip_gradients(new_state, gradient_clipping)

                input_state[unfinished] = new_state
                output[offset_unfinished] = new_state

        return output

    @staticmethod
    def forward_triton(
        ctx,
        input: torch.Tensor,
        weight: torch.Tensor,
        input_state: torch.Tensor | None,
        gradient_clipping: float | None,
        cu_seqlens: torch.Tensor | None,
        max_seqlen: torch.Tensor | int | None,
    ) -> torch.Tensor:
        output = empty_like_contiguous(input)
        max_seqlen_tensor, max_seqlen = get_max_seqlen_and_max_seqlen_tensor(max_seqlen)

        rnn_forward_triton(
            input=input,
            weight=weight,
            input_state=input_state,
            output=output,
            cu_seqlens=cu_seqlens,
            max_seqlen_tensor=max_seqlen_tensor,
            max_seqlen=max_seqlen,
        )

        ctx_save_for_backward(ctx, weight, output, input_state, cu_seqlens, max_seqlen_tensor)
        ctx.max_seqlen = max_seqlen
        ctx.gradient_clipping = gradient_clipping

        return output

    @staticmethod
    def backward_triton(ctx, output_grad: torch.Tensor) -> tuple[torch.Tensor]:
        weight, output, input_state, cu_seqlens, max_seqlen_tensor = ctx.saved_tensors
        input_grad = empty_like_contiguous(output)
        weight_grad = zeros_like_contiguous(weight, dtype=torch.float32)

        rnn_backward_triton(
            weight=weight,
            output=output,
            input_state=input_state,
            output_grad=output_grad,
            input_grad=input_grad,
            weight_grad=weight_grad,
            cu_seqlens=cu_seqlens,
            max_seqlen_tensor=max_seqlen_tensor,
            max_seqlen=ctx.max_seqlen,
            gradient_clipping=ctx.gradient_clipping,
        )

        weight_grad = weight_grad.type_as(weight)

        return input_grad, weight_grad, *[None] * 4


def rnn(
    input: torch.Tensor,
    weight: torch.Tensor,
    input_state: torch.Tensor | None = None,
    gradient_clipping: float | None = None,
    cu_seqlens: torch.Tensor | None = None,
    max_seqlen: torch.Tensor | int | None = None,
    *,
    kernel_backend: KernelBackend | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
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

    Returns:
        tuple[torch.Tensor, torch.Tensor]: output tensor of shape (B, S, N, H) and output state tensor of shape (B, N, H)
    """

    if cu_seqlens is None:
        assert max_seqlen is None
        assert input.dim() == 4

        B, _, Nx, H = input.size()
    else:
        assert max_seqlen is not None
        assert input.dim() == 3

        _, Nx, H = input.size()
        B = cu_seqlens.size(0) - 1

    Nw = weight.size(0)
    N = max(Nx, Nw)

    assert weight.dim() == 3
    assert weight.size() == (Nw, H, H)
    assert N % Nx == 0
    assert N % Nw == 0

    if input_state is not None:
        assert input_state.size() == (B, N, H)

    if gradient_clipping is not None and gradient_clipping < 0:
        gradient_clipping = -gradient_clipping

    output = _RNN.run(
        input=input,
        weight=weight,
        input_state=input_state,
        gradient_clipping=gradient_clipping,
        cu_seqlens=cu_seqlens,
        max_seqlen=max_seqlen,
        kernel_backend=kernel_backend,
    )

    output_state = output[:, -1] if cu_seqlens is None else output[cu_seqlens[1:] - 1]

    return output, output_state
