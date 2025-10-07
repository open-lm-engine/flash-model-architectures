# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import torch
import torch.nn as nn

from ...cutotune import CutoTuneParameter
from ...enums import KernelBackend
from ...math import divide_if_divisible
from ...torch_math import clip_gradients, tanh
from ...utils import ensure_contiguous
from .triton_implementation import (
    rnn_backward_triton,
    rnn_forward_triton,
    rnn_varlen_backward_triton,
    rnn_varlen_forward_triton,
)


class _RNN(torch.autograd.Function):
    @staticmethod
    @ensure_contiguous
    def forward(
        ctx,
        input: torch.Tensor,
        weight: torch.Tensor,
        input_state: torch.Tensor | None,
        gradient_clipping: float | None,
    ) -> torch.Tensor:
        output = torch.empty_like(input)

        rnn_forward_triton(input=input, weight=weight, input_state=input_state, output=output)

        ctx.save_for_backward(weight, output, input_state)
        ctx.gradient_clipping = gradient_clipping

        return output

    @staticmethod
    @ensure_contiguous
    def backward(ctx, output_grad: torch.Tensor) -> tuple[torch.Tensor]:
        weight, output, input_state = ctx.saved_tensors

        input_grad = torch.empty_like(output)
        weight_grad = torch.zeros_like(weight, dtype=torch.float32)

        rnn_backward_triton(
            weight=weight,
            output=output,
            input_state=input_state,
            output_grad=output_grad,
            input_grad=input_grad,
            weight_grad=weight_grad,
            gradient_clipping=ctx.gradient_clipping,
        )

        return input_grad, weight_grad, None, None


class _RNN_Varlen(torch.autograd.Function):
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
    ) -> torch.Tensor:
        output = torch.empty_like(input)

        is_max_seqlen_tensor = isinstance(max_seqlen, torch.Tensor)

        rnn_varlen_forward_triton(
            input=input,
            weight=weight,
            input_state=input_state,
            output=output,
            cu_seqlens=cu_seqlens,
            max_seqlen_tensor=max_seqlen if is_max_seqlen_tensor else None,
            max_seqlen=None if is_max_seqlen_tensor else max_seqlen,
        )

        if is_max_seqlen_tensor:
            ctx.save_for_backward(weight, output, input_state, cu_seqlens, max_seqlen)
        else:
            ctx.save_for_backward(weight, output, input_state, cu_seqlens)
            ctx.max_seqlen = max_seqlen

        ctx.gradient_clipping = gradient_clipping
        ctx.is_max_seqlen_tensor = is_max_seqlen_tensor

        return output

    @staticmethod
    @ensure_contiguous
    def backward(ctx, output_grad: torch.Tensor) -> tuple[torch.Tensor]:
        is_max_seqlen_tensor = ctx.is_max_seqlen_tensor

        if is_max_seqlen_tensor:
            weight, output, input_state, cu_seqlens, max_seqlen = ctx.saved_tensors
        else:
            weight, output, input_state, cu_seqlens = ctx.saved_tensors
            max_seqlen = ctx.max_seqlen

        input_grad = torch.empty_like(output)
        weight_grad = torch.zeros_like(weight, dtype=torch.float32)

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
            gradient_clipping=ctx.gradient_clipping,
        )

        return input_grad, weight_grad, *[None] * 4


def rnn(
    input: torch.Tensor,
    weight: torch.Tensor,
    input_state: torch.Tensor | None = None,
    gradient_clipping: float | None = None,
    cu_seqlens: torch.Tensor | None = None,
    max_seqlen: torch.Tensor | int | None = None,
    *,
    kernel_backend: KernelBackend | CutoTuneParameter = KernelBackend.triton,
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
        kernel_backend (KernelBackend | CutoTuneParameter, optional): kernel backend to prioritize.
            Defaults to KernelBackend.triton.

    Returns:
        torch.Tensor: output tensor of shape (B, S, N, H)
    """

    assert input.dim() in [3, 4]
    assert weight.dim() == 3

    N, H = input.size()[-2:]
    assert weight.size() == (N, H, H)

    if gradient_clipping is not None and gradient_clipping < 0:
        gradient_clipping = -gradient_clipping

    if kernel_backend == KernelBackend.torch:
        output = torch.empty_like(input)

        if cu_seqlens is None:
            assert max_seqlen is None
            B, S, N, H = input.size()

            if input_state is None:
                input_state = torch.zeros(B, N, H, device=input.device, dtype=input.dtype)

            # input -> (B, S, N, H)
            # weight -> (N, H, H)
            # input_state -> (B, N, H)

            for s in range(S):
                # (B, N, 1, H) = (B, N, 1, H) @ (1, N, H, H) + (B, N, 1, H)
                input_state = input_state.unsqueeze(-2) @ weight.unsqueeze(0) + input[:, s].unsqueeze(-2)
                input_state = tanh(input_state)
                input_state = input_state.squeeze(-2)

                if gradient_clipping is not None:
                    input_state = clip_gradients(input_state, gradient_clipping)

                output[:, s] = input_state
        else:
            assert max_seqlen is not None
            B = cu_seqlens.numel() - 1
            _, N, H = input.size()

            if input_state is None:
                input_state = torch.zeros(B, N, H, device=input.device, dtype=input.dtype)
            else:
                input_state = input_state.clone()

            # input -> (cu_seqlens[-1], N, H)
            # weight -> (N, H, H)
            # input_state -> (B, N, H)

            start = cu_seqlens[:-1]
            end = cu_seqlens[1:]

            for s in range(max_seqlen):
                offset = start + s
                unfinished = offset < end

                offset_unfinished = offset[unfinished]

                # don't update the finished sequences
                # (B, N, 1, H) @ (1, N, H, H) + (B, N, 1, H)
                new_state = input_state[unfinished].unsqueeze(-2) @ weight.unsqueeze(0) + input[
                    offset_unfinished
                ].unsqueeze(-2)

                new_state = tanh(new_state)

                if gradient_clipping is not None:
                    new_state = clip_gradients(new_state, gradient_clipping)

                new_state = new_state.squeeze(-2)

                output[offset_unfinished] = new_state
                input_state[unfinished] = new_state
    else:
        if cu_seqlens is None:
            output = _RNN.apply(input, weight, input_state, gradient_clipping)
        else:
            output = _RNN_Varlen.apply(input, weight, input_state, gradient_clipping, cu_seqlens, max_seqlen)

    return output


class RNN(nn.Module):
    def __init__(
        self,
        input_size: int,
        state_size: int,
        output_size: int,
        num_heads: int,
        add_bias: bool,
        gradient_clipping: float | None,
    ) -> None:
        super().__init__()

        self.num_heads = num_heads
        self.gradient_clipping = gradient_clipping
        self.state_head_dim = divide_if_divisible(state_size, self.num_heads)

        self.input_projection = nn.Linear(input_size, state_size, bias=add_bias)
        self.state_weight = nn.Parameter(torch.empty(self.num_heads, self.state_head_dim, self.state_head_dim))
        self.output_projection = nn.Linear(state_size, output_size, bias=False)

        self.reset_parameters()

    def forward(
        self,
        input: torch.Tensor,
        input_state: torch.Tensor | None = None,
        cu_seqlens: torch.Tensor | None = None,
        max_seqlen: int | None = None,
        kernel_backend: KernelBackend = KernelBackend.triton,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        input = self.input_projection(input)
        input = input.view(*input.size()[:-1], self.num_heads, self.state_head_dim)

        if input_state is not None:
            input_state = input_state.view(-1, self.num_heads, self.state_head_dim)

        input = rnn(
            input=input,
            weight=self.state_weight,
            input_state=input_state,
            gradient_clipping=self.gradient_clipping,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            kernel_backend=kernel_backend,
        )

        if cu_seqlens is None:
            input_state = input[:, -1]
        else:
            input_state = input[cu_seqlens[1:] - 1]

        input_state = input_state.view(input_state.size(0), -1)

        input = input.view(*input.size()[:-2], -1)
        input = self.output_projection(input)

        return input, input_state

    @torch.no_grad()
    def reset_parameters(self) -> None:
        nn.init.normal_(self.state_weight)
