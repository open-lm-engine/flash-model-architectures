# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import torch
import torch.nn as nn

from ...cutotune import CutoTuneParameter
from ...kernel_backend import KernelBackend
from ...math import divide_if_divisible
from ...torch_math import sigmoid, tanh
from ...utils import ensure_contiguous
from .triton_implementation import (
    gru_backward_triton,
    gru_forward_triton,
    gru_varlen_backward_triton,
    gru_varlen_forward_triton,
)


class _GRU_Cute(torch.autograd.Function):
    @staticmethod
    @ensure_contiguous
    def forward(
        ctx,
        input: torch.Tensor,
        weight: torch.Tensor,
        forget_input: torch.Tensor,
        forget_weight: torch.Tensor,
        reset_input: torch.Tensor,
        reset_weight: torch.Tensor,
        input_state: torch.Tensor | None,
        gradient_clipping: float | None,
        cu_seqlens: torch.Tensor | None,
        max_seqlen: torch.Tensor | int | None,
        kernel_backend: KernelBackend | CutoTuneParameter,
    ) -> torch.Tensor:
        assert input.dim() in [3, 4]
        assert weight.dim() == 3
        assert kernel_backend == KernelBackend.triton or isinstance(kernel_backend, CutoTuneParameter)

        N, H = input.size()[-2:]
        assert weight.size() == (N, H, H)

        if gradient_clipping is not None and gradient_clipping < 0:
            gradient_clipping = -gradient_clipping

        output = torch.empty_like(input)
        forget_gate = torch.empty_like(input)
        reset_gate = torch.empty_like(input)
        output_update = torch.empty_like(input)

        kwargs = {
            "input": input,
            "weight": weight,
            "forget_input": forget_input,
            "forget_weight": forget_weight,
            "forget_gate": forget_gate,
            "reset_input": reset_input,
            "reset_weight": reset_weight,
            "reset_gate": reset_gate,
            "output_update": output_update,
            "input_state": input_state,
            "output": output,
        }

        if cu_seqlens is None:
            assert max_seqlen is None
            gru_forward_triton(**kwargs)
        else:
            assert max_seqlen is not None
            is_max_seqlen_tensor = isinstance(max_seqlen, torch.Tensor)

            gru_varlen_forward_triton(
                **kwargs,
                cu_seqlens=cu_seqlens,
                max_seqlen_tensor=max_seqlen if is_max_seqlen_tensor else None,
                max_seqlen=None if is_max_seqlen_tensor else max_seqlen,
            )

        ctx.save_for_backward(
            weight,
            forget_weight,
            forget_gate,
            reset_weight,
            reset_gate,
            output_update,
            output,
            input_state,
            cu_seqlens,
            max_seqlen,
        )

        ctx.gradient_clipping = gradient_clipping

        return output

    @staticmethod
    @ensure_contiguous
    def backward(ctx, output_grad: torch.Tensor) -> tuple[torch.Tensor | None]:
        (
            weight,
            forget_weight,
            forget_gate,
            reset_weight,
            reset_gate,
            output_update,
            output,
            input_state,
            cu_seqlens,
            max_seqlen,
        ) = ctx.saved_tensors

        input_grad = torch.empty_like(output)
        forget_input_grad = torch.empty_like(output)
        reset_input_grad = torch.empty_like(output)
        weight_grad = torch.zeros_like(weight, dtype=torch.float32)
        forget_weight_grad = torch.zeros_like(weight, dtype=torch.float32)
        reset_weight_grad = torch.zeros_like(weight, dtype=torch.float32)

        kwargs = {
            "weight": weight,
            "output": output,
            "forget_weight": forget_weight,
            "forget_gate": forget_gate,
            "forget_input_grad": forget_input_grad,
            "forget_weight_grad": forget_weight_grad,
            "reset_weight": reset_weight,
            "reset_gate": reset_gate,
            "reset_input_grad": reset_input_grad,
            "reset_weight_grad": reset_weight_grad,
            "output_update": output_update,
            "input_state": input_state,
            "output_grad": output_grad,
            "input_grad": input_grad,
            "weight_grad": weight_grad,
            "gradient_clipping": ctx.gradient_clipping,
            "output": output,
        }

        if cu_seqlens is None:
            gru_backward_triton(**kwargs)
        else:
            is_max_seqlen_tensor = isinstance(max_seqlen, torch.Tensor)

            gru_varlen_backward_triton(
                **kwargs,
                cu_seqlens=cu_seqlens,
                max_seqlen_tensor=max_seqlen if is_max_seqlen_tensor else None,
                max_seqlen=None if is_max_seqlen_tensor else max_seqlen,
            )

        weight_grad = weight_grad.type_as(weight)
        forget_weight_grad = forget_weight_grad.type_as(forget_weight)
        reset_weight_grad = reset_weight_grad.type_as(reset_weight)

        return (
            input_grad,
            weight_grad,
            forget_input_grad,
            forget_weight_grad,
            reset_input_grad,
            reset_weight_grad,
            *[None] * 5,
        )


def gru_cute(
    input: torch.Tensor,
    weight: torch.Tensor,
    forget_input: torch.Tensor,
    forget_weight: torch.Tensor,
    reset_input: torch.Tensor,
    reset_weight: torch.Tensor,
    input_state: torch.Tensor | None = None,
    gradient_clipping: float | None = None,
    cu_seqlens: torch.Tensor | None = None,
    max_seqlen: torch.Tensor | int | None = None,
    *,
    kernel_backend: KernelBackend | CutoTuneParameter = KernelBackend.triton,
) -> torch.Tensor:
    """computes multihead RNN: tanh(`input_state` @ `weight` + `input`)

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
        torch.Tensor: output tensor of shape (B, S, N, H)
    """

    if kernel_backend == KernelBackend.torch:
        if gradient_clipping is not None and gradient_clipping < 0:
            gradient_clipping = -gradient_clipping

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
                forget_gate = input_state.unsqueeze(-2) @ forget_weight.unsqueeze(0) + forget_input[:, s].unsqueeze(-2)
                forget_gate = sigmoid(forget_gate)

                # (B, N, 1, H) = (B, N, 1, H) @ (1, N, H, H) + (B, N, 1, H)
                reset_gate = input_state.unsqueeze(-2) @ reset_weight.unsqueeze(0) + reset_input[:, s].unsqueeze(-2)
                reset_gate = sigmoid(reset_gate)

                # (B, N, 1, H) = [(B, N, 1, H) * (B, N, 1, H)] @ (1, N, H, H) + (B, N, 1, H)
                possible_new_state = (input_state.unsqueeze(-2) * reset_gate) @ weight.unsqueeze(0) + input[
                    :, s
                ].unsqueeze(-2)
                possible_new_state = tanh(possible_new_state)

                input_state = forget_gate * input_state.unsqueeze(-2) + (1 - forget_gate) * possible_new_state
                input_state = input_state.squeeze(-2)

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

                new_state = input_state[unfinished].unsqueeze(-2)
                offset_unfinished = offset[unfinished]

                # don't update the finished sequences
                # (B, N, 1, H) = (B, N, 1, H) @ (1, N, H, H) + (B, N, 1, H)
                forget_gate = new_state @ forget_weight.unsqueeze(0) + forget_input[offset_unfinished].unsqueeze(-2)
                forget_gate = sigmoid(forget_gate)

                # (B, N, 1, H) = (B, N, 1, H) @ (1, N, H, H) + (B, N, 1, H)
                reset_gate = new_state @ reset_weight.unsqueeze(0) + reset_input[offset_unfinished].unsqueeze(-2)
                reset_gate = sigmoid(reset_gate)

                # (B, N, 1, H) = [(B, N, 1, H) * (B, N, 1, H)] @ (1, N, H, H) + (B, N, 1, H)
                possible_new_state = (new_state * reset_gate) @ weight.unsqueeze(0) + input[
                    offset_unfinished
                ].unsqueeze(-2)
                possible_new_state = tanh(possible_new_state)

                new_state = forget_gate * new_state + (1 - forget_gate) * possible_new_state
                new_state = new_state.squeeze(-2)

                output[offset_unfinished] = new_state
                input_state[unfinished] = new_state
    else:
        output = _GRU_Cute.apply(
            input,
            weight,
            forget_input,
            forget_weight,
            reset_input,
            reset_weight,
            input_state,
            gradient_clipping,
            cu_seqlens,
            max_seqlen,
            kernel_backend,
        )

    return output


class GRU(nn.Module):
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

        self.input_projection = nn.Linear(input_size, 3 * state_size, bias=add_bias)
        self.state_weight = nn.Parameter(torch.empty(self.num_heads, self.state_head_dim, self.state_head_dim))
        self.forget_weight = nn.Parameter(torch.empty(self.num_heads, self.state_head_dim, self.state_head_dim))
        self.reset_weight = nn.Parameter(torch.empty(self.num_heads, self.state_head_dim, self.state_head_dim))
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
        input, forget_gate, reset_gate = input.chunk(3, dim=-1)

        input, forget_gate, reset_gate = [
            i.view(*input.size()[:-1], self.num_heads, self.state_head_dim) for i in (input, forget_gate, reset_gate)
        ]

        if input_state is not None:
            input_state = input_state.view(-1, self.num_heads, self.state_head_dim)

        input = gru_cute(
            input=input,
            weight=self.state_weight,
            forget_input=forget_gate,
            forget_weight=self.forget_weight,
            reset_input=reset_gate,
            reset_weight=self.reset_weight,
            input_state=input_state,
            gradient_clipping=self.gradient_clipping,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            kernel_backend=kernel_backend,
        )

        del forget_gate, reset_gate

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
