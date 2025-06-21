# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import torch
import torch.nn as nn

from ...cutotune import CutoTuneParameter
from ...kernel_backend import KernelBackend
from ...math import divide_if_divisible
from ...torch_math import tanh
from ...utils import ensure_contiguous
from .triton_implementation import diagonal_rnn_backward_triton, diagonal_rnn_forward_triton


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
    ) -> torch.Tensor:
        assert input.dim() in [3, 4]
        assert weight.dim() == 3

        N, H = input.size()[-2:]
        assert weight.size() == (N, H, H)

        if gradient_clipping is not None and gradient_clipping < 0:
            gradient_clipping = -gradient_clipping

        output = torch.empty_like(input)

        kwargs = {"input": input, "weight": weight, "input_state": input_state, "output": output}

        if cu_seqlens is None:
            assert max_seqlen is None

            if H == 1:
                diagonal_rnn_forward_triton(**kwargs)
            else:
                rnn_forward_triton(**kwargs)
        else:
            assert max_seqlen is not None
            is_max_seqlen_tensor = isinstance(max_seqlen, torch.Tensor)

            kwargs["cu_seqlens"] = cu_seqlens
            kwargs["max_seqlen_tensor"] = max_seqlen if is_max_seqlen_tensor else None
            kwargs["max_seqlen"] = None if is_max_seqlen_tensor else max_seqlen

            if H == 1:
                diagonal_rnn_varlen_forward_triton(**kwargs)
            else:
                rnn_varlen_forward_triton(**kwargs)

        ctx.save_for_backward(weight, output, input_state, cu_seqlens, max_seqlen)
        ctx.gradient_clipping = gradient_clipping

        return output

    @staticmethod
    @ensure_contiguous
    def backward(ctx, output_grad: torch.Tensor) -> tuple[torch.Tensor]:
        weight, output, input_state, cu_seqlens, max_seqlen = ctx.saved_tensors
        input_grad = torch.empty_like(output)
        weight_grad = torch.zeros_like(weight, dtype=torch.float32)

        H = weight.size(-1)

        kwargs = {
            "weight": weight,
            "output": output,
            "input_state": input_state,
            "output_grad": output_grad,
            "input_grad": input_grad,
            "weight_grad": weight_grad,
            "gradient_clipping": ctx.gradient_clipping,
        }

        if cu_seqlens is None:
            if H == 1:
                diagonal_rnn_backward_triton(**kwargs)
            else:
                rnn_backward_triton(**kwargs)
        else:
            is_max_seqlen_tensor = isinstance(max_seqlen, torch.Tensor)

            kwargs["cu_seqlens"] = cu_seqlens
            kwargs["max_seqlen_tensor"] = max_seqlen if is_max_seqlen_tensor else None
            kwargs["max_seqlen"] = None if is_max_seqlen_tensor else max_seqlen

            if H == 1:
                diagonal_rnn_varlen_backward_triton(**kwargs)
            else:
                rnn_varlen_backward_triton(**kwargs)

        return input_grad, weight_grad, *[None] * 8


class _GradientClipping(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, gradient_clipping: float) -> torch.Tensor:
        ctx.gradient_clipping = gradient_clipping
        return x

    @staticmethod
    def backward(ctx, x_grad: torch.Tensor) -> tuple[torch.Tensor, None]:
        gradient_clipping = ctx.gradient_clipping
        x_grad = x_grad.clip(-gradient_clipping, gradient_clipping)
        return x_grad, None


def hippo_rnn_cute(
    input: torch.Tensor,
    weight: torch.Tensor,
    hippo_weight: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    input_state: torch.Tensor | None = None,
    hippo_state: torch.Tensor | None = None,
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
        hippo_weight (torch.Tensor): weight tensor of shape (N, D)
        A (torch.Tensor): weight tensor of shape (D, D)
        B (torch.Tensor): weight tensor of shape (D, 1)
        input_state (torch.Tensor | None, optional): starting state of shape (B, N, H), None means starting state
            is 0 tensor. Defaults to None.
        hippo_state (torch.Tensor | None, optional): starting state of shape (B, N, H, D), None means starting state
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

    if kernel_backend == KernelBackend.torch:
        if gradient_clipping is not None and gradient_clipping < 0:
            gradient_clipping = -gradient_clipping

        output = torch.empty_like(input)

        weight = weight.unsqueeze(0)
        hippo_weight = hippo_weight.unsqueeze(0).unsqueeze(-1)
        input = input.unsqueeze(-2)

        if cu_seqlens is None:
            assert max_seqlen is None
            B, S, N, _, H = input.size()
            D = A.size(0)

            if input_state is None:
                input_state = torch.zeros(B, N, H, device=input.device, dtype=input.dtype)

            if hippo_state is None:
                hippo_state = torch.zeros(B, N, H, D, device=input.device, dtype=input.dtype)

            input_state = input_state.unsqueeze(-2)
            hippo_state = hippo_state.unsqueeze(-2)

            # input -> (B, S, N, 1, H)
            # weight -> (1, N, H, H)
            # input_state -> (B, N, 1, H)

            for s in range(S):
                # (B, N, 1, H) @ (1, N, H, H) + (B, N, 1, H)
                input_state = input_state @ weight + input[:, s]
                # (B, N, H, D) @ (1, N, D, 1) + (B, N, 1, H)
                input_state = (hippo_state @ hippo_weight).transpose(-1, -2) + input_state
                input_state = tanh(input_state)

                if gradient_clipping is not None:
                    input_state = _GradientClipping.apply(input_state, gradient_clipping)

                output[:, s] = input_state.squeeze(-2)
        else:
            raise NotImplementedError()
    else:
        output = _RNN_Cute.apply(input, weight, input_state, gradient_clipping, cu_seqlens, max_seqlen)

    return output


class HiPPO_RNN(nn.Module):
    def __init__(
        self,
        input_size: int,
        state_size: int,
        output_size: int,
        num_heads: int,
        hippo_size: int,
        hippo_measure: str,
        add_bias: bool,
        gradient_clipping: float | None,
    ) -> None:
        super().__init__()

        assert hippo_size == 1
        assert hippo_measure == "legs"

        self.num_heads = num_heads
        self.gradient_clipping = gradient_clipping
        self.hippo_size = hippo_size
        self.state_head_dim = divide_if_divisible(state_size, self.num_heads)

        assert self.state_head_dim == 1

        self.input_projection = nn.Linear(input_size, state_size, bias=add_bias)
        self.state_weight = nn.Parameter(torch.empty(self.num_heads, self.state_head_dim, self.state_head_dim))
        self.hippo_weight = nn.Parameter(torch.empty(self.num_heads, self.hippo_size))
        self.output_projection = nn.Linear(state_size, output_size, bias=False)

        self.register_buffer("A", torch.empty(hippo_size, hippo_size))
        self.register_buffer("B", torch.empty(hippo_size, 1))

        self.reset_parameters()

    def forward(
        self,
        input: torch.Tensor,
        input_state: torch.Tensor | None = None,
        hippo_state: torch.Tensor | None = None,
        cu_seqlens: torch.Tensor | None = None,
        max_seqlen: int | None = None,
        kernel_backend: KernelBackend = KernelBackend.triton,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        input = self.input_projection(input)
        input = input.view(*input.size()[:-1], self.num_heads, self.state_head_dim)

        if input_state is not None:
            input_state = input_state.view(-1, self.num_heads, self.state_head_dim)

        weight, hippo_weight = self.state_weight.chunk(2)

        input = hippo_rnn_cute(
            input=input,
            weight=weight,
            hippo_weight=hippo_weight,
            A=self.A,
            B=self.B,
            input_state=input_state,
            hippo_state=hippo_state,
            gradient_clipping=self.gradient_clipping,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            kernel_backend=kernel_backend,
        )

        del input_state

        if cu_seqlens is None:
            output_state = input[:, -1]
        else:
            output_state = input[cu_seqlens[1:] - 1]

        output_state = output_state.view(output_state.size(0), -1)

        input = input.view(*input.size()[:-2], -1)
        input = self.output_projection(input)

        return input, output_state

    @torch.no_grad()
    def reset_parameters(self) -> None:
        nn.init.normal_(self.state_weight)

        self.A.fill_(1)
        self.B.fill_(1)
