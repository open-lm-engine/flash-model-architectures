# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import numpy as np
import torch
import torch.nn as nn

from ...cutotune import CutoTuneParameter
from ...kernel_backend import KernelBackend
from ...math import divide_if_divisible
from ...torch_math import tanh
from ...utils import ensure_contiguous
from .triton_implementation import hippo_rnn_backward_triton, hippo_rnn_forward_triton


class _HiPPO_RNN_Cute(torch.autograd.Function):
    @staticmethod
    @ensure_contiguous
    def forward(
        ctx,
        input: torch.Tensor,
        weight: torch.Tensor,
        hippo_weight: torch.Tensor,
        compress_weight: torch.Tensor,
        hippo_A: torch.Tensor,
        hippo_B: torch.Tensor,
        input_state: torch.Tensor | None,
        hippo_state: torch.Tensor | None,
        gradient_clipping: float | None,
        cu_seqlens: torch.Tensor | None,
        max_seqlen: torch.Tensor | int | None,
    ) -> torch.Tensor:
        assert input.dim() in [3, 4]
        assert weight.dim() == 3

        B, S, N, H = input.size()
        D = hippo_weight.size(1)

        assert weight.size() == (N, H, H)
        assert hippo_weight.size() == (N, D, H)
        assert hippo_A.size() == (D, D)
        assert hippo_B.size(0) == D

        if gradient_clipping is not None and gradient_clipping < 0:
            gradient_clipping = -gradient_clipping

        output = torch.empty_like(input)
        hippo_output = torch.empty(B, S, N, D, device=input.device, dtype=input.dtype)

        kwargs = {
            "input": input,
            "weight": weight,
            "hippo_weight": hippo_weight,
            "compress_weight": compress_weight,
            "hippo_A": hippo_A,
            "hippo_B": hippo_B,
            "input_state": input_state,
            "hippo_state": hippo_state,
            "output": output,
            "hippo_output": hippo_output,
        }

        if cu_seqlens is None:
            assert max_seqlen is None
            hippo_rnn_forward_triton(**kwargs)
        else:
            raise NotImplementedError()

        ctx.save_for_backward(
            weight,
            hippo_weight,
            compress_weight,
            hippo_A,
            hippo_B,
            output,
            hippo_output,
            input_state,
            hippo_state,
            cu_seqlens,
            max_seqlen,
        )

        ctx.gradient_clipping = gradient_clipping

        return output

    @staticmethod
    @ensure_contiguous
    def backward(ctx, output_grad: torch.Tensor) -> tuple[torch.Tensor]:
        (
            weight,
            hippo_weight,
            compress_weight,
            hippo_A,
            hippo_B,
            output,
            hippo_output,
            input_state,
            hippo_state,
            cu_seqlens,
            max_seqlen,
        ) = ctx.saved_tensors

        input_grad = torch.empty_like(output)
        weight_grad = torch.zeros_like(weight, dtype=torch.float32)
        hippo_weight_grad = torch.zeros_like(hippo_weight, dtype=torch.float32)
        compress_weight_grad = torch.zeros_like(compress_weight, dtype=torch.float32)

        kwargs = {
            "weight": weight,
            "hippo_weight": hippo_weight,
            "compress_weight": compress_weight,
            "hippo_A": hippo_A,
            "hippo_B": hippo_B,
            "output": output,
            "hippo_output": hippo_output,
            "input_state": input_state,
            "hippo_state": hippo_state,
            "output_grad": output_grad,
            "input_grad": input_grad,
            "weight_grad": weight_grad,
            "hippo_weight_grad": hippo_weight_grad,
            "compress_weight_grad": compress_weight_grad,
            "gradient_clipping": ctx.gradient_clipping,
        }

        if cu_seqlens is None:
            hippo_rnn_backward_triton(**kwargs)
        else:
            raise NotImplementedError()

        return input_grad, weight_grad, hippo_weight_grad, compress_weight_grad, *[None] * 7


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
    compress_weight: torch.Tensor,
    hippo_A: torch.Tensor,
    hippo_B: torch.Tensor,
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
        hippo_weight (torch.Tensor): weight tensor of shape (N, D, H)
        compress_weight (torch.Tensor): weight tensor of shape (N, H)
        hippo_A (torch.Tensor): weight tensor of shape (D, D)
        hippo_B (torch.Tensor): weight tensor of shape (D,)
        input_state (torch.Tensor | None, optional): starting state of shape (B, N, H), None means starting state
            is 0 tensor. Defaults to None.
        hippo_state (torch.Tensor | None, optional): starting state of shape (B, N, D), None means starting state
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

        D = hippo_A.size(0)
        I = torch.eye(D, device=hippo_A.device, dtype=hippo_A.dtype)

        if cu_seqlens is None:
            assert max_seqlen is None
            B, S, N, H = input.size()

            if input_state is None:
                input_state = torch.zeros(B, N, H, device=input.device, dtype=input.dtype)

            if hippo_state is None:
                hippo_state = torch.zeros(B, N, D, device=input.device, dtype=input.dtype)

            # input -> (B, S, N, H)
            # weight -> (N, H, H)
            # input_state -> (B, N, H)
            # hippo_state -> (B, N, D)

            for s in range(S):
                # (B, N, 1, H) = (B, N, 1, H) @ (1, N, H, H) + (B, N, 1, H)
                input_state = input_state.unsqueeze(-2) @ weight.unsqueeze(0) + input[:, s].unsqueeze(-2)
                # (B, N, 1, H) = (B, N, 1, D) @ (1, N, D, H) + (B, N, 1, H)
                input_state = hippo_state.unsqueeze(-2) @ hippo_weight.unsqueeze(0) + input_state
                input_state = tanh(input_state)

                # (B, N, 1, 1) = (B, N, 1, H) @ (1, N, H, 1)
                compressed_input = input_state @ compress_weight.unsqueeze(0).unsqueeze(-1)

                # (B, N, D, 1) = (1, 1, D, D) @ (B, N, D, 1) + (1, 1, D, 1) @ (B, N, 1, 1)
                hippo_state = (I - hippo_A / (s + 1)).unsqueeze(0).unsqueeze(0) @ hippo_state.unsqueeze(-1) + (
                    hippo_B / (s + 1)
                ).unsqueeze(0).unsqueeze(0).unsqueeze(-1) @ compressed_input

                input_state = input_state.squeeze(-2)
                hippo_state = hippo_state.squeeze(-1)

                if gradient_clipping is not None:
                    input_state = _GradientClipping.apply(input_state, gradient_clipping)

                output[:, s] = input_state
        else:
            raise NotImplementedError()
    else:
        output = _HiPPO_RNN_Cute.apply(
            input,
            weight,
            hippo_weight,
            compress_weight,
            hippo_A,
            hippo_B,
            input_state,
            hippo_state,
            gradient_clipping,
            cu_seqlens,
            max_seqlen,
        )

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

        assert hippo_measure == "legs"

        self.num_heads = num_heads
        self.gradient_clipping = gradient_clipping
        self.hippo_size = hippo_size
        self.state_head_dim = divide_if_divisible(state_size, self.num_heads)

        self.input_projection = nn.Linear(input_size, state_size, bias=add_bias)

        self.state_weight = nn.Parameter(torch.empty(self.num_heads, self.state_head_dim, self.state_head_dim))
        self.hippo_weight = nn.Parameter(torch.empty(self.num_heads, self.hippo_size, self.state_head_dim))
        self.compress_weight = nn.Parameter(torch.empty(self.num_heads, self.state_head_dim))

        self.output_projection = nn.Linear(state_size, output_size, bias=False)

        self.register_buffer("A", torch.empty(hippo_size, hippo_size))
        self.register_buffer("B", torch.empty(hippo_size))

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

        if hippo_state is not None:
            hippo_state = hippo_state.view(-1, self.num_heads, self.hippo_size)

        input = hippo_rnn_cute(
            input=input,
            weight=self.state_weight,
            hippo_weight=self.hippo_weight,
            compress_weight=self.compress_weight,
            hippo_A=self.A,
            hippo_B=self.B,
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
        nn.init.normal_(self.hippo_weight)
        nn.init.normal_(self.compress_weight)

        arange = np.arange(self.hippo_size, dtype=np.float64)
        r = 2 * arange + 1

        B = np.sqrt(r)

        M = np.tril(r) - np.diag(arange)
        T = B[:, None]
        T_inv = 1 / B[None, :]
        r = 2 * arange + 1

        A = T * T_inv * M

        A = torch.tensor(A)
        B = torch.tensor(B)

        self.A.copy_(A)
        self.B.copy_(B)
