import torch
import torch.nn as nn
import torch.nn.functional as F

from ...cutotune import CutoTuneParameter
from .triton_implementation import rnn_forward_triton


class _RNN_Cute(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        state_weight: torch.Tensor,
        input_state: torch.Tensor | None,
        BLOCK_SIZE_B: int,
        BLOCK_SIZE_H: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        y = torch.empty_like(x)

        rnn_forward_triton(
            x=x,
            weight=state_weight,
            y=y,
            input_state=input_state,
            BLOCK_SIZE_B=BLOCK_SIZE_B,
            BLOCK_SIZE_H=BLOCK_SIZE_H,
        )

        return y

    @staticmethod
    def backward(ctx, output_grad: torch.Tensor) -> tuple[torch.Tensor]: ...


def rnn_cute(
    input: torch.Tensor,
    state_weight: torch.Tensor,
    input_state: torch.Tensor | None = None,
    BLOCK_SIZE_B: int = CutoTuneParameter(),
    BLOCK_SIZE_H: int = CutoTuneParameter(),
) -> torch.Tensor:
    return _RNN_Cute.apply(input, state_weight, input_state, BLOCK_SIZE_B, BLOCK_SIZE_H)


class RNNCute(nn.Module):
    def __init__(
        self, input_size: int, state_size: int, output_size: int, num_heads: int, add_bias: bool = True
    ) -> None:
        self.input_size = input_size
        self.state_size = state_size
        self.output_size = output_size
        self.num_heads = num_heads

        self.input_weight = nn.Parameter(torch.empty(self.num_heads * self.state_size, self.input_size))
        self.state_weight = nn.Parameter(torch.empty(self.num_heads, self.state_size, self.state_size))
        self.state_bias = nn.Parameter(torch.empty(self.num_heads, self.state_size)) if add_bias else None
        self.output_weight = nn.Parameter(torch.empty(self.output_size, self.num_heads * self.state_size))

    def forward(self, x: torch.Tensor, input_state: torch.Tensor | None = None) -> torch.Tensor:
        x = F.linear(x, self.input_weight)
        x = rnn_cute(input=x, state_weight=self.state_weight, state_bias=self.state_bias, input_state=input_state)
        x = F.linear(x, self.output_weight)
        return x
