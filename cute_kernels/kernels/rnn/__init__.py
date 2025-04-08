import torch

from ...cutotune import CutoTuneParameter
from .torch_implementation import RNNTorch, rnn_torch
from .triton_implementation import rnn_forward_triton


class _RNN_Cute(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        input: torch.Tensor,
        weight: torch.Tensor,
        input_state: torch.Tensor | None,
        BLOCK_SIZE_B: int,
        BLOCK_SIZE_H: int,
        BLOCK_SIZE_I: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        output = torch.empty_like(input)

        rnn_forward_triton(
            input=input,
            weight=weight,
            output=output,
            input_state=input_state,
            BLOCK_SIZE_B=BLOCK_SIZE_B,
            BLOCK_SIZE_H=BLOCK_SIZE_H,
            BLOCK_SIZE_I=BLOCK_SIZE_I,
        )

        return y

    @staticmethod
    def backward(ctx, output_grad: torch.Tensor) -> tuple[torch.Tensor]: ...


def rnn_cute(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tenosr | None = None,
    input_state: torch.Tensor | None = None,
    BLOCK_SIZE_B: int = CutoTuneParameter(),
    BLOCK_SIZE_H: int = CutoTuneParameter(),
) -> torch.Tensor:
    return _RNN_Cute.apply(input, weight, bias, input_state, BLOCK_SIZE_B, BLOCK_SIZE_H)


class RNNCute(RNNTorch):
    def forward(self, x: torch.Tensor, input_state: torch.Tensor | None = None) -> torch.Tensor:
        x = self.input_projection(x)
        x = rnn_cute(input=x, state_weight=self.state_weight, state_bias=self.state_bias, input_state=input_state)
        x = self.output_projection(x)
        return x
