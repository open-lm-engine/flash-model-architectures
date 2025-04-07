import torch
import torch.nn.functional as F

from ...cutotune import CutoTuneParameter
from .triton_implementation import rnn_forward_triton


class _RNN_Cute(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        input: torch.Tensor,
        input_weight: torch.Tensor,
        state_weight: torch.Tensor,
        output_weight: torch.Tensor,
        input_state: torch.Tensor | None,
        BLOCK_SIZE_B: int,
        BLOCK_SIZE_H: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = F.linear(input, input_weight)
        y = torch.empty_like(x)

        rnn_forward_triton(
            x=x,
            weight=state_weight,
            y=y,
            input_state=input_state,
            BLOCK_SIZE_B=BLOCK_SIZE_B,
            BLOCK_SIZE_H=BLOCK_SIZE_H,
        )

        output = F.linear(y, output_weight)
        return output

    @staticmethod
    def backward(ctx, output_grad: torch.Tensor) -> tuple[torch.Tensor]: ...


def rnn_cute(
    input: torch.Tensor,
    input_weight: torch.Tensor,
    state_weight: torch.Tensor,
    output_weight: torch.Tensor,
    input_state: torch.Tensor | None = None,
    BLOCK_SIZE_B: int = CutoTuneParameter(),
    BLOCK_SIZE_H: int = CutoTuneParameter(),
) -> torch.Tensor:
    return _RNN_Cute.apply(input, input_weight, state_weight, output_weight, input_state, BLOCK_SIZE_B, BLOCK_SIZE_H)
