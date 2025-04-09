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
        BLOCK_SIZE_HI: int,
        BLOCK_SIZE_HO: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        output = torch.empty_like(input)

        rnn_forward_triton(
            input=input,
            weight=weight,
            output=output,
            input_state=input_state,
            BLOCK_SIZE_B=BLOCK_SIZE_B,
            BLOCK_SIZE_HI=BLOCK_SIZE_HI,
            BLOCK_SIZE_HO=BLOCK_SIZE_HO,
        )

        return output

    @staticmethod
    def backward(ctx, output_grad: torch.Tensor) -> tuple[torch.Tensor]: ...


def rnn_cute(
    input: torch.Tensor,
    weight: torch.Tensor,
    input_state: torch.Tensor | None = None,
    # BLOCK_SIZE_B: int = CutoTuneParameter(),
    # BLOCK_SIZE_H: int = CutoTuneParameter(),
    # BLOCK_SIZE_I: int = CutoTuneParameter(),
    BLOCK_SIZE_B: int = 16,
    BLOCK_SIZE_HI: int = 16,
    BLOCK_SIZE_HO: int = 16,
) -> torch.Tensor:
    return _RNN_Cute.apply(input, weight, input_state, BLOCK_SIZE_B, BLOCK_SIZE_HI, BLOCK_SIZE_HO)


class RNNCute(RNNTorch):
    def _rnn(self, input: torch.Tensor, input_state: torch.Tensor | None) -> torch.Tensor:
        return rnn_cute(input=input, weight=self.state_weight, input_state=input_state)
