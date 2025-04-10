import torch

from ...cutotune import CutoTuneParameter
from ...utils import ensure_contiguous
from .torch_implementation import RNNTorch, rnn_torch
from .triton_implementation import rnn_backward_triton, rnn_forward_triton


class _RNN_Cute(torch.autograd.Function):
    @staticmethod
    @ensure_contiguous
    def forward(
        ctx,
        input: torch.Tensor,
        weight: torch.Tensor,
        input_state: torch.Tensor | None,
        BLOCK_SIZE_B_forward: int,
        BLOCK_SIZE_B_backward: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        output = torch.empty_like(input)

        rnn_forward_triton(
            input=input,
            weight=weight,
            output=output,
            input_state=input_state,
            BLOCK_SIZE_B=BLOCK_SIZE_B_forward,
        )

        ctx.save_for_backward(input, weight, output)
        ctx.BLOCK_SIZE_B_backward = BLOCK_SIZE_B_backward

        return output

    @staticmethod
    @ensure_contiguous
    def backward(ctx, output_grad: torch.Tensor) -> tuple[torch.Tensor]:
        input, weight, output = ctx.saved_tensors

        input_grad = torch.empty_like(input)
        weight_grad = torch.empty_like(weight)

        rnn_backward_triton(
            input=input,
            weight=weight,
            output=output,
            output_grad=output_grad,
            input_grad=input_grad,
            BLOCK_SIZE_B=ctx.BLOCK_SIZE_B_backward,
        )

        return input_grad, weight_grad, *[None] * 3


def rnn_cute(
    input: torch.Tensor,
    weight: torch.Tensor,
    input_state: torch.Tensor | None = None,
    BLOCK_SIZE_B_forward: int = 16,
    BLOCK_SIZE_B_backward: int = 16,
) -> torch.Tensor:
    return _RNN_Cute.apply(input, weight, input_state, BLOCK_SIZE_B_forward, BLOCK_SIZE_B_backward)


class RNNCute(RNNTorch):
    def _rnn(self, input: torch.Tensor, input_state: torch.Tensor | None) -> torch.Tensor:
        return rnn_cute(input=input, weight=self.state_weight, input_state=input_state)
