import torch

from .torch_implementation import linear_torch


class _Linear_Cute(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
        output = torch.empty(input.size()[:-1], weight.size(0), dtype=input.dtype, device=input.device)

        # TODO

        return output


def linear_cute(input: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
    return _Linear_Cute.apply(input, weight, bias)
