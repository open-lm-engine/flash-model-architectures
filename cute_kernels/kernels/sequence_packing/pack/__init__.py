import torch

from .torch_implementation import pack_sequence_torch


class _PackSequence_Cute(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, x: torch.Tensor, cu_seqlens: torch.Tensor, max_seqlen: torch.Tensor | int, padding_side: str
    ) -> torch.Tensor:
        assert padding_side in ["left", "right"]

        seqlens = cu_seqlens[1:] - cu_seqlens[:-1]
        B = seqlens.numel()

        return x

    @staticmethod
    def backward(ctx, output_grad: torch.Tensor) -> tuple[torch.Tensor | None]:
        return x_grad, None, None


def pack_sequence_cute(
    x: torch.Tensor, cu_seqlens: torch.Tensor, max_seqlen: torch.Tensor | int, padding_side: str = "left"
) -> torch.Tensor:
    return _PackSequence_Cute.apply(x, cu_seqlens, max_seqlen, padding_side)
