import torch

from .cuda_implementation import pack_sequence_cuda
from .torch_implementation import pack_sequence_torch


class _PackSequence_Cute(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: torch.Tensor | int,
        padding_side: str,
        BLOCK_SIZE: int,
    ) -> torch.Tensor:
        assert padding_side in ["left", "right"]
        assert x.dim() >= 2

        output = torch.empty(cu_seqlens[-1], *x.size()[2:], device=x.device, dtype=x.dtype)

        pack_sequence_cuda(x=x, output=output, cu_seqlens=cu_seqlens, padding_side=padding_side, BLOCK_SIZE=BLOCK_SIZE)

        return output

    @staticmethod
    def backward(ctx, output_grad: torch.Tensor) -> tuple[torch.Tensor | None]:
        return output_grad, *[None] * 4


def pack_sequence_cute(
    x: torch.Tensor,
    cu_seqlens: torch.Tensor,
    max_seqlen: torch.Tensor | int,
    padding_side: str = "left",
    BLOCK_SIZE: int = 1024,
) -> torch.Tensor:
    return _PackSequence_Cute.apply(x, cu_seqlens, max_seqlen, padding_side, BLOCK_SIZE)
