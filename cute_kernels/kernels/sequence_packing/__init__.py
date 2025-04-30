import torch

from ...utils import ensure_contiguous
from .cuda_implementation import pack_unpack_sequence_cuda
from .torch_implementation import pack_sequence_torch


class _PackSequence_Cute(torch.autograd.Function):
    @staticmethod
    @ensure_contiguous
    def forward(
        ctx,
        x: torch.Tensor,
        cu_seqlens: torch.Tensor,
        padding_side: str,
        BLOCK_SIZE_forward: int,
        BLOCK_SIZE_backward: int,
    ) -> torch.Tensor:
        assert padding_side in ["left", "right"]
        assert x.dim() >= 2

        output = torch.empty(cu_seqlens[-1], *x.size()[2:], device=x.device, dtype=x.dtype)
        pack_unpack_sequence_cuda(
            x=x,
            output=output,
            cu_seqlens=cu_seqlens,
            padding_side=padding_side,
            pack=True,
            BLOCK_SIZE=BLOCK_SIZE_forward,
        )

        ctx.save_for_backward(cu_seqlens)
        ctx.x_size = x.size()
        ctx.BLOCK_SIZE_backward = BLOCK_SIZE_backward

        return output

    @staticmethod
    @ensure_contiguous
    def backward(ctx, output_grad: torch.Tensor) -> tuple[torch.Tensor | None]:
        B, S = ctx.x_size[:2]

        cu_seqlens = ctx.saved_tensors[0]
        x_grad = torch.zeros(B, S, *ctx.x_size[2:], device=output_grad.device, dtype=output_grad.dtype)

        pack_unpack_sequence_cuda(
            x=output_grad,
            output=x_grad,
            cu_seqlens=cu_seqlens,
            padding_side=ctx.padding_side,
            pack=False,
            BLOCK_SIZE=ctx.BLOCK_SIZE_backward,
        )

        return x_grad, *[None] * 5


def pack_sequence_cute(
    x: torch.Tensor,
    cu_seqlens: torch.Tensor,
    padding_side: str = "left",
    BLOCK_SIZE_forward: int = 1024,
    BLOCK_SIZE_backward: int = 1024,
) -> torch.Tensor:
    return _PackSequence_Cute.apply(x, cu_seqlens, padding_side, BLOCK_SIZE_forward, BLOCK_SIZE_backward)
