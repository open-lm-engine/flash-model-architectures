import torch

from ...utils import ensure_contiguous
from .cuda_implementation import pack_unpack_sequence_cuda
from .torch_implementation import pack_sequence_torch


def _pack_sequence(x: torch.Tensor, cu_seqlens: torch.Tensor, padding_side: str, BLOCK_SIZE: int) -> torch.Tensor:
    output = torch.empty(cu_seqlens[-1], *x.size()[2:], device=x.device, dtype=x.dtype)
    pack_unpack_sequence_cuda(
        x=x,
        output=output,
        cu_seqlens=cu_seqlens,
        padding_side=padding_side,
        pack=True,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return output


def _unpack_sequence(
    x: torch.Tensor, cu_seqlens: torch.Tensor, padding_side: str, desired_shape: tuple[int], BLOCK_SIZE: int
) -> torch.Tensor:
    B, S = desired_shape[:2]

    output = torch.zeros(B, S, *desired_shape[2:], device=x.device, dtype=x.dtype)
    pack_unpack_sequence_cuda(
        x=x,
        output=output,
        cu_seqlens=cu_seqlens,
        padding_side=padding_side,
        pack=False,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return output


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

        ctx.save_for_backward(cu_seqlens)
        ctx.padding_side = padding_side
        ctx.x_shape = x.size()
        ctx.BLOCK_SIZE_backward = BLOCK_SIZE_backward

        return _pack_sequence(x=x, cu_seqlens=cu_seqlens, padding_side=padding_side, BLOCK_SIZE=BLOCK_SIZE_forward)

    @staticmethod
    @ensure_contiguous
    def backward(ctx, output_grad: torch.Tensor) -> tuple[torch.Tensor | None]:
        x_grad = _unpack_sequence(
            x=output_grad,
            cu_seqlens=ctx.saved_tensors[0],
            padding_side=ctx.padding_side,
            desired_shape=ctx.x_shape,
            BLOCK_SIZE=ctx.BLOCK_SIZE_backward,
        )

        return x_grad, *[None] * 4


class _UnpackSequence_Cute(torch.autograd.Function):
    @staticmethod
    @ensure_contiguous
    def forward(
        ctx,
        x: torch.Tensor,
        cu_seqlens: torch.Tensor,
        desired_shape: tuple[int],
        padding_side: str,
        BLOCK_SIZE_forward: int,
        BLOCK_SIZE_backward: int,
    ) -> torch.Tensor:
        assert padding_side in ["left", "right"]
        assert x.dim() >= 2
        assert desired_shape[2:] == x.size()[1:]

        ctx.save_for_backward(cu_seqlens)
        ctx.padding_side = padding_side
        ctx.BLOCK_SIZE_backward = BLOCK_SIZE_backward

        output = _unpack_sequence(
            x=x,
            cu_seqlens=cu_seqlens,
            padding_side=padding_side,
            desired_shape=desired_shape,
            BLOCK_SIZE=BLOCK_SIZE_forward,
        )

        return output

    @staticmethod
    @ensure_contiguous
    def backward(ctx, output_grad: torch.Tensor) -> tuple[torch.Tensor | None]:
        x_grad = _pack_sequence(
            x=output_grad,
            cu_seqlens=ctx.saved_tensors[0],
            padding_side=ctx.padding_side,
            BLOCK_SIZE=ctx.BLOCK_SIZE_backward,
        )

        return x_grad, *[None] * 4


def pack_sequence_cute(
    x: torch.Tensor,
    cu_seqlens: torch.Tensor,
    padding_side: str = "left",
    BLOCK_SIZE_forward: int = 1024,
    BLOCK_SIZE_backward: int = 1024,
) -> torch.Tensor:
    return _PackSequence_Cute.apply(x, cu_seqlens, padding_side, BLOCK_SIZE_forward, BLOCK_SIZE_backward)


def unpack_sequence_cute(
    x: torch.Tensor,
    cu_seqlens: torch.Tensor,
    desired_shape: tuple[int],
    padding_side: str = "left",
    BLOCK_SIZE_forward: int = 1024,
    BLOCK_SIZE_backward: int = 1024,
) -> torch.Tensor:
    return _PackSequence_Cute.apply(
        x, cu_seqlens, desired_shape, padding_side, BLOCK_SIZE_forward, BLOCK_SIZE_backward
    )
