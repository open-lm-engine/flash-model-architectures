# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

from typing import Sequence

import torch

from ...custom_op import CustomOp, ctx_save_for_backward
from ...enums import KernelBackend
from ...utils import ensure_contiguous
from .cuda_implementation import pack_unpack_sequence_cuda
from .triton_implementation import pack_unpack_sequence_triton


def _pack_sequence(
    x: torch.Tensor,
    cu_seqlens: torch.Tensor,
    output_shape: tuple[int],
    padding_side: str,
    kernel_backend: KernelBackend,
) -> torch.Tensor:
    output = torch.empty(output_shape, device=x.device, dtype=x.dtype)

    if kernel_backend == KernelBackend.cuda:
        pack_unpack_sequence_cuda(
            x=x, output=output, cu_seqlens=cu_seqlens, padding_side=padding_side, pack=True, BLOCK_SIZE=1024
        )
    elif kernel_backend == KernelBackend.triton:
        pack_unpack_sequence_triton(x=x, output=output, cu_seqlens=cu_seqlens, padding_side=padding_side, pack=True)
    else:
        raise ValueError(f"unexpected kernel_backend ({kernel_backend})")

    return output


def _unpack_sequence(
    x: torch.Tensor,
    cu_seqlens: torch.Tensor,
    output_shape: tuple[int],
    padding_side: str,
    kernel_backend: KernelBackend,
) -> torch.Tensor:
    output = torch.zeros(*output_shape, device=x.device, dtype=x.dtype)

    if kernel_backend == KernelBackend.cuda:
        pack_unpack_sequence_cuda(
            x=x, output=output, cu_seqlens=cu_seqlens, padding_side=padding_side, pack=False, BLOCK_SIZE=1024
        )
    elif kernel_backend == KernelBackend.triton:
        pack_unpack_sequence_triton(x=x, output=output, cu_seqlens=cu_seqlens, padding_side=padding_side, pack=False)
    else:
        raise ValueError(f"unexpected kernel_backend ({kernel_backend})")

    return output


class _PackSequence(CustomOp):
    @staticmethod
    def forward_backward_torch(
        x: torch.Tensor, cu_seqlens: torch.Tensor, output_shape: tuple[int], padding_side: str
    ) -> torch.Tensor:
        B, S = x.size()[:2]
        seqlens = cu_seqlens[1:] - cu_seqlens[:-1]
        batch_indices = torch.arange(B, device=x.device).repeat_interleave(seqlens)

        if padding_side == "left":
            pad_tokens = S - seqlens
            seq_indices = torch.cat([torch.arange(sl, S, device=x.device) for sl in pad_tokens])
        elif padding_side == "right":
            seq_indices = torch.cat([torch.arange(sl, device=x.device) for sl in seqlens])
        else:
            raise ValueError(f"unexpected padding_side ({padding_side})")

        x = x[batch_indices, seq_indices]

        return x

    @staticmethod
    @ensure_contiguous
    def forward_cuda(
        ctx, x: torch.Tensor, cu_seqlens: torch.Tensor, output_shape: tuple[int], padding_side: str
    ) -> torch.Tensor:
        ctx_save_for_backward(ctx, cu_seqlens)
        ctx.padding_side = padding_side
        ctx.x_shape = x.size()

        output = torch.empty(output_shape, device=x.device, dtype=x.dtype)

        pack_unpack_sequence_cuda(
            x=x, output=output, cu_seqlens=cu_seqlens, padding_side=padding_side, pack=True, BLOCK_SIZE=1024
        )

        return output

    @staticmethod
    @ensure_contiguous
    def backward_cuda(ctx, output_grad: torch.Tensor) -> tuple[torch.Tensor, None, None, None]:
        x_grad = torch.zeros(*ctx.x_shape, device=output_grad.device, dtype=output_grad.dtype)
        cu_seqlens = ctx.saved_tensors[0]

        pack_unpack_sequence_cuda(
            x=output_grad,
            output=x_grad,
            cu_seqlens=cu_seqlens,
            padding_side=ctx.padding_side,
            pack=False,
            BLOCK_SIZE=1024,
        )

        return x_grad, None, None, None

    @staticmethod
    def forward_triton(
        ctx, x: torch.Tensor, cu_seqlens: torch.Tensor, output_shape: tuple[int], padding_side: str
    ) -> torch.Tensor:
        ctx_save_for_backward(ctx, cu_seqlens)
        ctx.padding_side = padding_side
        ctx.x_shape = x.size()

        output = torch.empty(output_shape, device=x.device, dtype=x.dtype)
        pack_unpack_sequence_triton(x=x, output=output, cu_seqlens=cu_seqlens, padding_side=padding_side, pack=True)

        return output

    @staticmethod
    def backward_triton(ctx, output_grad: torch.Tensor) -> tuple[torch.Tensor, None, None, None]:
        x_grad = torch.zeros(*ctx.x_shape, device=output_grad.device, dtype=output_grad.dtype)
        cu_seqlens = ctx.saved_tensors[0]

        pack_unpack_sequence_triton(
            x=output_grad, output=x_grad, cu_seqlens=cu_seqlens, padding_side=ctx.padding_side, pack=False
        )

        return x_grad, None, None, None


class _UnpackSequence(CustomOp):
    @staticmethod
    @ensure_contiguous
    def forward_cuda(
        ctx, x: torch.Tensor, cu_seqlens: torch.Tensor, output_shape: tuple[int], padding_side: str
    ) -> torch.Tensor:
        ctx_save_for_backward(ctx, cu_seqlens)
        ctx.padding_side = padding_side
        ctx.x_shape = x.size()

        output = torch.zeros(*output_shape, device=x.device, dtype=x.dtype)

        pack_unpack_sequence_cuda(
            x=x, output=output, cu_seqlens=cu_seqlens, padding_side=padding_side, pack=False, BLOCK_SIZE=1024
        )

        return output

    @staticmethod
    @ensure_contiguous
    def backward_cuda(ctx, output_grad: torch.Tensor) -> tuple[torch.Tensor | None]:
        x_grad = torch.empty(ctx.x_shape, device=output_grad.device, dtype=output_grad.dtype)
        cu_seqlens = ctx.saved_tensors[0]

        pack_unpack_sequence_cuda(
            x=output_grad,
            output=x_grad,
            cu_seqlens=cu_seqlens,
            padding_side=ctx.padding_side,
            pack=True,
            BLOCK_SIZE=1024,
        )

        return x_grad, None, None, None


def pack_sequence(
    inputs: Sequence[torch.Tensor],
    cu_seqlens: torch.Tensor,
    total_tokens: int,
    padding_side: str = "left",
    *,
    kernel_backend: KernelBackend | None = None,
) -> Sequence[torch.Tensor]:
    assert padding_side in ["left", "right"]
    assert isinstance(inputs, (list, tuple))

    outputs = []

    for x in inputs:
        assert x.dim() >= 2
        assert x.size(0) == cu_seqlens.size(0) - 1

        x = _PackSequence.run(
            x=x,
            cu_seqlens=cu_seqlens,
            output_shape=(total_tokens, *x.size()[2:]),
            padding_side=padding_side,
            kernel_backend=kernel_backend,
        )

        outputs.append(x)

    return outputs


def unpack_sequence(
    inputs: Sequence[torch.Tensor],
    cu_seqlens: torch.Tensor,
    batch_size: int,
    sequence_length: int,
    padding_side: str = "left",
    *,
    kernel_backend: KernelBackend | None = None,
) -> Sequence[torch.Tensor]:
    assert padding_side in ["left", "right"]
    assert isinstance(inputs, (list, tuple))

    outputs = []
    B = batch_size
    S = sequence_length

    for x in inputs:
        assert x.dim() >= 2
        output_shape = (B, S, *x.size()[1:])

        if kernel_backend == KernelBackend.torch:
            assert cu_seqlens.size(0) - 1 == B

            seqlens = cu_seqlens[1:] - cu_seqlens[:-1]
            batch_indices = torch.arange(B, device=x.device).repeat_interleave(seqlens)

            if padding_side == "left":
                pad_tokens = S - seqlens
                seq_indices = torch.cat([torch.arange(sl, S, device=x.device) for sl in pad_tokens])
            elif padding_side == "right":
                seq_indices = torch.cat([torch.arange(sl, device=x.device) for sl in seqlens])
            else:
                raise ValueError(f"unexpected padding_side ({padding_side})")

            padded = torch.zeros(output_shape, dtype=x.dtype, device=x.device)
            padded[batch_indices, seq_indices] = x
        else:
            padded = _UnpackSequence.run(
                x=x,
                cu_seqlens=cu_seqlens,
                output_shape=output_shape,
                padding_side=padding_side,
                kernel_backend=kernel_backend,
            )

        outputs.append(padded)

    return outputs
