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
    def forward_backward_torch(
        x: torch.Tensor, cu_seqlens: torch.Tensor, output_shape: tuple[int], padding_side: str
    ) -> torch.Tensor:
        B = cu_seqlens.size(0) - 1
        S = output_shape[1]

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

        return padded

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

    @staticmethod
    def forward_triton(
        ctx, x: torch.Tensor, cu_seqlens: torch.Tensor, output_shape: tuple[int], padding_side: str
    ) -> torch.Tensor:
        ctx_save_for_backward(ctx, cu_seqlens)
        ctx.padding_side = padding_side
        ctx.x_shape = x.size()

        output = torch.zeros(*output_shape, device=x.device, dtype=x.dtype)

        pack_unpack_sequence_triton(x=x, output=output, cu_seqlens=cu_seqlens, padding_side=padding_side, pack=False)

        return output

    @staticmethod
    def backward_triton(ctx, output_grad: torch.Tensor) -> tuple[torch.Tensor | None]:
        x_grad = torch.empty(ctx.x_shape, device=output_grad.device, dtype=output_grad.dtype)
        cu_seqlens = ctx.saved_tensors[0]

        pack_unpack_sequence_triton(
            x=output_grad, output=x_grad, cu_seqlens=cu_seqlens, padding_side=ctx.padding_side, pack=True
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

    for x in inputs:
        assert x.dim() >= 2
        assert cu_seqlens.size(0) - 1 == batch_size

        x = _UnpackSequence.run(
            x=x,
            cu_seqlens=cu_seqlens,
            output_shape=(batch_size, sequence_length, *x.size()[1:]),
            padding_side=padding_side,
            kernel_backend=kernel_backend,
        )

        outputs.append(x)

    return outputs
