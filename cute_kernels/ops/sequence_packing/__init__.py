# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import torch

from ...kernel_backend import KernelBackend
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


class _PackSequence_Cute(torch.autograd.Function):
    @staticmethod
    @ensure_contiguous
    def forward(
        ctx,
        x: torch.Tensor,
        cu_seqlens: torch.Tensor,
        output_shape: tuple[int],
        padding_side: str,
        kernel_backend_forward: KernelBackend,
        kernel_backend_backward: KernelBackend,
    ) -> torch.Tensor:
        assert padding_side in ["left", "right"]
        assert x.dim() >= 2
        assert x.size(0) == cu_seqlens.size(0) - 1

        ctx.save_for_backward(cu_seqlens)
        ctx.padding_side = padding_side
        ctx.x_shape = x.size()
        ctx.kernel_backend_backward = kernel_backend_backward

        output = _pack_sequence(
            x=x,
            cu_seqlens=cu_seqlens,
            output_shape=output_shape,
            padding_side=padding_side,
            kernel_backend=kernel_backend_forward,
        )

        return output

    @staticmethod
    @ensure_contiguous
    def backward(ctx, output_grad: torch.Tensor) -> tuple[torch.Tensor | None]:
        x_grad = _unpack_sequence(
            x=output_grad,
            cu_seqlens=ctx.saved_tensors[0],
            output_shape=ctx.x_shape,
            padding_side=ctx.padding_side,
            kernel_backend=ctx.kernel_backend_backward,
        )

        return x_grad, *[None] * 5


class _UnpackSequence_Cute(torch.autograd.Function):
    @staticmethod
    @ensure_contiguous
    def forward(
        ctx,
        x: torch.Tensor,
        cu_seqlens: torch.Tensor,
        output_shape: tuple[int],
        padding_side: str,
        kernel_backend_forward: KernelBackend,
        kernel_backend_backward: KernelBackend,
    ) -> torch.Tensor:
        assert padding_side in ["left", "right"]
        assert x.dim() >= 2
        assert output_shape[0] == cu_seqlens.size(0) - 1
        assert output_shape[2:] == x.size()[1:]

        ctx.save_for_backward(cu_seqlens)
        ctx.padding_side = padding_side
        ctx.kernel_backend_backward = kernel_backend_backward
        # saving shape in forward can avoid allocating a tensor of shape depending on cu_seqlens[-1]
        # this avoids synchronization with CPU
        ctx.x_shape = x.size()

        output = _unpack_sequence(
            x=x,
            cu_seqlens=cu_seqlens,
            padding_side=padding_side,
            output_shape=output_shape,
            kernel_backend=kernel_backend_forward,
        )

        return output

    @staticmethod
    @ensure_contiguous
    def backward(ctx, output_grad: torch.Tensor) -> tuple[torch.Tensor | None]:
        x_grad = _pack_sequence(
            x=output_grad,
            cu_seqlens=ctx.saved_tensors[0],
            output_shape=ctx.x_shape,
            padding_side=ctx.padding_side,
            kernel_backend=ctx.kernel_backend_backward,
        )

        return x_grad, *[None] * 5


def pack_sequence_cute(
    inputs: torch.Tensor | list[torch.Tensor],
    cu_seqlens: torch.Tensor,
    output_shape: tuple[int] | None = None,
    padding_side: str = "left",
    *,
    kernel_backend_forward: KernelBackend = KernelBackend.cuda,
    kernel_backend_backward: KernelBackend = KernelBackend.cuda,
) -> torch.Tensor | list[torch.Tensor]:
    is_list = isinstance(inputs, (list, tuple))
    if not is_list:
        inputs = [inputs]

    outputs = []

    for x in inputs:
        if kernel_backend_forward == KernelBackend.torch:
            assert kernel_backend_backward == KernelBackend.torch

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

            if output_shape is not None:
                assert x.size() == output_shape
        else:
            if output_shape is None:
                # NOTE we call .item() outside because data dependent (cu_seqlens[-1]) memory allocation calls sync anyways
                # so, we do it once for all the tensors
                output_shape = (cu_seqlens[-1].item(), *x.size()[2:])

            x = _PackSequence_Cute.apply(
                x, cu_seqlens, output_shape, padding_side, kernel_backend_forward, kernel_backend_backward
            )

        outputs.append(x)

    if not is_list:
        outputs = outputs[0]

    return outputs


def unpack_sequence_cute(
    inputs: torch.Tensor | list[torch.Tensor],
    cu_seqlens: torch.Tensor,
    output_shape: tuple[int],
    padding_side: str = "left",
    *,
    kernel_backend_forward: KernelBackend = KernelBackend.cuda,
    kernel_backend_backward: KernelBackend = KernelBackend.cuda,
) -> torch.Tensor | list[torch.Tensor]:
    is_list = isinstance(inputs, (list, tuple))
    if not is_list:
        inputs = [inputs]

    outputs = []

    for x in inputs:
        if kernel_backend_forward == KernelBackend.torch:
            assert kernel_backend_backward == KernelBackend.torch

            B, S = output_shape[:2]
            assert cu_seqlens.size(0) - 1 == B
            assert output_shape[2:] == x.size()[1:]

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
            padded = _UnpackSequence_Cute.apply(
                x, cu_seqlens, output_shape, padding_side, kernel_backend_forward, kernel_backend_backward
            )

        outputs.append(padded)

    if not is_list:
        outputs = outputs[0]

    return outputs
