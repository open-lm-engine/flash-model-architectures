import torch

from ...kernel_backend import KernelBackend
from ...utils import ensure_contiguous
from .cuda_implementation import pack_unpack_sequence_cuda
from .torch_implementation import pack_sequence_torch, unpack_sequence_torch
from .triton_implementation import pack_unpack_sequence_triton_kernel


def _pack_sequence(
    x: torch.Tensor,
    cu_seqlens: torch.Tensor,
    padding_side: str,
    kernel_backend: KernelBackend,
    BLOCK_SIZE_CUDA: int,
    BLOCK_SIZE_TRITON: int,
    NUM_WARPS_TRITON: int,
) -> torch.Tensor:
    output = torch.empty(cu_seqlens[-1], *x.size()[2:], device=x.device, dtype=x.dtype)

    if kernel_backend == KernelBackend.cuda:
        pack_unpack_sequence_cuda(
            x=x,
            output=output,
            cu_seqlens=cu_seqlens,
            padding_side=padding_side,
            pack=True,
            BLOCK_SIZE=BLOCK_SIZE_CUDA,
        )
    elif kernel_backend == KernelBackend.triton:
        B, S = x.size()[:2]
        N = x.numel() // (B * S)

        with torch.cuda.device(x.device):
            pack_unpack_sequence_triton_kernel[S, B](
                x_ptr=x,
                output_ptr=output,
                cu_seqlens_ptr=cu_seqlens,
                S=S,
                N=N,
                padding_side=padding_side,
                pack=True,
                BLOCK_SIZE=BLOCK_SIZE_TRITON,
                num_warps=NUM_WARPS_TRITON,
            )
    else:
        raise ValueError(f"unexpected kernel_backend ({kernel_backend})")

    return output


def _unpack_sequence(
    x: torch.Tensor,
    cu_seqlens: torch.Tensor,
    padding_side: str,
    desired_shape: tuple[int],
    kernel_backend: KernelBackend,
    BLOCK_SIZE_CUDA: int,
    BLOCK_SIZE_TRITON: int,
    NUM_WARPS_TRITON: int,
) -> torch.Tensor:
    B, S = desired_shape[:2]

    output = torch.zeros(B, S, *desired_shape[2:], device=x.device, dtype=x.dtype)

    if kernel_backend == KernelBackend.cuda:
        pack_unpack_sequence_cuda(
            x=x,
            output=output,
            cu_seqlens=cu_seqlens,
            padding_side=padding_side,
            pack=False,
            BLOCK_SIZE=BLOCK_SIZE_CUDA,
        )
    elif kernel_backend == KernelBackend.triton:
        N = output.numel() // (B * S)

        with torch.cuda.device(x.device):
            pack_unpack_sequence_triton_kernel[S, B](
                x_ptr=x,
                output_ptr=output,
                cu_seqlens_ptr=cu_seqlens,
                S=S,
                N=N,
                padding_side=padding_side,
                pack=False,
                BLOCK_SIZE=BLOCK_SIZE_TRITON,
                num_warps=NUM_WARPS_TRITON,
            )
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
        padding_side: str,
        kernel_backend_forward: KernelBackend,
        BLOCK_SIZE_CUDA_forward: int,
        BLOCK_SIZE_TRITON_forward: int,
        NUM_WARPS_TRITON_forward: int,
        kernel_backend_backward: KernelBackend,
        BLOCK_SIZE_CUDA_backward: int,
        BLOCK_SIZE_TRITON_backward: int,
        NUM_WARPS_TRITON_backward: int,
    ) -> torch.Tensor:
        assert padding_side in ["left", "right"]
        assert x.dim() >= 2

        ctx.save_for_backward(cu_seqlens)
        ctx.padding_side = padding_side
        ctx.x_shape = x.size()
        ctx.kernel_backend_backward = kernel_backend_backward
        ctx.BLOCK_SIZE_CUDA_backward = BLOCK_SIZE_CUDA_backward
        ctx.BLOCK_SIZE_TRITON_backward = BLOCK_SIZE_TRITON_backward
        ctx.NUM_WARPS_TRITON_backward = NUM_WARPS_TRITON_backward

        output = _pack_sequence(
            x=x,
            cu_seqlens=cu_seqlens,
            padding_side=padding_side,
            kernel_backend=kernel_backend_forward,
            BLOCK_SIZE_CUDA=BLOCK_SIZE_CUDA_forward,
            BLOCK_SIZE_TRITON=BLOCK_SIZE_TRITON_forward,
            NUM_WARPS_TRITON=NUM_WARPS_TRITON_forward,
        )

        return output

    @staticmethod
    @ensure_contiguous
    def backward(ctx, output_grad: torch.Tensor) -> tuple[torch.Tensor | None]:
        x_grad = _unpack_sequence(
            x=output_grad,
            cu_seqlens=ctx.saved_tensors[0],
            padding_side=ctx.padding_side,
            desired_shape=ctx.x_shape,
            kernel_backend=ctx.kernel_backend_backward,
            BLOCK_SIZE_CUDA=ctx.BLOCK_SIZE_CUDA_backward,
            BLOCK_SIZE_TRITON=ctx.BLOCK_SIZE_TRITON_backward,
            NUM_WARPS_TRITON=ctx.NUM_WARPS_TRITON_backward,
        )

        return x_grad, *[None] * 10


class _UnpackSequence_Cute(torch.autograd.Function):
    @staticmethod
    @ensure_contiguous
    def forward(
        ctx,
        x: torch.Tensor,
        cu_seqlens: torch.Tensor,
        desired_shape: tuple[int],
        padding_side: str,
        kernel_backend_forward: KernelBackend,
        BLOCK_SIZE_CUDA_forward: int,
        BLOCK_SIZE_TRITON_forward: int,
        NUM_WARPS_TRITON_forward: int,
        kernel_backend_backward: KernelBackend,
        BLOCK_SIZE_CUDA_backward: int,
        BLOCK_SIZE_TRITON_backward: int,
        NUM_WARPS_TRITON_backward: int,
    ) -> torch.Tensor:
        assert padding_side in ["left", "right"]
        assert x.dim() >= 2
        assert desired_shape[0] == cu_seqlens.size(0) - 1
        assert desired_shape[2:] == x.size()[1:]

        ctx.save_for_backward(cu_seqlens)
        ctx.padding_side = padding_side
        ctx.kernel_backend_backward = kernel_backend_backward
        ctx.BLOCK_SIZE_CUDA_backward = BLOCK_SIZE_CUDA_backward
        ctx.BLOCK_SIZE_TRITON_backward = BLOCK_SIZE_TRITON_backward
        ctx.NUM_WARPS_TRITON_backward = NUM_WARPS_TRITON_backward

        output = _unpack_sequence(
            x=x,
            cu_seqlens=cu_seqlens,
            padding_side=padding_side,
            desired_shape=desired_shape,
            kernel_backend=kernel_backend_forward,
            BLOCK_SIZE_CUDA=BLOCK_SIZE_CUDA_forward,
            BLOCK_SIZE_TRITON=BLOCK_SIZE_TRITON_forward,
            NUM_WARPS_TRITON=NUM_WARPS_TRITON_forward,
        )

        return output

    @staticmethod
    @ensure_contiguous
    def backward(ctx, output_grad: torch.Tensor) -> tuple[torch.Tensor | None]:
        x_grad = _pack_sequence(
            x=output_grad,
            cu_seqlens=ctx.saved_tensors[0],
            padding_side=ctx.padding_side,
            kernel_backend=ctx.kernel_backend_backward,
            BLOCK_SIZE_CUDA=ctx.BLOCK_SIZE_CUDA_backward,
            BLOCK_SIZE_TRITON=ctx.BLOCK_SIZE_TRITON_backward,
            NUM_WARPS_TRITON=ctx.NUM_WARPS_TRITON_backward,
        )

        return x_grad, *[None] * 11


def pack_sequence_cute(
    x: torch.Tensor,
    cu_seqlens: torch.Tensor,
    padding_side: str = "left",
    *,
    kernel_backend_forward: KernelBackend = KernelBackend.cuda,
    BLOCK_SIZE_CUDA_forward: int = 1024,
    BLOCK_SIZE_TRITON_forward: int = 4096,
    NUM_WARPS_TRITON_forward: int = 32,
    kernel_backend_backward: KernelBackend = KernelBackend.cuda,
    BLOCK_SIZE_CUDA_backward: int = 1024,
    BLOCK_SIZE_TRITON_backward: int = 4096,
    NUM_WARPS_TRITON_backward: int = 32,
) -> torch.Tensor:
    return _PackSequence_Cute.apply(
        x,
        cu_seqlens,
        padding_side,
        kernel_backend_forward,
        BLOCK_SIZE_CUDA_forward,
        BLOCK_SIZE_TRITON_forward,
        NUM_WARPS_TRITON_forward,
        kernel_backend_backward,
        BLOCK_SIZE_CUDA_backward,
        BLOCK_SIZE_TRITON_backward,
        NUM_WARPS_TRITON_backward,
    )


def unpack_sequence_cute(
    x: torch.Tensor,
    cu_seqlens: torch.Tensor,
    desired_shape: tuple[int],
    padding_side: str = "left",
    *,
    kernel_backend_forward: KernelBackend = KernelBackend.cuda,
    BLOCK_SIZE_CUDA_forward: int = 1024,
    BLOCK_SIZE_TRITON_forward: int = 4096,
    NUM_WARPS_TRITON_forward: int = 32,
    kernel_backend_backward: KernelBackend = KernelBackend.cuda,
    BLOCK_SIZE_CUDA_backward: int = 1024,
    BLOCK_SIZE_TRITON_backward: int = 4096,
    NUM_WARPS_TRITON_backward: int = 32,
) -> torch.Tensor:
    return _UnpackSequence_Cute.apply(
        x,
        cu_seqlens,
        desired_shape,
        padding_side,
        kernel_backend_forward,
        BLOCK_SIZE_CUDA_forward,
        BLOCK_SIZE_TRITON_forward,
        NUM_WARPS_TRITON_forward,
        kernel_backend_backward,
        BLOCK_SIZE_CUDA_backward,
        BLOCK_SIZE_TRITON_backward,
        NUM_WARPS_TRITON_backward,
    )
