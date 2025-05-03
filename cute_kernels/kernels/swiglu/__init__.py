import torch

from ...kernel_backend import KernelBackend, is_cuda_kernel_backend_allowed, is_triton_kernel_backend_allowed
from ...math import ceil_divide
from ...utils import ensure_contiguous, is_nvidia_gpu
from .cuda_implementation import swiglu_backward_cuda, swiglu_forward_cuda
from .torch_implementation import swiglu_torch
from .triton_implementation import swiglu_backward_triton, swiglu_forward_triton


class _Swiglu_Cute(torch.autograd.Function):
    @staticmethod
    @ensure_contiguous
    def forward(
        ctx,
        gate: torch.Tensor,
        up: torch.Tensor,
        kernel_backend_forward: KernelBackend,
        BLOCK_SIZE_CUDA_forward: int,
        BLOCK_SIZE_TRITON_forward: int,
        NUM_WARPS_TRITON_forward: int,
        kernel_backend_backward: KernelBackend,
        BLOCK_SIZE_CUDA_backward: int,
        BLOCK_SIZE_TRITON_backward: int,
        NUM_WARPS_TRITON_backward: int,
    ) -> torch.Tensor:
        assert gate.size() == up.size(), "tensors gate and up should have same shape"
        assert gate.type() == up.type(), "tensors gate and up should have same dtype"

        ctx.save_for_backward(gate, up)
        ctx.kernel_backend_backward = kernel_backend_backward
        ctx.BLOCK_SIZE_CUDA_backward = BLOCK_SIZE_CUDA_backward
        ctx.BLOCK_SIZE_TRITON_backward = BLOCK_SIZE_TRITON_backward
        ctx.NUM_WARPS_TRITON_backward = NUM_WARPS_TRITON_backward

        output = torch.empty_like(gate)

        if is_cuda_kernel_backend_allowed(kernel_backend_forward) and is_nvidia_gpu() and gate.is_cuda and up.is_cuda:
            swiglu_forward_cuda(gate=gate, up=up, output=output, BLOCK_SIZE=BLOCK_SIZE_CUDA_forward)
        elif is_triton_kernel_backend_allowed(kernel_backend_forward):
            swiglu_forward_triton(
                gate=gate,
                up=up,
                output=output,
                BLOCK_SIZE=BLOCK_SIZE_TRITON_forward,
                NUM_WARPS=NUM_WARPS_TRITON_forward,
            )
        else:
            raise ValueError("unexpected kernel_backend")

        return output

    @staticmethod
    @ensure_contiguous
    def backward(ctx, output_grad: torch.Tensor) -> tuple[torch.Tensor | None]:
        gate, up = ctx.saved_tensors
        gate_grad = torch.empty_like(gate)
        up_grad = torch.empty_like(up)

        kernel_backend_backward = ctx.kernel_backend_backward

        if is_cuda_kernel_backend_allowed(kernel_backend_backward) and is_nvidia_gpu() and gate.is_cuda and up.is_cuda:
            swiglu_backward_cuda(
                gate=gate,
                up=up,
                output_grad=output_grad,
                gate_grad=gate_grad,
                up_grad=up_grad,
                BLOCK_SIZE=ctx.BLOCK_SIZE_CUDA_backward,
            )
        elif is_triton_kernel_backend_allowed(kernel_backend_backward):
            swiglu_backward_triton(
                gate=gate,
                up=up,
                output_grad=output_grad,
                gate_grad=gate_grad,
                up_grad=up_grad,
                BLOCK_SIZE=ctx.BLOCK_SIZE_TRITON_backward,
                NUM_WARPS=ctx.NUM_WARPS_TRITON_backward,
            )
        else:
            raise ValueError("unexpected kernel_backend")

        return gate_grad, up_grad, *[None] * 8


def swiglu_cute(
    gate: torch.Tensor,
    up: torch.Tensor,
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
    """computes swiglu activation as `up` * `gate` * sigmoid(`gate`)

    Args:
        gate (torch.Tensor): `gate` activation tensor
        up (torch.Tensor): `up` activation tensor
        kernel_backend_forward (KernelBackend, optional): kernel backend to prioritize. Defaults to
            KernelBackend.cuda.
        BLOCK_SIZE_CUDA_forward (int, optional): block size for CUDA backend. Defaults to 1024.
        BLOCK_SIZE_TRITON_forward (int, optional): block size for triton backend. Defaults to 4096.
        NUM_WARPS_TRITON_forward (int, optional): warps for triton backend. Defaults to 32.
        kernel_backend_backward (KernelBackend, optional): kernel backend to prioritize. Defaults to
            KernelBackend.cuda.
        BLOCK_SIZE_CUDA_backward (int, optional): block size for CUDA backend. Defaults to 1024.
        BLOCK_SIZE_TRITON_backward (int, optional): block size for triton backend. Defaults to 4096.
        NUM_WARPS_TRITON_backward (int, optional): warps for triton backend. Defaults to 32.

    Returns:
        torch.Tensor: output tensor
    """

    return _Swiglu_Cute.apply(
        gate,
        up,
        kernel_backend_forward,
        BLOCK_SIZE_CUDA_forward,
        BLOCK_SIZE_TRITON_forward,
        NUM_WARPS_TRITON_forward,
        kernel_backend_backward,
        BLOCK_SIZE_CUDA_backward,
        BLOCK_SIZE_TRITON_backward,
        NUM_WARPS_TRITON_backward,
    )
