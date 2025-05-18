import torch

from ...kernel_backend import KernelBackend, is_cuda_kernel_backend_allowed, is_triton_kernel_backend_allowed
from ...math import divide_if_divisible
from ...utils import ensure_contiguous, is_nvidia_gpu
from .cuda_implementation import swiglu_backward_cuda, swiglu_forward_cuda
from .torch_implementation import swiglu_packed_torch, swiglu_torch
from .triton_implementation import swiglu_backward_triton, swiglu_forward_triton, swiglu_packed_backward_triton


class _Swiglu_Cute(torch.autograd.Function):
    @staticmethod
    @ensure_contiguous
    def forward(
        ctx,
        gate: torch.Tensor,
        up: torch.Tensor,
        kernel_backend_forward: KernelBackend,
        kernel_backend_backward: KernelBackend,
    ) -> torch.Tensor:
        assert gate.size() == up.size(), "tensors gate and up should have same shape"
        assert gate.type() == up.type(), "tensors gate and up should have same dtype"

        ctx.save_for_backward(gate, up)
        ctx.kernel_backend_backward = kernel_backend_backward

        output = torch.empty_like(gate)

        if is_cuda_kernel_backend_allowed(kernel_backend_forward) and is_nvidia_gpu() and gate.is_cuda and up.is_cuda:
            swiglu_forward_cuda(gate=gate, up=up, output=output, BLOCK_SIZE=1024)
        elif is_triton_kernel_backend_allowed(kernel_backend_forward):
            swiglu_forward_triton(gate=gate, up=up, output=output, BLOCK_SIZE_B=64, BLOCK_SIZE_H=64)
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
                gate=gate, up=up, output_grad=output_grad, gate_grad=gate_grad, up_grad=up_grad, BLOCK_SIZE=1024
            )
        elif is_triton_kernel_backend_allowed(kernel_backend_backward):
            swiglu_backward_triton(
                gate=gate,
                up=up,
                output_grad=output_grad,
                gate_grad=gate_grad,
                up_grad=up_grad,
                BLOCK_SIZE=4096,
                NUM_WARPS=32,
            )
        else:
            raise ValueError("unexpected kernel_backend")

        return gate_grad, up_grad, None, None


class _SwigluPacked_Cute(torch.autograd.Function):
    @staticmethod
    @ensure_contiguous
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(x)

        output = torch.empty(*x.size()[:-1], divide_if_divisible(x.size(-1), 2), device=x.device, dtype=x.dtype)
        up, gate = x.chunk(2, dim=-1)

        swiglu_forward_triton(gate=gate, up=up, output=output, BLOCK_SIZE_B=64, BLOCK_SIZE_H=64)

        return output

    @staticmethod
    @ensure_contiguous
    def backward(ctx, output_grad: torch.Tensor) -> tuple[torch.Tensor | None]:
        x = ctx.saved_tensors[0]
        x_grad = torch.empty_like(x)

        swiglu_packed_backward_triton(x=x, output_grad=output_grad, x_grad=x_grad, BLOCK_SIZE_B=64, BLOCK_SIZE_H=64)

        return x_grad


def swiglu_cute(
    gate: torch.Tensor,
    up: torch.Tensor,
    *,
    kernel_backend_forward: KernelBackend = KernelBackend.cuda,
    kernel_backend_backward: KernelBackend = KernelBackend.cuda,
) -> torch.Tensor:
    """computes swiglu activation as `up` * `gate` * sigmoid(`gate`)

    Args:
        gate (torch.Tensor): `gate` activation tensor
        up (torch.Tensor): `up` activation tensor
        kernel_backend_forward (KernelBackend, optional): kernel backend to prioritize. Defaults to
            KernelBackend.cuda.
        kernel_backend_backward (KernelBackend, optional): kernel backend to prioritize. Defaults to
            KernelBackend.cuda.

    Returns:
        torch.Tensor: output tensor
    """

    return _Swiglu_Cute.apply(gate, up, kernel_backend_forward, kernel_backend_backward)


def swiglu_packed_cute(x: torch.Tensor) -> torch.Tensor:
    """computes swiglu activation by splitting the tensor `x` into 2 parts: gate and up activations

    Args:
        x (torch.Tensor): input activation

    Returns:
        torch.Tensor: output tensor
    """

    return _SwigluPacked_Cute.apply(x)
