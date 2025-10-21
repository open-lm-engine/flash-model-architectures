# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import torch
import torch.nn.functional as F

from ...enums import KernelBackend
from ...math import divide_if_divisible
from ...utils import empty_like_contiguous, ensure_contiguous
from .cuda_implementation import swiglu_backward_cuda, swiglu_forward_cuda
from .triton_implementation import swiglu_backward_triton, swiglu_forward_triton


class _Swiglu(torch.autograd.Function):
    @staticmethod
    @ensure_contiguous
    def forward(ctx, gate: torch.Tensor, up: torch.Tensor, kernel_backend: KernelBackend) -> torch.Tensor:
        output = empty_like_contiguous(gate)

        if kernel_backend == KernelBackend.cuda:
            swiglu_forward_cuda(gate=gate, up=up, output=output, BLOCK_SIZE=1024)
        elif kernel_backend == KernelBackend.triton:
            swiglu_forward_triton(gate=gate, up=up, output=output)
        else:
            raise ValueError(f"unexpected kernel_backend ({kernel_backend})")

        ctx.save_for_backward(gate, up)
        ctx.kernel_backend = kernel_backend

        return output

    @staticmethod
    @ensure_contiguous
    def backward(ctx, output_grad: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, None]:
        gate, up = ctx.saved_tensors
        gate_grad = empty_like_contiguous(gate)
        up_grad = empty_like_contiguous(up)
        kernel_backend = ctx.kernel_backend

        if kernel_backend == KernelBackend.cuda:
            swiglu_backward_cuda(
                gate=gate, up=up, output_grad=output_grad, gate_grad=gate_grad, up_grad=up_grad, BLOCK_SIZE=1024
            )
        elif kernel_backend == KernelBackend.triton:
            swiglu_backward_triton(gate=gate, up=up, output_grad=output_grad, gate_grad=gate_grad, up_grad=up_grad)
        else:
            raise ValueError("unexpected kernel_backend")

        return gate_grad, up_grad, None


class _SwigluPacked(torch.autograd.Function):
    @staticmethod
    @ensure_contiguous
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(x)

        output = torch.empty(*x.size()[:-1], divide_if_divisible(x.size(-1), 2), device=x.device, dtype=x.dtype)
        up, gate = x.chunk(2, dim=-1)

        swiglu_forward_triton(gate=gate, up=up, output=output)

        return output

    @staticmethod
    @ensure_contiguous
    def backward(ctx, output_grad: torch.Tensor) -> tuple[torch.Tensor | None]:
        x: torch.Tensor = ctx.saved_tensors[0]
        x_grad = empty_like_contiguous(x)

        up, gate = x.chunk(2, dim=-1)
        up_grad, gate_grad = x_grad.chunk(2, dim=-1)

        swiglu_backward_triton(gate=gate, up=up, output_grad=output_grad, gate_grad=gate_grad, up_grad=up_grad)

        return x_grad


def swiglu(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """computes swiglu activation as `up` * `gate` * sigmoid(`gate`)

    Args:
        gate (torch.Tensor): `gate` activation tensor
        up (torch.Tensor): `up` activation tensor

    Returns:
        torch.Tensor: output tensor
    """

    assert gate.size() == up.size(), "tensors gate and up should have same shape"
    assert gate.type() == up.type(), "tensors gate and up should have same dtype"

    kernel_backend = KernelBackend.get_kernel_backend_from_device(gate)

    if kernel_backend == KernelBackend.torch:
        dtype = gate.dtype

        gate = gate.float()
        up = up.float()

        output = up * F.silu(gate)
        output = output.to(dtype)
    else:
        output = _Swiglu.apply(gate, up, kernel_backend)

    return output


def swiglu_packed(
    x: torch.Tensor,
    *,
    kernel_backend_forward: KernelBackend = KernelBackend.triton,
    kernel_backend_backward: KernelBackend = KernelBackend.triton,
) -> torch.Tensor:
    """computes swiglu activation by splitting the tensor `x` into 2 parts: gate and up activations

    Args:
        x (torch.Tensor): input activation
        kernel_backend_forward (KernelBackend, optional): kernel backend to prioritize. Defaults
            to KernelBackend.triton.
        kernel_backend_backward (KernelBackend, optional): kernel backend to prioritize. Defaults
            to KernelBackend.triton.

    Returns:
        torch.Tensor: output tensor
    """

    if kernel_backend_forward == KernelBackend.torch:
        up, gate = x.chunk(2, dim=-1)

        output = swiglu(
            gate=gate,
            up=up,
            kernel_backend_forward=kernel_backend_forward,
            kernel_backend_backward=kernel_backend_backward,
        )
    else:
        assert kernel_backend_forward == KernelBackend.triton
        assert kernel_backend_backward == KernelBackend.triton

        output = _SwigluPacked.apply(x)

    return output
