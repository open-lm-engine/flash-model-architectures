# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import torch
import torch.nn.functional as F

from ...kernel_backend import KernelBackend
from ...math import divide_if_divisible
from ...utils import ensure_contiguous
from .cuda_implementation import swiglu_backward_cuda, swiglu_forward_cuda
from .triton_implementation import swiglu_backward_triton, swiglu_forward_triton


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

        output = torch.empty_like(gate)

        if kernel_backend_forward == KernelBackend.cuda:
            swiglu_forward_cuda(gate=gate, up=up, output=output, BLOCK_SIZE=1024)
        elif kernel_backend_forward == KernelBackend.triton:
            swiglu_forward_triton(gate=gate, up=up, output=output)
        else:
            raise ValueError(f"unexpected kernel_backend ({kernel_backend_forward})")

        ctx.save_for_backward(gate, up)
        ctx.kernel_backend_backward = kernel_backend_backward

        return output

    @staticmethod
    @ensure_contiguous
    def backward(ctx, output_grad: torch.Tensor) -> tuple[torch.Tensor | None]:
        gate, up = ctx.saved_tensors
        gate_grad = torch.empty_like(gate)
        up_grad = torch.empty_like(up)
        kernel_backend_backward = ctx.kernel_backend_backward

        if kernel_backend_backward == KernelBackend.cuda:
            swiglu_backward_cuda(
                gate=gate, up=up, output_grad=output_grad, gate_grad=gate_grad, up_grad=up_grad, BLOCK_SIZE=1024
            )
        elif kernel_backend_backward == KernelBackend.triton:
            swiglu_backward_triton(gate=gate, up=up, output_grad=output_grad, gate_grad=gate_grad, up_grad=up_grad)
        else:
            raise ValueError("unexpected kernel_backend")

        return gate_grad, up_grad, None, None


class _SwigluPacked_Cute(torch.autograd.Function):
    @staticmethod
    @ensure_contiguous
    def forward(
        ctx,
        x: torch.Tensor,
        kernel_backend_forward: KernelBackend = KernelBackend.cuda,
        kernel_backend_backward: KernelBackend = KernelBackend.cuda,
    ) -> torch.Tensor:
        assert kernel_backend_forward == KernelBackend.triton
        assert kernel_backend_backward == KernelBackend.triton

        ctx.save_for_backward(x)

        output = torch.empty(*x.size()[:-1], divide_if_divisible(x.size(-1), 2), device=x.device, dtype=x.dtype)
        up, gate = x.chunk(2, dim=-1)

        swiglu_forward_triton(gate=gate, up=up, output=output)

        return output

    @staticmethod
    @ensure_contiguous
    def backward(ctx, output_grad: torch.Tensor) -> tuple[torch.Tensor | None]:
        x: torch.Tensor = ctx.saved_tensors[0]
        x_grad = torch.empty_like(x)

        up, gate = x.chunk(2, dim=-1)
        up_grad, gate_grad = x_grad.chunk(2, dim=-1)

        swiglu_backward_triton(gate=gate, up=up, output_grad=output_grad, gate_grad=gate_grad, up_grad=up_grad)

        return x_grad, None, None


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
        kernel_backend_forward (KernelBackend, optional): kernel backend to prioritize. Defaults
            to KernelBackend.cuda.
        kernel_backend_backward (KernelBackend, optional): kernel backend to prioritize. Defaults
            to KernelBackend.cuda.

    Returns:
        torch.Tensor: output tensor
    """

    if kernel_backend_forward == KernelBackend.torch:
        assert kernel_backend_backward == KernelBackend.torch
        dtype = gate.dtype

        gate = gate.float()
        up = up.float()

        output = up * F.silu(gate)
        output = output.to(dtype)
    else:
        output = _Swiglu_Cute.apply(gate, up, kernel_backend_forward, kernel_backend_backward)

    return output


def swiglu_packed_cute(
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

        output = swiglu_cute(
            gate=gate,
            up=up,
            kernel_backend_forward=kernel_backend_forward,
            kernel_backend_backward=kernel_backend_backward,
        )
    else:
        output = _SwigluPacked_Cute.apply(x, kernel_backend_forward, kernel_backend_backward)

    return output
