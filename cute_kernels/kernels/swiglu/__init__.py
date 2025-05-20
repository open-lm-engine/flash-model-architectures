# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import torch

from ...cutotune import CutoTuneConfig, CutoTuneParameter, cutotune
from ...kernel_backend import KernelBackend, is_cuda_kernel_backend_allowed, is_triton_kernel_backend_allowed
from ...math import divide_if_divisible
from ...utils import ensure_contiguous, is_nvidia_gpu
from .cuda_implementation import swiglu_backward_cuda, swiglu_forward_cuda
from .torch_implementation import swiglu_packed_torch, swiglu_torch
from .triton_implementation import swiglu_backward_triton, swiglu_forward_triton


@cutotune(
    configs=[
        CutoTuneConfig(
            {"kernel_backend": KernelBackend.cuda},
            condition=lambda **kwargs: is_cuda_kernel_backend_allowed(kwargs["kernel_backend"]) and is_nvidia_gpu(),
        ),
        CutoTuneConfig(
            {"kernel_backend": KernelBackend.triton},
            condition=lambda **kwargs: is_triton_kernel_backend_allowed(kwargs["kernel_backend"]),
        ),
    ],
    triggers={"gate.dtype"},
)
def _forward(gate: torch.Tensor, up: torch.Tensor, output: torch.Tensor, kernel_backend: torch.Tensor) -> None:
    if kernel_backend == KernelBackend.cuda:
        swiglu_forward_cuda(gate=gate, up=up, output=output, BLOCK_SIZE=CutoTuneParameter())
    elif kernel_backend == KernelBackend.triton:
        swiglu_forward_triton(gate=gate, up=up, output=output)
    else:
        raise ValueError("unexpected kernel_backend")


@cutotune(
    configs=[
        CutoTuneConfig(
            {"kernel_backend": KernelBackend.cuda},
            condition=lambda **kwargs: is_cuda_kernel_backend_allowed(kwargs["kernel_backend"]) and is_nvidia_gpu(),
        ),
        CutoTuneConfig(
            {"kernel_backend": KernelBackend.triton},
            condition=lambda **kwargs: is_triton_kernel_backend_allowed(kwargs["kernel_backend"]),
        ),
    ],
    triggers={"gate.dtype"},
)
def _backward(
    gate: torch.Tensor,
    up: torch.Tensor,
    output_grad: torch.Tensor,
    gate_grad: torch.Tensor,
    up_grad: torch.Tensor,
    kernel_backend: torch.Tensor,
) -> None:
    if kernel_backend == KernelBackend.cuda:
        swiglu_backward_cuda(
            gate=gate,
            up=up,
            output_grad=output_grad,
            gate_grad=gate_grad,
            up_grad=up_grad,
            BLOCK_SIZE=CutoTuneParameter(),
        )
    elif kernel_backend == KernelBackend.triton:
        swiglu_backward_triton(gate=gate, up=up, output_grad=output_grad, gate_grad=gate_grad, up_grad=up_grad)
    else:
        raise ValueError("unexpected kernel_backend")


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
        _forward(gate=gate, up=up, output=output, kernel_backend=kernel_backend_forward)

        return output

    @staticmethod
    @ensure_contiguous
    def backward(ctx, output_grad: torch.Tensor) -> tuple[torch.Tensor | None]:
        gate, up = ctx.saved_tensors
        gate_grad = torch.empty_like(gate)
        up_grad = torch.empty_like(up)

        _backward(
            gate=gate,
            up=up,
            output_grad=output_grad,
            gate_grad=gate_grad,
            up_grad=up_grad,
            kernel_backend=ctx.kernel_backend_backward,
        )

        return gate_grad, up_grad, None, None


class _SwigluPacked_Cute(torch.autograd.Function):
    @staticmethod
    @ensure_contiguous
    def forward(
        ctx,
        x: torch.Tensor,
        kernel_backend_forward: KernelBackend | CutoTuneParameter,
        kernel_backend_backward: KernelBackend | CutoTuneParameter,
    ) -> torch.Tensor:
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
    kernel_backend_forward: KernelBackend | CutoTuneParameter = CutoTuneParameter(),
    kernel_backend_backward: KernelBackend | CutoTuneParameter = CutoTuneParameter(),
) -> torch.Tensor:
    """computes swiglu activation as `up` * `gate` * sigmoid(`gate`)

    Args:
        gate (torch.Tensor): `gate` activation tensor
        up (torch.Tensor): `up` activation tensor
        kernel_backend_forward (KernelBackend | CutoTuneParameter, optional): kernel backend to prioritize. Defaults
            to CutoTuneParameter().
        kernel_backend_backward (KernelBackend | CutoTuneParameter, optional): kernel backend to prioritize. Defaults
            to CutoTuneParameter().

    Returns:
        torch.Tensor: output tensor
    """

    return _Swiglu_Cute.apply(gate, up, kernel_backend_forward, kernel_backend_backward)


def swiglu_packed_cute(
    x: torch.Tensor,
    *,
    kernel_backend_forward: KernelBackend | CutoTuneParameter = CutoTuneParameter(),
    kernel_backend_backward: KernelBackend | CutoTuneParameter = CutoTuneParameter(),
) -> torch.Tensor:
    """computes swiglu activation by splitting the tensor `x` into 2 parts: gate and up activations

    Args:
        x (torch.Tensor): input activation
        kernel_backend_forward (KernelBackend | CutoTuneParameter, optional): kernel backend to prioritize. Defaults
            to CutoTuneParameter().
        kernel_backend_backward (KernelBackend | CutoTuneParameter, optional): kernel backend to prioritize. Defaults
            to CutoTuneParameter().

    Returns:
        torch.Tensor: output tensor
    """

    return _SwigluPacked_Cute.apply(x, kernel_backend_forward, kernel_backend_backward)
