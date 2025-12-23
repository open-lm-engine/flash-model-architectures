# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import torch
import torch.nn.functional as F

from ...accelerator import KernelBackend
from ...custom_op import CustomOp, ctx_save_for_backward
from ...utils import (
    empty_like_contiguous,
    is_cute_dsl_available,
    is_torch_neuronx_available,
    is_torch_xla_available,
    is_triton_available,
    make_contiguous,
)


if is_cute_dsl_available():
    from .cuda_implementation import swiglu_backward_cuda, swiglu_forward_cuda

if is_torch_neuronx_available():
    from .nki_implementation import swiglu_backward_nki, swiglu_forward_nki

if is_torch_xla_available():
    from .pallas_implementation import swiglu_backward_pallas, swiglu_forward_pallas

if is_triton_available():
    from .triton_implementation import swiglu_backward_triton, swiglu_forward_triton


class _Swiglu(CustomOp):
    @staticmethod
    def forward_backward_torch(g: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        dtype = g.dtype

        g = g.float()
        u = u.float()

        y = u * F.silu(g)
        y = y.to(dtype)

        return y

    @staticmethod
    def forward(ctx, g: torch.Tensor, u: torch.Tensor, kernel_backend: KernelBackend) -> torch.Tensor:
        ctx.kernel_backend = kernel_backend

        if kernel_backend in [KernelBackend.cuda, KernelBackend.pallas]:
            g = make_contiguous(g)
            u = make_contiguous(u)

        ctx_save_for_backward(ctx, g, u)

        if kernel_backend == KernelBackend.pallas:
            return swiglu_forward_pallas(g=g, u=u)

        y = empty_like_contiguous(g)

        if kernel_backend == KernelBackend.cuda:
            swiglu_forward_cuda(g=g, u=u, y=y)
        elif kernel_backend == KernelBackend.nki:
            swiglu_forward_nki(g=g, u=u, y=y)
        elif kernel_backend == KernelBackend.triton:
            swiglu_forward_triton(g=g, u=u, y=y)
        else:
            raise ValueError(f"unexpected kernel_backend ({kernel_backend})")

        return y

    @staticmethod
    def backward(ctx, dy: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        g, u = ctx.saved_tensors
        kernel_backend = ctx.kernel_backend

        if kernel_backend == KernelBackend.pallas:
            dg, du = swiglu_backward_pallas(g=g, u=u, dy=dy)
            return dg, du, None

        if kernel_backend == KernelBackend.cuda:
            dy = make_contiguous(dy)

        dg = empty_like_contiguous(g)
        du = empty_like_contiguous(u)

        if kernel_backend == KernelBackend.cuda:
            swiglu_backward_cuda(g=g, u=u, dy=dy, dg=dg, du=du)
        elif kernel_backend == KernelBackend.nki:
            swiglu_backward_nki(g=g, u=u, dy=dy, dg=dg, du=du)
        elif kernel_backend == KernelBackend.triton:
            swiglu_backward_triton(g=g, u=u, dy=dy, dg=dg, du=du)
        else:
            raise ValueError(f"unexpected kernel_backend ({kernel_backend})")

        return dg, du, None


def swiglu(gate: torch.Tensor, up: torch.Tensor, *, kernel_backend: KernelBackend | None = None) -> torch.Tensor:
    """computes swiglu activation as `up` * `gate` * sigmoid(`gate`)

    Args:
        gate (torch.Tensor): `gate` activation tensor
        up (torch.Tensor): `up` activation tensor

    Returns:
        torch.Tensor: output tensor
    """

    assert gate.size() == up.size(), "tensors gate and up should have same shape"
    assert gate.type() == up.type(), "tensors gate and up should have same dtype"

    original_shape = gate.size()
    gate = gate.flatten(0, -2)
    up = up.flatten(0, -2)

    y = _Swiglu.run(g=gate, u=up, kernel_backend=kernel_backend)
    y = y.view(original_shape)

    return y
