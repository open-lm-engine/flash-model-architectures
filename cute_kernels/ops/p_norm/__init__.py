# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import torch
import torch.nn.functional as F

from ...cutotune import CutoTuneParameter
from ...kernel_backend import KernelBackend
from ...utils import ensure_contiguous, get_num_elements_and_hidden_size
from .triton_implementation import norm_2_backward_triton, norm_2_forward_triton


class _P_Norm_Cute(torch.autograd.Function):
    @staticmethod
    @ensure_contiguous
    def forward(
        ctx,
        x: torch.Tensor,
        weight: torch.Tensor | None,
        p: int | str,
        eps: float | None,
        memory_efficient: bool,
        kernel_backend: KernelBackend | CutoTuneParameter,
    ) -> torch.Tensor:
        assert p == 2
        assert kernel_backend == KernelBackend.triton or isinstance(kernel_backend, CutoTuneParameter)

        if weight is not None:
            assert weight.dim() == 1, "weight should be 1D"
            assert weight.size(-1) == x.size(-1), "hidden size for x and weight tensor is different"
            assert weight.type() == x.type(), "tensors weight and y should have same dtype"

        if eps is None:
            eps = torch.finfo(x.dtype).eps

        B, _ = get_num_elements_and_hidden_size(x)

        output = torch.empty_like(x)
        p_norm_denominator = None if memory_efficient else torch.empty(B, device=x.device, dtype=torch.float32)

        norm_2_forward_triton(x=x, weight=weight, output=output, eps=eps, p_norm_denominator=p_norm_denominator)

        ctx.save_for_backward(x, weight, p_norm_denominator)
        ctx.p = p
        ctx.eps = eps

        return output

    @staticmethod
    @ensure_contiguous
    def backward(ctx, output_grad: torch.Tensor) -> tuple[torch.Tensor | None]:
        x, weight, p_norm_denominator = ctx.saved_tensors
        x_grad = torch.empty_like(x)
        weight_grad = None if weight is None else torch.zeros_like(weight, dtype=torch.float32)

        norm_2_backward_triton(
            x=x,
            weight=weight,
            output_grad=output_grad,
            p_norm_denominator=p_norm_denominator,
            x_grad=x_grad,
            weight_grad=weight_grad,
            eps=ctx.eps,
        )

        if weight_grad is not None:
            weight_grad = weight_grad.type_as(weight)

        return x_grad, weight_grad, *[None] * 4


def p_norm_cute(
    x: torch.Tensor,
    p: int,
    weight: torch.Tensor | None,
    eps: float | None,
    memory_efficient: bool = False,
    *,
    kernel_backend: KernelBackend | CutoTuneParameter = KernelBackend.triton,
) -> torch.Tensor:
    """P Norm computation

    Args:
        x (torch.Tensor): input activation
        weight (torch.Tensor | None): RMSNorm weight
        eps (float | None): epsilon
        memory_efficient (bool, optional): memory efficient = False caches RMSNorm's denominator in the forward.
            Defaults to False.
        kernel_backend (KernelBackend | CutoTuneParameter, optional): kernel backend to prioritize.
            Defaults to KernelBackend.triton.

    Returns:
        torch.Tensor: output tensor
    """

    if kernel_backend == KernelBackend.torch:
        if p == "inf":
            x = x / x.abs().max(dim=-1, keepdim=True).values
        else:
            x = F.normalize(x, p=p, dim=-1, eps=eps)

        if weight is not None:
            x = x * weight
    else:
        x = _P_Norm_Cute.apply(x, weight, p, eps, memory_efficient, kernel_backend)

    return x
