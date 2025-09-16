# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import torch
import torch.nn.functional as F

from ...constants import MAX_TRITON_BLOCK_SIZE
from ...cutotune import CutoTuneParameter
from ...kernel_backend import KernelBackend
from ...math import ceil_divide, get_next_power_of_2
from ...utils import ensure_contiguous, get_num_elements_and_hidden_size, get_sm_count
from .triton_implementation import rmsnorm_backward_triton_kernel, rmsnorm_forward_triton_kernel


class _RMSNorm(torch.autograd.Function):
    @staticmethod
    @ensure_contiguous
    def forward(
        ctx,
        x: torch.Tensor,
        weight: torch.Tensor | None,
        eps: float | None,
        memory_efficient: bool,
        kernel_backend: KernelBackend | CutoTuneParameter,
    ) -> torch.Tensor:
        assert kernel_backend == KernelBackend.triton or isinstance(kernel_backend, CutoTuneParameter)

        if weight is not None:
            assert weight.dim() == 1, "weight should be 1D"
            assert weight.size(-1) == x.size(-1), "hidden size for x and weight tensor is different"
            assert weight.type() == x.type(), "tensors weight and y should have same dtype"

        if eps is None:
            eps = torch.finfo(x.dtype).eps

        B, H = get_num_elements_and_hidden_size(x)

        output = torch.empty_like(x)
        rmsnorm_denominator = None if memory_efficient else torch.empty(B, device=x.device, dtype=torch.float32)

        BLOCK_SIZE_B = 1
        BLOCK_SIZE_H = get_next_power_of_2(H)
        assert BLOCK_SIZE_H <= MAX_TRITON_BLOCK_SIZE
        NUM_WARPS = 8

        rmsnorm_forward_triton_kernel[ceil_divide(B, BLOCK_SIZE_B),](
            x_ptr=x,
            weight_ptr=weight,
            output_ptr=output,
            eps=eps,
            rmsnorm_denominator_ptr=rmsnorm_denominator,
            B=B,
            H=H,
            BLOCK_SIZE_B=BLOCK_SIZE_B,
            BLOCK_SIZE_H=BLOCK_SIZE_H,
            num_warps=NUM_WARPS,
        )

        ctx.save_for_backward(x, weight, rmsnorm_denominator)
        ctx.eps = eps

        return output

    @staticmethod
    @ensure_contiguous
    def backward(ctx, output_grad: torch.Tensor) -> tuple[torch.Tensor | None]:
        x, weight, rmsnorm_denominator = ctx.saved_tensors
        x_grad = torch.empty_like(x)
        weight_grad = None if weight is None else torch.zeros_like(weight, dtype=torch.float32)

        B, H = get_num_elements_and_hidden_size(x)

        BLOCK_SIZE_B = 1
        BLOCK_SIZE_H = get_next_power_of_2(H)
        assert BLOCK_SIZE_H <= MAX_TRITON_BLOCK_SIZE
        NUM_WARPS = 8

        sm_count = get_sm_count(x.device)
        GRID = lambda meta: (min(sm_count, ceil_divide(B, meta["BLOCK_SIZE_B"])),)

        rmsnorm_backward_triton_kernel[GRID](
            x_ptr=x,
            weight_ptr=weight,
            output_grad_ptr=output_grad,
            x_grad_ptr=x_grad,
            weight_grad_ptr=weight_grad,
            eps=ctx.eps,
            rmsnorm_denominator_ptr=rmsnorm_denominator,
            B=B,
            H=H,
            BLOCK_SIZE_B=BLOCK_SIZE_B,
            BLOCK_SIZE_H=BLOCK_SIZE_H,
            num_warps=NUM_WARPS,
        )

        if weight_grad is not None:
            weight_grad = weight_grad.type_as(weight)

        return x_grad, weight_grad, *[None] * 3


def rmsnorm(
    x: torch.Tensor,
    weight: torch.Tensor | None,
    eps: float | None,
    memory_efficient: bool = False,
    *,
    kernel_backend: KernelBackend | CutoTuneParameter = KernelBackend.triton,
) -> torch.Tensor:
    """RMSNorm computation

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
        x = F.rms_norm(x, normalized_shape=x.size(-1), weight=weight, eps=eps)
    else:
        x = _RMSNorm.apply(x, weight, eps, memory_efficient, kernel_backend)

    return x
