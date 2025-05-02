import torch

from ...constants import MAX_TRITON_BLOCK_SIZE
from ...math import ceil_divide, get_next_power_of_2
from ...utils import ensure_contiguous, get_num_elements_and_hidden_size, get_sm_count
from .torch_implementation import rmsnorm_torch
from .triton_implementation import rmsnorm_backward_triton_kernel, rmsnorm_forward_triton_kernel


class _RMSNorm_Cute(torch.autograd.Function):
    @staticmethod
    @ensure_contiguous
    def forward(
        ctx,
        x: torch.Tensor,
        weight: torch.Tensor | None,
        eps: float | None,
        memory_efficient: bool,
        BLOCK_SIZE_B_forward: int,
        BLOCK_SIZE_B_backward: int,
    ) -> torch.Tensor:
        if weight is not None:
            assert weight.dim() == 1, "weight should be 1D"
            assert weight.size(-1) == x.size(-1), "hidden size for x and weight tensor is different"
            assert weight.type() == x.type(), "tensors weight and y should have same dtype"

        if eps is None:
            eps = torch.finfo(x.dtype).eps

        B, H = get_num_elements_and_hidden_size(x)

        BLOCK_SIZE_H = get_next_power_of_2(H)
        assert BLOCK_SIZE_H <= MAX_TRITON_BLOCK_SIZE

        output = torch.empty_like(x)
        rmsnorm_denominator = None if memory_efficient else torch.empty(B, device=x.device, dtype=torch.float32)

        with torch.cuda.device(x.device):
            rmsnorm_forward_triton_kernel[ceil_divide(B, BLOCK_SIZE_B_forward),](
                x_ptr=x,
                has_weight=weight is not None,
                weight_ptr=weight,
                output_ptr=output,
                eps=eps,
                has_rmsnorm_denominator=rmsnorm_denominator is not None,
                rmsnorm_denominator_ptr=rmsnorm_denominator,
                B=B,
                H=H,
                BLOCK_SIZE_B=BLOCK_SIZE_B_forward,
                BLOCK_SIZE_H=BLOCK_SIZE_H,
            )

        ctx.save_for_backward(x, weight, rmsnorm_denominator)
        ctx.eps = eps
        ctx.BLOCK_SIZE_B_backward = BLOCK_SIZE_B_backward
        ctx.BLOCK_SIZE_H = BLOCK_SIZE_H

        return output

    @staticmethod
    @ensure_contiguous
    def backward(ctx, output_grad: torch.Tensor) -> tuple[torch.Tensor | None]:
        x, weight, rmsnorm_denominator = ctx.saved_tensors
        x_grad = torch.empty_like(x)
        weight_grad = None if weight is None else torch.zeros_like(weight, dtype=torch.float32)

        B, H = get_num_elements_and_hidden_size(x)
        BLOCK_SIZE_B = ctx.BLOCK_SIZE_B_backward
        BLOCK_SIZE_H = ctx.BLOCK_SIZE_H

        sm_count = get_sm_count(x.device)
        num_programs = min(sm_count, ceil_divide(B, BLOCK_SIZE_B))

        with torch.cuda.device(x.device):
            rmsnorm_backward_triton_kernel[num_programs,](
                x_ptr=x,
                has_weight=weight is not None,
                weight_ptr=weight,
                output_grad_ptr=output_grad,
                x_grad_ptr=x_grad,
                weight_grad_ptr=weight_grad,
                eps=ctx.eps,
                has_rmsnorm_denominator=rmsnorm_denominator is not None,
                rmsnorm_denominator_ptr=rmsnorm_denominator,
                B=B,
                H=H,
                BLOCK_SIZE_B=BLOCK_SIZE_B,
                BLOCK_SIZE_H=BLOCK_SIZE_H,
            )

        if weight_grad is not None:
            weight_grad = weight_grad.type_as(weight)

        return x_grad, weight_grad, *[None] * 4


def rmsnorm_cute(
    x: torch.Tensor,
    weight: torch.Tensor | None,
    eps: float | None,
    memory_efficient: bool = False,
    *,
    BLOCK_SIZE_B_forward: int = 1,
    BLOCK_SIZE_B_backward: int = 1,
) -> torch.Tensor:
    """RMSNorm computation

    Args:
        x (torch.Tensor): input activation
        weight (torch.Tensor | None): RMSNorm weight
        eps (float | None): epsilon
        memory_efficient (bool, optional): memory efficient = False caches RMSNorm's denominator in the forward.
            Defaults to False.
        BLOCK_SIZE_B_forward (int, optional): block size along the batch dimension for forward. Defaults to 1.
        BLOCK_SIZE_B_backward (int, optional): block size along the batch dimension for backward. Defaults to 1.

    Returns:
        torch.Tensor: output tensor
    """

    return _RMSNorm_Cute.apply(x, weight, eps, memory_efficient, BLOCK_SIZE_B_forward, BLOCK_SIZE_B_backward)
