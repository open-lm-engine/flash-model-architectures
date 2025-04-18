import torch

from ...constants import MAX_TRITON_BLOCK_SIZE
from ...math import ceil_divide, get_next_power_of_2
from ...utils import ensure_contiguous, get_num_elements_and_hidden_size, get_sm_count
from .torch_implementation import rmsnorm_torch
from .triton_implementation import _rmsnorm_backward_triton_kernel, _rmsnorm_forward_triton_kernel


class _RMSNorm_Cute(torch.autograd.Function):
    @staticmethod
    @ensure_contiguous
    def forward(
        ctx, x: torch.Tensor, weight: torch.Tensor | None, eps: float | None, memory_efficient: bool
    ) -> torch.Tensor:
        if weight is not None:
            assert weight.dim() == 1, "weight should be 1D"
            assert weight.size(-1) == x.size(-1), "hidden size for x and weight tensor is different"
            assert weight.type() == x.type(), "tensors weight and y should have same dtype"

        is_x_1d = x.dim() == 1
        if is_x_1d:
            x = x.unsqueeze(0)

        if eps is None:
            eps = torch.finfo(x.dtype).eps

        B, H = get_num_elements_and_hidden_size(x)
        BLOCK_SIZE_B = 1
        BLOCK_SIZE_H = get_next_power_of_2(H)
        assert BLOCK_SIZE_H <= MAX_TRITON_BLOCK_SIZE

        output = torch.empty_like(x)
        rmsnorm_denominator = None if memory_efficient else torch.empty(B, device=x.device, dtype=torch.float32)

        with torch.cuda.device(x.device):
            _rmsnorm_forward_triton_kernel[ceil_divide(B, BLOCK_SIZE_B),](
                x_ptr=x,
                has_weight=weight is not None,
                weight_ptr=weight,
                output_ptr=output,
                eps=eps,
                has_rmsnorm_denominator=rmsnorm_denominator is not None,
                rmsnorm_denominator_ptr=rmsnorm_denominator,
                B=B,
                H=H,
                BLOCK_SIZE_B=BLOCK_SIZE_B,
                BLOCK_SIZE_H=BLOCK_SIZE_H,
            )

        if is_x_1d:
            output = output.squeeze(0)

        ctx.save_for_backward(x, weight, rmsnorm_denominator)
        ctx.is_x_1d = is_x_1d
        ctx.eps = eps

        return output

    @staticmethod
    @ensure_contiguous
    def backward(ctx, output_grad: torch.Tensor) -> tuple[torch.Tensor | None]:
        x, weight, rmsnorm_denominator = ctx.saved_tensors

        B, H = get_num_elements_and_hidden_size(x)
        BLOCK_SIZE_B = 1
        BLOCK_SIZE_H = get_next_power_of_2(H)
        assert BLOCK_SIZE_H <= MAX_TRITON_BLOCK_SIZE

        x_grad = torch.empty_like(x)
        weight_grad = None if weight is None else torch.zeros_like(weight, dtype=torch.float32)

        sm_count = get_sm_count(x.device)
        num_programs = min(sm_count, ceil_divide(B, BLOCK_SIZE_B))

        with torch.cuda.device(x.device):
            _rmsnorm_backward_triton_kernel[num_programs,](
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

        if ctx.is_x_1d:
            x_grad = x_grad.squeeze(0)

        return x_grad, weight_grad, *[None] * 2


def rmsnorm_cute(
    x: torch.Tensor, weight: torch.Tensor | None, eps: float | None, memory_efficient: bool = False
) -> torch.Tensor:
    return _RMSNorm_Cute.apply(x, weight, eps, memory_efficient)
