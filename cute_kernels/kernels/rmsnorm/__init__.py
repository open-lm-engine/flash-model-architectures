import torch

from ...constants import MAX_TRITON_BLOCK_SIZE
from ...cutotune import CutoTuneParameter
from ...math import ceil_divide, get_next_power_of_2
from ...utils import ensure_contiguous, get_num_elements_and_hidden_size
from .backward import _backward
from .torch_implementation import rmsnorm_torch
from .triton_implementation import _rmsnorm_forward_triton_kernel


class _RMSNorm_Cute(torch.autograd.Function):
    @staticmethod
    @ensure_contiguous
    def forward(
        ctx,
        x: torch.Tensor,
        weight: torch.Tensor | None,
        eps: float | None,
        memory_efficient: bool,
        kernel_backend_backward: str,
        BLOCK_SIZE_B_backward: int,
        BLOCK_SIZE_H_backward: int,
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

        num_elements, hidden_size = get_num_elements_and_hidden_size(x)

        output = torch.empty_like(x)
        rmsnorm_denominator = (
            None if memory_efficient else torch.empty(num_elements, device=x.device, dtype=torch.float32)
        )

        BLOCK_SIZE_B = 1
        BLOCK_SIZE_H = get_next_power_of_2(hidden_size)
        assert BLOCK_SIZE_H <= MAX_TRITON_BLOCK_SIZE

        num_elements, hidden_size = get_num_elements_and_hidden_size(x)

        if BLOCK_SIZE_H < hidden_size:
            raise ValueError(f"hidden_size should be more than the BLOCK_SIZE_H")

        with torch.cuda.device(x.device):
            _rmsnorm_forward_triton_kernel[(ceil_divide(num_elements, BLOCK_SIZE_B),)](
                x_ptr=x,
                has_weight=weight is not None,
                weight_ptr=weight,
                output_ptr=output,
                eps=eps,
                has_rmsnorm_denominator=rmsnorm_denominator is not None,
                rmsnorm_denominator_ptr=rmsnorm_denominator,
                B=num_elements,
                H=hidden_size,
                BLOCK_SIZE_B=BLOCK_SIZE_B,
                BLOCK_SIZE_H=BLOCK_SIZE_H,
            )

        if is_x_1d:
            output = output.squeeze(0)

        ctx.save_for_backward(x, weight, rmsnorm_denominator)
        ctx.is_x_1d = is_x_1d
        ctx.kernel_backend_backward = kernel_backend_backward
        ctx.eps = eps
        ctx.BLOCK_SIZE_B_backward = BLOCK_SIZE_B_backward
        ctx.BLOCK_SIZE_H_backward = BLOCK_SIZE_H_backward

        return output

    @staticmethod
    @ensure_contiguous
    def backward(ctx, output_grad: torch.Tensor) -> tuple[torch.Tensor | None]:
        x, weight, rmsnorm_denominator = ctx.saved_tensors

        x_grad, weight_grad = _backward(
            x=x,
            weight=weight,
            eps=ctx.eps,
            rmsnorm_denominator=rmsnorm_denominator,
            output_grad=output_grad,
            kernel_backend=ctx.kernel_backend_backward,
            BLOCK_SIZE_B=ctx.BLOCK_SIZE_B_backward,
            BLOCK_SIZE_H=ctx.BLOCK_SIZE_H_backward,
        )

        if ctx.is_x_1d:
            x_grad = x_grad.squeeze(0)

        return x_grad, weight_grad, *[None] * 8


def rmsnorm_cute(
    x: torch.Tensor,
    weight: torch.Tensor | None,
    eps: float | None,
    memory_efficient: bool = False,
    kernel_backend_backward: str = CutoTuneParameter(),
    BLOCK_SIZE_B_backward: int = CutoTuneParameter(),
    BLOCK_SIZE_H_backward: int = CutoTuneParameter(),
) -> torch.Tensor:
    return _RMSNorm_Cute.apply(
        x,
        weight,
        eps,
        memory_efficient,
        kernel_backend_backward,
        BLOCK_SIZE_B_backward,
        BLOCK_SIZE_H_backward,
    )
