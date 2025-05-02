import torch

from ...math import ceil_divide, divide_if_divisible
from ...utils import ensure_contiguous, get_num_elements_and_hidden_size
from .torch_implementation import swiglu_unchunked_torch
from .triton_implementation import swiglu_unchunked_backward_triton_kernel, swiglu_unchunked_forward_triton_kernel


class _SwigluUnchunked_Cute(torch.autograd.Function):
    @staticmethod
    @ensure_contiguous
    def forward(
        ctx,
        x: torch.Tensor,
        BLOCK_SIZE_B_forward: int,
        BLOCK_SIZE_H_forward: int,
        BLOCK_SIZE_B_backward: int,
        BLOCK_SIZE_H_backward: int,
    ) -> torch.Tensor:
        ctx.save_for_backward(x)
        ctx.BLOCK_SIZE_B_backward = BLOCK_SIZE_B_backward
        ctx.BLOCK_SIZE_H_backward = BLOCK_SIZE_H_backward

        B, H = get_num_elements_and_hidden_size(x)
        output = torch.empty(*x.size()[:-1], divide_if_divisible(H, 2), device=x.device, dtype=x.dtype)

        with torch.cuda.device(x.device):
            swiglu_unchunked_forward_triton_kernel[
                ceil_divide(B, BLOCK_SIZE_B_forward), ceil_divide(H, BLOCK_SIZE_H_forward)
            ](
                x_ptr=x,
                output_ptr=output,
                B=B,
                H=H,
                BLOCK_SIZE_B=BLOCK_SIZE_B_forward,
                BLOCK_SIZE_H=BLOCK_SIZE_H_forward,
            )

        return output

    @staticmethod
    @ensure_contiguous
    def backward(ctx, output_grad: torch.Tensor) -> tuple[torch.Tensor | None]:
        x = ctx.saved_tensors[0]
        x_grad = torch.empty_like(x)

        B, H = get_num_elements_and_hidden_size(x)
        BLOCK_SIZE_B = ctx.BLOCK_SIZE_B_backward
        BLOCK_SIZE_H = ctx.BLOCK_SIZE_H_backward

        with torch.cuda.device(x.device):
            swiglu_unchunked_backward_triton_kernel[ceil_divide(B, BLOCK_SIZE_B), ceil_divide(H, BLOCK_SIZE_H)](
                x_ptr=x,
                output_grad_ptr=output_grad,
                x_grad_ptr=x_grad,
                B=B,
                H=H,
                BLOCK_SIZE_B=BLOCK_SIZE_B,
                BLOCK_SIZE_H=BLOCK_SIZE_H,
            )

        return x_grad, *[None] * 4


def swiglu_unchunked_cute(
    x: torch.Tensor,
    *,
    BLOCK_SIZE_B_forward: int = 64,
    BLOCK_SIZE_H_forward: int = 64,
    BLOCK_SIZE_B_backward: int = 64,
    BLOCK_SIZE_H_backward: int = 64,
) -> torch.Tensor:
    """computes swiglu activation by splitting the tensor `x` into 2 parts: gate and up activations

    Args:
        x (torch.Tensor): input activation
        BLOCK_SIZE_B_forward (int, optional): block size for forward along batch dimension for forward. Defaults to
            64.
        BLOCK_SIZE_H_forward (int, optional): block size for forward along hidden dimension for forward. Defaults to
            64.
        BLOCK_SIZE_B_backward (int, optional): block size for forward along batch dimension for forward. Defaults to
            64.
        BLOCK_SIZE_H_backward (int, optional): block size for forward along hidden dimension for forward. Defaults to
            64.

    Returns:
        torch.Tensor: output tensor
    """

    return _SwigluUnchunked_Cute.apply(
        x, BLOCK_SIZE_B_forward, BLOCK_SIZE_H_forward, BLOCK_SIZE_B_backward, BLOCK_SIZE_H_backward
    )
