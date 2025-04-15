import torch

from ...math import ceil_divide, divide_if_divisible
from ...utils import ensure_contiguous, get_num_elements_and_hidden_size
from .backward import _backward
from .torch_implementation import swiglu_unchunked_torch
from .triton_implementation import _swiglu_unchunked_forward_triton_kernel


class _SwigluUnchunked_Cute(torch.autograd.Function):
    @staticmethod
    @ensure_contiguous
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(x)

        B, H = get_num_elements_and_hidden_size(x)
        BLOCK_SIZE_B = 64
        BLOCK_SIZE_H = 64

        output = torch.empty(*x.size()[:-1], divide_if_divisible(H, 2), device=x.device, dtype=x.dtype)

        with torch.cuda.device(x.device):
            _swiglu_unchunked_forward_triton_kernel[(ceil_divide(B, BLOCK_SIZE_B), ceil_divide(H, BLOCK_SIZE_H))](
                x_ptr=x,
                output_ptr=output,
                B=B,
                H=H,
                BLOCK_SIZE_B=BLOCK_SIZE_B,
                BLOCK_SIZE_H=BLOCK_SIZE_H,
            )

        return output

    @staticmethod
    @ensure_contiguous
    def backward(ctx, output_grad: torch.Tensor) -> tuple[torch.Tensor | None]:
        x = ctx.saved_tensors[0]

        x_grad = _backward(
            x=x,
            output_grad=output_grad,
            BLOCK_SIZE_B=64,
            BLOCK_SIZE_H=64,
        )

        return x_grad


def swiglu_unchunked_cute(x: torch.Tensor) -> torch.Tensor:
    return _SwigluUnchunked_Cute.apply(x)
