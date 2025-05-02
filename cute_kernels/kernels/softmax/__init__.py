import torch

from ...math import ceil_divide
from ...utils import ensure_contiguous, get_num_elements_and_hidden_size
from .torch_implementation import softmax_torch
from .triton_implementation import softmax_backward_triton, softmax_forward_triton_kernel


class _Softmax_Cute(torch.autograd.Function):
    @staticmethod
    @ensure_contiguous
    def forward(
        ctx,
        x: torch.Tensor,
        logits_multiplier: float | None,
        BLOCK_SIZE_B_forward: int,
        BLOCK_SIZE_H_forward: int,
        BLOCK_SIZE_B_backward: int,
        BLOCK_SIZE_H_backward: int,
    ) -> torch.Tensor:
        output = torch.empty_like(x)

        if x.dim() == 1:
            B = 1
            H = x.size(-1)
        else:
            B, H = get_num_elements_and_hidden_size(x)

        with torch.cuda.device(x.device):
            softmax_forward_triton_kernel[ceil_divide(B, BLOCK_SIZE_B_forward),](
                x_ptr=x,
                output_ptr=output,
                has_logits_multiplier=logits_multiplier not in [None, 1],
                logits_multiplier=logits_multiplier,
                B=B,
                H=H,
                BLOCK_SIZE_B=BLOCK_SIZE_B_forward,
                BLOCK_SIZE_H=BLOCK_SIZE_H_forward,
            )

        ctx.save_for_backward(output)
        ctx.logits_multiplier = logits_multiplier
        ctx.BLOCK_SIZE_B_backward = BLOCK_SIZE_B_backward
        ctx.BLOCK_SIZE_H_backward = BLOCK_SIZE_H_backward

        return output

    @staticmethod
    @ensure_contiguous
    def backward(ctx, output_grad: torch.Tensor) -> tuple[torch.Tensor | None]:
        output = ctx.saved_tensors[0]
        x_grad = torch.empty_like(output)

        softmax_backward_triton(
            output=output,
            output_grad=output_grad,
            x_grad=x_grad,
            logits_multiplier=ctx.logits_multiplier,
            BLOCK_SIZE_B=ctx.BLOCK_SIZE_B_backward,
            BLOCK_SIZE_H=ctx.BLOCK_SIZE_H_backward,
        )

        return x_grad, *[None] * 5


def softmax_cute(
    x: torch.Tensor,
    logits_multiplier: float | None = None,
    *,
    BLOCK_SIZE_B_forward: int = 1,
    BLOCK_SIZE_H_forward: int = 8192,
    BLOCK_SIZE_B_backward: int = 1,
    BLOCK_SIZE_H_backward: int = 8192,
) -> torch.Tensor:
    """computes softmax activation

    Args:
        x (torch.Tensor): input activation tensor
        logits_multiplier (float, optional): pre-multiplies `x` with `logits_multiplier` before computing softmax.
            Defaults to None.
        BLOCK_SIZE_B_forward (int, optional): block size for forward along batch dimension for forward. Defaults to 1.
        BLOCK_SIZE_H_forward (int, optional): block size for forward along hidden dimension for forward. Defaults to
            1.
        BLOCK_SIZE_B_backward (int, optional): block size for backward along batch dimension for backward. Defaults to
            8192.
        BLOCK_SIZE_H_backward (int, optional): block size for backward along hidden dimension for backward. Defaults
            to 8192.

    Returns:
        torch.Tensor: output tensor
    """

    return _Softmax_Cute.apply(
        x, logits_multiplier, BLOCK_SIZE_B_forward, BLOCK_SIZE_H_forward, BLOCK_SIZE_B_backward, BLOCK_SIZE_H_backward
    )
