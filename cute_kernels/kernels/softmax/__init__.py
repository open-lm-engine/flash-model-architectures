import torch

from ...math import ceil_divide
from ...utils import ensure_contiguous, get_num_elements_and_hidden_size
from .torch_implementation import softmax_torch
from .triton_implementation import _softmax_forward_triton_kernel, softmax_backward_triton


class _Softmax_Cute(torch.autograd.Function):
    @staticmethod
    @ensure_contiguous
    def forward(ctx, x: torch.Tensor, logits_multiplier: float) -> torch.Tensor:
        if x.size(-1) == 1:
            return torch.ones_like(x)

        ctx.save_for_backward(x)

        is_x_1d = x.dim() == 1
        if is_x_1d:
            x = x.unsqueeze(0)

        output = torch.empty_like(x)
        B, H = get_num_elements_and_hidden_size(x)
        BLOCK_SIZE_B = 1
        BLOCK_SIZE_H = 8192

        with torch.cuda.device(x.device):
            _softmax_forward_triton_kernel[ceil_divide(B, BLOCK_SIZE_B),](
                x_ptr=x,
                output_ptr=output,
                logits_multiplier=logits_multiplier,
                B=B,
                H=H,
                BLOCK_SIZE_B=BLOCK_SIZE_B,
                BLOCK_SIZE_H=BLOCK_SIZE_H,
            )

        if is_x_1d:
            output = output.squeeze(0)

        ctx.save_for_backward(output)
        ctx.logits_multiplier = logits_multiplier

        return output

    @staticmethod
    @ensure_contiguous
    def backward(ctx, output_grad: torch.Tensor) -> tuple[torch.Tensor | None]:
        if output_grad.size(-1) == 1:
            x_grad = torch.zeros_like(output_grad)
        else:
            output = ctx.saved_tensors[0]
            x_grad = torch.empty_like(output)
            BLOCK_SIZE_B = 1
            BLOCK_SIZE_H = 8192

            softmax_backward_triton(
                output=output,
                output_grad=output_grad,
                x_grad=x_grad,
                logits_multiplier=ctx.logits_multiplier,
                BLOCK_SIZE_B=BLOCK_SIZE_B,
                BLOCK_SIZE_H=BLOCK_SIZE_H,
            )

        return x_grad, *[None] * 8


def softmax_cute(x: torch.Tensor, logits_multiplier: float = 1) -> torch.Tensor:
    """_summary_

    Args:
        x (torch.Tensor): _description_
        logits_multiplier (float, optional): _description_. Defaults to 1.

    Returns:
        torch.Tensor: _description_
    """

    return _Softmax_Cute.apply(x, logits_multiplier)
