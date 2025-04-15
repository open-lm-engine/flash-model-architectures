import torch

from ...math import ceil_divide
from ...utils import ensure_contiguous
from .torch_implementation import embedding_torch
from .triton_implementation import _embedding_backward_triton_kernel, _embedding_forward_triton_kernel


class _Embedding_Cute(torch.autograd.Function):
    @staticmethod
    @ensure_contiguous
    def forward(ctx, input_ids: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        B = input_ids.numel()
        H = weight.size(-1)
        BLOCK_SIZE_B = 128
        BLOCK_SIZE_H = 128

        output = torch.empty(B, H, dtype=weight.dtype, device=input_ids.device)

        with torch.cuda.device(input_ids.device):
            _embedding_forward_triton_kernel[(ceil_divide(B, BLOCK_SIZE_B), ceil_divide(H, BLOCK_SIZE_H))](
                x_ptr=input_ids,
                weight_ptr=weight,
                output_ptr=output,
                B=B,
                H=H,
                BLOCK_SIZE_B=BLOCK_SIZE_B,
                BLOCK_SIZE_H=BLOCK_SIZE_H,
            )

        output = output.view(*input_ids.size(), H)

        ctx.save_for_backward(input_ids, weight)

        return output

    @staticmethod
    @ensure_contiguous
    def backward(ctx, output_grad: torch.Tensor) -> torch.Tensor:
        input_ids, weight = ctx.saved_tensors

        B = input_ids.numel()
        H = weight_grad.size(-1)
        BLOCK_SIZE_B = 128
        BLOCK_SIZE_H = 128

        weight_grad = torch.zeros_like(weight)

        accumulate_in_fp32 = weight_grad.dtype == torch.bfloat16
        if accumulate_in_fp32:
            weight_grad = weight_grad.float()

        with torch.cuda.device(input_ids.device):
            _embedding_backward_triton_kernel[(ceil_divide(B, BLOCK_SIZE_B), ceil_divide(H, BLOCK_SIZE_H))](
                x_ptr=input_ids,
                output_grad_ptr=output_grad,
                weight_grad_ptr=weight_grad,
                B=B,
                H=H,
                accumulate_in_fp32=accumulate_in_fp32,
                BLOCK_SIZE_B=BLOCK_SIZE_B,
                BLOCK_SIZE_H=BLOCK_SIZE_H,
            )

        return None, weight_grad, *[None] * 6


def embedding_cute(input_ids: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return _Embedding_Cute.apply(input_ids, weight)
