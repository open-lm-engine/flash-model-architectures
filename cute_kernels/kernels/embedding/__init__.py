import torch

from ...math import ceil_divide
from ...utils import ensure_contiguous
from .torch_implementation import embedding_torch
from .triton_implementation import embedding_backward_triton_kernel, embedding_forward_triton_kernel


class _Embedding_Cute(torch.autograd.Function):
    @staticmethod
    @ensure_contiguous
    def forward(
        ctx,
        input_ids: torch.Tensor,
        weight: torch.Tensor,
        BLOCK_SIZE_B_forward: int,
        BLOCK_SIZE_H_forward: int,
        BLOCK_SIZE_B_backward: int,
        BLOCK_SIZE_H_backward: int,
    ) -> torch.Tensor:
        output = torch.empty(*input_ids.size(), weight.size(-1), dtype=weight.dtype, device=input_ids.device)

        B = input_ids.numel()
        H = weight.size(-1)

        with torch.cuda.device(input_ids.device):
            embedding_forward_triton_kernel[
                ceil_divide(B, BLOCK_SIZE_B_forward), ceil_divide(H, BLOCK_SIZE_H_forward)
            ](
                x_ptr=input_ids,
                weight_ptr=weight,
                output_ptr=output,
                B=B,
                H=H,
                BLOCK_SIZE_B=BLOCK_SIZE_B_forward,
                BLOCK_SIZE_H=BLOCK_SIZE_H_forward,
            )

        ctx.save_for_backward(input_ids, weight)
        ctx.BLOCK_SIZE_B_backward = BLOCK_SIZE_B_backward
        ctx.BLOCK_SIZE_H_backward = BLOCK_SIZE_H_backward

        return output

    @staticmethod
    @ensure_contiguous
    def backward(ctx, output_grad: torch.Tensor) -> torch.Tensor:
        input_ids, weight = ctx.saved_tensors
        weight_grad = torch.zeros_like(weight, dtype=torch.float32)

        B = input_ids.numel()
        H = weight_grad.size(-1)
        BLOCK_SIZE_B = ctx.BLOCK_SIZE_B_backward
        BLOCK_SIZE_H = ctx.BLOCK_SIZE_H_backward

        with torch.cuda.device(input_ids.device):
            embedding_backward_triton_kernel[ceil_divide(B, BLOCK_SIZE_B), ceil_divide(H, BLOCK_SIZE_H)](
                x_ptr=input_ids,
                output_grad_ptr=output_grad,
                weight_grad_ptr=weight_grad,
                B=B,
                H=H,
                BLOCK_SIZE_B=BLOCK_SIZE_B,
                BLOCK_SIZE_H=BLOCK_SIZE_H,
            )

        weight_grad = weight_grad.type_as(weight)

        return None, weight_grad, *[None] * 4


def embedding_cute(
    input_ids: torch.Tensor,
    weight: torch.Tensor,
    *,
    BLOCK_SIZE_B_forward: int = 128,
    BLOCK_SIZE_H_forward: int = 128,
    BLOCK_SIZE_B_backward: int = 128,
    BLOCK_SIZE_H_backward: int = 128,
) -> torch.Tensor:
    """computes word embeddings

    Args:
        input_ids (torch.Tensor): input ids
        weight (torch.Tensor): embedding matrix
        BLOCK_SIZE_B_forward (int, optional): block size for forward along batch dimension. Defaults to 128.
        BLOCK_SIZE_H_forward (int, optional): block size for forward along vocabulary dimension. Defaults to 128.
        BLOCK_SIZE_B_backward (int, optional): block size for backward along batch dimension. Defaults to 128.
        BLOCK_SIZE_H_backward (int, optional): block size for backward along vocabulary dimension. Defaults to 128.

    Returns:
        torch.Tensor: word embeddings
    """

    return _Embedding_Cute.apply(
        input_ids, weight, BLOCK_SIZE_B_forward, BLOCK_SIZE_H_forward, BLOCK_SIZE_B_backward, BLOCK_SIZE_H_backward
    )
