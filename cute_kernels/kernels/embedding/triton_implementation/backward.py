import torch
import triton
import triton.language as tl

from ....constants import LIBRARY_NAME
from ....math import ceil_divide
from ....utils import cute_op


@triton.jit
def embedding_backward_triton_kernel(
    x_ptr,
    output_grad_ptr,
    weight_grad_ptr,
    B,
    H,
    accumulate_in_fp32: tl.constexpr,
    BLOCK_SIZE_B: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
):
    pid_b = tl.program_id(axis=0)
    pid_h = tl.program_id(axis=1)

    indices_b = pid_b * BLOCK_SIZE_B + tl.arange(0, BLOCK_SIZE_B)
    indices_h = pid_h * BLOCK_SIZE_H + tl.arange(0, BLOCK_SIZE_H)

    mask_b = indices_b < B
    mask_h = indices_h < H
    mask_bh = mask_b[:, None] & mask_h[None, :]

    x_ptrs = x_ptr + indices_b
    x = tl.load(x_ptrs, mask=mask_b)

    output_grad_ptrs = output_grad_ptr + indices_b[:, None] * H + indices_h[None, :]
    output_grad = tl.load(output_grad_ptrs, mask=mask_bh)

    weight_grad_ptrs = weight_grad_ptr + x[:, None] * H + indices_h[None, :]

    if accumulate_in_fp32:
        output_grad = output_grad.to(tl.float32)

    tl.atomic_add(weight_grad_ptrs, output_grad, mask=mask_bh)


@cute_op(f"{LIBRARY_NAME}::embedding_backward_triton", mutates_args={"weight_grad"})
def embedding_backward_triton(
    input_ids: torch.Tensor,
    output_grad: torch.Tensor,
    weight_grad: torch.Tensor,
    BLOCK_SIZE_B: int,
    BLOCK_SIZE_H: int,
) -> None:
    B = input_ids.numel()
    H = weight_grad.size(-1)

    accumulate_in_fp32 = weight_grad.dtype == torch.bfloat16
    if accumulate_in_fp32:
        weight_grad = weight_grad.float()

    with torch.device(input_ids.device):
        embedding_backward_triton_kernel[ceil_divide(B, BLOCK_SIZE_B), ceil_divide(H, BLOCK_SIZE_H)](
            x_ptr=input_ids,
            output_grad_ptr=output_grad,
            weight_grad_ptr=weight_grad,
            B=B,
            H=H,
            accumulate_in_fp32=accumulate_in_fp32,
            BLOCK_SIZE_B=BLOCK_SIZE_B,
            BLOCK_SIZE_H=BLOCK_SIZE_H,
        )
