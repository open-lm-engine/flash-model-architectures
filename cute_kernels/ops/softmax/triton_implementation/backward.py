# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import torch
import triton
import triton.language as tl
from torch.library import custom_op

from ....constants import LIBRARY_NAME
from ....math import ceil_divide, get_next_power_of_2
from ....utils import get_num_elements_and_hidden_size


@triton.jit
def _load_output_output_grad(output_ptr, output_grad_ptr, h, H, BLOCK_SIZE_H, indices_b, mask_b):
    indices_h = h * BLOCK_SIZE_H + tl.arange(0, BLOCK_SIZE_H)
    mask_h = indices_h < H

    indices = indices_b[:, None] * H + indices_h[None, :]
    mask_bh = mask_b[:, None] & mask_h[None, :]

    output = tl.load(output_ptr + indices, mask=mask_bh)
    output_grad = tl.load(output_grad_ptr + indices, mask=mask_bh)

    return output, output_grad, indices, mask_bh


@triton.jit
def softmax_backward_triton_kernel(
    output_ptr,
    output_grad_ptr,
    x_grad_ptr,
    logits_multiplier,
    B,
    H,
    BLOCK_SIZE_B: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
):
    pid = tl.program_id(axis=0)

    indices_b = pid * BLOCK_SIZE_B + tl.arange(0, BLOCK_SIZE_B)
    mask_b = indices_b < B

    accumulator = tl.zeros((BLOCK_SIZE_B, 1), dtype=tl.float32)
    num_blocks_h = tl.cdiv(H, BLOCK_SIZE_H)

    for h in range(num_blocks_h):
        output, output_grad, indices, mask_bh = _load_output_output_grad(
            output_ptr=output_ptr,
            output_grad_ptr=output_grad_ptr,
            h=h,
            H=H,
            BLOCK_SIZE_H=BLOCK_SIZE_H,
            indices_b=indices_b,
            mask_b=mask_b,
        )

        acc = output_grad * output
        acc = acc.to(tl.float32)
        accumulator += tl.sum(acc, axis=1, keep_dims=True)

    for h in range(num_blocks_h):
        output, output_grad, indices, mask_bh = _load_output_output_grad(
            output_ptr=output_ptr,
            output_grad_ptr=output_grad_ptr,
            h=h,
            H=H,
            BLOCK_SIZE_H=BLOCK_SIZE_H,
            indices_b=indices_b,
            mask_b=mask_b,
        )

        output_grad -= accumulator
        output *= output_grad
        if logits_multiplier is not None:
            output *= logits_multiplier

        tl.store(x_grad_ptr + indices, output, mask=mask_bh)


@custom_op(f"{LIBRARY_NAME}::softmax_backward_triton", mutates_args={"x_grad"})
def softmax_backward_triton(
    output: torch.Tensor, output_grad: torch.Tensor, x_grad: torch.Tensor, logits_multiplier: float | None
) -> None:
    B, H = get_num_elements_and_hidden_size(x_grad)

    BLOCK_SIZE_B = 1
    BLOCK_SIZE_H = min(get_next_power_of_2(H), 4096 if output.dtype == torch.float32 else 8192)

    with torch.device(x_grad.device):
        softmax_backward_triton_kernel[ceil_divide(B, BLOCK_SIZE_B),](
            output_ptr=output,
            output_grad_ptr=output_grad,
            x_grad_ptr=x_grad,
            logits_multiplier=logits_multiplier,
            B=B,
            H=H,
            BLOCK_SIZE_B=BLOCK_SIZE_B,
            BLOCK_SIZE_H=BLOCK_SIZE_H,
        )
