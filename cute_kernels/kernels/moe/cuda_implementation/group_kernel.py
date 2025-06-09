# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import torch
import triton
import triton.language as tl

from ....math import ceil_divide
from ....utils import ensure_contiguous
from .padded_expert_frequency_kernel import padded_expert_frequency_triton_kernel


@triton.jit
def group_with_padding_triton_kernel(
    x_ptr,
    expert_padding_frequency_ptr,
    scattered_idxs_ptr,
    y_ptr,
    B,
    H,
    K,
    BLOCK_SIZE_B: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
):
    BLOCK_ID = tl.program_id(axis=0)

    indices_b = BLOCK_ID * BLOCK_SIZE_B + tl.arange(0, BLOCK_SIZE_B)
    mask_b = indices_b < B

    if expert_padding_frequency_ptr is not None:
        expert_padding_frequency = tl.load(expert_padding_frequency_ptr + indices_b, mask=mask_b)

    scattered_idxs = tl.load(
        scattered_idxs_ptr + indices_b,
    )

    for h in range(tl.cdiv(H, BLOCK_SIZE_H)):
        indices_h = h * BLOCK_SIZE_H + tl.arange(0, BLOCK_SIZE_H)
        mask_h = indices_h < H


class _GroupWithPadding(torch.autograd.Function):
    @staticmethod
    @ensure_contiguous
    def forward(
        ctx,
        x: torch.Tensor,
        expert_frequency: torch.Tensor,
        scattered_idxs: torch.Tensor,
        topk: int,
        pad_to_multiple_of: int,
    ) -> torch.Tensor:
        assert x.dim() == 2

        T, H = x.size()
        E = expert_frequency.size(0)

        if pad_to_multiple_of == 1:
            output = torch.empty_like(x)
            expert_padding_frequency = None
        else:
            # we pad to max possible shape to make tensor shape independent of data
            output = torch.empty(T + pad_to_multiple_of * E, H, device=x.device, dtype=x.dtype)
            expert_padding_frequency = torch.empty_like(expert_frequency)

            BLOCK_SIZE = 4096
            NUM_WARPS = 32

            with torch.cuda.device(expert_frequency.device):
                padded_expert_frequency_triton_kernel[ceil_divide(E, BLOCK_SIZE),](
                    x_ptr=expert_frequency,
                    y_ptr=expert_padding_frequency,
                    pad_to_multiple_of=pad_to_multiple_of,
                    N=E,
                    BLOCK_SIZE=BLOCK_SIZE,
                    num_warps=NUM_WARPS,
                )

            expert_padding_frequency = expert_padding_frequency.cumsum(-1)

        BLOCK_SIZE_B = 1
        BLOCK_SIZE_H = 4096
        NUM_WARPS = 32

        with torch.cuda.device(x.device):
            group_with_padding_triton_kernel[ceil_divide(T * topk, BLOCK_SIZE_B),](
                x_ptr=x,
                expert_padding_frequency_ptr=expert_padding_frequency,
                scattered_idxs_ptr=scattered_idxs,
                y_ptr=output,
                B=T,
                H=H,
                K=topk,
                BLOCK_SIZE_B=BLOCK_SIZE_B,
                BLOCK_SIZE_H=BLOCK_SIZE_H,
                num_warps=NUM_WARPS,
            )

        return output


def group_with_padding(
    x: torch.Tensor, expert_frequency: torch.Tensor, scattered_idxs: torch.Tensor, pad_to_multiple_of: int = 1
) -> torch.Tensor:
    return _GroupWithPadding.apply(x, expert_frequency, scattered_idxs, pad_to_multiple_of)
