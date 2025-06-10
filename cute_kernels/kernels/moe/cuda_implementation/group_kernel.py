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
    x_stride_s,
    expert_padding_offset_ptr,
    sorted_idxs_ptr,
    scattered_idxs_ptr,
    y_ptr,
    y_stride_s,
    T,
    H,
    K,
    BLOCK_SIZE_B: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
):
    BLOCK_ID = tl.program_id(axis=0)
    B = T * K

    indices_b = BLOCK_ID * BLOCK_SIZE_B + tl.arange(0, BLOCK_SIZE_B)
    mask_b = indices_b < B

    scattered_idxs = tl.load(scattered_idxs_ptr + indices_b, mask=mask_b)

    NUM_BLOCKS_H = tl.cdiv(H, BLOCK_SIZE_H)

    x_ptrs = x_ptr + (scattered_idxs // K)[:, None] * x_stride_s
    y_ptrs = y_ptr + indices_b[:, None] * y_stride_s

    if expert_padding_offset_ptr is not None:
        sorted_idxs = tl.load(sorted_idxs_ptr + indices_b, mask=mask_b)
        expert_padding_offset = tl.load(expert_padding_offset_ptr + sorted_idxs)

        y_ptrs += expert_padding_offset * y_stride_s

    for h in range(NUM_BLOCKS_H):
        indices_h = h * BLOCK_SIZE_H + tl.arange(0, BLOCK_SIZE_H)

        if h < NUM_BLOCKS_H - 1:
            x = tl.load(x_ptrs + indices_h[None, :], mask=mask_b[:, None])
            tl.store(y_ptrs + indices_h[None, :], x, mask=mask_b[:, None])
        else:
            mask_h = indices_h < H
            mask_bh = mask_b[:, None] & mask_h[None, :]

            x = tl.load(x_ptrs + indices_h[None, :], mask=mask_bh)
            tl.store(y_ptrs + indices_h[None, :], x, mask=mask_bh)


class _GroupWithPadding(torch.autograd.Function):
    @staticmethod
    @ensure_contiguous
    def forward(
        ctx,
        x: torch.Tensor,
        expert_frequency: torch.Tensor,
        sorted_idxs: torch.Tensor,
        scattered_idxs: torch.Tensor,
        topk: int,
        pad_to_multiple_of: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert x.dim() == 2

        T, H = x.size()
        E = expert_frequency.size(0)
        K = topk

        if pad_to_multiple_of == 1:
            output = torch.empty(
                T * K, ceil_divide(H, pad_to_multiple_of) * pad_to_multiple_of, device=x.device, dtype=x.dtype
            )

            expert_padding_offset = None
            padded_expert_frequency = expert_frequency
        else:
            # we pad to max possible shape to make tensor shape independent of data
            output = torch.zeros(
                T * K + pad_to_multiple_of * E,
                ceil_divide(H, pad_to_multiple_of) * pad_to_multiple_of,
                device=x.device,
                dtype=x.dtype,
            )

            expert_padding_frequency = torch.empty_like(expert_frequency)
            padded_expert_frequency = expert_frequency.to(torch.int32) + expert_padding_frequency.to(torch.int32)
            padded_expert_frequency = padded_expert_frequency.to(torch.uint32)

            with torch.cuda.device(expert_frequency.device):
                BLOCK_SIZE = 4096
                NUM_WARPS = 32

                padded_expert_frequency_triton_kernel[ceil_divide(E, BLOCK_SIZE),](
                    x_ptr=expert_frequency,
                    y_ptr=expert_padding_frequency,
                    pad_to_multiple_of=pad_to_multiple_of,
                    N=E,
                    BLOCK_SIZE=BLOCK_SIZE,
                    num_warps=NUM_WARPS,
                )

            expert_padding_offset = expert_padding_frequency.cumsum(-1)
            expert_padding_offset = torch.cat(
                [
                    torch.tensor([0], device=expert_padding_offset.device, dtype=expert_padding_offset.dtype),
                    expert_padding_offset,
                ]
            )

        with torch.cuda.device(x.device):
            BLOCK_SIZE_B = 1
            BLOCK_SIZE_H = 4096
            NUM_WARPS = 32

            group_with_padding_triton_kernel[ceil_divide(T * K, BLOCK_SIZE_B),](
                x_ptr=x,
                x_stride_s=x.stride(0),
                expert_padding_offset_ptr=expert_padding_offset,
                sorted_idxs_ptr=sorted_idxs,
                scattered_idxs_ptr=scattered_idxs,
                y_ptr=output,
                y_stride_s=output.stride(0),
                T=T,
                H=H,
                K=topk,
                BLOCK_SIZE_B=BLOCK_SIZE_B,
                BLOCK_SIZE_H=BLOCK_SIZE_H,
                num_warps=NUM_WARPS,
            )

        return output, padded_expert_frequency


def group_with_padding(
    x: torch.Tensor,
    expert_frequency: torch.Tensor,
    sorted_idxs: torch.Tensor,
    scattered_idxs: torch.Tensor,
    topk: int,
    pad_to_multiple_of: int = 1,
) -> torch.Tensor:
    return _GroupWithPadding.apply(x, expert_frequency, sorted_idxs, scattered_idxs, topk, pad_to_multiple_of)
