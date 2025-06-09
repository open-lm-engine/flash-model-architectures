# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import torch
import triton
import triton.language as tl

from ....math import ceil_divide
from ....utils import ensure_contiguous


@triton.jit
def padded_expert_frequency_triton_kernel(x_ptr, y_ptr, pad_to_multiple_of, N, BLOCK_SIZE: tl.constexpr):
    BLOCK_ID = tl.program_id(axis=0)

    indices = BLOCK_ID * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = indices < N

    x = tl.load(x_ptr + indices, mask=mask)

    y = pad_to_multiple_of - (x % pad_to_multiple_of)
    tl.store(y_ptr + indices, y, mask=mask)


class _GroupWithPadding(torch.autograd.Function):
    @staticmethod
    @ensure_contiguous
    def forward(
        ctx, x: torch.Tensor, expert_frequency: torch.Tensor, scattered_idxs: torch.Tensor, pad_to_multiple_of: int
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
                    BLOCK_SIZE=BLOCK_SIZE,
                    num_warps=NUM_WARPS,
                )

            expert_padding_frequency.cumsum_(-1)

        return output


def group_with_padding(
    x: torch.Tensor, expert_frequency: torch.Tensor, scattered_idxs: torch.Tensor, pad_to_multiple_of: int = 1
) -> torch.Tensor:
    return _GroupWithPadding.apply(x, expert_frequency, scattered_idxs, pad_to_multiple_of)
