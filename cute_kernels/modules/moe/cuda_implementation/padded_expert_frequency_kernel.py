# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import torch
import triton
import triton.language as tl

from ....constants import LIBRARY_NAME
from ....math import get_next_power_of_2
from ....utils import cute_op


@triton.jit
def padded_expert_frequency_triton_kernel(x_ptr, y_ptr, pad_to_multiple_of, N, BLOCK_SIZE: tl.constexpr):
    indices = tl.arange(0, BLOCK_SIZE)
    mask = indices < N

    x = tl.load(x_ptr + indices, mask=mask)

    y = pad_to_multiple_of - (x % pad_to_multiple_of.to(tl.uint32))
    tl.store(y_ptr + indices, y, mask=mask)


@cute_op(f"{LIBRARY_NAME}::padded_expert_frequency_triton", mutates_args={"output"})
def padded_expert_frequency_triton(
    expert_frequency: torch.Tensor, output: torch.Tensor, pad_to_multiple_of: int
) -> None:
    E = expert_frequency.size(0)
    BLOCK_SIZE = get_next_power_of_2(E)
    NUM_WARPS = 32

    with torch.device(expert_frequency.device):
        padded_expert_frequency_triton_kernel[1,](
            x_ptr=expert_frequency,
            y_ptr=output,
            pad_to_multiple_of=pad_to_multiple_of,
            N=E,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=NUM_WARPS,
        )
