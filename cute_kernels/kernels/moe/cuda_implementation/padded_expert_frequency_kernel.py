# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import triton
import triton.language as tl


@triton.jit
def padded_expert_frequency_triton_kernel(x_ptr, y_ptr, pad_to_multiple_of, N, BLOCK_SIZE: tl.constexpr):
    BLOCK_ID = tl.program_id(axis=0)

    indices = BLOCK_ID * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = indices < N

    x = tl.load(x_ptr + indices, mask=mask)

    y = pad_to_multiple_of - (x % pad_to_multiple_of.to(tl.uint32))
    tl.store(y_ptr + indices, y, mask=mask)
