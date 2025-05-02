import triton
import triton.language as tl

from ....triton_math import sigmoid


@triton.jit
def swiglu_unchunked_forward_triton_kernel(
    x_ptr, output_ptr, B, H, BLOCK_SIZE_B: tl.constexpr, BLOCK_SIZE_H: tl.constexpr
):
    BLOCK_ID_B = tl.program_id(axis=0)
    BLOCK_ID_H = tl.program_id(axis=1)

    indices_b = BLOCK_ID_B * BLOCK_SIZE_B + tl.arange(0, BLOCK_SIZE_B)
    indices_h = BLOCK_ID_H * BLOCK_SIZE_H + tl.arange(0, BLOCK_SIZE_H)

    half_H = H >> 1

    mask_b = indices_b < B
    mask_h = indices_h < half_H
    mask_bh = mask_b[:, None] & mask_h[None, :]

    up_ptrs = x_ptr + indices_b[:, None] * H + indices_h[None, :]
    up = tl.load(up_ptrs, mask=mask_bh)

    gate_ptrs = up_ptrs + (H >> 1)
    gate = tl.load(gate_ptrs, mask=mask_bh).to(tl.float32)

    output = up * gate * sigmoid(gate)

    output_ptrs = output_ptr + indices_b[:, None] * half_H + indices_h[None, :]
    tl.store(output_ptrs, output, mask=mask_bh)
