import triton
import triton.language as tl


@triton.jit
def _swiglu_unchunked_forward_triton_kernel(
    x_ptr, output_ptr, B, H, BLOCK_SIZE_B: tl.constexpr, BLOCK_SIZE_H: tl.constexpr
):
    pid_b = tl.program_id(axis=0)
    pid_h = tl.program_id(axis=1)

    indices_b = pid_b * BLOCK_SIZE_B + tl.arange(0, BLOCK_SIZE_B)
    indices_h = pid_h * BLOCK_SIZE_H + tl.arange(0, BLOCK_SIZE_H)

    half_H = H >> 1

    mask_b = indices_b < B
    mask_h = indices_h < half_H
    mask_bh = mask_b[:, None] & mask_h[None, :]

    up_ptrs = x_ptr + indices_b[:, None] * H + indices_h[None, :]
    up = tl.load(up_ptrs, mask=mask_bh)

    gate_ptrs = up_ptrs + (H >> 1)
    gate = tl.load(gate_ptrs, mask=mask_bh).to(tl.float32)

    output = up * gate * tl.sigmoid(gate)

    output_ptrs = output_ptr + indices_b[:, None] * half_H + indices_h[None, :]
    tl.store(output_ptrs, output, mask=mask_bh)
