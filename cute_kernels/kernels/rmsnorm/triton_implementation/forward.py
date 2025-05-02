import triton
import triton.language as tl


@triton.jit
def rmsnorm_forward_triton_kernel(
    x_ptr,
    has_weight: tl.constexpr,
    weight_ptr,
    output_ptr,
    eps,
    has_rmsnorm_denominator: tl.constexpr,
    rmsnorm_denominator_ptr,
    B,
    H,
    BLOCK_SIZE_B: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
):
    pid_b = tl.program_id(axis=0)

    indices_b = pid_b * BLOCK_SIZE_B + tl.arange(0, BLOCK_SIZE_B)
    indices_h = tl.arange(0, BLOCK_SIZE_H)
    indices_bh = indices_b[:, None] * H + indices_h[None, :]

    mask_b = indices_b < B
    mask_h = indices_h < H

    mask_bh = mask_b[:, None] & mask_h[None, :]

    x_ptrs = x_ptr + indices_bh
    x = tl.load(x_ptrs, mask=mask_bh).to(tl.float32)

    squared_sum = tl.sum(x * x, axis=1)
    inverse_rms = tl.rsqrt((squared_sum / H) + eps)

    if has_rmsnorm_denominator:
        tl.store(rmsnorm_denominator_ptr + indices_b, inverse_rms, mask=mask_b)

    x *= inverse_rms[:, None]

    if has_weight:
        weight = tl.load(weight_ptr + indices_h, mask=mask_h)
        x = x.to(x_ptr.dtype.element_ty) * weight[None, :]

    output_ptrs = output_ptr + indices_bh
    tl.store(output_ptrs, x, mask=mask_bh)
