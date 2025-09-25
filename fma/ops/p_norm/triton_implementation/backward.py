# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import triton
import triton.language as tl


@triton.jit
def norm_2_backward_triton_kernel(
    x_ptr,
    weight_ptr,
    output_grad_ptr,
    x_grad_ptr,
    weight_grad_ptr,
    eps,
    p_norm_denominator_ptr,
    B,
    H,
    BLOCK_SIZE_B: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_programs = tl.num_programs(axis=0)

    num_elements_per_program = tl.cdiv(B, num_programs)

    indices_h = tl.arange(0, BLOCK_SIZE_H)
    mask_h = indices_h < H

    program_start = pid * num_elements_per_program
    program_end = min(program_start + num_elements_per_program, B)
    num_elements_in_current_program = program_end - program_start

    num_loops = tl.cdiv(num_elements_in_current_program, BLOCK_SIZE_B)

    x_dtype = x_ptr.dtype.element_ty

    if weight_ptr is not None:
        weight = tl.load(weight_ptr + indices_h, mask=mask_h)[None, :]
        weight_grad = tl.zeros((BLOCK_SIZE_H,), dtype=tl.float32)

    for i in range(num_loops):
        indices_b = program_start + i * BLOCK_SIZE_B + tl.arange(0, BLOCK_SIZE_B)
        indices_bh = indices_b[:, None] * H + indices_h[None, :]

        mask_b = indices_b < program_end
        mask_bh = mask_b[:, None] & mask_h[None, :]

        x = tl.load(x_ptr + indices_bh, mask=mask_bh).to(tl.float32)

        if p_norm_denominator_ptr is None:
            r = x * x
            r = tl.sum(r, axis=1)
            r = tl.rsqrt(r)
        else:
            r = tl.load(p_norm_denominator_ptr + indices_b, mask=mask_b)

        output_grad = tl.load(output_grad_ptr + indices_bh, mask=mask_bh)

        output_grad_weight = output_grad
        if weight_ptr is not None:
            output_grad_weight *= weight

        output_grad_weight = output_grad_weight.to(tl.float32)

        x_grad = r[:, None] * output_grad_weight
        x_grad -= r[:, None] * r[:, None] * r[:, None] * x * tl.sum(output_grad_weight * x, axis=1, keep_dims=True)
        x_grad = x_grad.to(x_dtype)

        tl.store(x_grad_ptr + indices_bh, x_grad, mask=mask_bh)

        if weight_ptr is not None:
            weight_grad += tl.sum(output_grad * (x * r[:, None]).to(x_dtype), axis=0)

    if weight_ptr is not None:
        tl.atomic_add(weight_grad_ptr + indices_h, weight_grad, mask=mask_h, sem="relaxed")
