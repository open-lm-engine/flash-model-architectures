import triton
import triton.language as tl


@triton.jit
def _load_output_output_grad(output_ptr, output_grad_ptr, h, H, BLOCK_SIZE_H, indices_b, mask_b):
    indices_h = h * BLOCK_SIZE_H + tl.arange(0, BLOCK_SIZE_H)
    mask_h = indices_h < H

    indices = indices_b[:, None] * H + indices_h[None, :]
    mask_bh = mask_b[:, None] & mask_h[None, :]

    output_ptrs = output_ptr + indices
    output = tl.load(output_ptrs, mask=mask_bh)

    output_grad_ptrs = output_grad_ptr + indices
    output_grad = tl.load(output_grad_ptrs, mask=mask_bh)

    return output, output_grad, indices, mask_bh


@triton.jit
def softmax_backward_triton_kernel(
    output_ptr,
    output_grad_ptr,
    x_grad_ptr,
    has_logits_multiplier: tl.constexpr,
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
        if has_logits_multiplier:
            output *= logits_multiplier

        x_grad_ptrs = x_grad_ptr + indices
        tl.store(x_grad_ptrs, output, mask=mask_bh)
