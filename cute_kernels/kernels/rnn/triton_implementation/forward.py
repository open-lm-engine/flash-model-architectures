import triton
import triton.language as tl

from ....triton_math import tanh


@triton.jit
def rnn_forward_triton_kernel(
    input_ptr,
    input_stride_b,
    input_stride_s,
    input_stride_n,
    weight_ptr,
    weight_stride_n,
    weight_stride_h,
    has_input_state: tl.constexpr,
    input_state_ptr,
    output_ptr,
    B,
    S,
    H,
    BLOCK_SIZE_B: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
):
    pid_b = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    indices_b = pid_b * BLOCK_SIZE_B + tl.arange(0, BLOCK_SIZE_B)
    indices_h = tl.arange(0, BLOCK_SIZE_H)

    mask_b = indices_b < B
    mask_h = indices_h < H
    mask_bh = mask_b[:, None] & mask_h[None, :]

    weight_ptrs = weight_ptr + pid_n * weight_stride_n + indices_h[:, None] * weight_stride_h + indices_h[None, :]
    weight = tl.load(weight_ptrs, mask=mask_h[:, None] & mask_h[None, :], other=0)

    indices = indices_b[:, None] * input_stride_b + pid_n * input_stride_n + indices_h[None, :]

    if has_input_state:
        input_state_ptrs = input_state_ptr + indices
        input_state = tl.load(input_state_ptrs, mask=mask_bh)
    else:
        input_state = tl.zeros((BLOCK_SIZE_B, BLOCK_SIZE_H), dtype=input_ptr.dtype.element_ty)

    for _ in range(S):
        input_ptrs = input_ptr + indices
        input = tl.load(input_ptrs, mask=mask_bh, other=0)

        input_state = tl.dot(input_state, weight, input, allow_tf32=True).to(input_state.dtype)
        input_state = tanh(input_state)

        output_ptrs = output_ptr + indices
        tl.store(output_ptrs, input_state, mask=mask_bh)

        indices += input_stride_s
