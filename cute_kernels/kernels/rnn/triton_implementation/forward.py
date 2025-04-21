import triton
import triton.language as tl

from ....triton_math import tanh


@triton.jit
def rnn_forward_triton(
    input_ptr,
    input_stride_b,
    input_stride_s,
    weight_ptr,
    weight_stride_n,
    has_input_state: tl.constexpr,
    input_state_ptr,
    input_state_stride_b,
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

    weight_ptrs = weight_ptr + pid_n * weight_stride_n + indices_h[:, None] * H + indices_h[None, :]
    weight = tl.load(weight_ptrs, mask=mask_h[:, None] & mask_h[None, :], other=0)

    if has_input_state:
        input_state_ptrs = input_state_ptr + indices_b[:, None] * input_state_stride_b + pid_n * H + indices_h[None, :]
        input_state = tl.load(input_state_ptrs, mask=mask_bh)
    else:
        input_state = tl.zeros((BLOCK_SIZE_B, BLOCK_SIZE_H), dtype=input_ptr.dtype.element_ty)

    indices = indices_b[:, None] * input_stride_b + pid_n * H + indices_h[None, :]

    for _ in range(S):
        input_ptrs = input_ptr + indices
        input = tl.load(input_ptrs, mask=mask_bh, other=0)

        input_state = tl.dot(input_state, weight, input, allow_tf32=True).to(input_state.dtype)
        input_state = tanh(input_state)

        output_ptrs = output_ptr + indices
        tl.store(output_ptrs, input_state, mask=mask_bh)

        indices += input_stride_s


@triton.jit
def rnn_varlen_forward_triton(
    input_ptr,
    input_stride_s,
    weight_ptr,
    weight_stride_n,
    has_input_state: tl.constexpr,
    input_state_ptr,
    input_state_stride_b,
    output_ptr,
    cu_seqlens_ptr,
    is_max_seqlen_tensor: tl.constexpr,
    max_seqlen_ptr,
    B,
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

    weight_ptrs = weight_ptr + pid_n * weight_stride_n + indices_h[:, None] * H + indices_h[None, :]
    weight = tl.load(weight_ptrs, mask=mask_h[:, None] & mask_h[None, :], other=0)

    if has_input_state:
        input_state_ptrs = input_state_ptr + indices_b[:, None] * input_state_stride_b + pid_n * H + indices_h[None, :]
        input_state = tl.load(input_state_ptrs, mask=mask_bh)
    else:
        input_state = tl.zeros((BLOCK_SIZE_B, BLOCK_SIZE_H), dtype=input_ptr.dtype.element_ty)

    cu_seqlens_ptrs = cu_seqlens_ptr + indices_b
    start = tl.load(cu_seqlens_ptrs, mask=mask_b)
    end = tl.load(cu_seqlens_ptrs + 1, mask=mask_b)

    if is_max_seqlen_tensor:
        max_seqlen = tl.load(max_seqlen_ptr)
    else:
        max_seqlen = max_seqlen_ptr

    indices = start * input_stride_s + pid_n * H + indices_h[None, :]

    for _ in range(max_seqlen):
        unfinished = indices < end
        mask = unfinished[:, None] & mask_h[None, :]

        input_ptrs = input_ptr + indices
        input = tl.load(input_ptrs, mask=mask, other=0)

        new_state = tl.dot(input_state, weight, input, allow_tf32=True).to(input_state.dtype)
        new_state = tanh(new_state)

        input_state = new_state * unfinished[:, None] + input_state * (1 - unfinished)[:, None]

        output_ptrs = output_ptr + indices
        tl.store(output_ptrs, new_state, mask=mask)

        indices += input_stride_s
