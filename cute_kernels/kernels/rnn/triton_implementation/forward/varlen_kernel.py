import torch
import triton
import triton.language as tl

from .....constants import LIBRARY_NAME
from .....math import ceil_divide, get_next_power_of_2
from .....triton_math import tanh
from .....utils import cute_op


@triton.jit
def rnn_varlen_forward_triton_kernel(
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

    indices = start[:, None] * input_stride_s + pid_n * H + indices_h[None, :]

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


@cute_op(f"{LIBRARY_NAME}::rnn_forward_triton", mutates_args={"output"})
def rnn_varlen_forward_triton(
    input: torch.Tensor,
    weight: torch.Tensor,
    input_state: torch.Tensor | None,
    output: torch.Tensor,
    BLOCK_SIZE_B: int,
) -> None:
    B, S, N, H = input.size()

    BLOCK_SIZE_H = get_next_power_of_2(H)
    BLOCK_SIZE_H = max(16, BLOCK_SIZE_H)

    with torch.device(input.device):
        rnn_forward_triton_kernel[ceil_divide(B, BLOCK_SIZE_B), N](
            input_ptr=input,
            input_stride_b=input.stride(0),
            input_stride_s=input.stride(1),
            input_stride_n=input.stride(2),
            weight_ptr=weight,
            weight_stride_n=weight.stride(0),
            weight_stride_h=weight.stride(1),
            has_input_state=input_state is not None,
            input_state_ptr=input_state,
            output_ptr=output,
            B=B,
            S=S,
            H=H,
            BLOCK_SIZE_B=BLOCK_SIZE_B,
            BLOCK_SIZE_H=BLOCK_SIZE_H,
        )
