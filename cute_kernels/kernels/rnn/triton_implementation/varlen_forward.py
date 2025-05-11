import torch
import triton
import triton.language as tl

from ....constants import LIBRARY_NAME
from ....math import ceil_divide, get_next_power_of_2
from ....utils import cute_op
from .forward import _rnn_forward_update


@triton.jit
def rnn_varlen_forward_triton_kernel(
    input_ptr,
    input_stride_t,
    weight_ptr,
    weight_stride_n,
    HAS_INPUT_STATE: tl.constexpr,
    input_state_ptr,
    input_state_stride_b,
    output_ptr,
    cu_seqlens_ptr,
    IS_MAX_SEQLEN_TENSOR: tl.constexpr,
    max_seqlen_ptr,
    ACTIVATION_FUNCTION: tl.constexpr,
    relu_negative_slope,
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

    if HAS_INPUT_STATE:
        input_state_ptrs = input_state_ptr + indices_b[:, None] * input_state_stride_b + pid_n * H + indices_h[None, :]
        input_state = tl.load(input_state_ptrs, mask=mask_bh)
    else:
        input_state = tl.zeros((BLOCK_SIZE_B, BLOCK_SIZE_H), dtype=input_ptr.dtype.element_ty)

    cu_seqlens_ptrs = cu_seqlens_ptr + indices_b[:, None]
    start = tl.load(cu_seqlens_ptrs, mask=mask_b[:, None])
    end = tl.load(cu_seqlens_ptrs + 1, mask=mask_b[:, None])

    if IS_MAX_SEQLEN_TENSOR:
        max_seqlen = tl.load(max_seqlen_ptr)
    else:
        max_seqlen = max_seqlen_ptr

    indices = start * input_stride_t + pid_n * H + indices_h[None, :]

    input_dtype = input_ptr.dtype.element_ty
    out_dtype = input_dtype
    cast_dtype = input_dtype
    if input_dtype == tl.bfloat16:
        input_dtype = tl.float32
        out_dtype = tl.float32
        cast_dtype = tl.bfloat16

    for _ in range(max_seqlen):
        unfinished = start < end
        mask = unfinished & mask_h[None, :]

        input_state = _rnn_forward_update(
            input_state=input_state,
            weight=weight,
            input=tl.load(input_ptr + indices, mask=mask, other=0).to(input_dtype),
            out_dtype=out_dtype,
            cast_dtype=cast_dtype,
            ACTIVATION_FUNCTION=ACTIVATION_FUNCTION,
            relu_negative_slope=relu_negative_slope,
        )

        tl.store(output_ptr + indices, input_state, mask=mask)
        indices += input_stride_t
        start += 1


@cute_op(f"{LIBRARY_NAME}::rnn_varlen_forward_triton", mutates_args={"output"})
def rnn_varlen_forward_triton(
    input: torch.Tensor,
    weight: torch.Tensor,
    input_state: torch.Tensor | None,
    output: torch.Tensor,
    cu_seqlens: torch.Tensor,
    max_seqlen_tensor: torch.Tensor | None,
    max_seqlen: int | None,
    activation_function: str,
    relu_negative_slope: float | None,
    BLOCK_SIZE_B: int,
) -> None:
    _, N, H = input.size()
    B = cu_seqlens.size(0) - 1

    BLOCK_SIZE_H = get_next_power_of_2(H)
    BLOCK_SIZE_H = max(16, BLOCK_SIZE_H)

    has_input_state = input_state is not None
    is_max_seqlen_tensor = max_seqlen_tensor is not None

    with torch.device(input.device):
        rnn_varlen_forward_triton_kernel[ceil_divide(B, BLOCK_SIZE_B), N](
            input_ptr=input,
            input_stride_t=input.stride(0),
            weight_ptr=weight,
            weight_stride_n=weight.stride(0),
            HAS_INPUT_STATE=has_input_state,
            input_state_ptr=input_state,
            input_state_stride_b=input_state.stride(0) if has_input_state else None,
            output_ptr=output,
            cu_seqlens_ptr=cu_seqlens,
            IS_MAX_SEQLEN_TENSOR=is_max_seqlen_tensor,
            max_seqlen_ptr=max_seqlen_tensor if is_max_seqlen_tensor else max_seqlen,
            ACTIVATION_FUNCTION=activation_function,
            relu_negative_slope=relu_negative_slope,
            B=B,
            H=H,
            BLOCK_SIZE_B=BLOCK_SIZE_B,
            BLOCK_SIZE_H=BLOCK_SIZE_H,
        )
