import torch
import triton
import triton.language as tl

from .....constants import LIBRARY_NAME
from .....math import ceil_divide, get_next_power_of_2
from .....triton_math import clamp
from .....utils import cute_op
from .backward import _tanh_backward


@triton.jit
def _load_input_state(
    has_input_state,
    input_state_ptr,
    input_state_stride_b,
    pid_n,
    indices,
    indices_h,
    mask,
    H,
    BLOCK_SIZE_B,
    BLOCK_SIZE_H,
    dtype,
):
    if has_input_state:
        output_ptrs = input_state_ptr + indices * input_state_stride_b + pid_n * H + indices_h[None, :]
        output_prev = tl.load(output_ptrs, mask=mask, other=0)
    else:
        output_prev = tl.zeros((BLOCK_SIZE_B, BLOCK_SIZE_H), dtype=dtype)

    return output_prev


@triton.jit
def rnn_varlen_backward_triton_kernel(
    weight_ptr,
    weight_stride_n,
    output_ptr,
    output_stride_t,
    has_input_state: tl.constexpr,
    input_state_ptr,
    input_state_stride_b,
    output_grad_ptr,
    cu_seqlens_ptr,
    is_max_seqlen_tensor: tl.constexpr,
    max_seqlen_ptr,
    input_grad_ptr,
    weight_grad_ptr,
    has_gradient_clipping: tl.constexpr,
    gradient_clipping,
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

    input_state_grad = tl.zeros((BLOCK_SIZE_B, BLOCK_SIZE_H), dtype=weight_ptr.dtype.element_ty)
    weight_grad = tl.zeros((BLOCK_SIZE_H, BLOCK_SIZE_H), dtype=tl.float32)

    weight_ptrs = weight_ptr + pid_n * weight_stride_n + indices_h[:, None] * H + indices_h[None, :]
    weight = tl.load(weight_ptrs, mask=mask_h[:, None] & mask_h[None, :], other=0)

    cu_seqlens_ptrs = cu_seqlens_ptr + indices_b[:, None]
    start = tl.load(cu_seqlens_ptrs, mask=mask_b[:, None])
    end = tl.load(cu_seqlens_ptrs + 1, mask=mask_b[:, None])

    if is_max_seqlen_tensor:
        max_seqlen = tl.load(max_seqlen_ptr)
    else:
        max_seqlen = max_seqlen_ptr

    end -= 1

    indices = end * output_stride_t + pid_n * H + indices_h[None, :]

    output_ptrs = output_ptr + indices
    output = tl.load(output_ptrs, mask=mask_bh, other=0)

    # backward counting reduces 1 instruction since we need to compare s == 0, otherwise we have to compare s == S - 1
    for _ in range(max_seqlen - 1, -1, -1):
        unfinished = end >= start
        mask = unfinished & mask_h[None, :]

        output_grad_ptrs = output_grad_ptr + indices
        output_grad = tl.load(output_grad_ptrs, mask=mask, other=0)

        if has_gradient_clipping:
            input_state_grad = clamp(input_state_grad, min_value=-gradient_clipping, max_value=gradient_clipping)

        input_grad = (output_grad + input_state_grad) * _tanh_backward(output)

        input_grad_ptrs = input_grad_ptr + indices
        tl.store(input_grad_ptrs, input_grad, mask=mask)

        input_state_grad = tl.dot(input_grad, weight.T, allow_tf32=True).to(input_state_grad.dtype)

        output_ptrs -= output_stride_t
        output_prev = tl.where(
            start == end,
            _load_input_state(
                has_input_state=has_input_state,
                input_state_ptr=input_state_ptr,
                input_state_stride_b=input_state_stride_b,
                pid_n=pid_n,
                indices=indices,
                indices_h=indices_h,
                mask=mask,
                H=H,
                BLOCK_SIZE_B=BLOCK_SIZE_B,
                BLOCK_SIZE_H=BLOCK_SIZE_H,
                dtype=weight.dtype,
            ),
            tl.load(output_ptrs, mask=mask, other=0),
        )

        weight_grad = tl.dot(output_prev.T, input_grad, weight_grad, allow_tf32=True)
        output = output_prev

        indices -= output_stride_t
        end -= 1

    weight_grad_ptrs = weight_grad_ptr + pid_n * weight_stride_n + indices_h[:, None] * H + indices_h[None, :]
    tl.store(weight_grad_ptrs, weight_grad, mask=mask_h[:, None] & mask_h[None, :])


@cute_op(f"{LIBRARY_NAME}::rnn_varlen_backward_triton", mutates_args={"input_grad", "weight_grad"})
def rnn_varlen_backward_triton(
    weight: torch.Tensor,
    output: torch.Tensor,
    input_state: torch.Tensor | None,
    output_grad: torch.Tensor,
    input_grad: torch.Tensor,
    weight_grad: torch.Tensor,
    cu_seqlens: torch.Tensor,
    max_seqlen_tensor: torch.Tensor | None,
    max_seqlen: int | None,
    gradient_clipping: float | None,
    BLOCK_SIZE_B: int,
) -> None:
    _, N, H = output.size()
    B = cu_seqlens.size(0) - 1

    BLOCK_SIZE_H = get_next_power_of_2(H)
    BLOCK_SIZE_H = max(16, BLOCK_SIZE_H)

    has_input_state = input_state is not None
    is_max_seqlen_tensor = max_seqlen_tensor is not None

    with torch.device(output.device):
        rnn_varlen_backward_triton_kernel[ceil_divide(B, BLOCK_SIZE_B), N](
            weight_ptr=weight,
            weight_stride_n=weight.stride(0),
            output_ptr=output,
            output_stride_t=output.stride(0),
            has_input_state=has_input_state,
            input_state_ptr=input_state if has_input_state else None,
            input_state_stride_b=input_state.stride(0) if has_input_state else None,
            output_grad_ptr=output_grad,
            cu_seqlens_ptr=cu_seqlens,
            is_max_seqlen_tensor=is_max_seqlen_tensor,
            max_seqlen_ptr=max_seqlen_tensor if is_max_seqlen_tensor else max_seqlen,
            input_grad_ptr=input_grad,
            weight_grad_ptr=weight_grad,
            has_gradient_clipping=gradient_clipping is not None,
            gradient_clipping=gradient_clipping,
            B=B,
            H=H,
            BLOCK_SIZE_B=BLOCK_SIZE_B,
            BLOCK_SIZE_H=BLOCK_SIZE_H,
        )
