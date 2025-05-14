import torch
import triton
import triton.language as tl

from ....constants import LIBRARY_NAME
from ....math import ceil_divide
from ....triton_math import clamp
from ....utils import cute_op
from .backward_scalar import _rnn_backward_update


@triton.jit
def _load_input_state(
    HAS_INPUT_STATE,
    input_state_ptr,
    indices_b,
    indices_n,
    mask_bn,
    N,
    BLOCK_SIZE_B,
    BLOCK_SIZE_N,
    dtype,
):
    if HAS_INPUT_STATE:
        output_prev = tl.load(input_state_ptr + indices_b[:, None] * N + indices_n[None, :], mask=mask_bn)
    else:
        output_prev = tl.zeros((BLOCK_SIZE_B, BLOCK_SIZE_N), dtype=dtype)

    return output_prev


@triton.jit
def scalar_rnn_varlen_backward_triton_kernel(
    weight_ptr,
    output_ptr,
    output_stride_t,
    HAS_INPUT_STATE: tl.constexpr,
    input_state_ptr,
    output_grad_ptr,
    cu_seqlens_ptr,
    IS_MAX_SEQLEN_TENSOR: tl.constexpr,
    max_seqlen_ptr,
    input_grad_ptr,
    weight_grad_ptr,
    HAS_GRADIENT_CLIPPING: tl.constexpr,
    gradient_clipping,
    ACTIVATION_FUNCTION: tl.constexpr,
    relu_negative_slope,
    B,
    N,
    BLOCK_SIZE_B: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    pid_b = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    indices_b = pid_b * BLOCK_SIZE_B + tl.arange(0, BLOCK_SIZE_B)
    indices_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    mask_b = indices_b < B
    mask_n = indices_n < N
    mask_bn = mask_b[:, None] & mask_n[None, :]

    input_state_grad = tl.zeros((BLOCK_SIZE_B, BLOCK_SIZE_N), dtype=weight_ptr.dtype.element_ty)
    weight_grad = tl.zeros((BLOCK_SIZE_N,), dtype=tl.float32)

    weight = tl.load(weight_ptr + indices_n, mask=mask_n)

    cu_seqlens_ptrs = cu_seqlens_ptr + indices_b[:, None]
    start = tl.load(cu_seqlens_ptrs, mask=mask_b[:, None])
    end = tl.load(cu_seqlens_ptrs + 1, mask=mask_b[:, None])

    if IS_MAX_SEQLEN_TENSOR:
        max_seqlen = tl.load(max_seqlen_ptr)
    else:
        max_seqlen = max_seqlen_ptr

    end -= 1

    indices = end * output_stride_t + indices_n[None, :]
    output = tl.load(output_ptr + indices, mask=mask_bn)

    # backward counting reduces 1 instruction since we need to compare s == 0, otherwise we have to compare s == S - 1
    for _ in range(max_seqlen - 1, -1, -1):
        if HAS_GRADIENT_CLIPPING:
            input_state_grad = clamp(input_state_grad, min_value=-gradient_clipping, max_value=gradient_clipping)

        unfinished = end >= start
        mask = unfinished & mask_n[None, :]

        output_grad = tl.load(output_grad_ptr + indices, mask=mask)
        output_grad += input_state_grad

        input_grad_ptrs = input_grad_ptr + indices
        indices -= output_stride_t

        output_prev = tl.where(
            start == end,
            _load_input_state(
                HAS_INPUT_STATE=HAS_INPUT_STATE,
                input_state_ptr=input_state_ptr,
                indices_b=indices_b,
                indices_n=indices_n,
                mask_bn=mask_bn,
                N=N,
                BLOCK_SIZE_B=BLOCK_SIZE_B,
                BLOCK_SIZE_N=BLOCK_SIZE_N,
                dtype=weight.dtype,
            ),
            tl.load(output_ptr + indices, mask=mask & (indices >= 0)),
        )

        input_grad, weight_grad, input_state_grad = _rnn_backward_update(
            output=output,
            weight=weight,
            output_grad=output_grad,
            weight_grad=weight_grad,
            output_prev=output_prev,
            ACTIVATION_FUNCTION=ACTIVATION_FUNCTION,
            relu_negative_slope=relu_negative_slope,
        )

        tl.store(input_grad_ptrs, input_grad, mask=mask)
        output = output_prev

        end -= 1

    tl.atomic_add(weight_grad_ptr + indices_n, weight_grad, mask=mask_n)


@cute_op(f"{LIBRARY_NAME}::scalar_rnn_varlen_backward_triton", mutates_args={"input_grad", "weight_grad"})
def scalar_rnn_varlen_backward_triton(
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
    activation_function: str,
    relu_negative_slope: float | None,
    BLOCK_SIZE_B: int,
    BLOCK_SIZE_N: int,
) -> None:
    N = output.size(1)
    B = cu_seqlens.size(0) - 1

    has_input_state = input_state is not None
    is_max_seqlen_tensor = max_seqlen_tensor is not None

    with torch.device(output.device):
        scalar_rnn_varlen_backward_triton_kernel[ceil_divide(B, BLOCK_SIZE_B), ceil_divide(N, BLOCK_SIZE_N)](
            weight_ptr=weight,
            output_ptr=output,
            output_stride_t=output.stride(0),
            HAS_INPUT_STATE=has_input_state,
            input_state_ptr=input_state if has_input_state else None,
            output_grad_ptr=output_grad,
            cu_seqlens_ptr=cu_seqlens,
            IS_MAX_SEQLEN_TENSOR=is_max_seqlen_tensor,
            max_seqlen_ptr=max_seqlen_tensor if is_max_seqlen_tensor else max_seqlen,
            input_grad_ptr=input_grad,
            weight_grad_ptr=weight_grad,
            HAS_GRADIENT_CLIPPING=gradient_clipping is not None,
            gradient_clipping=gradient_clipping,
            ACTIVATION_FUNCTION=activation_function,
            relu_negative_slope=relu_negative_slope,
            B=B,
            N=N,
            BLOCK_SIZE_B=BLOCK_SIZE_B,
            BLOCK_SIZE_N=BLOCK_SIZE_N,
        )
