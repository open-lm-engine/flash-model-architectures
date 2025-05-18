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
    h_ptr,
    indices_b,
    indices_n,
    mask_bn,
    N,
    BLOCK_SIZE_B,
    BLOCK_SIZE_N,
    dtype,
):
    if HAS_INPUT_STATE:
        y_prev = tl.load(h_ptr + indices_b[:, None] * N + indices_n[None, :], mask=mask_bn)
    else:
        y_prev = tl.zeros((BLOCK_SIZE_B, BLOCK_SIZE_N), dtype=dtype)

    return y_prev


@triton.jit
def scalar_rnn_varlen_backward_triton_kernel(
    W_ptr,
    y_ptr,
    y_stride_t,
    HAS_INPUT_STATE: tl.constexpr,
    h_ptr,
    dy_ptr,
    cu_seqlens_ptr,
    IS_MAX_SEQLEN_TENSOR: tl.constexpr,
    max_seqlen_ptr,
    dx_ptr,
    dW_ptr,
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

    dh = tl.zeros((BLOCK_SIZE_B, BLOCK_SIZE_N), dtype=W_ptr.dtype.element_ty)
    dW = tl.zeros((BLOCK_SIZE_N,), dtype=tl.float32)

    W = tl.load(W_ptr + indices_n, mask=mask_n)

    cu_seqlens_ptrs = cu_seqlens_ptr + indices_b[:, None]
    start = tl.load(cu_seqlens_ptrs, mask=mask_b[:, None])
    end = tl.load(cu_seqlens_ptrs + 1, mask=mask_b[:, None])

    if IS_MAX_SEQLEN_TENSOR:
        max_seqlen = tl.load(max_seqlen_ptr)
    else:
        max_seqlen = max_seqlen_ptr

    end -= 1

    indices = end * y_stride_t + indices_n[None, :]
    y = tl.load(y_ptr + indices, mask=mask_bn)

    # backward counting reduces 1 instruction since we need to compare s == 0, otherwise we have to compare s == S - 1
    for _ in range(max_seqlen - 1, -1, -1):
        if HAS_GRADIENT_CLIPPING:
            dh = clamp(dh, min_value=-gradient_clipping, max_value=gradient_clipping)

        unfinished = end >= start
        mask = unfinished & mask_n[None, :]

        dy = tl.load(dy_ptr + indices, mask=mask) + dh

        dx_ptrs = dx_ptr + indices
        indices -= y_stride_t

        y_prev = tl.where(
            start == end,
            _load_input_state(
                HAS_INPUT_STATE=HAS_INPUT_STATE,
                h_ptr=h_ptr,
                indices_b=indices_b,
                indices_n=indices_n,
                mask_bn=mask_bn,
                N=N,
                BLOCK_SIZE_B=BLOCK_SIZE_B,
                BLOCK_SIZE_N=BLOCK_SIZE_N,
                dtype=W.dtype,
            ),
            tl.load(y_ptr + indices, mask=mask & (indices >= 0)),
        )

        dx, dW, dh = _rnn_backward_update(
            y=y,
            W=W,
            dy=dy,
            dW=dW,
            y_prev=y_prev,
            ACTIVATION_FUNCTION=ACTIVATION_FUNCTION,
            relu_negative_slope=relu_negative_slope,
        )

        tl.store(dx_ptrs, dx, mask=mask)
        y = y_prev

        end -= 1

    tl.atomic_add(dW_ptr + indices_n, dW, mask=mask_n)


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
) -> None:
    N = output.size(1)
    B = cu_seqlens.size(0) - 1

    BLOCK_SIZE_B = 32
    BLOCK_SIZE_N = 32

    has_input_state = input_state is not None
    is_max_seqlen_tensor = max_seqlen_tensor is not None

    with torch.device(output.device):
        scalar_rnn_varlen_backward_triton_kernel[ceil_divide(B, BLOCK_SIZE_B), ceil_divide(N, BLOCK_SIZE_N)](
            W_ptr=weight,
            y_ptr=output,
            y_stride_t=output.stride(0),
            HAS_INPUT_STATE=has_input_state,
            h_ptr=input_state if has_input_state else None,
            dy_ptr=output_grad,
            cu_seqlens_ptr=cu_seqlens,
            IS_MAX_SEQLEN_TENSOR=is_max_seqlen_tensor,
            max_seqlen_ptr=max_seqlen_tensor if is_max_seqlen_tensor else max_seqlen,
            dx_ptr=input_grad,
            dW_ptr=weight_grad,
            HAS_GRADIENT_CLIPPING=gradient_clipping is not None,
            gradient_clipping=gradient_clipping,
            ACTIVATION_FUNCTION=activation_function,
            relu_negative_slope=relu_negative_slope,
            B=B,
            N=N,
            BLOCK_SIZE_B=BLOCK_SIZE_B,
            BLOCK_SIZE_N=BLOCK_SIZE_N,
        )
