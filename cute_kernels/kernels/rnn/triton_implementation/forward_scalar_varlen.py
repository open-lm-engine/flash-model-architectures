import torch
import triton
import triton.language as tl

from ....constants import LIBRARY_NAME
from ....math import ceil_divide
from ....utils import cute_op
from .forward_scalar import _rnn_forward_update


@triton.jit
def rnn_varlen_forward_triton_kernel(
    input_ptr,
    input_stride_t,
    weight_ptr,
    HAS_INPUT_STATE: tl.constexpr,
    input_state_ptr,
    output_ptr,
    cu_seqlens_ptr,
    IS_MAX_SEQLEN_TENSOR: tl.constexpr,
    max_seqlen_ptr,
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

    weight = tl.load(weight_ptr + indices_n, mask=mask_n)

    if HAS_INPUT_STATE:
        input_state = tl.load(input_state_ptr + indices_b[:, None] * N + indices_n[None, :], mask=mask_bn)
    else:
        input_state = tl.zeros((BLOCK_SIZE_B, BLOCK_SIZE_N), dtype=input_ptr.dtype.element_ty)

    cu_seqlens_ptrs = cu_seqlens_ptr + indices_b[:, None]
    start = tl.load(cu_seqlens_ptrs, mask=mask_b[:, None])
    end = tl.load(cu_seqlens_ptrs + 1, mask=mask_b[:, None])

    if IS_MAX_SEQLEN_TENSOR:
        max_seqlen = tl.load(max_seqlen_ptr)
    else:
        max_seqlen = max_seqlen_ptr

    indices = start * input_stride_t + indices_n[None, :]

    for _ in range(max_seqlen):
        unfinished = start < end
        mask = unfinished & mask_n[None, :]

        input_state = _rnn_forward_update(
            input_state=input_state,
            weight=weight,
            input=tl.load(input_ptr + indices, mask=mask),
            ACTIVATION_FUNCTION=ACTIVATION_FUNCTION,
            relu_negative_slope=relu_negative_slope,
        )

        tl.store(output_ptr + indices, input_state, mask=mask)

        indices += input_stride_t
        start += 1


@cute_op(f"{LIBRARY_NAME}::scalar_rnn_varlen_forward_triton", mutates_args={"output"})
def scalar_rnn_varlen_forward_triton(
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
    BLOCK_SIZE_N: int,
) -> None:
    N = input.size(1)
    B = cu_seqlens.size(0) - 1

    is_max_seqlen_tensor = max_seqlen_tensor is not None

    with torch.device(input.device):
        rnn_varlen_forward_triton_kernel[ceil_divide(B, BLOCK_SIZE_B), ceil_divide(N, BLOCK_SIZE_N)](
            input_ptr=input,
            input_stride_t=input.stride(0),
            weight_ptr=weight,
            HAS_INPUT_STATE=input_state is not None,
            input_state_ptr=input_state,
            output_ptr=output,
            cu_seqlens_ptr=cu_seqlens,
            IS_MAX_SEQLEN_TENSOR=is_max_seqlen_tensor,
            max_seqlen_ptr=max_seqlen_tensor if is_max_seqlen_tensor else max_seqlen,
            ACTIVATION_FUNCTION=activation_function,
            relu_negative_slope=relu_negative_slope,
            B=B,
            N=N,
            BLOCK_SIZE_B=BLOCK_SIZE_B,
            BLOCK_SIZE_N=BLOCK_SIZE_N,
        )
