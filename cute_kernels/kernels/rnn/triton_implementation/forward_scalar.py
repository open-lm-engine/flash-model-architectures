import torch
import triton
import triton.language as tl

from ....constants import LIBRARY_NAME
from ....math import ceil_divide
from ....utils import cute_op
from .forward import _activation


@triton.jit
def _rnn_forward_update(input_state, weight, input, ACTIVATION_FUNCTION, relu_negative_slope):
    input_state = weight[None, :] * input_state + input
    input_state = _activation(
        x=input_state, ACTIVATION_FUNCTION=ACTIVATION_FUNCTION, relu_negative_slope=relu_negative_slope
    )
    return input_state


@triton.jit
def scalar_rnn_forward_triton_kernel(
    input_ptr,
    input_stride_b,
    weight_ptr,
    HAS_INPUT_STATE: tl.constexpr,
    input_state_ptr,
    output_ptr,
    ACTIVATION_FUNCTION: tl.constexpr,
    relu_negative_slope,
    B,
    S,
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

    indices = indices_b[:, None] * input_stride_b + indices_n[None, :]

    for _ in range(S):
        input_state = _rnn_forward_update(
            input_state=input_state,
            weight=weight,
            input=tl.load(input_ptr + indices, mask=mask_bn),
            ACTIVATION_FUNCTION=ACTIVATION_FUNCTION,
            relu_negative_slope=relu_negative_slope,
        )

        tl.store(output_ptr + indices, input_state, mask=mask_bn)
        indices += N


@cute_op(f"{LIBRARY_NAME}::scalar_rnn_forward_triton", mutates_args={"output"})
def scalar_rnn_forward_triton(
    input: torch.Tensor,
    weight: torch.Tensor,
    input_state: torch.Tensor | None,
    output: torch.Tensor,
    activation_function: str,
    relu_negative_slope: float | None,
    BLOCK_SIZE_B: int,
    BLOCK_SIZE_N: int,
) -> None:
    B, S, N, _ = input.size()

    with torch.device(input.device):
        scalar_rnn_forward_triton_kernel[ceil_divide(B, BLOCK_SIZE_B), ceil_divide(N, BLOCK_SIZE_N)](
            input_ptr=input,
            input_stride_b=input.stride(0),
            weight_ptr=weight,
            HAS_INPUT_STATE=input_state is not None,
            input_state_ptr=input_state,
            output_ptr=output,
            ACTIVATION_FUNCTION=activation_function,
            relu_negative_slope=relu_negative_slope,
            B=B,
            S=S,
            N=N,
            BLOCK_SIZE_B=BLOCK_SIZE_B,
            BLOCK_SIZE_N=BLOCK_SIZE_N,
        )
