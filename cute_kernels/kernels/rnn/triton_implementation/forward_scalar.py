import torch
import triton
import triton.language as tl

from ....constants import LIBRARY_NAME
from ....math import ceil_divide
from ....utils import cute_op
from .forward import _activation


@triton.jit
def _rnn_forward_update(h, W, x, ACTIVATION_FUNCTION, relu_negative_slope):
    h = W[None, :] * h + x
    h = _activation(x=h, ACTIVATION_FUNCTION=ACTIVATION_FUNCTION, relu_negative_slope=relu_negative_slope)
    return h


@triton.jit
def scalar_rnn_forward_triton_kernel(
    x_ptr,
    x_stride_b,
    W_ptr,
    HAS_INPUT_STATE: tl.constexpr,
    h_ptr,
    y_ptr,
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

    W = tl.load(W_ptr + indices_n, mask=mask_n)

    if HAS_INPUT_STATE:
        h = tl.load(h_ptr + indices_b[:, None] * N + indices_n[None, :], mask=mask_bn)
    else:
        h = tl.zeros((BLOCK_SIZE_B, BLOCK_SIZE_N), dtype=x_ptr.dtype.element_ty)

    indices = indices_b[:, None] * x_stride_b + indices_n[None, :]

    for _ in range(S):
        h = _rnn_forward_update(
            h=h,
            W=W,
            x=tl.load(x_ptr + indices, mask=mask_bn),
            ACTIVATION_FUNCTION=ACTIVATION_FUNCTION,
            relu_negative_slope=relu_negative_slope,
        )

        tl.store(y_ptr + indices, h, mask=mask_bn)

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
            x_ptr=input,
            x_stride_b=input.stride(0),
            W_ptr=weight,
            HAS_INPUT_STATE=input_state is not None,
            h_ptr=input_state,
            y_ptr=output,
            ACTIVATION_FUNCTION=activation_function,
            relu_negative_slope=relu_negative_slope,
            B=B,
            S=S,
            N=N,
            BLOCK_SIZE_B=BLOCK_SIZE_B,
            BLOCK_SIZE_N=BLOCK_SIZE_N,
        )
