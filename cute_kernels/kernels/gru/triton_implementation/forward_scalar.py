import torch
import triton
import triton.language as tl

from ....constants import LIBRARY_NAME
from ....math import ceil_divide
from ....utils import cute_op
from ...rnn.triton_implementation.forward_scalar import _rnn_forward_update


@triton.jit
def gru_forward_triton_kernel(
    x_ptr,
    x_stride_b,
    W_ptr,
    xf_ptr,
    Wf_ptr,
    f_ptr,
    xr_ptr,
    Wr_ptr,
    r_ptr,
    z_ptr,
    HAS_INPUT_STATE: tl.constexpr,
    h_ptr,
    y_ptr,
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
    Wf = tl.load(Wf_ptr + indices_n, mask=mask_n)
    Wr = tl.load(Wr_ptr + indices_n, mask=mask_n)

    if HAS_INPUT_STATE:
        h = tl.load(h_ptr + indices_b[:, None] * N + indices_n[None, :], mask=mask_bn)
    else:
        h = tl.zeros((BLOCK_SIZE_B, BLOCK_SIZE_N), dtype=h_ptr.dtype.element_ty)

    indices = indices_b[:, None] * x_stride_b + indices_n[None, :]

    for _ in range(S):
        r = _rnn_forward_update(
            h=h,
            W=Wr,
            x=tl.load(xr_ptr + indices, mask=mask_bn),
            ACTIVATION_FUNCTION="sigmoid",
            relu_negative_slope=None,
        )

        tl.store(r_ptr + indices, r, mask=mask_bn)

        z = _rnn_forward_update(
            h=h * r,
            W=W,
            x=tl.load(x_ptr + indices, mask=mask_bn),
            ACTIVATION_FUNCTION="tanh",
            relu_negative_slope=None,
        )

        tl.store(z_ptr + indices, z, mask=mask_bn)

        f = _rnn_forward_update(
            h=h,
            W=Wf,
            x=tl.load(xf_ptr + indices, mask=mask_bn),
            ACTIVATION_FUNCTION="sigmoid",
            relu_negative_slope=None,
        )

        tl.store(f_ptr + indices, f, mask=mask_bn)

        h = f * h + (1 - f) * z
        tl.store(y_ptr + indices, h, mask=mask_bn)

        indices += N


@cute_op(f"{LIBRARY_NAME}::scalar_gru_forward_triton", mutates_args={"forget_gate", "reset_gate", "output"})
def scalar_gru_forward_triton(
    input: torch.Tensor,
    weight: torch.Tensor,
    forget_input: torch.Tensor,
    forget_weight: torch.Tensor,
    forget_gate: torch.Tensor,
    reset_input: torch.Tensor,
    reset_weight: torch.Tensor,
    reset_gate: torch.Tensor,
    output_update: torch.Tensor,
    input_state: torch.Tensor | None,
    output: torch.Tensor,
    BLOCK_SIZE_B: int,
    BLOCK_SIZE_N: int,
) -> None:
    B, S, N, _ = input.size()

    with torch.device(input.device):
        gru_forward_triton_kernel[ceil_divide(B, BLOCK_SIZE_B), N](
            x_ptr=input,
            x_stride_b=input.stride(0),
            W_ptr=weight,
            xf_ptr=forget_input,
            Wf_ptr=forget_weight,
            f_ptr=forget_gate,
            xr_ptr=reset_input,
            Wr_ptr=reset_weight,
            r_ptr=reset_gate,
            z_ptr=output_update,
            HAS_INPUT_STATE=input_state is not None,
            h_ptr=input_state,
            y_ptr=output,
            B=B,
            S=S,
            N=N,
            BLOCK_SIZE_B=BLOCK_SIZE_B,
            BLOCK_SIZE_N=BLOCK_SIZE_N,
        )
