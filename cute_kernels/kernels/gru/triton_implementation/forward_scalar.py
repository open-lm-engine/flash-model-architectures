import torch
import triton
import triton.language as tl

from ....constants import LIBRARY_NAME
from ....math import ceil_divide
from ....utils import cute_op
from ...rnn.triton_implementation.forward_scalar import _rnn_forward_update


@triton.jit
def scalar_gru_forward_triton_kernel(
    input_ptr,
    input_stride_b,
    weight_ptr,
    forget_input_ptr,
    forget_weight_ptr,
    forget_gate_ptr,
    reset_input_ptr,
    reset_weight_ptr,
    reset_gate_ptr,
    output_update_ptr,
    HAS_INPUT_STATE: tl.constexpr,
    input_state_ptr,
    output_ptr,
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
    forget_weight = tl.load(forget_weight_ptr + indices_n, mask=mask_n)
    reset_weight = tl.load(reset_weight_ptr + indices_n, mask=mask_n)

    if HAS_INPUT_STATE:
        input_state = tl.load(input_state_ptr + indices_b[:, None] * N + indices_n[None, :], mask=mask_bn)
    else:
        input_state = tl.zeros((BLOCK_SIZE_B, BLOCK_SIZE_N), dtype=input_ptr.dtype.element_ty)

    indices = indices_b[:, None] * input_stride_b + indices_n[None, :]

    for _ in range(S):
        reset_gate = _rnn_forward_update(
            input_state=input_state,
            weight=reset_weight,
            input=tl.load(reset_input_ptr + indices, mask=mask_bn),
            ACTIVATION_FUNCTION="sigmoid",
            relu_negative_slope=None,
        )

        tl.store(reset_gate_ptr + indices, reset_gate, mask=mask_bn)

        output_update = _rnn_forward_update(
            input_state=input_state * reset_gate,
            weight=weight,
            input=tl.load(input_ptr + indices, mask=mask_bn),
            ACTIVATION_FUNCTION="tanh",
            relu_negative_slope=None,
        )

        tl.store(output_update_ptr + indices, output_update, mask=mask_bn)

        forget_gate = _rnn_forward_update(
            input_state=input_state,
            weight=forget_weight,
            input=tl.load(forget_input_ptr + indices, mask=mask_bn),
            ACTIVATION_FUNCTION="sigmoid",
            relu_negative_slope=None,
        )

        tl.store(forget_gate_ptr + indices, forget_gate, mask=mask_bn)

        input_state = forget_gate * input_state + (1 - forget_gate) * output_update
        tl.store(output_ptr + indices, input_state, mask=mask_bn)

        indices += N


@cute_op(
    f"{LIBRARY_NAME}::scalar_gru_forward_triton", mutates_args={"forget_gate", "reset_gate", "output_update", "output"}
)
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
        scalar_gru_forward_triton_kernel[ceil_divide(B, BLOCK_SIZE_B), ceil_divide(N, BLOCK_SIZE_N)](
            input_ptr=input,
            input_stride_b=input.stride(0),
            weight_ptr=weight,
            forget_input_ptr=forget_input,
            forget_weight_ptr=forget_weight,
            forget_gate_ptr=forget_gate,
            reset_input_ptr=reset_input,
            reset_weight_ptr=reset_weight,
            reset_gate_ptr=reset_gate,
            output_update_ptr=output_update,
            HAS_INPUT_STATE=input_state is not None,
            input_state_ptr=input_state,
            output_ptr=output,
            B=B,
            S=S,
            N=N,
            BLOCK_SIZE_B=BLOCK_SIZE_B,
            BLOCK_SIZE_N=BLOCK_SIZE_N,
        )
