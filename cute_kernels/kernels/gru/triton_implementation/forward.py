import torch
import triton
import triton.language as tl

from ....constants import LIBRARY_NAME
from ....math import ceil_divide, get_next_power_of_2
from ....utils import cute_op
from ...rnn.triton_implementation.forward import _rnn_forward_update


@triton.jit
def gru_forward_triton_kernel(
    input_ptr,
    input_stride_b,
    input_stride_s,
    weight_ptr,
    weight_stride_n,
    forget_input_ptr,
    forget_weight_ptr,
    forget_gate_ptr,
    reset_input_ptr,
    reset_weight_ptr,
    reset_gate_ptr,
    output_update_ptr,
    HAS_INPUT_STATE: tl.constexpr,
    input_state_ptr,
    input_state_stride_b,
    output_ptr,
    B,
    S,
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

    indices = pid_n * weight_stride_n + indices_h[:, None] * H + indices_h[None, :]
    mask_hh = mask_h[:, None] & mask_h[None, :]

    weight = tl.load(weight_ptr + indices, mask=mask_hh)
    forget_weight = tl.load(forget_weight_ptr + indices, mask=mask_hh)
    reset_weight = tl.load(reset_weight_ptr + indices, mask=mask_hh)

    if HAS_INPUT_STATE:
        input_state = tl.load(
            input_state_ptr + indices_b[:, None] * input_state_stride_b + pid_n * H + indices_h[None, :], mask=mask_bh
        )
    else:
        input_state = tl.zeros((BLOCK_SIZE_B, BLOCK_SIZE_H), dtype=input_ptr.dtype.element_ty)

    indices = indices_b[:, None] * input_stride_b + pid_n * H + indices_h[None, :]

    input_dtype = input_ptr.dtype.element_ty
    cast_dtype = input_dtype
    if input_dtype == tl.bfloat16:
        input_dtype = tl.float32
        cast_dtype = tl.bfloat16

    out_dtype = input_dtype

    for _ in range(S):
        reset_gate = _rnn_forward_update(
            h=input_state,
            W=reset_weight,
            x=tl.load(reset_input_ptr + indices, mask=mask_bh).to(input_dtype),
            out_dtype=out_dtype,
            cast_dtype=cast_dtype,
            ACTIVATION_FUNCTION="sigmoid",
            relu_negative_slope=None,
        )

        tl.store(reset_gate_ptr + indices, reset_gate, mask=mask_bh)

        output_update = _rnn_forward_update(
            h=input_state * reset_gate,
            W=weight,
            x=tl.load(input_ptr + indices, mask=mask_bh).to(input_dtype),
            out_dtype=out_dtype,
            cast_dtype=cast_dtype,
            ACTIVATION_FUNCTION="tanh",
            relu_negative_slope=None,
        )

        tl.store(output_update_ptr + indices, output_update, mask=mask_bh)

        forget_gate = _rnn_forward_update(
            h=input_state,
            W=forget_weight,
            x=tl.load(forget_input_ptr + indices, mask=mask_bh).to(input_dtype),
            out_dtype=out_dtype,
            cast_dtype=cast_dtype,
            ACTIVATION_FUNCTION="sigmoid",
            relu_negative_slope=None,
        )

        tl.store(forget_gate_ptr + indices, forget_gate, mask=mask_bh)

        input_state = forget_gate * input_state + (1 - forget_gate) * output_update
        tl.store(output_ptr + indices, input_state, mask=mask_bh)

        indices += input_stride_s


@cute_op(f"{LIBRARY_NAME}::gru_forward_triton", mutates_args={"forget_gate", "reset_gate", "output_update", "output"})
def gru_forward_triton(
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
) -> None:
    B, S, N, H = input.size()

    BLOCK_SIZE_H = get_next_power_of_2(H)
    BLOCK_SIZE_H = max(16, BLOCK_SIZE_H)

    has_input_state = input_state is not None

    with torch.device(input.device):
        gru_forward_triton_kernel[ceil_divide(B, BLOCK_SIZE_B), N](
            input_ptr=input,
            input_stride_b=input.stride(0),
            input_stride_s=input.stride(1),
            weight_ptr=weight,
            weight_stride_n=weight.stride(0),
            forget_input_ptr=forget_input,
            forget_weight_ptr=forget_weight,
            forget_gate_ptr=forget_gate,
            reset_input_ptr=reset_input,
            reset_weight_ptr=reset_weight,
            reset_gate_ptr=reset_gate,
            output_update_ptr=output_update,
            HAS_INPUT_STATE=has_input_state,
            input_state_ptr=input_state,
            input_state_stride_b=input_state.stride(0) if has_input_state else None,
            output_ptr=output,
            B=B,
            S=S,
            H=H,
            BLOCK_SIZE_B=BLOCK_SIZE_B,
            BLOCK_SIZE_H=BLOCK_SIZE_H,
        )
