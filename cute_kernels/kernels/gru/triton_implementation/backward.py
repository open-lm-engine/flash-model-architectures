import torch
import triton
import triton.language as tl

from ....constants import LIBRARY_NAME
from ....math import ceil_divide, get_next_power_of_2
from ....triton_math import clamp, sigmoid_backward, tanh_backward
from ....utils import cute_op
from ...rnn.triton_implementation.backward import _load_previous_output, _rnn_backward_update


@triton.jit
def gru_backward_triton_kernel(
    weight_ptr,
    weight_stride_n,
    output_ptr,
    output_stride_b,
    output_stride_s,
    forget_weight_ptr,
    forget_gate_ptr,
    reset_weight_ptr,
    reset_gate_ptr,
    output_update_ptr,
    HAS_INPUT_STATE: tl.constexpr,
    input_state_ptr,
    input_state_stride_b,
    output_grad_ptr,
    input_grad_ptr,
    weight_grad_ptr,
    HAS_GRADIENT_CLIPPING: tl.constexpr,
    gradient_clipping,
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

    input_state_grad = tl.zeros((BLOCK_SIZE_B, BLOCK_SIZE_H), dtype=weight_ptr.dtype.element_ty)
    weight_grad = tl.zeros((BLOCK_SIZE_H, BLOCK_SIZE_H), dtype=tl.float32)

    indices = pid_n * weight_stride_n + indices_h[:, None] * H + indices_h[None, :]
    mask_hh = mask_h[:, None] & mask_h[None, :]

    weight_ptrs = weight_ptr + indices
    weight = tl.load(weight_ptrs, mask=mask_hh, other=0)

    forget_weight_ptrs = forget_weight_ptr + indices
    forget_weight = tl.load(forget_weight_ptrs, mask=mask_hh, other=0)

    reset_weight_ptrs = reset_weight_ptr + indices
    reset_weight = tl.load(reset_weight_ptrs, mask=mask_hh, other=0)

    weight_ptrs = weight_ptr + indices
    weight = tl.load(weight_ptrs, mask=mask_hh, other=0)

    indices = indices_b[:, None] * output_stride_b + (S - 1) * output_stride_s + pid_n * H + indices_h[None, :]

    output_ptrs = output_ptr + indices
    output = tl.load(output_ptrs, mask=mask_bh, other=0)

    # backward counting reduces 1 instruction since we need to compare s == 0, otherwise we have to compare s == S - 1
    for s in range(S - 1, -1, -1):
        output_grad_ptrs = output_grad_ptr + indices
        output_grad = tl.load(output_grad_ptrs, mask=mask_bh, other=0)

        if HAS_GRADIENT_CLIPPING:
            input_state_grad = clamp(input_state_grad, min_value=-gradient_clipping, max_value=gradient_clipping)

        forget_gate_ptrs = forget_gate_ptr + indices
        forget_gate = tl.load(forget_gate_ptrs, mask=mask_bh)

        output_update_ptrs = output_update_ptr + indices
        output_update = tl.load(output_update_ptrs, mask=mask_bh)

        output_grad += input_state_grad
        forget_gate_grad = output_grad * (forget_gate - output_update)

        output_ptrs -= output_stride_s
        output_prev = _load_previous_output(
            HAS_INPUT_STATE=HAS_INPUT_STATE,
            input_state_ptr=input_state_ptr,
            input_state_stride_b=input_state_stride_b,
            output_ptrs=output_ptrs,
            pid_n=pid_n,
            indices_b=indices_b,
            indices_h=indices_h,
            mask_bh=mask_bh,
            BLOCK_SIZE_B=BLOCK_SIZE_B,
            BLOCK_SIZE_H=BLOCK_SIZE_H,
            s=s,
            dtype=weight.dtype,
        )

        forget_input_grad, forger_weight_grad, input_state_grad = _backward_rnn_update(
            output=forget_gate,
            weight=forget_weight,
            output_grad=forget_gate_grad,
            weight_grad=forget_weight_grad,
            output_prev=output_prev,
            ACTIVATION_FUNCTION="sigmoid",
            relu_negative_slope=None,
        )

        input_grad_ptrs = input_grad_ptr + indices
        tl.store(input_grad_ptrs, input_grad, mask=mask_bh)

        input_state_grad = tl.dot(input_grad, weight.T, allow_tf32=True).to(input_state_grad.dtype)

        weight_grad = tl.dot(output_prev.T, input_grad, weight_grad, allow_tf32=True)
        output = output_prev

        indices -= output_stride_s

    weight_grad_ptrs = weight_grad_ptr + pid_n * weight_stride_n + indices_h[:, None] * H + indices_h[None, :]
    tl.store(weight_grad_ptrs, weight_grad, mask=mask_h[:, None] & mask_h[None, :])


@cute_op(f"{LIBRARY_NAME}::gru_backward_triton", mutates_args={"input_grad", "weight_grad"})
def gru_backward_triton(
    weight: torch.Tensor,
    output: torch.Tensor,
    forget_weight: torch.Tensor,
    forget_gate: torch.Tensor,
    reset_weight: torch.Tensor,
    reset_gate: torch.Tensor,
    output_update: torch.Tensor,
    input_state: torch.Tensor | None,
    output_grad: torch.Tensor,
    input_grad: torch.Tensor,
    weight_grad: torch.Tensor,
    gradient_clipping: float | None,
    activation_function: str,
    relu_negative_slope: float | None,
    BLOCK_SIZE_B: int,
) -> None:
    B, S, N, H = output.size()

    BLOCK_SIZE_H = get_next_power_of_2(H)
    BLOCK_SIZE_H = max(16, BLOCK_SIZE_H)

    with torch.device(output.device):
        gru_backward_triton_kernel[ceil_divide(B, BLOCK_SIZE_B), N](
            weight_ptr=weight,
            weight_stride_n=weight.stride(0),
            output_ptr=output,
            output_stride_b=output.stride(0),
            output_stride_s=output.stride(1),
            forget_weight_str=forget_weight,
            forget_gate_str=forget_gate,
            reset_weight_str=reset_weight,
            reset_gate_str=reset_gate,
            output_update_ptr=output_update,
            HAS_INPUT_STATE=input_state is not None,
            input_state_ptr=input_state,
            input_state_stride_b=None if input_state is None else input_state.stride(0),
            output_grad_ptr=output_grad,
            input_grad_ptr=input_grad,
            weight_grad_ptr=weight_grad,
            HAS_GRADIENT_CLIPPING=gradient_clipping is not None,
            gradient_clipping=gradient_clipping,
            ACTIVATION_FUNCTION=activation_function,
            relu_negative_slope=relu_negative_slope,
            B=B,
            S=S,
            H=H,
            BLOCK_SIZE_B=BLOCK_SIZE_B,
            BLOCK_SIZE_H=BLOCK_SIZE_H,
        )
