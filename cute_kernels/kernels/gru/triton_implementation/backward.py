import torch
import triton
import triton.language as tl

from ....constants import LIBRARY_NAME
from ....math import ceil_divide, get_next_power_of_2
from ....triton_math import clamp
from ....utils import cute_op
from ...rnn.triton_implementation.backward import _load_previous_output, _rnn_backward_update


@triton.jit
def gru_backward_triton_kernel(
    W_ptr,
    W_stride_n,
    y_ptr,
    y_stride_b,
    y_stride_s,
    Wf_ptr,
    f_ptr,
    dxf_ptr,
    dWf_ptr,
    Wr_ptr,
    r_ptr,
    reset_input_grad_ptr,
    reset_weight_grad_ptr,
    output_update_ptr,
    HAS_INPUT_STATE: tl.constexpr,
    input_state_ptr,
    input_state_stride_b,
    dy_ptr,
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
    indices_weight = pid_n * W_stride_n + indices_h[:, None] * H + indices_h[None, :]

    mask_b = indices_b < B
    mask_h = indices_h < H
    mask_bh = mask_b[:, None] & mask_h[None, :]
    mask_hh = mask_h[:, None] & mask_h[None, :]

    input_state_grad = tl.zeros((BLOCK_SIZE_B, BLOCK_SIZE_H), dtype=W_ptr.dtype.element_ty)
    weight_grad = tl.zeros((BLOCK_SIZE_H, BLOCK_SIZE_H), dtype=tl.float32)
    forget_weight_grad = tl.zeros((BLOCK_SIZE_H, BLOCK_SIZE_H), dtype=tl.float32)
    reset_weight_grad = tl.zeros((BLOCK_SIZE_H, BLOCK_SIZE_H), dtype=tl.float32)

    weight = tl.load(W_ptr + indices_weight, mask=mask_hh)
    forget_weight = tl.load(Wf_ptr + indices_weight, mask=mask_hh)
    reset_weight = tl.load(Wr_ptr + indices_weight, mask=mask_hh)

    indices = indices_b[:, None] * y_stride_b + (S - 1) * y_stride_s + pid_n * H + indices_h[None, :]

    # backward counting reduces 1 instruction since we need to compare s == 0, otherwise we have to compare s == S - 1
    for s in range(S - 1, -1, -1):
        if HAS_GRADIENT_CLIPPING:
            input_state_grad = clamp(input_state_grad, min_value=-gradient_clipping, max_value=gradient_clipping)

        dy = tl.load(dy_ptr + indices, mask=mask_bh)
        f = tl.load(f_ptr + indices, mask=mask_bh)
        r = tl.load(r_ptr + indices, mask=mask_bh)
        output_update = tl.load(output_update_ptr + indices, mask=mask_bh)

        input_grad_ptrs = input_grad_ptr + indices
        dxf_ptrs = dxf_ptr + indices
        reset_input_grad_ptrs = reset_input_grad_ptr + indices

        dy += f * input_state_grad
        input_state_grad = dy

        indices -= y_stride_s

        output_prev = _load_previous_output(
            HAS_INPUT_STATE=HAS_INPUT_STATE,
            h_ptr=input_state_ptr,
            h_stride_b=input_state_stride_b,
            y_ptrs=y_ptr + indices,
            pid_n=pid_n,
            H=H,
            indices_b=indices_b,
            indices_h=indices_h,
            mask_bh=mask_bh,
            BLOCK_SIZE_B=BLOCK_SIZE_B,
            BLOCK_SIZE_H=BLOCK_SIZE_H,
            s=s,
            dtype=weight.dtype,
        )

        input_grad, weight_grad, r_times_input_state_grad = _rnn_backward_update(
            y=output_update,
            W=weight,
            dy=dy * (1 - f),
            dW=weight_grad,
            y_prev=r * output_prev,
            ACTIVATION_FUNCTION="tanh",
            relu_negative_slope=None,
        )

        input_state_grad += r_times_input_state_grad * r
        tl.store(input_grad_ptrs, input_grad, mask=mask_bh)

        forget_input_grad, forget_weight_grad, input_state_grad_from_f = _rnn_backward_update(
            y=f,
            W=forget_weight,
            dy=dy * (output_prev - output_update),
            dW=forget_weight_grad,
            y_prev=output_prev,
            ACTIVATION_FUNCTION="sigmoid",
            relu_negative_slope=None,
        )

        input_state_grad += input_state_grad_from_f
        tl.store(dxf_ptrs, forget_input_grad, mask=mask_bh)

        reset_input_grad, reset_weight_grad, input_state_grad_from_r = _rnn_backward_update(
            y=r,
            W=reset_weight,
            dy=r_times_input_state_grad * output_prev,
            dW=reset_weight_grad,
            y_prev=output_prev,
            ACTIVATION_FUNCTION="sigmoid",
            relu_negative_slope=None,
        )

        input_state_grad += input_state_grad_from_r
        tl.store(reset_input_grad_ptrs, reset_input_grad, mask=mask_bh)

    tl.atomic_add(weight_grad_ptr + indices_weight, weight_grad, mask=mask_hh)
    tl.atomic_add(dWf_ptr + indices_weight, forget_weight_grad, mask=mask_hh)
    tl.atomic_add(reset_weight_grad_ptr + indices_weight, reset_weight_grad, mask=mask_hh)


@cute_op(
    f"{LIBRARY_NAME}::gru_backward_triton",
    mutates_args={
        "forget_input_grad",
        "forget_weight_grad",
        "reset_input_grad",
        "reset_weight_grad",
        "input_grad",
        "weight_grad",
    },
)
def gru_backward_triton(
    weight: torch.Tensor,
    output: torch.Tensor,
    forget_weight: torch.Tensor,
    forget_gate: torch.Tensor,
    forget_input_grad: torch.Tensor,
    forget_weight_grad: torch.Tensor,
    reset_weight: torch.Tensor,
    reset_gate: torch.Tensor,
    reset_input_grad: torch.Tensor,
    reset_weight_grad: torch.Tensor,
    output_update: torch.Tensor,
    input_state: torch.Tensor | None,
    output_grad: torch.Tensor,
    input_grad: torch.Tensor,
    weight_grad: torch.Tensor,
    gradient_clipping: float | None,
    BLOCK_SIZE_B: int,
) -> None:
    B, S, N, H = output.size()

    BLOCK_SIZE_H = get_next_power_of_2(H)
    BLOCK_SIZE_H = max(16, BLOCK_SIZE_H)

    with torch.device(output.device):
        gru_backward_triton_kernel[ceil_divide(B, BLOCK_SIZE_B), N](
            W_ptr=weight,
            W_stride_n=weight.stride(0),
            y_ptr=output,
            y_stride_b=output.stride(0),
            y_stride_s=output.stride(1),
            Wf_ptr=forget_weight,
            f_ptr=forget_gate,
            dxf_ptr=forget_input_grad,
            dWf_ptr=forget_weight_grad,
            Wr_ptr=reset_weight,
            r_ptr=reset_gate,
            reset_input_grad_ptr=reset_input_grad,
            reset_weight_grad_ptr=reset_weight_grad,
            output_update_ptr=output_update,
            HAS_INPUT_STATE=input_state is not None,
            input_state_ptr=input_state,
            input_state_stride_b=None if input_state is None else input_state.stride(0),
            dy_ptr=output_grad,
            input_grad_ptr=input_grad,
            weight_grad_ptr=weight_grad,
            HAS_GRADIENT_CLIPPING=gradient_clipping is not None,
            gradient_clipping=gradient_clipping,
            B=B,
            S=S,
            H=H,
            BLOCK_SIZE_B=BLOCK_SIZE_B,
            BLOCK_SIZE_H=BLOCK_SIZE_H,
        )
