import torch
import triton
import triton.language as tl

from ....constants import LIBRARY_NAME
from ....math import ceil_divide
from ....triton_math import clamp
from ....utils import cute_op
from ...rnn.triton_implementation.backward_scalar import _load_previous_output, _rnn_backward_update


@triton.jit
def scalar_gru_backward_triton_kernel(
    weight_ptr,
    output_ptr,
    output_stride_b,
    forget_weight_ptr,
    forget_gate_ptr,
    forget_input_grad_ptr,
    forget_weight_grad_ptr,
    reset_weight_ptr,
    reset_gate_ptr,
    reset_input_grad_ptr,
    reset_weight_grad_ptr,
    output_update_ptr,
    HAS_INPUT_STATE: tl.constexpr,
    input_state_ptr,
    output_grad_ptr,
    input_grad_ptr,
    weight_grad_ptr,
    HAS_GRADIENT_CLIPPING: tl.constexpr,
    gradient_clipping,
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

    input_state_grad = tl.zeros((BLOCK_SIZE_B, BLOCK_SIZE_N), dtype=weight_ptr.dtype.element_ty)
    weight_grad = tl.zeros((BLOCK_SIZE_N,), dtype=tl.float32)
    forget_weight_grad = tl.zeros((BLOCK_SIZE_N,), dtype=tl.float32)
    reset_weight_grad = tl.zeros((BLOCK_SIZE_N,), dtype=tl.float32)

    weight = tl.load(weight_ptr + indices_n, mask=mask_n)
    forget_weight = tl.load(forget_weight_ptr + indices_n, mask=mask_n)
    reset_weight = tl.load(reset_weight_ptr + indices_n, mask=mask_n)

    indices = indices_b[:, None] * output_stride_b + (S - 1) * N + indices_n[None, :]

    # backward counting reduces 1 instruction since we need to compare s == 0, otherwise we have to compare s == S - 1
    for s in range(S - 1, -1, -1):
        if HAS_GRADIENT_CLIPPING:
            input_state_grad = clamp(input_state_grad, min_value=-gradient_clipping, max_value=gradient_clipping)

        output_grad = tl.load(output_grad_ptr + indices, mask=mask_bn)
        forget_gate = tl.load(forget_gate_ptr + indices, mask=mask_bn)
        reset_gate = tl.load(reset_gate_ptr + indices, mask=mask_bn)
        output_update = tl.load(output_update_ptr + indices, mask=mask_bn)

        input_grad_ptrs = input_grad_ptr + indices
        forget_input_grad_ptrs = forget_input_grad_ptr + indices
        reset_input_grad_ptrs = reset_input_grad_ptr + indices

        output_grad += forget_gate * input_state_grad
        input_state_grad = output_grad

        indices -= N

        output_prev = _load_previous_output(
            HAS_INPUT_STATE=HAS_INPUT_STATE,
            input_state_ptr=input_state_ptr,
            output_ptrs=output_ptr + indices,
            N=N,
            indices_b=indices_b,
            indices_n=indices_n,
            mask_bn=mask_bn,
            BLOCK_SIZE_B=BLOCK_SIZE_B,
            BLOCK_SIZE_N=BLOCK_SIZE_N,
            s=s,
            dtype=weight.dtype,
        )

        input_grad, weight_grad, reset_gate_times_input_state_grad = _rnn_backward_update(
            output=output_update,
            weight=weight,
            output_grad=output_grad * (1 - forget_gate),
            weight_grad=weight_grad,
            output_prev=reset_gate * output_prev,
            ACTIVATION_FUNCTION="tanh",
            relu_negative_slope=None,
        )

        input_state_grad += reset_gate_times_input_state_grad * reset_gate
        tl.store(input_grad_ptrs, input_grad, mask=mask_bn)

        forget_input_grad, forget_weight_grad, input_state_grad_from_forget_gate = _rnn_backward_update(
            output=forget_gate,
            weight=forget_weight,
            output_grad=output_grad * (output_prev - output_update),
            weight_grad=forget_weight_grad,
            output_prev=output_prev,
            ACTIVATION_FUNCTION="sigmoid",
            relu_negative_slope=None,
        )

        input_state_grad += input_state_grad_from_forget_gate
        tl.store(forget_input_grad_ptrs, forget_input_grad, mask=mask_bn)

        reset_input_grad, reset_weight_grad, input_state_grad_from_reset_gate = _rnn_backward_update(
            output=reset_gate,
            weight=reset_weight,
            output_grad=reset_gate_times_input_state_grad * output_prev,
            weight_grad=reset_weight_grad,
            output_prev=output_prev,
            ACTIVATION_FUNCTION="sigmoid",
            relu_negative_slope=None,
        )

        input_state_grad += input_state_grad_from_reset_gate
        tl.store(reset_input_grad_ptrs, reset_input_grad, mask=mask_bn)

    tl.atomic_add(weight_grad_ptr + indices_n, weight_grad, mask=mask_n)
    tl.atomic_add(forget_weight_grad_ptr + indices_n, forget_weight_grad, mask=mask_n)
    tl.atomic_add(reset_weight_grad_ptr + indices_n, reset_weight_grad, mask=mask_n)


@cute_op(
    f"{LIBRARY_NAME}::scalar_gru_backward_triton",
    mutates_args={
        "forget_input_grad",
        "forget_weight_grad",
        "reset_input_grad",
        "reset_weight_grad",
        "input_grad",
        "weight_grad",
    },
)
def scalar_gru_backward_triton(
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
    BLOCK_SIZE_N: int,
) -> None:
    B, S, N, _ = output.size()

    with torch.device(output.device):
        scalar_gru_backward_triton_kernel[ceil_divide(B, BLOCK_SIZE_B), ceil_divide(N, BLOCK_SIZE_N)](
            weight_ptr=weight,
            output_ptr=output,
            output_stride_b=output.stride(0),
            forget_weight_ptr=forget_weight,
            forget_gate_ptr=forget_gate,
            forget_input_grad_ptr=forget_input_grad,
            forget_weight_grad_ptr=forget_weight_grad,
            reset_weight_ptr=reset_weight,
            reset_gate_ptr=reset_gate,
            reset_input_grad_ptr=reset_input_grad,
            reset_weight_grad_ptr=reset_weight_grad,
            output_update_ptr=output_update,
            HAS_INPUT_STATE=input_state is not None,
            input_state_ptr=input_state,
            output_grad_ptr=output_grad,
            input_grad_ptr=input_grad,
            weight_grad_ptr=weight_grad,
            HAS_GRADIENT_CLIPPING=gradient_clipping is not None,
            gradient_clipping=gradient_clipping,
            B=B,
            S=S,
            N=N,
            BLOCK_SIZE_B=BLOCK_SIZE_B,
            BLOCK_SIZE_N=BLOCK_SIZE_N,
        )
