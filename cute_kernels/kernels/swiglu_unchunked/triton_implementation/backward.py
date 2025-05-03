import torch
import triton
import triton.language as tl

from ....constants import LIBRARY_NAME
from ....math import ceil_divide
from ....triton_math import sigmoid
from ....utils import cute_op, get_num_elements_and_hidden_size


@triton.jit
def swiglu_unchunked_backward_triton_kernel(
    x_ptr, output_grad_ptr, x_grad_ptr, B, H, BLOCK_SIZE_B: tl.constexpr, BLOCK_SIZE_H: tl.constexpr
):
    pid_b = tl.program_id(axis=0)
    pid_h = tl.program_id(axis=1)

    indices_b = pid_b * BLOCK_SIZE_B + tl.arange(0, BLOCK_SIZE_B)
    indices_h = pid_h * BLOCK_SIZE_H + tl.arange(0, BLOCK_SIZE_H)

    half_H = H >> 1

    mask_b = indices_b < B
    mask_h = indices_h < half_H
    mask_bh = mask_b[:, None] & mask_h[None, :]

    up_ptrs = x_ptr + indices_b[:, None] * H + indices_h[None, :]
    up = tl.load(up_ptrs, mask=mask_bh)

    gate_ptrs = up_ptrs + (H >> 1)
    gate = tl.load(gate_ptrs, mask=mask_bh).to(tl.float32)

    output_grad_ptrs = output_grad_ptr + indices_b[:, None] * half_H + indices_h[None, :]
    output_grad = tl.load(output_grad_ptrs, mask=mask_bh)

    gate_sigmoid = sigmoid(gate)
    gate_silu = gate * gate_sigmoid

    gate_grad = output_grad * up * (gate_sigmoid + gate_silu * (1 - gate_sigmoid))
    up_grad = output_grad * gate_silu

    up_grad_ptrs = x_grad_ptr + indices_b[:, None] * H + indices_h[None, :]
    tl.store(up_grad_ptrs, up_grad, mask=mask_bh)

    gate_grad_ptrs = up_grad_ptrs + (H >> 1)
    tl.store(gate_grad_ptrs, gate_grad, mask=mask_bh)


@cute_op(f"{LIBRARY_NAME}::swiglu_unchunked_backward_triton", mutates_args={"x_grad"})
def swiglu_unchunked_backward_triton(
    x: torch.Tensor,
    output_grad: torch.Tensor,
    x_grad: torch.Tensor,
    BLOCK_SIZE_B: int,
    BLOCK_SIZE_H: int,
) -> None:
    B, H = get_num_elements_and_hidden_size(x)

    with torch.device(x.device):
        swiglu_unchunked_backward_triton_kernel[ceil_divide(B, BLOCK_SIZE_B), ceil_divide(H, BLOCK_SIZE_H)](
            x_ptr=x,
            output_grad_ptr=output_grad,
            x_grad_ptr=x_grad,
            B=B,
            H=H,
            BLOCK_SIZE_B=BLOCK_SIZE_B,
            BLOCK_SIZE_H=BLOCK_SIZE_H,
        )
