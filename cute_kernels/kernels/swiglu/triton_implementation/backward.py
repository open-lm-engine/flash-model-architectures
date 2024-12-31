import torch
import triton
import triton.language as tl

from ....constants import LIBRARY_NAME
from ....math import ceil_divide
from ....utils import cute_op


_KERNEL_NAME = "swiglu_backward_triton"


@triton.jit
def _swiglu_backward_triton_kernel(
    gate_ptr, up_ptr, output_grad_ptr, gate_grad_ptr, up_grad_ptr, num_elements, BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)

    indices = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = indices < num_elements

    gate = tl.load(gate_ptr + indices, mask=mask).to(tl.float32)
    up = tl.load(up_ptr + indices, mask=mask)
    output_grad = tl.load(output_grad_ptr + indices, mask=mask)

    gate_sigmoid = tl.sigmoid(gate)
    gate_silu = gate * gate_sigmoid

    gate_grad = output_grad * up * (gate_sigmoid + gate_silu * (1 - gate_sigmoid))
    up_grad = output_grad * gate_silu

    tl.store(gate_grad_ptr + indices, gate_grad, mask=mask)
    tl.store(up_grad_ptr + indices, up_grad, mask=mask)


@cute_op(f"{LIBRARY_NAME}::{_KERNEL_NAME}", mutates_args={"gate_grad", "up_grad"})
def swiglu_backward_triton(
    gate: torch.Tensor,
    up: torch.Tensor,
    output_grad: torch.Tensor,
    gate_grad: torch.Tensor,
    up_grad: torch.Tensor,
    BLOCK_SIZE: int,
) -> None:
    num_elements = gate.numel()

    with torch.device(gate.device):
        _swiglu_backward_triton_kernel[(ceil_divide(num_elements, BLOCK_SIZE),)](
            gate_ptr=gate,
            up_ptr=up,
            output_grad_ptr=output_grad,
            gate_grad_ptr=gate_grad,
            up_grad_ptr=up_grad,
            num_elements=num_elements,
            BLOCK_SIZE=BLOCK_SIZE,
        )
