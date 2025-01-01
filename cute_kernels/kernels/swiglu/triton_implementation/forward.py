import torch
import triton
import triton.language as tl

from ....constants import LIBRARY_NAME
from ....math import ceil_divide
from ....utils import cute_op


_KERNEL_NAME = "swiglu_forward_triton"


@triton.jit
def _swiglu_forward_triton_kernel(gate_ptr, up_ptr, output_ptr, num_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)

    indices = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = indices < num_elements

    gate = tl.load(gate_ptr + indices, mask=mask).to(tl.float32)
    up = tl.load(up_ptr + indices, mask=mask)

    output = up * gate * tl.sigmoid(gate)

    tl.store(output_ptr + indices, output, mask=mask)


@cute_op(f"{LIBRARY_NAME}::{_KERNEL_NAME}", mutates_args={"output"})
def swiglu_forward_triton(gate: torch.Tensor, up: torch.Tensor, output: torch.Tensor, BLOCK_SIZE: int) -> None:
    num_elements = gate.numel()

    with torch.device(gate.device):
        _swiglu_forward_triton_kernel[(ceil_divide(num_elements, BLOCK_SIZE),)](
            gate_ptr=gate,
            up_ptr=up,
            output_ptr=output,
            num_elements=num_elements,
            BLOCK_SIZE=BLOCK_SIZE,
        )
