import torch
import triton
import triton.language as tl

from ....constants import LIBRARY_NAME
from ....math import ceil_divide
from ....triton_math import sigmoid
from ....utils import cute_op


@triton.jit
def swiglu_forward_triton_kernel(gate_ptr, up_ptr, output_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)

    indices = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = indices < N

    gate = tl.load(gate_ptr + indices, mask=mask).to(tl.float32)
    up = tl.load(up_ptr + indices, mask=mask)

    output = up * gate * sigmoid(gate)

    tl.store(output_ptr + indices, output, mask=mask)


@cute_op(f"{LIBRARY_NAME}::swiglu_forward_triton", mutates_args={"output"})
def swiglu_forward_triton(
    gate: torch.Tensor, up: torch.Tensor, output: torch.Tensor, BLOCK_SIZE: int, NUM_WARPS: int
) -> None:
    N = gate.numel()

    with torch.device(gate.device):
        swiglu_forward_triton_kernel[ceil_divide(N, BLOCK_SIZE),](
            gate_ptr=gate,
            up_ptr=up,
            output_ptr=output,
            N=N,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=NUM_WARPS,
        )
