import triton
import triton.language as tl

from ....triton_math import sigmoid


@triton.jit
def swiglu_forward_triton_kernel(gate_ptr, up_ptr, output_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)

    indices = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = indices < N

    gate = tl.load(gate_ptr + indices, mask=mask).to(tl.float32)
    up = tl.load(up_ptr + indices, mask=mask)

    output = up * gate * sigmoid(gate)

    tl.store(output_ptr + indices, output, mask=mask)
