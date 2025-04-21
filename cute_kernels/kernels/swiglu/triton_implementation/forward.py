import triton
import triton.language as tl

from ....triton_math import sigmoid


@triton.jit
def _swiglu(gate_ptr, up_ptr, output_ptr, indices, mask):
    gate = tl.load(gate_ptr + indices, mask=mask).to(tl.float32)
    up = tl.load(up_ptr + indices, mask=mask)

    output = up * gate * sigmoid(gate)
    tl.store(output_ptr + indices, output, mask=mask)


@triton.jit
def swiglu_forward_triton(gate_ptr, up_ptr, output_ptr, N, BLOCK_SIZE: tl.constexpr):
    BLOCK_ID = tl.program_id(axis=0)
    NUM_BLOCKS = tl.num_programs(axis=0)

    indices = BLOCK_ID * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    if BLOCK_ID < NUM_BLOCKS - 1:
        _swiglu(gate_ptr=gate_ptr, up_ptr=up_ptr, output_ptr=output_ptr, indices=indices, mask=None)
    else:
        _swiglu(gate_ptr=gate_ptr, up_ptr=up_ptr, output_ptr=output_ptr, indices=indices, mask=indices < N)
