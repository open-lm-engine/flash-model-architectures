import triton
import triton.language as tl

from ....triton_math import sigmoid


@triton.jit
def _swiglu_backward(gate_ptr, up_ptr, output_grad_ptr, gate_grad_ptr, up_grad_ptr, indices, mask):
    gate = tl.load(gate_ptr + indices, mask=mask).to(tl.float32)
    up = tl.load(up_ptr + indices, mask=mask)
    output_grad = tl.load(output_grad_ptr + indices, mask=mask)

    gate_sigmoid = sigmoid(gate)
    gate_silu = gate * gate_sigmoid

    gate_grad = output_grad * up * (gate_sigmoid + gate_silu * (1 - gate_sigmoid))
    up_grad = output_grad * gate_silu

    tl.store(gate_grad_ptr + indices, gate_grad, mask=mask)
    tl.store(up_grad_ptr + indices, up_grad, mask=mask)


@triton.jit
def swiglu_backward_triton(gate_ptr, up_ptr, output_grad_ptr, gate_grad_ptr, up_grad_ptr, N, BLOCK_SIZE: tl.constexpr):
    BLOCK_ID = tl.program_id(axis=0)
    NUM_BLOCKS = tl.num_programs(axis=0)

    indices = BLOCK_ID * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    if BLOCK_ID < NUM_BLOCKS - 1:
        _swiglu_backward(
            gate_ptr=gate_ptr,
            up_ptr=up_ptr,
            output_grad_ptr=output_grad_ptr,
            gate_grad_ptr=gate_grad_ptr,
            up_grad_ptr=up_grad_ptr,
            indices=indices,
            mask=None,
        )
    else:
        _swiglu_backward(
            gate_ptr=gate_ptr,
            up_ptr=up_ptr,
            output_grad_ptr=output_grad_ptr,
            gate_grad_ptr=gate_grad_ptr,
            up_grad_ptr=up_grad_ptr,
            indices=indices,
            mask=indices < N,
        )
