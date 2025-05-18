import torch
import triton
import triton.language as tl

from ....constants import LIBRARY_NAME
from ....math import ceil_divide
from ....triton_math import sigmoid
from ....utils import cute_op, get_num_elements_and_hidden_size


@triton.jit
def _swiglu(gate_ptr, up_ptr, output_ptr, indices, mask):
    gate = tl.load(gate_ptr + indices, mask=mask).to(tl.float32)
    up = tl.load(up_ptr + indices, mask=mask)

    output = up * gate * sigmoid(gate)
    tl.store(output_ptr + indices, output, mask=mask)


@triton.jit
def swiglu_forward_triton_kernel(gate_ptr, up_ptr, output_ptr, N, BLOCK_SIZE: tl.constexpr):
    BLOCK_ID = tl.program_id(axis=0)
    NUM_BLOCKS = tl.num_programs(axis=0)

    indices = BLOCK_ID * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    if BLOCK_ID < NUM_BLOCKS - 1:
        _swiglu(gate_ptr=gate_ptr, up_ptr=up_ptr, output_ptr=output_ptr, indices=indices, mask=None)
    else:
        _swiglu(gate_ptr=gate_ptr, up_ptr=up_ptr, output_ptr=output_ptr, indices=indices, mask=indices < N)


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


@triton.jit
def swiglu_packed_forward_triton_kernel(
    gate_ptr,
    gate_stride_b,
    up_ptr,
    output_ptr,
    output_stride_b,
    B,
    H,
    BLOCK_SIZE_B: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
):
    BLOCK_ID_B = tl.program_id(axis=0)
    BLOCK_ID_H = tl.program_id(axis=1)

    indices_b = BLOCK_ID_B * BLOCK_SIZE_B + tl.arange(0, BLOCK_SIZE_B)
    indices_h = BLOCK_ID_H * BLOCK_SIZE_H + tl.arange(0, BLOCK_SIZE_H)

    half_H = H >> 1

    mask_b = indices_b < B
    mask_h = indices_h < half_H
    mask_bh = mask_b[:, None] & mask_h[None, :]

    indices = indices_b[:, None] * gate_stride_b + indices_h[None, :]

    up_ptrs = up_ptr + indices
    up = tl.load(up_ptrs, mask=mask_bh)

    gate_ptrs = gate_ptr + indices
    gate = tl.load(gate_ptrs, mask=mask_bh).to(tl.float32)

    output = up * gate * sigmoid(gate)

    tl.store(output_ptr + indices_b[:, None] * output_stride_b + indices_h[None, :], output, mask=mask_bh)


@cute_op(f"{LIBRARY_NAME}::swiglu_packed_forward_triton", mutates_args={"output"})
def swiglu_packed_forward_triton(x: torch.Tensor, output: torch.Tensor, BLOCK_SIZE_B: int, BLOCK_SIZE_H: int) -> None:
    B, H = get_num_elements_and_hidden_size(x)
    up, gate = x.chunk(2, dim=-1)

    with torch.device(x.device):
        swiglu_packed_forward_triton_kernel[ceil_divide(B, BLOCK_SIZE_B), ceil_divide(H, BLOCK_SIZE_H)](
            gate_ptr=gate,
            gate_stride_b=gate.stride(0),
            up_ptr=up,
            output_ptr=output,
            B=B,
            H=H,
            BLOCK_SIZE_B=BLOCK_SIZE_B,
            BLOCK_SIZE_H=BLOCK_SIZE_H,
        )
