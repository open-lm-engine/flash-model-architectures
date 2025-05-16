import torch
import triton
import triton.language as tl

from ...constants import LIBRARY_NAME
from ...math import ceil_divide
from ...utils import cute_op


@triton.jit
def _add_scalar(x_ptr, y, output_ptr, indices, mask):
    x = tl.load(x_ptr + indices, mask=mask)
    tl.store(output_ptr + indices, x + y, mask=mask)


@triton.autotune(configs=[triton.Config({"BLOCK_SIZE": 4096}, num_warps=32)], key=[])
@triton.jit
def add_scalar_triton_kernel(x_ptr, y, output_ptr, N, BLOCK_SIZE: tl.constexpr):
    BLOCK_ID = tl.program_id(axis=0)
    NUM_BLOCKS = tl.num_programs(axis=0)

    indices = BLOCK_ID * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    if BLOCK_ID < NUM_BLOCKS - 1:
        _add_scalar(x_ptr=x_ptr, y=y, output_ptr=output_ptr, indices=indices, mask=None)
    else:
        _add_scalar(x_ptr=x_ptr, y=y, output_ptr=output_ptr, indices=indices, mask=indices < N)


@cute_op(f"{LIBRARY_NAME}::add_scalar_triton", mutates_args={"output"})
def add_scalar_triton(x: torch.Tensor, y: float, output: torch.Tensor) -> None:
    N = x.numel()
    NUM_BLOCKS = lambda meta: (ceil_divide(N, meta["BLOCK_SIZE"]),)

    with torch.device(x.device):
        add_scalar_triton_kernel[NUM_BLOCKS](x_ptr=x, y=y, output_ptr=output, N=N)
