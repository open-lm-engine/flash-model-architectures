import torch
import triton
import triton.language as tl

from ...constants import LIBRARY_NAME
from ...math import ceil_divide
from ...utils import cute_op, get_sm_count


_KERNEL_NAME = "continuous_count_triton"


@triton.jit
def _continuous_count_triton_kernel(x_ptr, output_ptr, B, C, BLOCK_SIZE_B: tl.constexpr, BLOCK_SIZE_C: tl.constexpr):
    pid = tl.program_id(axis=0)
    num_programs = tl.num_programs(axis=0)

    num_elements_per_program = tl.cdiv(B, num_programs)

    indices_c = tl.arange(0, BLOCK_SIZE_C)
    mask_c = indices_c < C

    program_start = pid * num_elements_per_program
    program_end = min(program_start + num_elements_per_program, B)
    num_elements_in_current_program = program_end - program_start

    if num_elements_in_current_program > 0:
        num_loops = tl.cdiv(num_elements_in_current_program, BLOCK_SIZE_B)
        counts = tl.zeros((BLOCK_SIZE_C,), dtype=tl.uint32)

        for i in range(num_loops):
            indices_b = program_start + i * BLOCK_SIZE_B + tl.arange(0, BLOCK_SIZE_B)
            mask_b = indices_b < program_end

            x = tl.load(x_ptr + indices_b, mask=mask_b, other=-1)

            equal = (x[:, None] == indices_c[None, :]).to(tl.uint32)
            counts += tl.sum(equal, axis=0)

        tl.atomic_add(output_ptr + indices_c, counts, mask=mask_c)


@cute_op(f"{LIBRARY_NAME}::{_KERNEL_NAME}", mutates_args={"output"})
def continuous_count_triton(
    x: torch.Tensor, output: torch.Tensor, size: int, BLOCK_SIZE: int, BLOCK_SIZE_C: int
) -> None:
    B = x.numel()

    sm_count = get_sm_count(x.device)
    num_programs = min(sm_count, ceil_divide(B, BLOCK_SIZE))

    with torch.device(x.device):
        _continuous_count_triton_kernel[(num_programs,)](
            x_ptr=x,
            output_ptr=output,
            B=B,
            C=size,
            BLOCK_SIZE_B=BLOCK_SIZE,
            BLOCK_SIZE_C=BLOCK_SIZE_C,
        )
