import triton
import triton.language as tl


@triton.jit
def _add_scalar_triton_kernel(x_ptr, y, output_ptr, num_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)

    indices = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = indices < num_elements

    x = tl.load(x_ptr + indices, mask=mask)
    output = x + y

    tl.store(output_ptr + indices, output, mask=mask)
