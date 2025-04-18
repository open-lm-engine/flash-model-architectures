import triton
import triton.language as tl


@triton.jit
def _add_tensor_triton_kernel(x_ptr, y_ptr, output_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)

    indices = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = indices < N

    x = tl.load(x_ptr + indices, mask=mask)
    y = tl.load(y_ptr + indices, mask=mask)

    output = x + y

    tl.store(output_ptr + indices, output, mask=mask)
