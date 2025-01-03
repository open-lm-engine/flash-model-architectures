import torch
import triton.language as tl

from .math import get_powers_of_2


LIBRARY_NAME = "cute"

MAX_CUDA_BLOCK_SIZE = 1024
COMMON_CUDA_BLOCK_SIZES_POWERS_OF_2 = get_powers_of_2(32, MAX_CUDA_BLOCK_SIZE)

MAX_TRITON_BLOCK_SIZE = 65536
COMMON_TRITON_BLOCK_SIZES_POWERS_OF_2 = get_powers_of_2(64, MAX_TRITON_BLOCK_SIZE)

MAX_FP32_INSTRUCTION_WIDTH = 4
MAX_FP16_BF16_INSTRUCTION_WIDTH = 8
COMMON_VECTOR_INSTRUCTION_WIDTHS = get_powers_of_2(1, MAX_FP32_INSTRUCTION_WIDTH)

THREAD_BLOCK_CLUSTER_SIZES = get_powers_of_2(1, 8)

TORCH_TO_TRITON_DTYPE = {
    torch.float32: tl.float32,
    torch.float16: tl.float16,
    torch.bfloat16: tl.bfloat16,
}
