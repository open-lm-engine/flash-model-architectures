import torch
import triton.language as tl

from .math import get_powers_of_2


LIBRARY_NAME = "cute"

MAX_CUDA_BLOCK_SIZE = 1024
COMMON_CUDA_BLOCK_SIZES_POWERS_OF_2 = get_powers_of_2(32, MAX_CUDA_BLOCK_SIZE)

MAX_TRITON_BLOCK_SIZE = 65536
COMMON_TRITON_BLOCK_SIZES_POWERS_OF_2 = get_powers_of_2(64, MAX_TRITON_BLOCK_SIZE)

THREAD_BLOCK_CLUSTER_SIZES = get_powers_of_2(1, 8)

TORCH_TO_TRITON_DTYPE = {
    torch.float32: tl.float32,
    torch.float16: tl.float16,
    torch.bfloat16: tl.bfloat16,
}

DTYPE_TO_SIZE = {
    torch.int: 4,
    torch.int32: 4,
    torch.uint32: 4,
    torch.long: 8,
    torch.int64: 8,
    torch.uint64: 8,
}
