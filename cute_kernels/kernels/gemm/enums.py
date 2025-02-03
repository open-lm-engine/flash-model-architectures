from enum import Enum


class CUDAKernelAlgorithm(Enum):
    cutlass_gemm_cuda = "cutlass_gemm_cuda"
    naive = "naive"
    shared_memory = "shared_memory"
