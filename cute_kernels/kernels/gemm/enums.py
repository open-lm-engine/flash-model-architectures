from enum import Enum


class CUDAKernelAlgorithm(Enum):
    naive = "naive"
    shared_memory = "shared_memory"
