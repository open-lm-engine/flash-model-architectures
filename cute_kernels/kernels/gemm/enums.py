from enum import Enum


class CUDAKernelAlgorithm(Enum):
    no_tile_quantization = "no_tile_quantization"
    naive = "naive"
