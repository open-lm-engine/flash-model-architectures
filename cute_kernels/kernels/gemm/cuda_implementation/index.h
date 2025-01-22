#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

#include "../../../include/dtypes/all.h"

inline __device__ uint64 get_matrix_index(const uint32 &row, const uint32 &col, const uint32 &N) {
    return row * N + col;
}
