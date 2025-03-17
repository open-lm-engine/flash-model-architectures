#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

#include "include/dtypes.h"

namespace ck = cute_kernels;

using uint32 = ck::uint32;

template <typename T>
inline __device__ T
get_matrix_index(const uint32 &row, const uint32 &col, const uint32 &M, const uint32 &N, const bool &is_transposed) {
    T index;
    if (is_transposed) {
        index = col * M + row;
    } else {
        index = row * N + col;
    }

    return index;
}
