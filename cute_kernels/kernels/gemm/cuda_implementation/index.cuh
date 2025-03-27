#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

#include "include/cute_kernels.h"

namespace ck = cute_kernels;

using uint32 = ck::uint32;

template <typename T, bool is_transposed>
inline __device__ T get_matrix_index(const uint32 &row, const uint32 &col, const uint32 &M, const uint32 &N) {
    T index;
    if constexpr (is_transposed) {
        index = col * M + row;
    } else {
        index = row * N + col;
    }

    return index;
}
