#include <cuda.h>
#include <cuda_runtime.h>

#include "../../../include/dtypes/all.h"

inline __device__ uint64
get_matrix_index(const uint32 &row, const uint32 &col, const uint32 &M, const uint32 &N, const bool row_major = true) {
    uint64 index;
    if (row_major) {
        index = row * N + col;
    } else {
        index = col * M + row;
    }

    return index;
}
