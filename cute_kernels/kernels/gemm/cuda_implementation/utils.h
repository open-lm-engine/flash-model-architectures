// **************************************************
// Copyright (c) 2025, Mayank Mishra
// **************************************************

#include <torch/extension.h>

#include "include/cute_kernels.h"

inline __host__ std::tuple<uint32, uint32, uint32> get_MNK(const torch::Tensor &A, const torch::Tensor &B) {
    uint32 M, K, N;
    if (is_A_transposed) {
        M = A.size(1);
        K = A.size(0);
    } else {
        M = A.size(0);
        K = A.size(1);
    }

    if (is_B_transposed) {
        N = B.size(0);
    } else {
        N = B.size(1);
    }

    return {M, N, K};
}
