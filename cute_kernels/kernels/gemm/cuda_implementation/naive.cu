#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include "../../../include/dtypes/all.h"
#include "../../../include/math.h"
#include "index.h"

template <typename scalar_t, bool is_a_transposed, bool is_b_transposed>
__global__ void _naive_gemm_cuda_kernel(
    const scalar_t *a, const scalar_t *b, scalar_t *c, const uint32 M, const uint32 K, const uint32 N) {
    const uint32 i = blockDim.x * blockIdx.x + threadIdx.x;
    const uint32 j = blockDim.y * blockIdx.y + threadIdx.y;

    if (i < M && j < N) {
        fp32 accumulator = 0;
        for (uint32 k = 0; k < K; k++) {
            uint64 a_index;
            if (is_a_tranposed) {
                a_index = get_matrix_index(k, i, M);
            } else {
                a_index = get_matrix_index(i, k, K);
            }

            uint64 b_index;
            if (is_b_tranposed) {
                b_index = get_matrix_index(j, k, K);
            } else {
                b_index = get_matrix_index(k, j, N);
            }

            accumulator += a[a_index] * b[b_index];
        }

        c[get_matrix_index(i, j, N)] = accumulator;
    }
}

void naive_gemm_cuda(const torch::Tensor &a,
                     const torch::Tensor &b,
                     torch::Tensor &c,
                     const bool &is_a_transposed,
                     const bool &is_b_transposed,
                     const uint32 &BLOCK_SIZE_M,
                     const uint32 &BLOCK_SIZE_N) {
    TORCH_CHECK(BLOCK_SIZE % WARP_SIZE == 0);

    uint32 M, K, N;
    if (is_a_transposed) {
        K = a.size(0);
        M = a.size(1);
    } else {
        K = a.size(-1);
        M = a.numel() / K;
    }

    if (is_b_tranposed) {
        N = b.size(0);
        TORCH_CHECK(b.size(1) == K);
    } else {
        N = b.size(1);
        TORCH_CHECK(b.size(0) == K);
    }

    AT_DISPATCH_CUSTOM_FLOAT_TYPES(
        a.scalar_type(), "naive_gemm_cuda_kernel", ([&] {
            dim3 NUM_BLOCKS = dim3(ceil_divide<uint32>(M, BLOCK_SIZE_M), ceil_divide<uint32>(N, BLOCK_SIZE_N), 1);
            dim3 BLOCK_SIZE = dim3(BLOCK_SIZE_M, BLOCK_SIZE_N, 1);

            _naive_gemm_cuda_kernel<scalar_t, is_a_transposed, is_b_transposed><<<NUM_BLOCKS, BLOCK_SIZE>>>(
                a.data_ptr<scalar_t>(), b.data_ptr<scalar_t>(), c.data_ptr<scalar_t>(), M, K, N);
        }));
}
