#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include "../../../include/dtypes/all.h"
#include "../../../include/math.h"
#include "index.h"

template <typename scalar_t>
__global__ void _naive_gemm_cuda_kernel(
    const scalar_t *a, const scalar_t *b, scalar_t *c, const uint32 M, const uint32 K, const uint32 N) {
    const uint32 col = blockDim.x * blockIdx.x + threadIdx.x;
    const uint32 row = blockDim.y * blockIdx.y + threadIdx.y;

    if (row < M && col < N) {
        fp32 accumulator = 0;
        for (uint32 k = 0; k < K; k++) {
            accumulator += a[get_matrix_index(row, k, M, K)] + b[get_matrix_index(k, col, K, N)];
        }

        c[get_matrix_index(row, col, M, N)] = accumulator;
    }
}

void naive_gemm_cuda(const torch::Tensor &a,
                     const torch::Tensor &b,
                     torch::Tensor &c,
                     const uint32 &BLOCK_SIZE_M,
                     const uint32 &BLOCK_SIZE_N) {
    assert(BLOCK_SIZE % WARP_SIZE == 0);

    const uint32 K = a.size(-1);
    const uint32 M = a.numel() / K;
    const uint32 N = b.size(1);

    AT_DISPATCH_CUSTOM_INT_TYPES(
        a.scalar_type(), "naive_gemm_cuda_kernel", ([&] {
            dim3 NUM_BLOCKS = dim3(ceil_divide<uint32>(M, BLOCK_SIZE_M), ceil_divide<uint32>(N, BLOCK_SIZE_N), 1);
            dim3 BLOCK_SIZE = dim3(BLOCK_SIZE_M, BLOCK_SIZE_N, 1);

            _naive_gemm_cuda_kernel<scalar_t><<<NUM_BLOCKS, BLOCK_SIZE>>>(
                a.data_ptr<scalar_t>(), b.data_ptr<scalar_t>(), c.data_ptr<scalar_t>(), M, K, N);
        }));
}
