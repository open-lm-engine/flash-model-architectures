#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include "../../../include/dtypes/all.h"
#include "../../../include/math.h"
#include "../../../include/threads.h"
#include "index.h"

template <typename scalar_t>
__global__ void _shared_memory_gemm_cuda_kernel(const scalar_t *a,
                                                const scalar_t *b,
                                                const scalar_t *c,
                                                scalar_t *output,
                                                const fp32 alpha,
                                                const fp32 beta,
                                                const uint32 M,
                                                const uint32 K,
                                                const uint32 N) {
    const uint32 i = get_thread_id_along_axis(blockDim.x, blockIdx.y, threadIdx.y);
    const uint32 j = get_thread_id_along_axis(blockDim.x, blockIdx.x, threadIdx.x);

    if (i < M && j < N) {
        fp32 accumulator = 0;
        uint32 k;

        for (k = 0; k < K; k += blockDim.x) {
            __shared__ scalar_t a_shared[blockDim.x][blockDim.x];
            __shared__ scalar_t b_shared[blockDim.x][blockDim.x];

            if (k < K) {
                a_shared[threadIdx.y][threadIdx.x] = a[get_matrix_index(i, k, M, K, false)];
                b_shared[threadIdx.y][threadIdx.x] = b[get_matrix_index(k, j, K, N, false)];
            }

            __syncthreads();

            const uint32 max_q = min(K - k, blockDim.x);
            for (uint32 q = 0; q < max_q; q++) {
                accumulator += a_shared[i][q] * b_shared[q][j];
            }

            __syncthreads();
        }

        accumulator *= alpha;
        const uint64 index = get_matrix_index(i, j, M, N, false);

        if (beta != 0) {
            accumulator += beta * c[index];
        }

        output[index] = accumulator;
    }
}

void shared_memory_gemm_cuda(const torch::Tensor &a,
                             const torch::Tensor &b,
                             std::optional<torch::Tensor> &c,
                             torch::Tensor &output,
                             const bool &is_a_transposed,
                             const bool &is_b_transposed,
                             const fp32 alpha,
                             const fp32 beta,
                             const uint32 &M,
                             const uint32 &K,
                             const uint32 &N,
                             const uint32 &BLOCK_SIZE) {
    TORCH_CHECK((BLOCK_SIZE * BLOCK_SIZE) % WARP_SIZE == 0);

    TORCH_CHECK(!is_a_transposed);
    TORCH_CHECK(!is_b_transposed);

    dim3 NUM_BLOCKS = dim3(ceil_divide<uint32>(N, BLOCK_SIZE), ceil_divide<uint32>(M, BLOCK_SIZE), 1);

    AT_DISPATCH_CUSTOM_FLOAT_TYPES(a.scalar_type(), "shared_memory_gemm_cuda_kernel", ([&] {
                                       _shared_memory_gemm_cuda_kernel<scalar_t><<<NUM_BLOCKS, BLOCK_SIZE>>>(
                                           a.data_ptr<scalar_t>(),
                                           b.data_ptr<scalar_t>(),
                                           c.has_value() ? c.value().data_ptr<scalar_t>() : nullptr,
                                           output.data_ptr<scalar_t>(),
                                           alpha,
                                           beta,
                                           M,
                                           K,
                                           N);
                                   }));
}
