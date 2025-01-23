#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include "../../../include/dtypes/all.h"
#include "../../../include/math.h"
#include "../../../include/threads.h"
#include "index.h"

template <typename scalar_t>
__global__ void _naive_gemm_cuda_kernel(const scalar_t *a,
                                        const scalar_t *b,
                                        const scalar_t *c,
                                        scalar_t *output,
                                        const bool is_a_transposed,
                                        const bool is_b_transposed,
                                        const fp32 alpha,
                                        const fp32 beta,
                                        const uint32 M,
                                        const uint32 K,
                                        const uint32 N,
                                        const uint32 BLOCK_SIZE_K) {
    const uint32 block_start_row = blockDim.y * blockIdx.y;
    const uint32 block_start_col = blockDim.x * blockIdx.x;

    extern __shared__ scalar_t shared_memory[];
    scalar_t *a_shared = shared_memory;
    scalar_t *b_shared = &shared_memory[blockDim.x * blockDim.y];

    const uint32 i = block_start_row + threadIdx.y;
    const uint32 j = block_start_col + threadIdx.x;

    if (i < M && j < N) {
        fp32 accumulator = 0;
        for (uint32 k = 0; k < K; k += BLOCK_SIZE_K) {
            const uint64 a_index = get_matrix_index(i, k, M, K, is_a_transposed);
            const uint64 b_index = get_matrix_index(k, j, K, N, is_b_transposed);

            accumulator += a[a_index] * b[b_index];
        }

        accumulator *= alpha;
        const uint64 index = get_matrix_index(i, j, M, N, false);

        if (beta != 0) {
            accumulator += beta * c[index];
        }

        output[index] = accumulator;
    }
}

void naive_gemm_cuda(const torch::Tensor &a,
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
                     const uint32 &BLOCK_SIZE_M,
                     const uint32 &BLOCK_SIZE_N) {
    TORCH_CHECK((BLOCK_SIZE_M * BLOCK_SIZE_N) % WARP_SIZE == 0);

    dim3 NUM_BLOCKS = dim3(ceil_divide<uint32>(N, BLOCK_SIZE_N), ceil_divide<uint32>(M, BLOCK_SIZE_M), 1);
    dim3 BLOCK_SIZE = dim3(BLOCK_SIZE_N, BLOCK_SIZE_M, 1);

    AT_DISPATCH_CUSTOM_FLOAT_TYPES(a.scalar_type(), "naive_gemm_cuda_kernel", ([&] {
                                       _naive_gemm_cuda_kernel<scalar_t><<<NUM_BLOCKS, BLOCK_SIZE>>>(
                                           a.data_ptr<scalar_t>(),
                                           b.data_ptr<scalar_t>(),
                                           c.has_value() ? c.value().data_ptr<scalar_t>() : nullptr,
                                           output.data_ptr<scalar_t>(),
                                           is_a_transposed,
                                           is_b_transposed,
                                           alpha,
                                           beta,
                                           M,
                                           K,
                                           N);
                                   }));
}
