#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include "include/dtypes/all.h"
#include "include/math.h"
#include "include/threads.h"
#include "index.h"

template <typename scalar_t>
__global__ void _naive_gemm_cuda_kernel(const scalar_t *A,
                                        const scalar_t *B,
                                        const scalar_t *C,
                                        scalar_t *output,
                                        const bool is_A_transposed,
                                        const bool is_B_transposed,
                                        const fp32 alpha,
                                        const fp32 beta,
                                        const uint32 M,
                                        const uint32 K,
                                        const uint32 N) {
    const uint32 i = get_thread_id_along_axis(blockDim.y, blockIdx.y, threadIdx.y);
    const uint32 j = get_thread_id_along_axis(blockDim.x, blockIdx.x, threadIdx.x);

    if (i < M && j < N) {
        fp32 accumulator = 0;

        // clang-format off
        #pragma unroll 128
        // clang-format on
        for (uint32 k = 0; k < K; k++) {
            const uint32 A_index = get_matrix_index<uint32>(i, k, M, K, is_A_transposed);
            const uint32 B_index = get_matrix_index<uint32>(k, j, K, N, is_B_transposed);

            accumulator += A[A_index] * B[B_index];
        }

        accumulator *= alpha;
        const uint32 index = get_matrix_index<uint32>(i, j, M, N, false);

        if (beta != 0) {
            accumulator += beta * C[index];
        }

        output[index] = accumulator;
    }
}

void naive_gemm_cuda(const torch::Tensor &A,
                     const torch::Tensor &B,
                     std::optional<torch::Tensor> &C,
                     torch::Tensor &output,
                     const bool &is_A_transposed,
                     const bool &is_B_transposed,
                     const fp32 &alpha,
                     const fp32 &beta,
                     const uint32 &M,
                     const uint32 &K,
                     const uint32 &N,
                     const uint32 &BLOCK_SIZE_M,
                     const uint32 &BLOCK_SIZE_N) {
    TORCH_CHECK((BLOCK_SIZE_M * BLOCK_SIZE_N) % WARP_SIZE == 0);

    dim3 NUM_BLOCKS = dim3(ceil_divide<uint32>(N, BLOCK_SIZE_N), ceil_divide<uint32>(M, BLOCK_SIZE_M), 1);
    dim3 BLOCK_SIZE = dim3(BLOCK_SIZE_N, BLOCK_SIZE_M, 1);

    AT_DISPATCH_CUSTOM_FLOAT_TYPES(A.scalar_type(), "naive_gemm_cuda_kernel", ([&] {
                                       _naive_gemm_cuda_kernel<scalar_t><<<NUM_BLOCKS, BLOCK_SIZE>>>(
                                           A.data_ptr<scalar_t>(),
                                           B.data_ptr<scalar_t>(),
                                           C.has_value() ? C.value().data_ptr<scalar_t>() : nullptr,
                                           output.data_ptr<scalar_t>(),
                                           is_A_transposed,
                                           is_B_transposed,
                                           alpha,
                                           beta,
                                           M,
                                           K,
                                           N);
                                   }));
}
