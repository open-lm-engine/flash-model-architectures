#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include "include/cute_kernels.h"
#include "index.cuh"

namespace ck = cute_kernels;

using uint32 = ck::uint32;
using fp32 = ck::fp32;

template <typename scalar_t, bool is_A_transposed, bool is_B_transposed>
__global__ void _naive_gemm_cuda_kernel(const scalar_t *A,
                                        const scalar_t *B,
                                        const scalar_t *C,
                                        scalar_t *output,
                                        const fp32 alpha,
                                        const fp32 beta,
                                        const uint32 M,
                                        const uint32 K,
                                        const uint32 N) {
    const uint32 i = blockIdx.y * blockDim.y + threadIdx.y;
    const uint32 j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < M && j < N) {
        fp32 accumulator = 0;

        // clang-format off
        #pragma unroll 128
        // clang-format on
        for (uint32 k = 0; k < K; k++) {
            const uint32 A_index = get_matrix_index<uint32, is_A_transposed>(i, k, M, K);
            const uint32 B_index = get_matrix_index<uint32, is_B_transposed>(k, j, K, N);

            accumulator += A[A_index] * B[B_index];
        }

        accumulator *= alpha;
        const uint32 index = get_matrix_index<uint32, false>(i, j, M, N);

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
    CHECK_CUDA_TENSOR(A);
    CHECK_CUDA_TENSOR(B);
    if (C.has_value()) {
        CHECK_CUDA_TENSOR(C.value());
    }
    CHECK_CUDA_TENSOR(output);

    CHECK_VALID_THREAD_BLOCK(BLOCK_SIZE_M * BLOCK_SIZE_N);

    dim3 NUM_BLOCKS = dim3(ck::ceil_divide<uint32>(N, BLOCK_SIZE_N), ck::ceil_divide<uint32>(M, BLOCK_SIZE_M), 1);
    dim3 BLOCK_SIZE = dim3(BLOCK_SIZE_N, BLOCK_SIZE_M, 1);

    DISPATCH_FLOAT_KERNEL(A.scalar_type(), "naive_gemm_cuda_kernel", scalar_t, ([&] {
                              if (is_A_transposed) {
                                  if (is_B_transposed) {
                                      _naive_gemm_cuda_kernel<scalar_t, true, true><<<NUM_BLOCKS, BLOCK_SIZE>>>(
                                          A.data_ptr<scalar_t>(),
                                          B.data_ptr<scalar_t>(),
                                          C.has_value() ? C.value().data_ptr<scalar_t>() : nullptr,
                                          output.data_ptr<scalar_t>(),
                                          alpha,
                                          beta,
                                          M,
                                          K,
                                          N);
                                  } else {
                                      _naive_gemm_cuda_kernel<scalar_t, true, false><<<NUM_BLOCKS, BLOCK_SIZE>>>(
                                          A.data_ptr<scalar_t>(),
                                          B.data_ptr<scalar_t>(),
                                          C.has_value() ? C.value().data_ptr<scalar_t>() : nullptr,
                                          output.data_ptr<scalar_t>(),
                                          alpha,
                                          beta,
                                          M,
                                          K,
                                          N);
                                  }
                              } else {
                                  if (is_B_transposed) {
                                      _naive_gemm_cuda_kernel<scalar_t, false, true><<<NUM_BLOCKS, BLOCK_SIZE>>>(
                                          A.data_ptr<scalar_t>(),
                                          B.data_ptr<scalar_t>(),
                                          C.has_value() ? C.value().data_ptr<scalar_t>() : nullptr,
                                          output.data_ptr<scalar_t>(),
                                          alpha,
                                          beta,
                                          M,
                                          K,
                                          N);
                                  } else {
                                      _naive_gemm_cuda_kernel<scalar_t, false, false><<<NUM_BLOCKS, BLOCK_SIZE>>>(
                                          A.data_ptr<scalar_t>(),
                                          B.data_ptr<scalar_t>(),
                                          C.has_value() ? C.value().data_ptr<scalar_t>() : nullptr,
                                          output.data_ptr<scalar_t>(),
                                          alpha,
                                          beta,
                                          M,
                                          K,
                                          N);
                                  }
                              }
                          }));
}
