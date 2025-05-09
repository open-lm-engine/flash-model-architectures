#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include "cute/tensor.hpp"
#include "include/cute_kernels.h"

namespace ck = cute_kernels;
using namespace cute;

using uint32 = ck::uint32;
using fp32 = ck::fp32;

template <typename scalar_t, bool is_A_transposed, bool is_B_transposed>
__global__ void _naive_gemm_cuda_kernel(const scalar_t *_A,
                                        const scalar_t *_B,
                                        const scalar_t *_C,
                                        scalar_t *_output,
                                        const fp32 alpha,
                                        const fp32 beta,
                                        const uint32 M,
                                        const uint32 K,
                                        const uint32 N) {
    const uint32 i = blockIdx.y * blockDim.y + threadIdx.y;
    const uint32 j = blockIdx.x * blockDim.x + threadIdx.x;

    Shape shape_A = is_A_transposed ? make_shape(K, M) : make_shape(M, K);
    Stride stride_A = make_stride(K, 1);
    Tensor A = make_tensor(make_gmem_ptr(_A), make_layout(shape_A, stride_A));

    Shape shape_B = is_B_transposed ? make_shape(N, K) : make_shape(K, N);
    Stride stride_B = make_stride(N, 1);
    Tensor B = make_tensor(make_gmem_ptr(_B), make_layout(shape_B, stride_B));

    Shape shape_C = make_shape(M, N);
    Stride stride_C = make_stride(N, 1);
    Tensor C = make_tensor(make_gmem_ptr(_C), make_layout(shape_C, stride_C));

    Tensor output = make_tensor(make_gmem_ptr(_output), make_layout(shape_C, stride_C));

    if (i < M && j < N) {
        fp32 accumulator = 0;

        // clang-format off
        #pragma unroll
        // clang-format on
        for (uint32 k = 0; k < K; k++) {
            const scalar_t a = is_A_transposed ? A(k, i) : A(i, k);
            const scalar_t b = is_B_transposed ? B(j, k) : B(k, j);
            accumulator += a * b;
        }

        accumulator *= alpha;

        if (beta != 0) {
            accumulator += beta * C(i, j);
        }

        output(i, j) = accumulator;
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
