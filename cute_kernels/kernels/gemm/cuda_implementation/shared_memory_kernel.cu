#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include "cute/tensor.hpp"
#include "include/cute_kernels.h"

namespace ck = cute_kernels;
namespace ck_mem = cute_kernels::memory;
using namespace cute;

using uint32 = ck::uint32;
using fp32 = ck::fp32;

template <typename scalar_t>
__global__ void _shared_memory_gemm_cuda_kernel(const scalar_t *_A,
                                                const scalar_t *_B,
                                                const scalar_t *_C,
                                                scalar_t *_output,
                                                const fp32 alpha,
                                                const fp32 beta,
                                                const uint32 M,
                                                const uint32 K,
                                                const uint32 N) {
    const uint32 i = blockIdx.y * blockDim.x + threadIdx.y;
    const uint32 j = blockIdx.x * blockDim.x + threadIdx.x;

    scalar_t *shared_memory = ck_mem::get_dynamic_shared_memory<scalar_t>();

    scalar_t *_A_shared = shared_memory;
    scalar_t *_B_shared = &shared_memory[blockDim.x * blockDim.x];

    Layout layout_A_shared = make_layout(make_shape(blockDim.x, blockDim.x), make_stride(blockDim.x, 1));
    Tensor A_shared = make_tensor(make_smem_ptr(_A_shared), layout_A_shared);
    Tensor B_shared = make_tensor(make_smem_ptr(_B_shared), layout_A_shared);

    Layout layout_A = make_layout(make_shape(M, K), make_stride(K, 1));
    Tensor A = make_tensor(make_gmem_ptr(_A), layout_A);

    Layout layout_B = make_layout(make_shape(K, N), make_stride(N, 1));
    Tensor B = make_tensor(make_gmem_ptr(_B), layout_B);

    Layout layout_C = make_layout(make_shape(M, N), make_stride(N, 1));
    Tensor C = make_tensor(make_gmem_ptr(_C), layout_C);

    Tensor output = make_tensor(make_gmem_ptr(_output), layout_C);

    fp32 accumulator = 0;

    // clang-format off
    #pragma unroll
    // clang-format on
    for (uint32 k = 0; k < K; k += blockDim.x) {
        // instead of looping over k dimension, we use the threads in the block to load the data to shared memory
        uint32 k_offset = k + threadIdx.x;
        if (i < M && k_offset < K) {
            A_shared(threadIdx.y, threadIdx.x) = A(i, k_offset);
        }

        // instead of looping over k dimension, we use the threads in the block to load the data to shared memory
        k_offset = k + threadIdx.y;
        if (j < N && k_offset < K) {
            B_shared(threadIdx.y, threadIdx.x) = B(k_offset, j);
        }

        __syncthreads();

        if (i < M && j < N) {
            const uint32 max_q = min(K - k, blockDim.x);
            for (uint32 q = 0; q < max_q; q++) {
                accumulator += A_shared(threadIdx.y, q) * B_shared(q, threadIdx.x);
            }
        }

        // needed for ensuring that shared memory buffers are not modified before the loop finishes for all threads
        __syncthreads();
    }

    if (i < M && j < N) {
        accumulator *= alpha;

        if (beta != 0) {
            accumulator += beta * C(i, j);
        }

        output(i, j) = accumulator;
    }
}

void shared_memory_gemm_cuda(const torch::Tensor &A,
                             const torch::Tensor &B,
                             std::optional<torch::Tensor> &_C,
                             torch::Tensor &output,
                             const bool &is_A_transposed,
                             const bool &is_B_transposed,
                             const fp32 &alpha,
                             const fp32 &beta,
                             const uint32 &M,
                             const uint32 &K,
                             const uint32 &N,
                             const uint32 &BLOCK_SIZE) {
    CHECK_CUDA_TENSOR(A);
    CHECK_CUDA_TENSOR(B);
    if (_C.has_value()) {
        CHECK_CUDA_TENSOR(_C.value());
    }
    CHECK_CUDA_TENSOR(output);

    CHECK_VALID_THREAD_BLOCK(BLOCK_SIZE);

    TORCH_CHECK(!is_A_transposed);
    TORCH_CHECK(!is_B_transposed);

    dim3 NUM_BLOCKS = dim3(ck::ceil_divide<uint32>(N, BLOCK_SIZE), ck::ceil_divide<uint32>(M, BLOCK_SIZE), 1);
    dim3 BLOCK_SIZE_dim = dim3(BLOCK_SIZE, BLOCK_SIZE, 1);

    DISPATCH_FLOAT_KERNEL(A.scalar_type(), "shared_memory_gemm_cuda_kernel", scalar_t, ([&] {
                              _shared_memory_gemm_cuda_kernel<scalar_t>
                                  <<<NUM_BLOCKS, BLOCK_SIZE_dim, 2 * BLOCK_SIZE * BLOCK_SIZE * sizeof(scalar_t)>>>(
                                      A.data_ptr<scalar_t>(),
                                      B.data_ptr<scalar_t>(),
                                      _C.has_value() ? _C.value().data_ptr<scalar_t>() : nullptr,
                                      output.data_ptr<scalar_t>(),
                                      alpha,
                                      beta,
                                      M,
                                      K,
                                      N);
                          }));
}
