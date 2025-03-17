#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include "include/dtypes.h"
#include "include/math.h"
#include "include/shared_memory.h"
#include "include/threads.h"
#include "index.h"

namespace ck = cute_kernels;

using uint32 = ck::uint32;
using fp32 = ck::fp32;

template <typename scalar_t>
__global__ void _shared_memory_gemm_cuda_kernel(const scalar_t *A,
                                                const scalar_t *B,
                                                const scalar_t *C,
                                                scalar_t *output,
                                                const fp32 alpha,
                                                const fp32 beta,
                                                const uint32 M,
                                                const uint32 K,
                                                const uint32 N) {
    const uint32 i = ck::get_thread_id_along_axis(blockDim.x, blockIdx.y, threadIdx.y);
    const uint32 j = ck::get_thread_id_along_axis(blockDim.x, blockIdx.x, threadIdx.x);

    scalar_t *shared_memory = ck::get_dynamic_shared_memory<scalar_t>();

    scalar_t *A_shared = shared_memory;
    scalar_t *B_shared = &shared_memory[blockDim.x * blockDim.x];

    fp32 accumulator = 0;

    // clang-format off
    #pragma unroll 128
    // clang-format on
    for (uint32 k = 0; k < K; k += blockDim.x) {
        const uint32 index = get_matrix_index<uint32>(threadIdx.y, threadIdx.x, blockDim.x, blockDim.x, false);

        // instead of looping over k dimension, we use the threads in the block to load the data to shared memory
        uint32 k_offset = k + threadIdx.x;
        if (i < M && k_offset < K) {
            A_shared[index] = A[get_matrix_index<uint32>(i, k_offset, M, K, false)];
        }

        // instead of looping over k dimension, we use the threads in the block to load the data to shared memory
        k_offset = k + threadIdx.y;
        if (j < N && k_offset < K) {
            B_shared[index] = B[get_matrix_index<uint32>(k_offset, j, K, N, false)];
        }

        __syncthreads();

        if (i < M && j < N) {
            const uint32 max_q = min(K - k, blockDim.x);
            for (uint32 q = 0; q < max_q; q++) {
                accumulator += A_shared[get_matrix_index<uint32>(threadIdx.y, q, blockDim.x, blockDim.x, false)] *
                               B_shared[get_matrix_index<uint32>(q, threadIdx.x, blockDim.x, blockDim.x, false)];
            }
        }

        // needed for ensuring that shared memory buffers are not modified before the loop finishes for all threads
        __syncthreads();
    }

    if (i < M && j < N) {
        accumulator *= alpha;
        const uint32 index = get_matrix_index<uint32>(i, j, M, N, false);

        if (beta != 0) {
            accumulator += beta * C[index];
        }

        output[index] = accumulator;
    }
}

void shared_memory_gemm_cuda(const torch::Tensor &A,
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
                             const uint32 &BLOCK_SIZE) {
    TORCH_CHECK((BLOCK_SIZE * BLOCK_SIZE) % WARP_SIZE == 0);

    TORCH_CHECK(!is_A_transposed);
    TORCH_CHECK(!is_B_transposed);

    dim3 NUM_BLOCKS = dim3(ck::ceil_divide<uint32>(N, BLOCK_SIZE), ck::ceil_divide<uint32>(M, BLOCK_SIZE), 1);
    dim3 BLOCK_SIZE_dim = dim3(BLOCK_SIZE, BLOCK_SIZE, 1);

    AT_DISPATCH_CUSTOM_FLOAT_TYPES(
        A.scalar_type(), "shared_memory_gemm_cuda_kernel", ([&] {
            _shared_memory_gemm_cuda_kernel<scalar_t>
                <<<NUM_BLOCKS, BLOCK_SIZE_dim, 2 * BLOCK_SIZE * BLOCK_SIZE * sizeof(scalar_t)>>>(
                    A.data_ptr<scalar_t>(),
                    B.data_ptr<scalar_t>(),
                    C.has_value() ? C.value().data_ptr<scalar_t>() : nullptr,
                    output.data_ptr<scalar_t>(),
                    alpha,
                    beta,
                    M,
                    K,
                    N);
        }));
}
