#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include "../../../include/dtypes/all.h"
#include "../../../include/math.h"
#include "../../../include/threads.h"
#include "naive.cuh"

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
                                        const uint32 N) {
    const uint32 i = get_thread_id_along_axis(blockDim.y, blockIdx.y, threadIdx.y);
    const uint32 j = get_thread_id_along_axis(blockDim.x, blockIdx.x, threadIdx.x);

    _run_matmul<scalar_t>(a, b, c, output, is_a_transposed, is_b_transposed, alpha, beta, i, j, M, K, N);
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
