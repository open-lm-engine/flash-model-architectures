#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include "../../../include/dtypes/all.h"
#include "../../../include/math.h"
#include "../../../include/threads.h"
#include "index.h"
#include "naive.cuh"

template <typename scalar_t>
__global__ void _no_tile_quantization_cuda_kernel(const scalar_t *a,
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
    const uint32 thread_id = get_global_thread_id();
    const uint32 i = thread_id / N;
    const uint32 j = thread_id % N;

    _run_matmul<scalar_t>(a, b, c, output, is_a_transposed, is_b_transposed, alpha, beta, i, j, M, K, N);
}

void no_tile_quantization_cuda(const torch::Tensor &a,
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
    const uint32 BLOCK_SIZE = BLOCK_SIZE_M * BLOCK_SIZE_N;
    TORCH_CHECK(BLOCK_SIZE % WARP_SIZE == 0);

    const uint64 NUM_BLOCKS = ceil_divide<uint64>(M * N, BLOCK_SIZE);

    AT_DISPATCH_CUSTOM_FLOAT_TYPES(a.scalar_type(), "no_tile_quantization_cuda_kernel", ([&] {
                                       _no_tile_quantization_cuda_kernel<scalar_t><<<NUM_BLOCKS, BLOCK_SIZE>>>(
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
