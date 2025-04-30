#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include "include/cute_kernels.h"

namespace ck = cute_kernels;
namespace ck_mem = ck::memory;

using fp32 = ck::fp32;
using uint32 = ck::uint32;
using uint64 = ck::uint64;

template <typename scalar_t>
__global__ void add_scalar_cuda_kernel(const scalar_t *x, const fp32 y, scalar_t *output, const uint64 N) {
    constexpr uint32 N_per_thread = ck_mem::get_num_elements_for_vector_load_stores<scalar_t>();

    const uint32 thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32 N_vec = N / N_per_thread;

    if (thread_id < N_vec) {
        const scalar_t *x_vec = ck_mem::load_128_bits<scalar_t>(x, thread_id);
        scalar_t output_buffer[N_per_thread];

        for (uint32 i = 0; i < N_per_thread; i++) {
            output_buffer[i] = x_vec[i] + y;
        }

        ck_mem::store_128_bits<scalar_t>(output_buffer, output, thread_id);
    }

    const uint32 index = N_vec * N_per_thread + thread_id;
    if (index < N) {
        output[index] = x[index] + y;
    }
}

void add_scalar_cuda(const torch::Tensor &x, const fp32 &y, torch::Tensor &output, const uint32 &BLOCK_SIZE) {
    CHECK_CUDA_TENSOR(x);
    CHECK_CUDA_TENSOR(output);

    CHECK_VALID_THREAD_BLOCK(BLOCK_SIZE);

    const uint64 total_elements = x.numel();

    DISPATCH_FLOAT_KERNEL(x.scalar_type(), "add_scalar_cuda_kernel", scalar_t, ([&] {
                              std::vector<ck::ChunkedArray<scalar_t>> x_chunks =
                                  ck::chunk_array<scalar_t>(x.data_ptr<scalar_t>(), total_elements);
                              std::vector<ck::ChunkedArray<scalar_t>> output_chunks =
                                  ck::chunk_array<scalar_t>(output.data_ptr<scalar_t>(), total_elements);

                              const uint32 N_per_thread = ck_mem::get_num_elements_for_vector_load_stores<scalar_t>();
                              const uint32 N_per_block = BLOCK_SIZE * N_per_thread;

                              for (uint32 i = 0; i < x_chunks.size(); i++) {
                                  ck::ChunkedArray<scalar_t> x_chunk = x_chunks[i];
                                  ck::ChunkedArray<scalar_t> output_chunk = output_chunks[i];

                                  const uint64 N = x_chunk.num_elements;
                                  const uint32 NUM_BLOCKS = ck::ceil_divide<uint64>(N, N_per_block);

                                  add_scalar_cuda_kernel<scalar_t>
                                      <<<NUM_BLOCKS, BLOCK_SIZE>>>(x_chunk.array, y, output_chunk.array, N);
                              }
                          }));
}
