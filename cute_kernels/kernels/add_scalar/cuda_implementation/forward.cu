#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include "include/cute_kernels.h"

namespace ck = cute_kernels;
namespace ck_mem = ck::memory;

using fp32 = ck::fp32;
using uint32 = ck::uint32;
using uint64 = ck::uint64;

template <typename scalar_t, bool has_trailing_elements>
__global__ void add_scalar_cuda_kernel(const scalar_t *x, const fp32 y, scalar_t *output, const uint64 N) {
    constexpr uint32 num_elements_per_thread = ck_mem::get_num_elements_for_vector_load_stores<scalar_t>();

    const uint32 thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32 N_vector = N / num_elements_per_thread;

    if (thread_id < N_vector) {
        const scalar_t *x_vec = ck_mem::vectorized_load<const scalar_t>(x, thread_id);
        scalar_t output_buffer[num_elements_per_thread];

        for (uint32 i = 0; i < num_elements_per_thread; i++) {
            output_buffer[i] = x_vec[i] + y;
        }

        ck_mem::vectorized_store<scalar_t>(output_buffer, output, thread_id);
    }

    if (has_trailing_elements) {
        const uint32 warp_id = thread_id >> LOG_WARP_SIZE;
        const uint32 num_warps = (gridDim.x * blockDim.x) >> LOG_WARP_SIZE;
        const bool is_last_warp = warp_id == num_warps - 1;

        if (is_last_warp) {
            const uint32 index = N_vector * num_elements_per_thread + (threadIdx.x % WARP_SIZE);
            if (index < N) {
                output[index] = x[index] + y;
            }
        }
    }
}

void add_scalar_cuda(const torch::Tensor &x, const fp32 &y, torch::Tensor &output, const uint32 &BLOCK_SIZE) {
    CHECK_CUDA_TENSOR(x);
    CHECK_CUDA_TENSOR(output);

    CHECK_VALID_THREAD_BLOCK(BLOCK_SIZE);

    const uint64 total_elements = x.numel();

    AT_DISPATCH_CUSTOM_FLOAT_TYPES(
        x.scalar_type(), "add_scalar_cuda_kernel", ([&] {
            std::vector<ck::ChunkedArray<scalar_t>> x_chunks =
                ck::chunk_array<scalar_t>(x.data_ptr<scalar_t>(), total_elements);
            std::vector<ck::ChunkedArray<scalar_t>> output_chunks =
                ck::chunk_array<scalar_t>(output.data_ptr<scalar_t>(), total_elements);

            for (int i = 0; i < x_chunks.size(); i++) {
                ck::ChunkedArray<scalar_t> x_chunk = x_chunks[i];
                ck::ChunkedArray<scalar_t> output_chunk = output_chunks[i];

                const uint64 num_elements = x_chunk.num_elements;

                constexpr uint32 bits = 32;
                const uint32 num_elements_per_thread =
                    ck_mem::get_num_elements_for_vector_load_stores<scalar_t, bits>();
                const uint32 num_elements_per_block = BLOCK_SIZE * num_elements_per_thread;

                const bool has_trailing_elements =
                    (i == x_chunks.size() - 1) && (num_elements % num_elements_per_thread != 0);

                if (has_trailing_elements) {
                    const uint32 num_elements_per_warp = num_elements_per_thread << LOG_WARP_SIZE;
                    const uint32 num_warps_per_block = BLOCK_SIZE >> LOG_WARP_SIZE;
                    // 1 extra warp to avoid thread divergence
                    const uint32 NUM_WARPS = ck::ceil_divide<uint64>(num_elements, num_elements_per_warp) + 1;
                    const uint32 NUM_BLOCKS = ck::ceil_divide<uint64>(NUM_WARPS, num_warps_per_block);

                    add_scalar_cuda_kernel<scalar_t, true, bits>
                        <<<NUM_BLOCKS, BLOCK_SIZE>>>(x_chunk.array, y, output_chunk.array, num_elements);
                } else {
                    const uint32 NUM_BLOCKS = ck::ceil_divide<uint64>(num_elements, num_elements_per_block);

                    add_scalar_cuda_kernel<scalar_t, false, bits>
                        <<<NUM_BLOCKS, BLOCK_SIZE>>>(x_chunk.array, y, output_chunk.array, num_elements);
                }
            }
        }));
}
