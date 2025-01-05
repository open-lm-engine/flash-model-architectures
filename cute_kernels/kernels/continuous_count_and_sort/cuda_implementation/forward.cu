#include <cooperative_groups.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include "../../../include/dtypes/all.h"
#include "../../../include/launch.h"
#include "../../../include/math.h"
#include "../../../include/threads.h"

#define MAX_ALLOWED_C 16384

namespace cg = cooperative_groups;

inline __device__ void _looped_atomic_add(uint32 *output_shared,
                                          uint32 *destination_output_shared,
                                          const uint32 &num_loops_C,
                                          const uint32 &C,
                                          const uint32 &local_thread_id) {
    for (int i = 0; i < num_loops_C; i++) {
        const int index = i * blockDim.x + local_thread_id;
        if (index < C) {
            atomicAdd(&destination_output_shared[index], output_shared[index]);
        }
    }
}

template <typename scalar_t>
__global__ void _continuous_count_and_sort_cuda_kernel(const scalar_t *x,
                                                       uint32 *count,
                                                       const uint64 num_elements,
                                                       const uint32 C) {
    const uint32 local_thread_id = get_local_thread_id();
    const uint32 num_loops_C = ceil_divide<uint32>(C, blockDim.x);

    extern __shared__ uint32 output_shared[];

    for (uint32 i = 0; i < num_loops_C; i++) {
        const uint32 index = i * blockDim.x + local_thread_id;
        if (index < C) {
            count[index] = 0;
            output_shared[index] = 0;
        }
    }

    __syncthreads();

    // count the number of occurances of each number in x
    const uint32 num_elements_per_block = ceil_divide<uint64>(num_elements, gridDim.x);

    const uint32 start = blockIdx.x * num_elements_per_block;
    uint64 end = start + num_elements_per_block;
    if (end > num_elements) {
        end = num_elements;
    }

    const int num_elements_in_current_block = end - start;

    if (num_elements_in_current_block > 0) {
        const uint32 num_loops = ceil_divide<uint32>(num_elements_in_current_block, blockDim.x);

        for (int i = 0; i < num_loops; i++) {
            const int index = start + i * blockDim.x + local_thread_id;
            if (index < end) {
                atomicAdd(&output_shared[x[index]], 1);
            }
        }

        __syncthreads();

        for (int i = 0; i < num_loops_C; i++) {
            const int index = i * blockDim.x + local_thread_id;
            if (index < C) {
                atomicAdd(&count[index], output_shared[index]);
            }
        }
    }
}

void continuous_count_and_sort_cuda(
    const torch::Tensor &x, torch::Tensor &count, const uint32 &sm_count, const uint32 &C, const uint32 &BLOCK_SIZE) {
    assert(BLOCK_SIZE % WARP_SIZE == 0);
    assert(C <= MAX_ALLOWED_C);

    const uint64 total_elements = x.numel();

    std::vector<ChunkedArray<uint32>> count_chunks = chunk_array<uint32>(count.data_ptr<uint32>(), total_elements);

    AT_DISPATCH_CUSTOM_INT_TYPES(x.scalar_type(), "continuous_count_and_sort_cuda_kernel", ([&] {
                                     cudaFuncSetAttribute(_continuous_count_cuda_kernel<scalar_t>,
                                                          cudaFuncAttributeMaxDynamicSharedMemorySize,
                                                          MAX_ALLOWED_C * sizeof(uint32));

                                     std::vector<ChunkedArray<scalar_t>> x_chunks =
                                         chunk_array<scalar_t>(x.data_ptr<scalar_t>(), total_elements);

                                     for (int i = 0; i < x_chunks.size(); i++) {
                                         ChunkedArray<scalar_t> x_chunk = x_chunks[i];
                                         ChunkedArray<uint32> count_chunk = count_chunks[i];

                                         const uint32 num_elements = x_chunk.num_elements;

                                         auto [NUM_BLOCKS, cluster_size] = get_num_blocks(
                                             num_elements, BLOCK_SIZE, sm_count, thread_block_cluster_size);

                                         _continuous_count_and_sort_cuda_kernel<scalar_t><<<NUM_BLOCKS, BLOCK_SIZE>>>(
                                             x_chunk.array, count_chunk.array, num_elements, C);
                                     }
                                 }));
}
