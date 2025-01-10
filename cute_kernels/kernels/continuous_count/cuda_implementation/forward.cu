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
__global__ void _continuous_count_cuda_kernel(const scalar_t *x,
                                              uint32 *output,
                                              const uint64 num_elements,
                                              const uint32 C) {
    const uint32 local_thread_id = get_local_thread_id();
    const uint32 num_loops_C = ceil_divide<uint32>(C, blockDim.x);

    extern __shared__ uint32 output_shared[];

    for (uint32 i = 0; i < num_loops_C; i++) {
        const uint32 index = i * blockDim.x + local_thread_id;
        if (index < C) {
            output_shared[index] = 0;
        }
    }

    __syncthreads();

    const uint32 virtual_num_blocks = ceil_divide<uint64>(num_elements, blockDim.x);

    // num blocks with full loops is either the
    uint32 num_blocks_with_full_loops;
    if (virtual_num_blocks == gridDim.x) {
        num_blocks_with_full_loops = gridDim.x;
    } else {
        const uint32 remainder = virtual_num_blocks % gridDim.x;
        if (remainder == 0) {
            num_blocks_with_full_loops = gridDim.x;
        } else {
            num_blocks_with_full_loops = remainder;
        }
    }

    uint32 num_loops = ceil_divide<uint32>(virtual_num_blocks, gridDim.x);
    if (blockIdx.x >= num_blocks_with_full_loops) {
        num_loops -= 1;
    }

    cg::cluster_group cluster = cg::this_cluster();
    const bool is_first_cluster_block = cluster.block_rank() == 0;

    for (int i = 0; i < num_loops; i++) {
        const uint32 start = (gridDim.x * i + blockIdx.x) * blockDim.x;

        uint64 end = start + blockDim.x;
        if (end >= num_elements) {
            end = num_elements;
        }

        if (end > start) {
            const uint32 index = start + local_thread_id;
            if (index < end) {
                atomicAdd(&output_shared[x[index]], 1);
            }
        }
    }

    __syncthreads();

    if (!is_first_cluster_block) {
        _looped_atomic_add(output_shared, cluster.map_shared_rank(output_shared, 0), num_loops_C, C, local_thread_id);
    }

    cluster.sync();

    // write the output to the global memory
    if (is_first_cluster_block) {
        _looped_atomic_add(output_shared, output, num_loops_C, C, local_thread_id);
    }
}

void continuous_count_cuda(const torch::Tensor &x,
                           torch::Tensor &output,
                           const uint32 &sm_count,
                           const uint32 &thread_block_cluster_size,
                           const uint32 &C,
                           const uint32 &BLOCK_SIZE) {
    assert(BLOCK_SIZE % WARP_SIZE == 0);
    assert(C <= MAX_ALLOWED_C);

    const uint64 total_elements = x.numel();
    const int max_num_blocks = get_max_thread_blocks(sm_count, thread_block_cluster_size);

    std::vector<ChunkedArray<uint32>> output_chunks = chunk_array<uint32>(output.data_ptr<uint32>(), total_elements);

    AT_DISPATCH_CUSTOM_INT_TYPES(x.scalar_type(), "continuous_count_cuda_kernel", ([&] {
                                     cudaFuncSetAttribute(_continuous_count_cuda_kernel<scalar_t>,
                                                          cudaFuncAttributeMaxDynamicSharedMemorySize,
                                                          MAX_ALLOWED_C * sizeof(uint32));

                                     std::vector<ChunkedArray<scalar_t>> x_chunks =
                                         chunk_array<scalar_t>(x.data_ptr<scalar_t>(), total_elements);

                                     for (int i = 0; i < x_chunks.size(); i++) {
                                         ChunkedArray<scalar_t> x_chunk = x_chunks[i];
                                         ChunkedArray<uint32> output_chunk = output_chunks[i];

                                         const uint64 num_elements = x_chunk.num_elements;

                                         auto [NUM_BLOCKS, cluster_size] = get_num_blocks(
                                             num_elements, BLOCK_SIZE, max_num_blocks, thread_block_cluster_size);

                                         // dynamically sized clusters need this stupid way of launching the kernel
                                         cudaLaunchConfig_t launch_config = {0};
                                         launch_config.blockDim = BLOCK_SIZE;
                                         launch_config.gridDim = NUM_BLOCKS;
                                         launch_config.dynamicSmemBytes = C * sizeof(uint32);

                                         cudaLaunchAttribute attributes[1];
                                         attributes[0].id = cudaLaunchAttributeClusterDimension;
                                         attributes[0].val.clusterDim.x = cluster_size;
                                         attributes[0].val.clusterDim.y = 1;
                                         attributes[0].val.clusterDim.z = 1;

                                         launch_config.attrs = attributes;
                                         launch_config.numAttrs = 1;

                                         cudaLaunchKernelEx(&launch_config,
                                                            _continuous_count_cuda_kernel<scalar_t>,
                                                            x_chunk.array,
                                                            output_chunk.array,
                                                            num_elements,
                                                            C);
                                     }
                                 }));
}
