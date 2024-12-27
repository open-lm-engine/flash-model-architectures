#include <cooperative_groups.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include "../../../include/dtypes/all.h"
#include "../../../include/threads.h"

#define MAX_ALLOWED_C 16384

namespace cg = cooperative_groups;

template <typename scalar_t>
__global__ void _contiguous_count_cuda_kernel(const scalar_t *x,
                                              uint32 *output,
                                              const uint64 num_elements,
                                              const uint32 C) {
    const int local_thread_id = get_local_thread_id();
    const int num_loops_C = (C + blockDim.x - 1) / blockDim.x;

    extern __shared__ uint32 output_shared[];

    // clang-format off
    #pragma unroll
    // clang-format on
    for (int i = 0; i < num_loops_C; i++) {
        const int index = i * blockDim.x + local_thread_id;
        if (index < C) {
            output_shared[index] = 0;
        }
    }

    __syncthreads();

    // count the number of occurances of each number in x
    const int num_elements_per_block = (num_elements + gridDim.x - 1) / gridDim.x;

    const int start = blockIdx.x * num_elements_per_block;
    int end = start + num_elements_per_block;
    if (end > num_elements) {
        end = num_elements;
    }

    const int num_elements_in_current_block = end - start;

    if (num_elements_in_current_block > 0) {
        const int num_loops = (num_elements_in_current_block + blockDim.x - 1) / blockDim.x;

        for (int i = 0; i < num_loops; i++) {
            const int index = start + i * blockDim.x + local_thread_id;
            if (index < end) {
                atomicAdd(&output_shared[x[index]], 1);
            }
        }

        __syncthreads();

        cg::cluster_group cluster = cg::this_cluster();
        const uint32 cluster_block_rank = cluster.block_rank();

        if (cluster_block_rank != 0) {
            uint32 *destination_output_shared = cluster.map_shared_rank(output_shared, 0);

            // clang-format off
            #pragma unroll
            // clang-format on
            for (int i = 0; i < num_loops_C; i++) {
                const int index = i * blockDim.x + local_thread_id;
                if (index < C) {
                    atomicAdd(&destination_output_shared[index], output_shared[index]);
                }
            }
        }

        cluster.sync();

        // write the output to the global memory
        if (cluster_block_rank == 0) {
            // clang-format off
            #pragma unroll
            // clang-format on
            for (int i = 0; i < num_loops_C; i++) {
                const int index = i * blockDim.x + local_thread_id;
                if (index < C) {
                    atomicAdd(&output[index], output_shared[index]);
                }
            }
        }
    }
}

void contiguous_count_cuda(const torch::Tensor &x,
                           torch::Tensor &output,
                           const int &sm_count,
                           const int &thread_block_cluster_size,
                           const int &C,
                           const int &BLOCK_SIZE) {
    assert(BLOCK_SIZE % WARP_SIZE == 0);
    assert(C < MAX_ALLOWED_C);

    const uint64 num_elements = x.numel();
    const int max_num_blocks = get_max_thread_blocks(sm_count, thread_block_cluster_size);

    auto [NUM_BLOCKS, cluster_size] =
        get_num_blocks(num_elements, BLOCK_SIZE, max_num_blocks, thread_block_cluster_size);

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

    AT_DISPATCH_CUSTOM_INT_TYPES(x.scalar_type(), "contiguous_count_cuda_kernel", ([&] {
                                     cudaLaunchKernelEx(&launch_config,
                                                        _contiguous_count_cuda_kernel<scalar_t>,
                                                        x.data_ptr<scalar_t>(),
                                                        output.data_ptr<uint32>(),
                                                        num_elements,
                                                        C);
                                 }));
}
