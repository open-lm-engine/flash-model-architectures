#include <cooperative_groups.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include "include/cute_kernels.h"

#define MAX_ALLOWED_C 16384

namespace cg = cooperative_groups;
namespace ck = cute_kernels;

using uint32 = ck::uint32;
using uint32_4 = ck::uint32_4;
using int32 = ck::int32;

using uint64 = ck::uint64;
using uint64_2 = ck::uint64_2;
using int64 = ck::int64;

inline __device__ void _looped_atomic_add(
    uint32 *source, uint32 *destination, const uint32 &num_loops_C, const uint32 &C, const uint32 &local_thread_id) {
    for (int i = 0; i < num_loops_C; i++) {
        const uint32 index = i * blockDim.x + local_thread_id;
        if (index < C) {
            atomicAdd(&destination[index], source[index]);
        }
    }
}

inline __device__ void _initialize_global_output(uint32 *output, const uint32 &C, const uint32 &global_thread_id) {
    const uint32 C4 = C >> 2;
    for (uint32 i = global_thread_id; i < C4; i += gridDim.x * blockDim.x) {
        ((uint32_4 *)output)[i] = ck::DType<uint32>::make4(0, 0, 0, 0);
    }

    const uint32 index = (C4 << 2) + global_thread_id;
    if (index < C) {
        output[index] = 0;
    }
}

template <typename scalar_t>
inline __device__ void _update_local_count(const scalar_t *x,
                                           uint32 *shared_memory,
                                           const uint64 &num_elements,
                                           const uint32 &global_thread_id) {
    const uint32 num_elements_per_thread = 16 / sizeof(scalar_t);
    const uint32 num_elements4 = num_elements / num_elements_per_thread;

    for (uint32 i = global_thread_id; i < num_elements4; i += gridDim.x * blockDim.x) {
        if constexpr (std::is_same_v<scalar_t, uint32> || std::is_same_v<scalar_t, int32>) {
            uint32_4 _x = ((uint32_4 *)x)[i];
            atomicAdd(&shared_memory[_x.x], 1);
            atomicAdd(&shared_memory[_x.y], 1);
            atomicAdd(&shared_memory[_x.z], 1);
            atomicAdd(&shared_memory[_x.w], 1);
        } else if constexpr (std::is_same_v<scalar_t, uint64> || std::is_same_v<scalar_t, int64>) {
            uint64_2 _x = ((uint64_2 *)x)[i];
            atomicAdd(&shared_memory[_x.x], 1);
            atomicAdd(&shared_memory[_x.y], 1);
        }
    }

    const uint32 index = (num_elements4 * num_elements_per_thread) + global_thread_id;
    if (index < num_elements) {
        atomicAdd(&shared_memory[x[index]], 1);
    }
}

template <typename scalar_t>
__global__ void _continuous_count_cuda_kernel(
    const scalar_t *x, uint32 *output, const uint64 num_elements, const uint32 C, const bool initialize_output) {
    const uint32 local_thread_id = ck::get_local_thread_id();
    const uint32 global_thread_id = ck::get_global_thread_id();
    const uint32 num_loops_C = ck::ceil_divide<uint32>(C, blockDim.x);

    extern __shared__ uint32 shared_memory[];

    for (uint32 i = 0; i < num_loops_C; i++) {
        const uint32 index = i * blockDim.x + local_thread_id;
        if (index < C) {
            shared_memory[index] = 0;
        }
    }

    if (initialize_output) {
        _initialize_global_output(output, C, global_thread_id);
        cg::this_grid().sync();
    }

    _update_local_count<scalar_t>(x, shared_memory, num_elements, global_thread_id);

    cg::cluster_group cluster = cg::this_cluster();
    const bool is_first_cluster_block = cluster.block_rank() == 0;

    __syncthreads();

    if (!is_first_cluster_block) {
        _looped_atomic_add(shared_memory, cluster.map_shared_rank(shared_memory, 0), num_loops_C, C, local_thread_id);
    }

    cluster.sync();

    // write the output to the global memory
    if (is_first_cluster_block) {
        _looped_atomic_add(shared_memory, output, num_loops_C, C, local_thread_id);
    }
}

void continuous_count_cuda(const torch::Tensor &x,
                           torch::Tensor &output,
                           const uint32 &sm_count,
                           const uint32 &thread_block_cluster_size,
                           const uint32 &C,
                           const uint32 &BLOCK_SIZE) {
    CHECK_CUDA_TENSOR(x);
    CHECK_CUDA_TENSOR(output);

    CHECK_VALID_THREAD_BLOCK(BLOCK_SIZE);

    TORCH_CHECK(C <= MAX_ALLOWED_C);

    const uint64 total_elements = x.numel();
    const int max_num_blocks = ck::get_max_thread_blocks(sm_count, thread_block_cluster_size);

    std::vector<ck::ChunkedArray<uint32>> output_chunks =
        ck::chunk_array<uint32>(output.data_ptr<uint32>(), total_elements);

    AT_DISPATCH_CUSTOM_INT_TYPES(x.scalar_type(), "continuous_count_cuda_kernel", ([&] {
                                     cudaFuncSetAttribute(_continuous_count_cuda_kernel<scalar_t>,
                                                          cudaFuncAttributeMaxDynamicSharedMemorySize,
                                                          MAX_ALLOWED_C * sizeof(uint32));

                                     std::vector<ck::ChunkedArray<scalar_t>> x_chunks =
                                         ck::chunk_array<scalar_t>(x.data_ptr<scalar_t>(), total_elements);

                                     for (int i = 0; i < x_chunks.size(); i++) {
                                         ck::ChunkedArray<scalar_t> x_chunk = x_chunks[i];
                                         ck::ChunkedArray<uint32> output_chunk = output_chunks[i];

                                         const uint64 num_elements = x_chunk.num_elements;

                                         auto [NUM_BLOCKS, cluster_size] = ck::get_num_blocks(
                                             num_elements, BLOCK_SIZE, max_num_blocks, thread_block_cluster_size);

                                         // dynamically sized clusters need this stupid way of launching the kernel
                                         cudaLaunchConfig_t launch_config = {0};
                                         launch_config.blockDim = BLOCK_SIZE;
                                         launch_config.gridDim = NUM_BLOCKS;
                                         launch_config.dynamicSmemBytes = C * sizeof(uint32);

                                         cudaLaunchAttribute attributes[2];

                                         attributes[0].id = cudaLaunchAttributeClusterDimension;
                                         attributes[0].val.clusterDim.x = cluster_size;
                                         attributes[0].val.clusterDim.y = 1;
                                         attributes[0].val.clusterDim.z = 1;

                                         attributes[1].id = cudaLaunchAttributeCooperative;
                                         attributes[1].val.cooperative = 1;

                                         launch_config.attrs = attributes;
                                         launch_config.numAttrs = 2;

                                         cudaLaunchKernelEx(&launch_config,
                                                            _continuous_count_cuda_kernel<scalar_t>,
                                                            x_chunk.array,
                                                            output_chunk.array,
                                                            num_elements,
                                                            C,
                                                            i == 0);
                                     }
                                 }));
}
