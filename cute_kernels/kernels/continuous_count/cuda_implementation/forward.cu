#include <cooperative_groups.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include "include/cute_kernels.h"

#define MAX_ALLOWED_C 16384

namespace cg = cooperative_groups;
namespace ck = cute_kernels;
namespace ck_mem = ck::memory;

using int32 = ck::int32;
using int64 = ck::int64;
using uint32 = ck::uint32;
using uint64 = ck::uint64;

inline __device__ void _looped_atomic_add(uint32 *source, uint32 *destination, const uint32 &C) {
    uint32 index = threadIdx.x;
    while (index < C) {
        atomicAdd(&destination[index], source[index]);
        index += blockDim.x;
    }
}

inline __device__ void _initialize_global_output(uint32 *output,
                                                 const uint32 &C,
                                                 const uint32 &global_thread_id,
                                                 const uint32 &total_threads) {
    const uint32 C4 = C >> 2;

    ck_mem::Packed128Array<uint32> output_vec = ck_mem::Packed128Array<uint32>(output);
    ck_mem::Packed128<uint32> init_value = ck_mem::Packed128<uint32>::constant(0);

    for (uint32 i = global_thread_id; i < C4; i += total_threads) {
        output_vec[i] = init_value;
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
                                           const uint32 &global_thread_id,
                                           const uint32 &total_threads) {
    const ck_mem::Packed128Array<scalar_t> x_vec_array = ck_mem::Packed128Array<scalar_t>(x);

    constexpr uint32 num_elements_per_thread = x_vec_array.size;
    const uint32 num_vector_elements = num_elements / num_elements_per_thread;

    for (uint32 i = global_thread_id; i < num_vector_elements; i += total_threads) {
        const ck_mem::Packed128<scalar_t> x_vec = x_vec_array[i];

        for (uint32 j = 0; j < num_elements_per_thread; j++) {
            atomicAdd(&shared_memory[x_vec[j]], 1);
        }
    }

    const uint32 index = (num_vector_elements * num_elements_per_thread) + global_thread_id;
    if (index < num_elements) {
        atomicAdd(&shared_memory[x[index]], 1);
    }
}

template <typename scalar_t>
__global__ void _continuous_count_cuda_kernel(
    const scalar_t *x, uint32 *output, const uint64 num_elements, const uint32 C, const bool initialize_output) {
    const uint32 global_thread_id = blockIdx.x * blockDim.x + threadIdx.x;

    extern __shared__ uint32 shared_memory[];

    uint32 index = threadIdx.x;
    while (index < C) {
        shared_memory[index] = 0;
        index += blockDim.x;
    }

    const uint32 grid_size = gridDim.x * blockDim.x;

    if (initialize_output) {
        _initialize_global_output(output, C, global_thread_id, grid_size);
        cg::this_grid().sync();
    }

    _update_local_count<scalar_t>(x, shared_memory, num_elements, global_thread_id, grid_size);

    cg::cluster_group cluster = cg::this_cluster();
    const bool is_first_cluster_block = cluster.block_rank() == 0;

    __syncthreads();

    if (!is_first_cluster_block) {
        _looped_atomic_add(shared_memory, cluster.map_shared_rank(shared_memory, 0), C);
    }

    cluster.sync();

    // write the output to the global memory
    if (is_first_cluster_block) {
        _looped_atomic_add(shared_memory, output, C);
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
