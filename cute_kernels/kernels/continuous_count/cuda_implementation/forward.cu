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

    uint32 init_value[] = {0, 0, 0, 0};

    for (uint32 i = global_thread_id; i < C4; i += total_threads) {
        ck_mem::store_128_bits<uint32>(init_value, output, i);
    }

    const uint32 index = (C4 << 2) + global_thread_id;
    if (index < C) {
        output[index] = 0;
    }
}

template <typename scalar_t>
inline __device__ void _update_local_count(const scalar_t *x,
                                           uint32 *shared_memory,
                                           const uint64 &N,
                                           const uint32 &global_thread_id,
                                           const uint32 &total_threads) {
    constexpr uint32 N_per_thread = ck_mem::get_num_elements_for_vector_load_stores<scalar_t>();
    const uint32 N_vec = N / N_per_thread;

    for (uint32 i = global_thread_id; i < N_vec; i += total_threads) {
        const scalar_t *x_vec = ck_mem::load_128_bits<scalar_t>(x, i);

        for (uint32 j = 0; j < N_per_thread; j++) {
            atomicAdd(&shared_memory[x_vec[j]], 1);
        }
    }

    const uint32 index = (N_vec * N_per_thread) + global_thread_id;
    if (index < N) {
        atomicAdd(&shared_memory[x[index]], 1);
    }
}

template <typename scalar_t, bool initialize_output>
__global__ void continuous_count_cuda_kernel(const scalar_t *x, uint32 *output, const uint64 N, const uint32 C) {
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

    _update_local_count<scalar_t>(x, shared_memory, N, global_thread_id, grid_size);

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
                           const uint32 &C,
                           const uint32 &THREAD_BLOCK_CLUSTER_SIZE,
                           const uint32 &BLOCK_SIZE) {
    CHECK_CUDA_TENSOR(x);
    CHECK_CUDA_TENSOR(output);

    CHECK_VALID_THREAD_BLOCK(BLOCK_SIZE);

    TORCH_CHECK(C <= MAX_ALLOWED_C);

    const uint64 total_elements = x.numel();

    const uint32 num_SMs = ck::get_num_SMs();
    const uint32 max_num_blocks = ck::get_max_thread_blocks(num_SMs, THREAD_BLOCK_CLUSTER_SIZE);

    std::vector<ck::ChunkedArray<uint32>> output_chunks =
        ck::chunk_array<uint32>(output.data_ptr<uint32>(), total_elements);

    DISPATCH_INT_KERNEL(x.scalar_type(), "continuous_count_cuda_kernel", scalar_t, ([&] {
                            cudaFuncSetAttribute(continuous_count_cuda_kernel<scalar_t, true>,
                                                 cudaFuncAttributeMaxDynamicSharedMemorySize,
                                                 MAX_ALLOWED_C * sizeof(uint32));
                            cudaFuncSetAttribute(continuous_count_cuda_kernel<scalar_t, false>,
                                                 cudaFuncAttributeMaxDynamicSharedMemorySize,
                                                 MAX_ALLOWED_C * sizeof(uint32));

                            std::vector<ck::ChunkedArray<scalar_t>> x_chunks =
                                ck::chunk_array<scalar_t>(x.data_ptr<scalar_t>(), total_elements);

                            for (int i = 0; i < x_chunks.size(); i++) {
                                ck::ChunkedArray<scalar_t> x_chunk = x_chunks[i];
                                ck::ChunkedArray<uint32> output_chunk = output_chunks[i];

                                const uint64 N = x_chunk.num_elements;

                                auto [NUM_BLOCKS, cluster_size] =
                                    ck::get_num_blocks(N, BLOCK_SIZE, max_num_blocks, THREAD_BLOCK_CLUSTER_SIZE);

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
                                                   (i == 0) ? continuous_count_cuda_kernel<scalar_t, true>
                                                            : continuous_count_cuda_kernel<scalar_t, false>,
                                                   x_chunk.array,
                                                   output_chunk.array,
                                                   N,
                                                   C);
                            }
                        }));
}
