#pragma once

#define WARP_SIZE 32
#define LOG_WARP_SIZE 5

#include <cuda.h>
#include <cuda_runtime.h>

#include "dtypes/all.h"
#include "math.h"

inline __device__ uint32 get_threads_per_block() { return blockDim.x * blockDim.y * blockDim.z; }

inline __device__ uint32 get_num_blocks() { return gridDim.x * gridDim.y * gridDim.z; }

inline __device__ uint32 get_block_id() {
    return gridDim.x * gridDim.y * blockIdx.z + gridDim.x * blockIdx.y + blockIdx.x;
}

inline __device__ uint32 get_local_thread_id() {
    return blockDim.x * blockDim.y * threadIdx.z + blockDim.x * threadIdx.y + threadIdx.x;
}

inline __device__ uint64 get_global_thread_id() {
    return get_threads_per_block() * get_block_id() + get_local_thread_id();
}

inline __host__ int get_max_thread_blocks(const int &sm_count, const int &thread_block_cluster_size) {
    int max_num_blocks = sm_count;
    if (max_num_blocks % thread_block_cluster_size != 0) {
        max_num_blocks = thread_block_cluster_size * (max_num_blocks / thread_block_cluster_size);
    }

    return max_num_blocks;
}

inline __host__ std::tuple<uint32, uint32, uint32> get_num_blocks(const uint32 &num_elements,
                                                                  const uint32 &BLOCK_SIZE,
                                                                  const uint32 &sm_count = 0,
                                                                  const uint32 &max_thread_block_cluster_size = 1) {
    uint32 virtual_NUM_BLOCKS = ceil_divide<uint32>(num_elements, BLOCK_SIZE);

    uint32 physical_NUM_BLOCKS = virtual_NUM_BLOCKS;
    if (sm_count != 0 && physical_NUM_BLOCKS > sm_count) {
        physical_NUM_BLOCKS = sm_count;
    }

    int thread_block_cluster_size = max_thread_block_cluster_size;
    if (thread_block_cluster_size > physical_NUM_BLOCKS) {
        thread_block_cluster_size = physical_NUM_BLOCKS;
    }

    physical_NUM_BLOCKS = thread_block_cluster_size * (physical_NUM_BLOCKS / thread_block_cluster_size);

    return std::make_tuple(virtual_NUM_BLOCKS, physical_NUM_BLOCKS, thread_block_cluster_size);
}
