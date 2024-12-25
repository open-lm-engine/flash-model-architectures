#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include "../../../include/device.h"
#include "../../../include/dtypes/all.h"
#include "../../../include/threads.h"

#define MAX_ALLOWED_C 16384

__global__ void _contiguous_count_cuda_kernel(const int32 *x,
                                              int32 *output,
                                              const uint64 num_elements,
                                              const uint32 C) {
    const uint64 local_thread_id = get_local_thread_id();
    const int num_loops_C = (C + blockDim.x - 1) / blockDim.x;

    extern __shared__ uint32 output_shared[];

    // initialize shared memory and output
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
    if (start == 8253) {
        printf("start: %d, blockIdx.x: %d, num_elements_per_block: %d\n", start, blockIdx.x, num_elements_per_block);
    }
    // printf("start: %d\n", start);
    int end = start + num_elements_per_block;
    // printf("start: %d, end: %d\n", start, end);
    if (end > num_elements) {
        end = num_elements;
    }

    const int num_elements_in_current_block = end - start;
    if (num_elements_in_current_block <= 0) {
        printf("start: %d, end: %d\n", start, end);
    }
    const int num_loops_B = (num_elements_in_current_block + blockDim.x - 1) / blockDim.x;
    // printf("num_loops_B: %d, num_elements_in_current_block: %d\n", num_loops_B, num_elements_in_current_block);

    for (int i = 0; i < num_loops_B; i++) {
        const int index = start + i * num_loops_B + local_thread_id;
        // printf("start: %d, num_loops_B: %d, local_thread_id: %d, index: %d\n", start, num_loops_B, local_thread_id,
        // index);
        if (index < end) {
            if (x[index] >= 0) {
                output[0] = 1;
                // atomicAdd(&output_shared[0], 1);
            }
            // atomicAdd(&output_shared[x[index]], 1);
        }
    }

    __syncthreads();

    // // write the output to the global memory
    // // clang-format off
    // #pragma unroll
    // // clang-format on
    // for (int i = 0; i < num_loops_C; i++) {
    //     const int index = i * blockDim.x + local_thread_id;
    //     if (index < C) {
    //         atomicAdd(&output[index], output_shared[index]);
    //     }
    // }
}

void contiguous_count_cuda(const torch::Tensor &x, const torch::Tensor &output, const int &C, const int &BLOCK_SIZE) {
    assert(BLOCK_SIZE % WARP_SIZE == 0);
    assert(C < MAX_ALLOWED_C);

    const uint64 num_elements = x.numel();

    // we use vector instructions of width 4
    int NUM_BLOCKS = (num_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;
    const int sm_count = get_sm_count();
    if (NUM_BLOCKS > sm_count) {
        NUM_BLOCKS = sm_count;
    }

    _contiguous_count_cuda_kernel<<<NUM_BLOCKS, BLOCK_SIZE, C * sizeof(int32)>>>(
        x.data_ptr<int32>(), output.data_ptr<int32>(), num_elements, C);
}
