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

template <typename scalar_t>
__global__ void _continuous_count_and_sort_cuda_kernel(const scalar_t *x,
                                                       uint32 *count_output,
                                                       uint32 *sorted_output,
                                                       uint32 *argsort_output,
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

    for (uint32 i = get_global_thread_id(); i < num_elements; i += gridDim.x * blockDim.x) {
        atomicAdd(&output_shared[x[i]], 1);
    }

    __syncthreads();

    for (int i = 0; i < num_loops_C; i++) {
        const int index = i * blockDim.x + local_thread_id;
        if (index < C) {
            atomicAdd(&count_output[index], output_shared[index]);
        }
    }
}

void continuous_count_and_sort_cuda(const torch::Tensor &x,
                                    torch::Tensor &count_output,
                                    torch::Tensor &sorted_output,
                                    torch::Tensor &argsort_output,
                                    const uint32 &sm_count,
                                    const uint32 &C,
                                    const uint32 &BLOCK_SIZE) {
    TORCH_CHECK(x.is_cuda());
    TORCH_CHECK(count_output.is_cuda());
    TORCH_CHECK(sorted_output.is_cuda());
    TORCH_CHECK(argsort_output.is_cuda());

    TORCH_CHECK(BLOCK_SIZE % WARP_SIZE == 0);
    TORCH_CHECK(C <= MAX_ALLOWED_C);

    const uint64 num_elements = x.numel();
    assert(num_elements <= std::numeric_limits<uint32>::max());

    auto [NUM_BLOCKS, q] = get_num_blocks(num_elements, BLOCK_SIZE, sm_count, 1);

    AT_DISPATCH_CUSTOM_INT_TYPES(x.scalar_type(), "continuous_count_cuda_kernel", ([&] {
                                     cudaFuncSetAttribute(_continuous_count_and_sort_cuda_kernel<scalar_t>,
                                                          cudaFuncAttributeMaxDynamicSharedMemorySize,
                                                          MAX_ALLOWED_C * sizeof(uint32));

                                     _continuous_count_and_sort_cuda_kernel<scalar_t>
                                         <<<NUM_BLOCKS, BLOCK_SIZE, C * sizeof(uint32)>>>(
                                             x.data_ptr<scalar_t>(),
                                             count_output.data_ptr<uint32>(),
                                             sorted_output.data_ptr<uint32>(),
                                             argsort_output.data_ptr<uint32>(),
                                             num_elements,
                                             C);
                                 }));
}
