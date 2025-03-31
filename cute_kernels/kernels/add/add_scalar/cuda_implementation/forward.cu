#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include "include/cute_kernels.h"

namespace ck = cute_kernels;
namespace ck_mem = ck::memory;

using fp32 = ck::fp32;
using uint32 = ck::uint32;
using uint64 = ck::uint64;
using fp32_2 = ck::fp32_2;

template <typename scalar_t>
__global__ void _add_scalar_cuda_kernel(const scalar_t *x, const fp32 y, scalar_t *output, const uint64 num_elements) {
    using dtype = ck::DType<scalar_t>;
    constexpr uint32 num_elements_per_thread = ck_mem::Packed128<scalar_t>::size;

    const uint32 thread_id = ck::get_global_thread_id();
    const uint32 num_vector_elements = num_elements / num_elements_per_thread;

    if (thread_id < num_vector_elements) {
        // packed array allows loading using vector loads, its just a syntactic sugar
        const ck_mem::Packed128<const scalar_t> x_vec = ck_mem::Packed128Array<const scalar_t>(x)[thread_id];
        ck_mem::Packed128<scalar_t> output_buffer;

        // clang-format off
        #pragma unroll
        // clang-format on
        for (uint32 i = 0; i < 4; i++) {
            if constexpr (std::is_same_v<scalar_t, fp32>) {
                output_buffer[i] = x_vec[i] + y;
            } else {
                using T2 = typename dtype::nv_dtype2;

                const uint32 index = i << 1;

                fp32_2 x2 = dtype::upcast(dtype::make2(x_vec[index], x_vec[index + 1]));
                x2.x = x2.x + y;
                x2.y = x2.y + y;
                T2 output2 = dtype::downcast(x2);

                output_buffer[index] = output2.x;
                output_buffer[index + 1] = output2.y;
            }
        }

        ck_mem::Packed128Array<scalar_t> output_vec = ck_mem::Packed128Array<scalar_t>(output);
        output_vec[thread_id] = output_buffer;
    }

    const uint32 index = num_vector_elements * num_elements_per_thread + thread_id;
    if (index < num_elements) {
        output[index] = x[index] + y;
    }
}

void add_scalar_cuda(const torch::Tensor &x, const fp32 &y, torch::Tensor &output, const uint32 &BLOCK_SIZE) {
    CHECK_CUDA_TENSOR(x);
    CHECK_CUDA_TENSOR(output);

    CHECK_VALID_THREAD_BLOCK(BLOCK_SIZE);

    const uint64 total_elements = x.numel();

    AT_DISPATCH_CUSTOM_FLOAT_TYPES(x.scalar_type(), "add_scalar_cuda_kernel", ([&] {
                                       const uint32 num_elements_per_thread = 16 / sizeof(scalar_t);
                                       const uint32 num_elements_per_block = BLOCK_SIZE * num_elements_per_thread;

                                       std::vector<ck::ChunkedArray<scalar_t>> x_chunks =
                                           ck::chunk_array<scalar_t>(x.data_ptr<scalar_t>(), total_elements);
                                       std::vector<ck::ChunkedArray<scalar_t>> output_chunks =
                                           ck::chunk_array<scalar_t>(output.data_ptr<scalar_t>(), total_elements);

                                       for (int i = 0; i < x_chunks.size(); i++) {
                                           ck::ChunkedArray<scalar_t> x_chunk = x_chunks[i];
                                           ck::ChunkedArray<scalar_t> output_chunk = output_chunks[i];

                                           const uint64 num_elements = x_chunk.num_elements;
                                           const uint32 NUM_BLOCKS =
                                               ck::ceil_divide<uint64>(num_elements, num_elements_per_block);

                                           _add_scalar_cuda_kernel<scalar_t><<<NUM_BLOCKS, BLOCK_SIZE>>>(
                                               x_chunk.array, y, output_chunk.array, num_elements);
                                       }
                                   }));
}
