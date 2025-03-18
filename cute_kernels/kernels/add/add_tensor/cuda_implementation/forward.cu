#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include "include/cute_kernels.h"

namespace ck = cute_kernels;

using fp32 = ck::fp32;
using fp32_2 = ck::fp32_2;
using fp32_4 = ck::fp32_4;

using uint32 = ck::uint32;
using uint64 = ck::uint64;

template <typename scalar_t>
__global__ void _add_tensor_cuda_kernel(const scalar_t *x,
                                        const scalar_t *y,
                                        scalar_t *output,
                                        const uint32 num_elements) {
    constexpr int num_elements_per_thread = 16 / sizeof(scalar_t);
    static_assert(num_elements_per_thread == 4 || num_elements_per_thread == 8);

    using dtype = ck::DType<scalar_t>;
    using T = typename dtype::nv_dtype;
    using T2 = typename dtype::nv_dtype2;

    const uint32 thread_id = ck::get_global_thread_id();
    const uint32 num_elements4 = num_elements / num_elements_per_thread;

    if (thread_id < num_elements4) {
        const fp32 *x_vec = (fp32 *)&((fp32_4 *)x)[thread_id];
        const fp32 *y_vec = (fp32 *)&((fp32_4 *)y)[thread_id];
        fp32 output_buffer[4];

        // clang-format off
        #pragma unroll
        // clang-format on
        for (int i = 0; i < 4; i++) {
            if constexpr (std::is_same_v<scalar_t, fp32>) {
                output_buffer[i] = x_vec[i] + y_vec[i];
            } else {
                T2 _x = dtype::reinterpret_32_bits_as_2x16(x_vec[i]);
                T2 _y = dtype::reinterpret_32_bits_as_2x16(y_vec[i]);

                _x = __hadd2(_x, _y);
                output_buffer[i] = dtype::reinterpret_2x16_as_32_bits(_x);
            }
        }

        ((fp32_4 *)output)[thread_id] = ck::DType<fp32>::make4(output_buffer);
    }

    const uint32 index = num_elements4 * num_elements_per_thread + thread_id;
    if (index < num_elements) {
        output[index] = x[index] + y[index];
    }
}

void add_tensor_cuda(const torch::Tensor &x, const torch::Tensor &y, torch::Tensor &output, const uint32 &BLOCK_SIZE) {
    CHECK_CUDA_TENSOR(x);
    CHECK_CUDA_TENSOR(y);
    CHECK_CUDA_TENSOR(output);

    CHECK_VALID_THREAD_BLOCK(BLOCK_SIZE);

    const uint64 total_elements = x.numel();

    AT_DISPATCH_CUSTOM_FLOAT_TYPES(x.scalar_type(), "add_tensor_cuda_kernel", ([&] {
                                       const uint32 num_elements_per_thread = 16 / sizeof(scalar_t);
                                       const uint32 num_elements_per_block = BLOCK_SIZE * num_elements_per_thread;

                                       std::vector<ck::ChunkedArray<scalar_t>> x_chunks =
                                           ck::chunk_array<scalar_t>(x.data_ptr<scalar_t>(), total_elements);
                                       std::vector<ck::ChunkedArray<scalar_t>> y_chunks =
                                           ck::chunk_array<scalar_t>(y.data_ptr<scalar_t>(), total_elements);
                                       std::vector<ck::ChunkedArray<scalar_t>> output_chunks =
                                           ck::chunk_array<scalar_t>(output.data_ptr<scalar_t>(), total_elements);

                                       for (int i = 0; i < x_chunks.size(); i++) {
                                           ck::ChunkedArray<scalar_t> x_chunk = x_chunks[i];
                                           ck::ChunkedArray<scalar_t> y_chunk = y_chunks[i];
                                           ck::ChunkedArray<scalar_t> output_chunk = output_chunks[i];

                                           const uint32 num_elements = x_chunk.num_elements;
                                           const uint32 NUM_BLOCKS =
                                               ck::ceil_divide<uint64>(num_elements, num_elements_per_block);

                                           _add_tensor_cuda_kernel<scalar_t><<<NUM_BLOCKS, BLOCK_SIZE>>>(
                                               x_chunk.array, y_chunk.array, output_chunk.array, num_elements);
                                       }
                                   }));
}
