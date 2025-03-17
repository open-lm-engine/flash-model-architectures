#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include "include/dtypes/dtypes.h"
#include "include/launch.h"
#include "include/math.h"
#include "include/threads.h"

using fp32 = cute_kernels::fp32;
using fp32_2 = cute_kernels::fp32_2;
using fp32_4 = cute_kernels::fp32_4;

using uint32 = cute_kernels::uint32;
using uint64 = cute_kernels::uint64;

template <typename scalar_t>
__global__ void _add_scalar_cuda_kernel(const scalar_t *x, const fp32 y, scalar_t *output, const uint64 num_elements) {
    constexpr int num_elements_per_thread = 16 / sizeof(scalar_t);
    static_assert(num_elements_per_thread == 4 || num_elements_per_thread == 8);

    using dtype = cute_kernels::DType<scalar_t>;
    using T = typename dtype::nv_dtype;
    using T2 = typename dtype::nv_dtype2;

    const uint32 thread_id = cute_kernels::get_global_thread_id();
    const uint32 num_elements4 = num_elements / num_elements_per_thread;

    if (thread_id < num_elements4) {
        const fp32 *x_vec = (fp32 *)&((fp32_4 *)x)[thread_id];
        fp32 output_buffer[4];

        // clang-format off
        #pragma unroll
        // clang-format on
        for (int i = 0; i < 4; i++) {
            if constexpr (std::is_same_v<scalar_t, fp32>) {
                output_buffer[i] = x_vec[i] + y;
            } else {
                fp32_2 _x_upcast = dtype::upcast(dtype::reinterpret_32_bits_as_2x16(x_vec[i]));
                _x_upcast = cute_kernels::DType<fp32>::make2(_x_upcast.x + y, _x_upcast.y + y);
                output_buffer[i] = dtype::reinterpret_2x16_as_32_bits(dtype::downcast(_x_upcast));
            }
        }

        ((fp32_4 *)output)[thread_id] = cute_kernels::DType<fp32>::make4(output_buffer);
    }

    const uint32 index = num_elements4 * num_elements_per_thread + thread_id;
    if (index < num_elements) {
        output[index] = x[index] + y;
    }
}

void add_scalar_cuda(const torch::Tensor &x, const float &y, torch::Tensor &output, const uint32 &BLOCK_SIZE) {
    TORCH_CHECK(x.is_cuda());
    TORCH_CHECK(output.is_cuda());

    TORCH_CHECK(BLOCK_SIZE % WARP_SIZE == 0);

    const uint64 total_elements = x.numel();

    AT_DISPATCH_CUSTOM_FLOAT_TYPES(
        x.scalar_type(), "add_scalar_cuda_kernel", ([&] {
            const uint32 num_elements_per_thread = 16 / sizeof(scalar_t);
            const uint32 num_elements_per_block = BLOCK_SIZE * num_elements_per_thread;

            std::vector<cute_kernels::ChunkedArray<scalar_t>> x_chunks =
                cute_kernels::chunk_array<scalar_t>(x.data_ptr<scalar_t>(), total_elements);
            std::vector<cute_kernels::ChunkedArray<scalar_t>> output_chunks =
                cute_kernels::chunk_array<scalar_t>(output.data_ptr<scalar_t>(), total_elements);

            for (int i = 0; i < x_chunks.size(); i++) {
                cute_kernels::ChunkedArray<scalar_t> x_chunk = x_chunks[i];
                cute_kernels::ChunkedArray<scalar_t> output_chunk = output_chunks[i];

                const uint64 num_elements = x_chunk.num_elements;
                const uint32 NUM_BLOCKS = cute_kernels::ceil_divide<uint64>(num_elements, num_elements_per_block);

                if constexpr (std::is_same_v<scalar_t, fp32>) {
                    _add_scalar_cuda_kernel<scalar_t>
                        <<<NUM_BLOCKS, BLOCK_SIZE>>>(x_chunk.array, y, output_chunk.array, num_elements);
                } else {
                    _add_scalar_cuda_kernel<scalar_t>
                        <<<NUM_BLOCKS, BLOCK_SIZE>>>(x_chunk.array, y, output_chunk.array, num_elements);
                }
            }
        }));
}
