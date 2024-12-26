#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include "../../../../include/dtypes/all.h"
#include "../../../../include/launch.h"
#include "../../../../include/math.h"
#include "../../../../include/threads.h"

template <typename scalar_t, typename vector_t>
__global__ void _add_scalar_forward_cuda_kernel(const scalar_t *x,
                                                const fp32 y,
                                                scalar_t *output,
                                                const uint num_elements) {
    constexpr int vector_instruction_width = sizeof(vector_t) / sizeof(scalar_t);
    static_assert(vector_instruction_width == 1 || vector_instruction_width == 2 || vector_instruction_width == 4 ||
                  vector_instruction_width == 8);

    using dtype = DType<scalar_t>;
    using T = typename dtype::nv_dtype;
    using T2 = typename dtype::nv_dtype2;

    const uint thread_id = get_global_thread_id();

    if constexpr (vector_instruction_width == 1) {
        if (thread_id < num_elements) {
            output[thread_id] = x[thread_id] + y;
        }
    } else {
        uint end = (thread_id + 1) * vector_instruction_width - 1;  // inclusive of last element

        if (end < num_elements) {
            vector_t *output_vec = (vector_t *)output;

            if constexpr (std::is_same_v<scalar_t, fp32>) {
                const fp32 *x_vec = (fp32 *)&((vector_t *)x)[thread_id];
                fp32 output_buffer[vector_instruction_width];

                // clang-format off
                #pragma unroll
                // clang-format on
                for (int i = 0; i < vector_instruction_width; i++) {
                    output_buffer[i] = x_vec[i] + y;
                }

                if constexpr (vector_instruction_width == 2) {
                    output_vec[thread_id] = dtype::make2(output_buffer);
                } else if constexpr (vector_instruction_width == 4) {
                    output_vec[thread_id] = dtype::make4(output_buffer);
                } else {
                    static_assert("vector_instruction_width is invalid for fp32");
                }
            } else {
                if constexpr (vector_instruction_width == 2) {
                    const T2 _x = ((vector_t *)x)[thread_id];
                    fp32_2 _x_upcast = dtype::upcast(_x);

                    _x_upcast = DType<fp32>::make2(_x_upcast.x + y, _x_upcast.y + y);
                    output_vec[thread_id] = dtype::downcast(_x_upcast);
                } else {
                    const fp32 *x_vec = (fp32 *)&((vector_t *)x)[thread_id];

                    constexpr int n = vector_instruction_width >> 1;
                    fp32 output_buffer[n];

                    // clang-format off
                    #pragma unroll
                    // clang-format on
                    for (int i = 0; i < n; i++) {
                        fp32_2 _x_upcast = dtype::upcast(dtype::reinterpret_32_bits_as_2x16(x_vec[i]));
                        _x_upcast = DType<fp32>::make2(_x_upcast.x + y, _x_upcast.y + y);
                        output_buffer[i] = dtype::reinterpret_2x16_as_32_bits(dtype::downcast(_x_upcast));
                    }

                    if constexpr (vector_instruction_width == 4) {
                        output_vec[thread_id] = DType<fp32>::make2(output_buffer);
                    } else if constexpr (vector_instruction_width == 8) {
                        output_vec[thread_id] = DType<fp32>::make4(output_buffer);
                    } else {
                        static_assert("vector_instruction_width is invalid for fp16 & bf16");
                    }
                }
            }
        }

        // use first warp for computing the last elements
        if (thread_id < WARP_SIZE) {
            // NOTE end is same as start since we don't use vector load stores here
            end = (num_elements / vector_instruction_width) * vector_instruction_width + thread_id;
            if (end < num_elements) {
                output[end] = x[end] + y;
            }
        }
    }
}

void add_scalar_forward_cuda(const torch::Tensor &x,
                             const float &y,
                             torch::Tensor &output,
                             const int &vector_instruction_width,
                             const int &BLOCK_SIZE) {
    assert(BLOCK_SIZE % WARP_SIZE == 0);
    const uint64 total_elements = x.numel();

    AT_DISPATCH_CUSTOM_FLOAT_TYPES(
        x.scalar_type(), "add_scalar_forward_cuda_kernel", ([&] {
            std::vector<ChunkedArray<scalar_t>> x_chunked =
                chunk_array<scalar_t>(x.data_ptr<scalar_t>(), total_elements);
            std::vector<ChunkedArray<scalar_t>> output_chunked =
                chunk_array<scalar_t>(output.data_ptr<scalar_t>(), total_elements);

            const uint num_elements = x_chunked.num_elements;

            scalar_t *x_chunk = x_chunked.array;
            scalar_t *output_chunk = output_chunked.array;

            const uint num_elements_per_block = BLOCK_SIZE * vector_instruction_width;
            const uint NUM_BLOCKS = ceil_divide<uint>(num_elements, num_elements_per_block);

            for (int i = 0; i < x_chunked.size(); i++) {
                ChunkedArray<scalar_t> x_chunk = x_chunked[i];
                ChunkedArray<scalar_t> output_chunk = output_chunked[i];

                switch (vector_instruction_width) {
                    case 1:
                        _add_scalar_forward_cuda_kernel<scalar_t, scalar_t>
                            <<<NUM_BLOCKS, BLOCK_SIZE>>>(x_chunk.array, y, output_chunk.array, x_chunk.num_elements);
                        break;
                    case 2:
                        using vector_t = typename DType<scalar_t>::nv_dtype2;
                        _add_scalar_forward_cuda_kernel<scalar_t, vector_t>
                            <<<NUM_BLOCKS, BLOCK_SIZE>>>(x_chunk.array, y, output_chunk.array, x_chunk.num_elements);
                        break;
                    case 4:
                        if constexpr (std::is_same_v<scalar_t, fp32>) {
                            _add_scalar_forward_cuda_kernel<scalar_t, fp32_4><<<NUM_BLOCKS, BLOCK_SIZE>>>(
                                x_chunk.array, y, output_chunk.array, x_chunk.num_elements);
                        } else {
                            _add_scalar_forward_cuda_kernel<scalar_t, fp32_2><<<NUM_BLOCKS, BLOCK_SIZE>>>(
                                x_chunk.array, y, output_chunk.array, x_chunk.num_elements);
                        }
                        break;
                    case 8:
                        if constexpr (std::is_same_v<scalar_t, fp32>) {
                            _add_scalar_forward_cuda_kernel<scalar_t, fp64_4><<<NUM_BLOCKS, BLOCK_SIZE>>>(
                                x_chunk.array, y, output_chunk.array, x_chunk.num_elements);
                        } else {
                            _add_scalar_forward_cuda_kernel<scalar_t, fp32_4><<<NUM_BLOCKS, BLOCK_SIZE>>>(
                                x_chunk.array, y, output_chunk.array, x_chunk.num_elements);
                        }
                        break;
                    default:
                        throw std::runtime_error("invalid vector_instruction_width");
                        break;
                }
            }
        }));
}
